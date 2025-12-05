import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ensure absolute import for local package
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))
from dialogsum.data import chunk_dialogues  # noqa: E402


@dataclass
class QLoRAConfig:
    model_name: str
    style_prompt: str
    encoder_max_len: int
    decoder_max_len: int
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    save_steps: int
    eval_steps: int
    logging_steps: int
    bf16: bool
    max_steps: Optional[int]
    early_stopping_patience: Optional[int]
    output_dir: str
    train_path: str
    dev_path: str
    chunk_overlap: bool
    chunk_overlap_tokens: int


def load_config(path: str) -> QLoRAConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    train_path = Path(data["train_path"]).expanduser().resolve()
    dev_path = Path(data["dev_path"]).expanduser().resolve()
    output_dir = Path(cfg["paths"]["checkpoint_dir"]).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)
    return QLoRAConfig(
        model_name=cfg["model"]["name"],
        style_prompt=cfg["style_prompt"],
        encoder_max_len=data["encoder_max_len"],
        decoder_max_len=data["decoder_max_len"],
        learning_rate=cfg["train"]["learning_rate"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        lora_r=cfg["train"]["lora_r"],
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        save_steps=cfg["train"]["save_steps"],
        eval_steps=cfg["train"]["eval_steps"],
        logging_steps=cfg["train"].get("logging_steps", 50),
        bf16=cfg["train"].get("bf16", True),
        max_steps=cfg["train"].get("max_steps"),
        early_stopping_patience=cfg["train"].get("early_stopping_patience"),
        output_dir=str(output_dir),
        train_path=str(train_path),
        dev_path=str(dev_path),
        chunk_overlap=data.get("chunk_overlap", False),
        chunk_overlap_tokens=data.get("chunk_overlap_tokens", 128),
    )


def build_prompt(style_prompt: str, dialogue: str) -> str:
    return f"{style_prompt}\n\n대화:\n{dialogue}\n\n답변:"


def make_sft_dataset(
    df: pd.DataFrame,
    tokenizer,
    cfg: QLoRAConfig,
) -> Dataset:
    texts = [build_prompt(cfg.style_prompt, str(row["dialogue"])) + " " + str(row.get("summary", "")) for _, row in df.iterrows()]
    return Dataset.from_dict({"text": texts})


def prepare_data(tokenizer, cfg: QLoRAConfig) -> Dict[str, Dataset]:
    train_df = pd.read_csv(cfg.train_path)
    dev_df = pd.read_csv(cfg.dev_path)
    chunk_kwargs = dict(
        tokenizer=tokenizer,
        encoder_max_len=cfg.encoder_max_len,
        chunk_overlap=cfg.chunk_overlap,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
    )
    train_df = chunk_dialogues(train_df, **chunk_kwargs)
    dev_df = chunk_dialogues(dev_df, **chunk_kwargs)
    return {
        "train": make_sft_dataset(train_df, tokenizer, cfg),
        "dev": make_sft_dataset(dev_df, tokenizer, cfg),
    }


def load_model_and_tokenizer(model_name: str, bnb_config: BitsAndBytesConfig, auth_token: Optional[str]):
    """
    모델별 공식 로딩 가이드를 반영한 분기 로더.
    """
    common_model_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=auth_token,
    )

    if model_name.startswith("naver-hyperclovax/"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_model_kwargs)
    elif model_name.startswith("kakaocorp/kanana-"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **common_model_kwargs,
        )
    elif model_name.startswith("beomi/gemma-ko-"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_model_kwargs)
    elif model_name.startswith("42dot/42dot_LLM-SFT-1.3B"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_model_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **common_model_kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer.padding_side or "right"
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="절대경로 YAML 설정 파일")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="스모크 테스트 등 짧은 학습용 max_steps override",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
        cfg.num_train_epochs = 1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )

    auth_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, bnb_config, auth_token)

    datasets = prepare_data(tokenizer, cfg)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None if cfg.model_name.startswith("beomi/gemma-ko-") else ["q_proj", "k_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.config.use_cache = False

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.encoder_max_len,
            padding="max_length",
        )

    tokenized_train = datasets["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dev = datasets["dev"].map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs if cfg.num_train_epochs is not None else 100,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        bf16=cfg.bf16,
        max_steps=-1,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience or 4)],
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
