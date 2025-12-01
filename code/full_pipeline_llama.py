import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_prompt(dialogue: str, summary: str | None = None) -> str:
    """Llama3용 프롬프트 템플릿 (decoder-only)."""
    system = "당신은 한국어 대화 요약 비서이다.\n대화를 읽고, 한두 문장으로 핵심 내용을 간결하게 요약하라.\n"
    user = f"요약: 대화의 핵심만 간결하게 한두 문장으로 정리하시오.\n\n{dialogue.strip()}\n"
    assistant = "" if summary is None else summary.strip()
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(data_dir / "train.csv"),
        "dev": pd.read_csv(data_dir / "dev.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
    }


def make_sft_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [build_prompt(r["dialogue"], r["summary"]) for _, r in df.iterrows()],
        }
    )


def make_infer_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [build_prompt(r["dialogue"], None) for _, r in df.iterrows()],
            "fname": df["fname"].tolist(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama3 8B fp16 + LoRA pipeline")
    parser.add_argument("--batch_size_override", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=896)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B"
    max_input_length = args.max_input_length  # 프롬프트+대화(+요약) 전체 길이 상한
    max_new_tokens = 80
    per_device_batch_size = args.batch_size_override
    grad_accum_steps = 8
    num_train_epochs = args.num_train_epochs
    learning_rate = 2e-4

    # run_id: 파일명/체크포인트 구분용
    kst = timezone(timedelta(hours=9))
    date_prefix = datetime.now(kst).strftime("%y%m%d")
    run_id = f"{date_prefix}_llama3-lora-fp16"

    print("### Load tokenizer & model (fp16) ###")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # gradient_checkpointing은 메모리 절약용 옵션인데, 현재 환경에서
    # backward 시 grad_fn이 없어지는 문제가 있어 일단 끕니다. 필요하면
    # TrainingArguments에서도 다시 켤 수 있습니다.

    data_dir = Path("data")
    splits = load_data(data_dir)
    train_dataset = make_sft_dataset(splits["train"])
    dev_dataset = make_sft_dataset(splits["dev"])
    test_dataset = make_infer_dataset(splits["test"])

    def tokenize_sft(batch):
        return tokenizer(
            batch["text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

    train_tokenized = train_dataset.map(tokenize_sft, batched=True, remove_columns=["text"])
    dev_tokenized = dev_dataset.map(tokenize_sft, batched=True, remove_columns=["text"])

    # label cut-off 미적용 (프롬프트+대화+요약 전체에 loss). 필요하면 여기서 cut-off 추가 가능.
    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    train_tokenized = train_tokenized.map(add_labels, batched=True)
    dev_tokenized = dev_tokenized.map(add_labels, batched=True)

    output_dir = Path("checkpoints") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        report_to=["wandb"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        data_collator=data_collator,
    )

    print("### Start training Llama3 LoRA fp16 ###")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("### Inference on test ###")
    model.eval()
    summaries: List[Dict[str, str]] = []
    for row in test_dataset:
        prompt = row["prompt"]
        fname = row["fname"]
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
            )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        if "<|assistant|>" in gen_text:
            summary = gen_text.split("<|assistant|>")[-1].strip()
        else:
            summary = gen_text.strip()
        summaries.append({"fname": fname, "summary": summary})

    pred_df = pd.DataFrame(summaries)
    pred_dir = Path("prediction")
    pred_dir.mkdir(exist_ok=True, parents=True)
    out_path = pred_dir / f"{run_id}.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()
