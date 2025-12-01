import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_prompt(dialogue: str, summary: str | None = None) -> str:
    system = (
        "당신은 한국어 대화 요약 비서이다.\n"
        "대화를 읽고, 한두 문장으로 핵심 내용을 간결하게 요약하라.\n"
    )
    user = f"요약: 대화의 핵심만 간결하게 한두 문장으로 정리하시오.\n\n{dialogue.strip()}\n"
    if summary is None:
        assistant = ""
    else:
        assistant = summary.strip()
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return {"train": train_df, "dev": dev_df, "test": test_df}


def make_sft_dataset(df: pd.DataFrame) -> Dataset:
    texts = [build_prompt(row["dialogue"], row["summary"]) for _, row in df.iterrows()]
    return Dataset.from_dict({"text": texts})


def make_infer_dataset(df: pd.DataFrame) -> Dataset:
    prompts = [build_prompt(row["dialogue"], None) for _, row in df.iterrows()]
    fnames = df["fname"].tolist()
    return Dataset.from_dict({"prompt": prompts, "fname": fnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=1,
        help="per_device_train_batch_size",
    )
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B"
    max_input_length = 1280
    max_new_tokens = 80
    learning_rate = 2e-4
    per_device_batch_size = args.batch_size_override
    grad_accum_steps = 8
    num_train_epochs = 3

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("### Load tokenizer & 4bit model ###")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

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

    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    train_tokenized = train_tokenized.map(add_labels, batched=True)
    dev_tokenized = dev_tokenized.map(add_labels, batched=True)

    output_dir = Path("checkpoints/llama3-lora")
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
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        data_collator=data_collator,
    )

    print("### Start training LoRA (Llama3) ###")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("### Inference on test ###")
    model.eval()
    summaries: list[dict[str, str]] = []
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
    out_path = pred_dir / "llama3_lora_test.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()

