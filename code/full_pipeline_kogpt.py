import argparse
from pathlib import Path

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
    """
    KoGPT용 프롬프트 템플릿.

    - 시스템 역할 + 사용자 역할을 명시해서 decoder-only 모델이 요약 태스크를 이해할 수 있게 합니다.
    """
    system = "당신은 한국어 대화 요약 비서이다.\n대화를 읽고, 한두 문장으로 핵심 내용을 간결하게 요약하라.\n"
    user = f"요약: 대화의 핵심만 간결하게 한두 문장으로 정리하시오.\n\n{dialogue.strip()}\n"
    if summary is None:
        assistant = ""
    else:
        assistant = summary.strip()
    return f"[시스템]\n{system}\n[사용자]\n{user}\n[답변]\n{assistant}"


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
    parser = argparse.ArgumentParser(description="KoGPT LoRA full pipeline")
    parser.add_argument(
        "--model_name",
        type=str,
        default="kakaobrain/kogpt",
        help="KoGPT 모델 이름",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="per_device_train_batch_size",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=768,
        help="입력 최대 토큰 길이",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="학습 epoch 수",
    )
    args = parser.parse_args()

    model_name = args.model_name
    max_input_length = args.max_input_length
    max_new_tokens = 80
    per_device_batch_size = args.batch_size
    grad_accum_steps = 4
    num_train_epochs = args.num_train_epochs
    learning_rate = 2e-4

    print(f"### Load tokenizer & model: {model_name} ###")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # LoRA 설정 (fp16 전용, bitsandbytes 사용 안 함)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],  # GPT2 계열 기본 어텐션/출력 모듈 이름
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

    output_dir = Path("checkpoints/kogpt-lora")
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

    print("### Start training KoGPT LoRA ###")
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
        summaries.append({"fname": fname, "summary": gen_text.strip()})

    pred_df = pd.DataFrame(summaries)
    pred_dir = Path("prediction")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "kogpt_lora_test.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()

