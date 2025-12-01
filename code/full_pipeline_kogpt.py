import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
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
    """KoGPT용 프롬프트 템플릿."""
    system = "당신은 한국어 대화 요약 비서이다.\n대화를 읽고, 한두 문장으로 핵심 내용을 간결하게 요약하라.\n"
    user = f"요약: 대화의 핵심만 간결하게 한두 문장으로 정리하시오.\n\n{dialogue.strip()}\n"
    if summary is None:
        assistant = ""
    else:
        assistant = summary.strip()
    return f"[시스템]\n{system}\n[사용자]\n{user}\n[답변]\n{assistant}"


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return {"train": train_df, "dev": dev_df, "test": test_df}


def make_sft_dataset(df: pd.DataFrame) -> Dataset:
    texts = []
    prompts_only = []
    for _, row in df.iterrows():
        texts.append(build_prompt(row["dialogue"], row["summary"]))
        prompts_only.append(build_prompt(row["dialogue"], None))
    return Dataset.from_dict({"text": texts, "prompt_only": prompts_only})


def make_infer_dataset(df: pd.DataFrame) -> Dataset:
    prompts = [build_prompt(row["dialogue"], None) for _, row in df.iterrows()]
    fnames = df["fname"].tolist()
    return Dataset.from_dict({"prompt": prompts, "fname": fnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="KoGPT LoRA full pipeline")
    parser.add_argument("--model_name", type=str, default="kakaobrain/kogpt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--run_both", action="store_true", help="cutoff 미적용/적용 두 버전을 순차 실행")
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
    # pad_token이 없는 모델이 많으므로 eos_token으로 대체해 패딩/라벨 계산 안정화
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 구조에 따라 LoRA 타겟 모듈이 다르다.
    # - GPT2 계열: c_attn, c_proj
    # - GPT-NeoX/polyglot 계열: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
    lm_lower = model_name.lower()
    if "polyglot" in lm_lower or "neox" in lm_lower:
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        target_modules = ["c_attn", "c_proj"]

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    data_dir = Path("data")
    splits = load_data(data_dir)
    train_dataset = make_sft_dataset(splits["train"])
    dev_dataset = make_sft_dataset(splits["dev"])
    test_dataset = make_infer_dataset(splits["test"])

    def tokenize_full(batch):
        return tokenizer(
            batch["text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

    def tokenize_prompt(batch):
        return tokenizer(
            batch["prompt_only"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

    train_tok_full = train_dataset.map(tokenize_full, batched=True, remove_columns=["text"])
    dev_tok_full = dev_dataset.map(tokenize_full, batched=True, remove_columns=["text"])
    train_tok_prompt = train_dataset.map(tokenize_prompt, batched=True, remove_columns=["prompt_only"])
    dev_tok_prompt = dev_dataset.map(tokenize_prompt, batched=True, remove_columns=["prompt_only"])

    def apply_labels(full_ds, prompt_ds, use_cutoff: bool):
        full = full_ds
        prompt = prompt_ds
        labels = []
        for f_ids, p_ids in zip(full["input_ids"], prompt["input_ids"]):
            lab = f_ids.copy()
            if use_cutoff:
                cutoff = 0
                for tok in p_ids:
                    if tok == tokenizer.pad_token_id:
                        break
                    cutoff += 1
                lab[:cutoff] = [-100] * cutoff
            labels.append(lab)
        return full.add_column("labels", labels)

    def run_experiment(tag: str, use_cutoff: bool):
        exp_name = f"kogpt-lora-{tag}"
        output_dir = Path("checkpoints") / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)

        train_tok = apply_labels(train_tok_full, train_tok_prompt, use_cutoff)
        dev_tok = apply_labels(dev_tok_full, dev_tok_prompt, use_cutoff)

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
            bf16=False,
            max_grad_norm=1.0,
            gradient_checkpointing=False,  # OOM 위험 낮추려면 max_input_length 더 줄이기
            load_best_model_at_end=True,
            report_to=[],
        )

        # 이미 라벨이 포함되어 있으므로 별도 collator 없이 그대로 사용
        data_collator = None

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=dev_tok,
            data_collator=data_collator,
        )

        print(f"\n### Start training KoGPT LoRA ({tag}, cutoff={use_cutoff}) ###")
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # dev ROUGE 평가
        rouge = evaluate.load("rouge")
        preds, refs = [], []
        for row in dev_dataset:
            prompt = row["prompt_only"]
            summary = row["text"].split("[답변]\n")[-1].strip()
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
            preds.append(gen_text.strip())
            refs.append(summary)
        result = rouge.compute(predictions=preds, references=refs)
        result = {k: round(v * 100, 2) for k, v in result.items()}
        print(f"[{tag}] Dev ROUGE: {result}")

        # test inference
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
            summaries.append({"fname": fname, "summary": gen_text.strip()})

        pred_df = pd.DataFrame(summaries)
        pred_dir = Path("prediction")
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_path = pred_dir / f"{exp_name}.csv"
        pred_df.to_csv(out_path, index=False)
        print(f"Saved prediction to {out_path}")
        return result

    results: List[Tuple[str, Dict[str, float]]] = []
    if args.run_both:
        res_no = run_experiment("nocut", use_cutoff=False)
        results.append(("nocut", res_no))
        res_cut = run_experiment("cut", use_cutoff=True)
        results.append(("cut", res_cut))
        print("\n=== 비교 결과 (Dev ROUGE) ===")
        for tag, res in results:
            print(tag, res)
    else:
        res = run_experiment("nocut", use_cutoff=False)
        print("\n=== 단일 실행 결과 (Dev ROUGE) ===", res)


if __name__ == "__main__":
    main()
