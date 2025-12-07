import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from dialogsum.data import DataConfig, chunk_dialogues, load_csv_splits
from inference.generate import run_generation


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_predictions(df: pd.DataFrame) -> pd.DataFrame:
    def split_sentences(text: str):
        import re
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def is_similar(a: str, b: str, thresh: float = 0.8) -> bool:
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return False
        return len(a_set & b_set) / len(a_set | b_set) >= thresh

    def dedup(sentences):
        kept = []
        for s in sentences:
            if any(is_similar(s, k) for k in kept):
                continue
            kept.append(s)
            if len(kept) >= 3:
                break
        return kept

    base_names = df["base_fname"] if "base_fname" in df.columns else df["fname"]
    expected = len(pd.unique(base_names))
    groups = {}
    for _, row in df.iterrows():
        base = str(row.get("base_fname", row["fname"]))
        groups.setdefault(base, []).append(str(row["summary"]))

    merged_rows = []
    for base, summaries in groups.items():
        sentences = []
        for s in summaries:
            sentences.extend(split_sentences(s))
        sentences = [s for s in sentences if s]
        sentences = dedup(sentences)
        merged_rows.append({"fname": base, "summary": " ".join(sentences)})
    merged_df = pd.DataFrame(merged_rows)
    if len(merged_df) != expected:
        raise ValueError(
            f"병합된 예측 개수({len(merged_df)})가 기대 개수({expected})와 다릅니다. chunk 처리/중복 제거 로직을 확인하세요."
        )
    return merged_df


def main():
    parser = argparse.ArgumentParser(description="Generate summaries from a trained decoder-only QLoRA checkpoint")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint directory")
    parser.add_argument("--run_name", default=None, help="Optional run name suffix for filename")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg["model"]["name"]
    encoder_max_len = cfg["data"]["encoder_max_len"]
    decoder_max_len = cfg["data"]["decoder_max_len"]
    prediction_dir = cfg.get("paths", {}).get("prediction_dir", "prediction")
    os.makedirs(prediction_dir, exist_ok=True)

    kst = timezone(timedelta(hours=9))
    date_prefix = datetime.now(kst).strftime("%y%m%d")
    run_name = args.run_name or Path(args.checkpoint).name
    base_filename = f"{date_prefix}_{run_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 모델 안전하게 left padding
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    data_cfg = DataConfig()
    data_cfg.train_path = cfg["data"]["train_path"]
    data_cfg.dev_path = cfg["data"]["dev_path"]
    data_cfg.test_path = cfg["data"]["test_path"] if "test_path" in cfg["data"] else str(
        Path(cfg["data"]["train_path"]).parent / "test.csv"
    )
    data_cfg.chunk_overlap = cfg["data"].get("chunk_overlap", False)
    data_cfg.chunk_overlap_tokens = cfg["data"].get("chunk_overlap_tokens", 128)

    splits = load_csv_splits(data_cfg)
    test_df_original = splits["test"]
    chunk_kwargs = dict(
        tokenizer=tokenizer,
        encoder_max_len=encoder_max_len,
        chunk_overlap=data_cfg.chunk_overlap,
        chunk_overlap_tokens=data_cfg.chunk_overlap_tokens,
    )
    test_df_chunked = chunk_dialogues(test_df_original, **chunk_kwargs)

    # chunked prediction
    chunked_df = run_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df_chunked,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=cfg.get("style_prompt"),
        model_type="decoder-only",
        batch_size=args.batch_size,
        beam_size=4,
        repetition_penalty=1.1,
        length_penalty=1.0,
    )
    chunked_path = os.path.join(prediction_dir, f"{base_filename}_chunked.csv")
    chunked_df.to_csv(chunked_path, index=False, encoding="utf-8-sig")
    print(f"Saved chunked predictions to {chunked_path}")

    merged_df = _merge_predictions(chunked_df)
    merged_path = os.path.join(prediction_dir, f"{base_filename}_chunked_merged.csv")
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged predictions to {merged_path}")

    # no-overlap prediction
    nooverlap_df = run_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df_original,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=cfg.get("style_prompt"),
        model_type="decoder-only",
        batch_size=args.batch_size,
        beam_size=4,
        repetition_penalty=1.1,
        length_penalty=1.0,
    )
    nooverlap_df = nooverlap_df[["fname", "summary"]]
    nooverlap_path = os.path.join(prediction_dir, f"{base_filename}_nooverlap.csv")
    nooverlap_df.to_csv(nooverlap_path, index=False, encoding="utf-8-sig")
    print(f"Saved no-overlap predictions to {nooverlap_path}")


if __name__ == "__main__":
    main()
