import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import re
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ensure local package import
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from dialogsum.data import DataConfig, chunk_dialogues, load_csv_splits
from dialogsum.train import run_sft_training
from dialogsum.utils import (
    ModelConfig,
    get_next_run_index,
    get_prediction_path,
    load_yaml_config,
    resolve_path,
)
from dialogsum.templates import encoder_templates, cot_templates
from inference.generate import run_generation, run_generation_with_references

def _merge_predictions(df: pd.DataFrame) -> pd.DataFrame:
    def split_sentences(text: str):
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

    groups = {}
    base_names = df["base_fname"] if "base_fname" in df.columns else df["fname"]
    expected = len(pd.unique(base_names))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Full dialogue summarization pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--prediction_name",
        type=str,
        default=None,
        help="Optional manual prediction filename.",
    )
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=None,
        help="Optional batch size override for this run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional custom run name (suffix).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience override.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation steps override.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save steps override.",
    )
    parser.add_argument(
        "--len_weight",
        type=float,
        default=None,
        help="Length-based sampling weight for top pct of dialogues.",
    )
    parser.add_argument(
        "--len_top_pct",
        type=float,
        default=0.3,
        help="Top percentage (0-1) of longest dialogues to weight.",
    )
    parser.add_argument(
        "--teacher_preds",
        type=str,
        default=None,
        help="Path to teacher prediction CSV for KD (must have fname, summary).",
    )
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=None,
        help="KD mixing weight (0-1).",
    )
    parser.add_argument(
        "--enc_key",
        type=str,
        default=None,
        help="Encoder template key (see src/dialogsum/templates.py).",
    )
    parser.add_argument(
        "--cot_key",
        type=str,
        default=None,
        help="Decoder CoT template key (see src/dialogsum/templates.py).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # 한국 시간 기준 날짜 + 인덱스로 run_id 생성 (체크포인트/파일명 공통 prefix)
    kst = timezone(timedelta(hours=9))
    date_prefix = datetime.now(kst).strftime("%y%m%d")
    prediction_dir = cfg.get("paths", {}).get("prediction_dir", "prediction")
    run_idx = get_next_run_index(prediction_dir, date_prefix)
    base_name = args.run_name if args.run_name else "kobart-base-style_prompt"
    run_id = f"{date_prefix}{run_idx}_{base_name}"

    model_cfg = ModelConfig(
        model_name=cfg["model"]["name"],
        model_type=cfg["model"]["type"],
        encoder_max_len=cfg["data"]["encoder_max_len"],
        decoder_max_len=cfg["data"]["decoder_max_len"],
        learning_rate=cfg["train"]["learning_rate"],
        batch_size=cfg["train"]["batch_size"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        fp16=cfg["train"].get("fp16", True),
        early_stopping_patience=cfg["train"].get("early_stopping_patience", 3),
        eval_steps=cfg["train"].get("eval_steps", 500),
        save_steps=cfg["train"].get("save_steps", 500),
        style_prompt=cfg.get("style_prompt", None),
        encoder_template=cfg.get("encoder_template", None),
        decoder_prefix=cfg.get("decoder_prefix", None),
        use_wandb=cfg.get("wandb", {}).get("use_wandb", False),
        wandb_project=cfg.get("wandb", {}).get("project", None),
        wandb_entity=cfg.get("wandb", {}).get("entity", None),
        wandb_run_name=run_id,
        max_steps=cfg["train"].get("max_steps"),
        kd_alpha=cfg["train"].get("kd_alpha"),
        teacher_preds_path=cfg.get("train", {}).get("teacher_preds_path"),
    )

    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    checkpoint_dir = resolve_path(checkpoint_dir)
    run_dir = os.path.join(checkpoint_dir, run_id)

    if args.batch_size_override is not None:
        model_cfg.batch_size = args.batch_size_override
    if args.patience is not None:
        model_cfg.early_stopping_patience = args.patience
    if args.kd_alpha is not None:
        model_cfg.kd_alpha = args.kd_alpha
    if args.teacher_preds is not None:
        model_cfg.teacher_preds_path = args.teacher_preds
    if args.eval_steps is not None:
        model_cfg.eval_steps = args.eval_steps
    if args.save_steps is not None:
        model_cfg.save_steps = args.save_steps
    if args.patience is not None:
        model_cfg.early_stopping_patience = args.patience
    if args.enc_key:
        if args.enc_key not in encoder_templates:
            raise ValueError(f"Unknown enc_key={args.enc_key}")
        model_cfg.encoder_template = encoder_templates[args.enc_key]
    if args.cot_key:
        if args.cot_key not in cot_templates:
            raise ValueError(f"Unknown cot_key={args.cot_key}")
        model_cfg.decoder_prefix = cot_templates[args.cot_key]

    data_cfg = DataConfig()
    data_cfg.truncate_tail = cfg.get("data", {}).get("truncate_tail", False)
    data_cfg.chunk_overlap = cfg.get("data", {}).get("chunk_overlap", False)
    data_cfg.chunk_overlap_tokens = cfg.get("data", {}).get("chunk_overlap_tokens", data_cfg.chunk_overlap_tokens)
    # optional custom data paths
    dp = cfg.get("data_paths", {})
    if dp:
        data_cfg.train_path = dp.get("train", data_cfg.train_path)
        data_cfg.dev_path = dp.get("dev", data_cfg.dev_path)
        data_cfg.test_path = dp.get("test", data_cfg.test_path)

    model, tokenizer = run_sft_training(
        model_cfg=model_cfg,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        checkpoint_base_dir=checkpoint_dir,
        data_cfg=data_cfg,
        use_wandb=model_cfg.use_wandb,
        wandb_project=model_cfg.wandb_project,
        wandb_run_name=model_cfg.wandb_run_name,
        len_weight=args.len_weight,
        len_top_pct=args.len_top_pct,
        teacher_preds_path=model_cfg.teacher_preds_path,
        kd_alpha=model_cfg.kd_alpha,
    )

    splits = load_csv_splits(data_cfg)
    chunk_kwargs = dict(
        tokenizer=tokenizer,
        encoder_max_len=model_cfg.encoder_max_len,
        chunk_overlap=data_cfg.chunk_overlap,
        chunk_overlap_tokens=data_cfg.chunk_overlap_tokens,
    )
    train_df = chunk_dialogues(splits["train"], **chunk_kwargs)
    dev_df = chunk_dialogues(splits["dev"], **chunk_kwargs)
    test_df = chunk_dialogues(splits["test"], **chunk_kwargs)
    test_df_original = splits["test"]

    pred_df = run_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        encoder_template=model_cfg.encoder_template,
        decoder_prefix=model_cfg.decoder_prefix,
        model_type=model_cfg.model_type,
        batch_size=model_cfg.batch_size,
        beam_size=4,
        max_new_tokens=96,
        min_length=30,
        repetition_penalty=1.1,
        length_penalty=1.0,
    )

    # 파일명 구성
    if args.prediction_name:
        chunked_filename = args.prediction_name
    else:
        chunked_filename = f"{run_id}_chunked.csv"

    # chunked prediction 저장
    chunked_path = get_prediction_path(prediction_dir, chunked_filename)
    pred_df.to_csv(chunked_path, index=False, encoding="utf-8-sig")
    print(f"Saved chunked prediction to {chunked_path}")

    # chunk 예측을 base fname으로 병합 (원본 행 수로 복원)
    merged_df = _merge_predictions(pred_df)
    merged_filename = chunked_filename.replace(".csv", "_merged.csv")
    merged_path = get_prediction_path(prediction_dir, merged_filename)
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged prediction to {merged_path}")

    # no-overlap (원본 test) 예측 추가 저장
    nooverlap_df = run_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df_original,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        encoder_template=model_cfg.encoder_template,
        decoder_prefix=model_cfg.decoder_prefix,
        model_type=model_cfg.model_type,
        batch_size=model_cfg.batch_size,
        beam_size=4,
        max_new_tokens=96,
        min_length=30,
        repetition_penalty=1.1,
        length_penalty=1.0,
    )
    # 제출 규격: fname, summary 두 컬럼만 유지
    nooverlap_df = nooverlap_df[["fname", "summary"]]
    nooverlap_filename = chunked_filename.replace("_chunked", "_nooverlap")
    nooverlap_path = get_prediction_path(prediction_dir, nooverlap_filename)
    nooverlap_df.to_csv(nooverlap_path, index=False, encoding="utf-8-sig")
    print(f"Saved no-overlap prediction to {nooverlap_path}")

    # 레퍼런스 포함 요약 결과 저장 (EDA용): dev + train
    full_pred_dir = resolve_path(cfg.get("paths", {}).get("full_pred_dir", "prediction_full"))
    os.makedirs(full_pred_dir, exist_ok=True)

    # dev
    dev_pred_df = run_generation_with_references(
        model=model,
        tokenizer=tokenizer,
        df=dev_df,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        encoder_template=model_cfg.encoder_template,
        decoder_prefix=model_cfg.decoder_prefix,
        model_type=model_cfg.model_type,
        batch_size=model_cfg.batch_size,
        beam_size=4,
        max_new_tokens=96,
        min_length=30,
    )
    dev_out_path = os.path.join(full_pred_dir, f"{run_id}_dev_full.csv")
    dev_pred_df.to_csv(dev_out_path, index=False, encoding="utf-8-sig")
    print(f"Saved dev predictions with references to {dev_out_path}")


if __name__ == "__main__":
    main()
