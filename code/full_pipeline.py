import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dialogsum.data import DataConfig, load_csv_splits
from dialogsum.train import run_sft_training
from dialogsum.utils import (
    ModelConfig,
    get_next_run_index,
    get_prediction_path,
    load_yaml_config,
)
from inference.generate import run_generation, run_generation_with_references


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
        use_wandb=cfg.get("wandb", {}).get("use_wandb", False),
        wandb_project=cfg.get("wandb", {}).get("project", None),
        wandb_entity=cfg.get("wandb", {}).get("entity", None),
        wandb_run_name=run_id,
    )

    if args.batch_size_override is not None:
        model_cfg.batch_size = args.batch_size_override
    if args.patience is not None:
        model_cfg.early_stopping_patience = args.patience
    if args.eval_steps is not None:
        model_cfg.eval_steps = args.eval_steps
    if args.save_steps is not None:
        model_cfg.save_steps = args.save_steps

    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")

    model, tokenizer = run_sft_training(
        model_cfg=model_cfg,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        checkpoint_base_dir=checkpoint_dir,
        use_wandb=model_cfg.use_wandb,
        wandb_project=model_cfg.wandb_project,
        wandb_run_name=model_cfg.wandb_run_name,
        len_weight=args.len_weight,
        len_top_pct=args.len_top_pct,
    )

    data_cfg = DataConfig()
    splits = load_csv_splits(data_cfg)
    dev_df = splits["dev"]
    test_df = splits["test"]

    if cfg["model"]["type"].lower() != "kobart":
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(checkpoint_dir, cfg["model"]["name"]))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, cfg["model"]["name"]))

    pred_df = run_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        model_type=model_cfg.model_type,
        batch_size=model_cfg.batch_size,
    )

    if args.prediction_name:
        filename = args.prediction_name
    else:
        filename = f"{run_id}.csv"

    out_path = get_prediction_path(prediction_dir, filename)
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved prediction to {out_path}")

    # dev셋에 대해 reference와 함께 요약 결과 저장 (학습 품질 확인용)
    dev_pred_df = run_generation_with_references(
        model=model,
        tokenizer=tokenizer,
        df=dev_df,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        model_type=model_cfg.model_type,
        batch_size=model_cfg.batch_size,
    )
    dev_out_path = get_prediction_path(prediction_dir, f"{run_id}_dev_full.csv")
    dev_pred_df.to_csv(dev_out_path, index=False, encoding="utf-8-sig")
    print(f"Saved dev predictions with references to {dev_out_path}")


if __name__ == "__main__":
    main()
