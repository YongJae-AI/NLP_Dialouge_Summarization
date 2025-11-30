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
from inference.generate import run_generation


def main():
    config_path = "config/config_kobart_style_prompt.yaml"
    cfg = load_yaml_config(config_path)

    model_cfg = ModelConfig(
        model_name=cfg["model"]["name"],
        model_type=cfg["model"]["type"],
        encoder_max_len=cfg["data"]["encoder_max_len"],
        decoder_max_len=cfg["data"]["decoder_max_len"],
        learning_rate=cfg["train"]["learning_rate"],
        batch_size=4,
        num_train_epochs=cfg["train"]["num_train_epochs"],
        fp16=cfg["train"].get("fp16", True),
        style_prompt=cfg.get("style_prompt", None),
        use_wandb=cfg.get("wandb", {}).get("use_wandb", False),
        wandb_project=cfg.get("wandb", {}).get("project", None),
        wandb_run_name=cfg.get("wandb", {}).get("run_name", None),
    )

    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    prediction_dir = cfg.get("paths", {}).get("prediction_dir", "prediction")

    model, tokenizer = run_sft_training(
        model_cfg=model_cfg,
        encoder_max_len=model_cfg.encoder_max_len,
        decoder_max_len=model_cfg.decoder_max_len,
        style_prompt=model_cfg.style_prompt,
        checkpoint_base_dir=checkpoint_dir,
        use_wandb=model_cfg.use_wandb,
        wandb_project=model_cfg.wandb_project,
        wandb_run_name=model_cfg.wandb_run_name or cfg["model"]["name"],
    )

    data_cfg = DataConfig()
    splits = load_csv_splits(data_cfg)
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

    kst = timezone(timedelta(hours=9))
    date_prefix = datetime.now(kst).strftime("%y%m%d")
    idx = get_next_run_index(prediction_dir, date_prefix)
    filename = f"{date_prefix}{idx}_kobart-base-style_prompt_bs4.csv"

    out_path = get_prediction_path(prediction_dir, filename)
    pred_df.to_csv(out_path, index=False)
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()
