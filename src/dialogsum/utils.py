import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# 프로젝트 루트 절대 경로
ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> str:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    return str(p.resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    model_name: str
    model_type: str
    encoder_max_len: int
    decoder_max_len: int
    learning_rate: float
    batch_size: int
    num_train_epochs: int
    fp16: bool = True
    early_stopping_patience: int | None = 3
    eval_steps: int = 500
    save_steps: int = 500
    style_prompt: str | None = None
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    max_steps: int | None = None


def get_checkpoint_dir(base_dir: str, run_name: str) -> str:
    base_dir = resolve_path(base_dir)
    path = os.path.join(base_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_prediction_path(base_dir: str, filename: str) -> str:
    base_dir = resolve_path(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def get_dialogue_special_tokens() -> List[str]:
    return [
        "#Person1#",
        "#Person2#",
        "#Person3#",
        "#Person4#",
        "#Person5#",
        "#Person6#",
        "#Person7#",
        "#SSN#",
        "#Email#",
        "#Address#",
        "#Reaction#",
        "#CarNumber#",
        "#Movietitle#",
        "#DateOfBirth#",
        "#CardNumber#",
        "#PhoneNumber#",
        "#PassportNumber#",
    ]


def get_next_run_index(prediction_dir: str, date_prefix: str) -> str:
    """
    날짜(YYMMDD)별로 001~999까지 순차적인 index를 반환한다.
    예) 251129 + 001 + _kobart-base-style_prompt.csv
    """
    os.makedirs(prediction_dir, exist_ok=True)
    max_idx = 0
    for fname in os.listdir(prediction_dir):
        if not fname.startswith(date_prefix):
            continue
        if len(fname) < 9:
            continue
        idx_str = fname[6:9]
        if idx_str.isdigit():
            max_idx = max(max_idx, int(idx_str))
    return f"{max_idx + 1:03d}"
