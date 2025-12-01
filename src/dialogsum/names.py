from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def normalize_names_in_dialogue(text: str) -> str:
    """
    이름 표기 정규화 훅 (dialogue용).

    - 현재는 아무 작업도 하지 않고 입력을 그대로 반환합니다.
    - 나중에 '존' -> 'John' 또는 그 반대와 같은 규칙을 여기에 구현하면 됩니다.
    """
    return text


def normalize_names_in_summary(text: str) -> str:
    """
    이름 표기 정규화 훅 (summary용).

    - 현재는 아무 작업도 하지 않고 입력을 그대로 반환합니다.
    - 학습/평가/추론에서 동일한 규칙을 적용할 수 있도록 이 함수만 수정하면 됩니다.
    """
    return text


@dataclass
class TextSplitPaths:
    train_src: Path
    train_out: Path


def create_train_text_csv(paths: TextSplitPaths) -> None:
    """
    전처리 실험을 위한 학습용 CSV (`train_text.csv`)를 생성합니다.

    - 원본 train.csv는 절대 수정/덮어쓰지 않습니다.
    - 현재는 normalize_* 훅을 거쳐서 그대로 복사하는 형태로만 동작합니다.
    - 나중에 이름 정규화 규칙을 넣으면 이 함수만 다시 실행하면 됩니다.
    """
    df = pd.read_csv(paths.train_src)

    if "dialogue" in df.columns:
        df["dialogue"] = df["dialogue"].astype(str).map(normalize_names_in_dialogue)
    if "summary" in df.columns:
        df["summary"] = df["summary"].astype(str).map(normalize_names_in_summary)

    paths.train_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.train_out, index=False)

