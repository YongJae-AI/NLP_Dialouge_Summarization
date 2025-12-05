from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .utils import ROOT_DIR, resolve_path


class DialogueSummaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        encoder_max_len: int,
        decoder_max_len: int,
        style_prompt: Optional[str] = None,
        model_type: str = "kobart",
        is_train: bool = True,
        truncate_tail: bool = False,
        teacher_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.style_prompt = style_prompt
        self.model_type = model_type
        self.is_train = is_train
        self.truncate_tail = truncate_tail
        self.teacher_map = teacher_map or {}

    def _trim_dialogue(self, dialogue: str) -> str:
        # 뒤에서부터 턴을 붙여 encoder_max_len을 넘지 않도록 자른다.
        turns = str(dialogue).split("#Person")
        turns = ["#Person" + t for t in turns if t.strip()]
        kept: List[str] = []
        for t in reversed(turns):
            cand = " ".join([t] + kept)
            ids = self.tokenizer(
                cand,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if len(ids) > self.encoder_max_len:
                break
            kept.insert(0, t)
        if not kept:
            return dialogue
        return " ".join(kept)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        base_fname = str(row.get("base_fname", row.get("fname", f"idx_{idx}")))
        chunk_id = int(row.get("chunk_id", 0)) if "chunk_id" in row and pd.notna(row["chunk_id"]) else 0
        dialogue = str(row["dialogue"])
        if self.truncate_tail:
            dialogue = self._trim_dialogue(dialogue)

        if self.style_prompt:
            # QA+스타일 프롬프트를 포함한 정형 입력
            src_text = (
                f"{self.style_prompt}\n\n"
                f"대화:\n{dialogue}\n\n"
                "답변:"
            )
        else:
            src_text = dialogue

        encoded = self.tokenizer(
            src_text,
            max_length=self.encoder_max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        model_inputs: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        if self.is_train:
            target = str(row["summary"])
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target,
                    max_length=self.decoder_max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None,
                )["input_ids"]
            model_inputs["labels"] = labels
            if "fname" in row and row["fname"] in self.teacher_map:
                t_sum = str(self.teacher_map[row["fname"]])
                with self.tokenizer.as_target_tokenizer():
                    t_labels = self.tokenizer(
                        t_sum,
                        max_length=self.decoder_max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors=None,
                    )["input_ids"]
                model_inputs["teacher_labels"] = t_labels

        # fname는 학습/검증 단계에서는 필요 없으므로,
        # is_train=False (주로 inference)일 때만 포함한다.
        if not self.is_train:
            model_inputs["fname"] = base_fname
            model_inputs["base_fname"] = base_fname
            model_inputs["chunk_id"] = chunk_id
            model_inputs["row_idx"] = int(row.get("row_idx", idx))
        return model_inputs


@dataclass
class DataConfig:
    train_path: str = str(ROOT_DIR / "data" / "train.csv")
    dev_path: str = str(ROOT_DIR / "data" / "dev.csv")
    test_path: str = str(ROOT_DIR / "data" / "test.csv")
    truncate_tail: bool = False
    chunk_overlap: bool = False
    chunk_overlap_tokens: int = 128


def load_csv_splits(cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    train_df = pd.read_csv(resolve_path(cfg.train_path))
    dev_df = pd.read_csv(resolve_path(cfg.dev_path))
    test_df = pd.read_csv(resolve_path(cfg.test_path))
    return {"train": train_df, "dev": dev_df, "test": test_df}


def chunk_dialogues(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    encoder_max_len: int,
    chunk_overlap: bool,
    chunk_overlap_tokens: int,
) -> pd.DataFrame:
    """
    턴 단위로 대화를 분할해 enc 길이를 넘지 않도록 잘라내고,
    overlap 구간을 chunk_overlap_tokens 만큼 유지한다.
    - fname은 원본 이름 그대로 유지하고, chunk_id/base_fname/row_idx를 추가한다.
    """
    if not chunk_overlap:
        df = df.copy()
        df["chunk_id"] = 0
        df["base_fname"] = df["fname"]
        df["row_idx"] = df.index
        return df

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        dialogue = str(row["dialogue"])
        base_fname = str(row["fname"])
        row_idx = row.name
        turns = ["#Person" + t for t in dialogue.split("#Person") if t.strip()]
        chunks: List[List[str]] = []
        cur: List[str] = []
        for t in turns:
            cand = " ".join(cur + [t])
            if len(
                tokenizer(
                    cand,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            ) > encoder_max_len:
                if cur:
                    chunks.append(cur)
                    ids = tokenizer(
                        " ".join(cur),
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                    keep = ids[-(encoder_max_len - chunk_overlap_tokens) :] if len(ids) > chunk_overlap_tokens else ids
                    cur = tokenizer.decode(keep, skip_special_tokens=True).split()
                else:
                    chunks.append([t])
                    cur = []
            else:
                cur.append(t)
        if cur:
            chunks.append(cur)
        for idx, ch in enumerate(chunks):
            rec = row.copy()
            rec["dialogue"] = " ".join(ch)
            rec["fname"] = base_fname
            rec["base_fname"] = base_fname
            rec["chunk_id"] = idx
            rec["row_idx"] = row_idx
            rows.append(rec)
    return pd.DataFrame(rows)


def load_datasets(
    tokenizer: PreTrainedTokenizerBase,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    model_type: str,
    data_cfg: Optional[DataConfig] = None,
    teacher_map: Optional[Dict[str, str]] = None,
) -> Dict[str, DialogueSummaryDataset]:
    if data_cfg is None:
        data_cfg = DataConfig()
    splits = load_csv_splits(data_cfg)
    train_df = chunk_dialogues(
        splits["train"],
        tokenizer=tokenizer,
        encoder_max_len=encoder_max_len,
        chunk_overlap=data_cfg.chunk_overlap,
        chunk_overlap_tokens=data_cfg.chunk_overlap_tokens,
    )
    dev_df = chunk_dialogues(
        splits["dev"],
        tokenizer=tokenizer,
        encoder_max_len=encoder_max_len,
        chunk_overlap=data_cfg.chunk_overlap,
        chunk_overlap_tokens=data_cfg.chunk_overlap_tokens,
    )
    test_df = chunk_dialogues(
        splits["test"],
        tokenizer=tokenizer,
        encoder_max_len=encoder_max_len,
        chunk_overlap=data_cfg.chunk_overlap,
        chunk_overlap_tokens=data_cfg.chunk_overlap_tokens,
    )
    train_ds = DialogueSummaryDataset(
        train_df,
        tokenizer,
        encoder_max_len,
        decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=True,
        truncate_tail=data_cfg.truncate_tail,
        teacher_map=teacher_map,
    )
    dev_ds = DialogueSummaryDataset(
        dev_df,
        tokenizer,
        encoder_max_len,
        decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=True,
        truncate_tail=data_cfg.truncate_tail,
        teacher_map=teacher_map,
    )
    test_ds = DialogueSummaryDataset(
        test_df,
        tokenizer,
        encoder_max_len,
        decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=False,
        truncate_tail=data_cfg.truncate_tail,
    )
    return {"train": train_ds, "dev": dev_ds, "test": test_ds}


def get_data_collator(tokenizer: PreTrainedTokenizerBase):
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)
