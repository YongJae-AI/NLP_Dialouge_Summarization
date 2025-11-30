from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


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
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.style_prompt = style_prompt
        self.model_type = model_type
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        dialogue = str(row["dialogue"])

        if self.style_prompt:
            if self.model_type == "kobart":
                src_text = f"{self.style_prompt} {dialogue}"
            else:
                src_text = f"{self.style_prompt}: {dialogue}"
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

        # fname는 학습/검증 단계에서는 필요 없으므로,
        # is_train=False (주로 inference)일 때만 포함한다.
        if not self.is_train:
            model_inputs["fname"] = row.get("fname", f"idx_{idx}")
        return model_inputs


@dataclass
class DataConfig:
    train_path: str = "data/train.csv"
    dev_path: str = "data/dev.csv"
    test_path: str = "data/test.csv"


def load_csv_splits(cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    train_df = pd.read_csv(cfg.train_path)
    dev_df = pd.read_csv(cfg.dev_path)
    test_df = pd.read_csv(cfg.test_path)
    return {"train": train_df, "dev": dev_df, "test": test_df}


def load_datasets(
    tokenizer: PreTrainedTokenizerBase,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    model_type: str,
    data_cfg: Optional[DataConfig] = None,
) -> Dict[str, DialogueSummaryDataset]:
    if data_cfg is None:
        data_cfg = DataConfig()
    splits = load_csv_splits(data_cfg)
    train_ds = DialogueSummaryDataset(
        splits["train"],
        tokenizer,
        encoder_max_len,
        decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=True,
    )
    dev_ds = DialogueSummaryDataset(
        splits["dev"],
        tokenizer,
        encoder_max_len,
        decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=True,
    )
    test_ds = DialogueSummaryDataset(
        splits["test"],
        tokenizer,
        encoder_max_len,
        decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=False,
    )
    return {"train": train_ds, "dev": dev_ds, "test": test_ds}


def get_data_collator(tokenizer: PreTrainedTokenizerBase):
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)
