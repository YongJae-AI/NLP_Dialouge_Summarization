from typing import Any, Dict, List, Optional

import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dialogsum.data import DialogueSummaryDataset


def run_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_df: pd.DataFrame,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    model_type: str,
    batch_size: int = 8,
    beam_size: int = 4,
) -> pd.DataFrame:
    dataset = DialogueSummaryDataset(
        test_df,
        tokenizer,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=False,
    )

    model.eval()
    all_summaries: List[str] = []
    fnames: List[str] = []

    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        fnames.extend([b["fname"] for b in batch])
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        import torch

        input_ids = torch.tensor(input_ids).to(model.device)
        attention_mask = torch.tensor(attention_mask).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=decoder_max_len,
                num_beams=beam_size,
            )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_summaries.extend(preds)

    return pd.DataFrame({"fname": fnames, "summary": all_summaries})


def run_generation_with_references(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    df: pd.DataFrame,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    model_type: str,
    batch_size: int = 8,
    beam_size: int = 4,
) -> pd.DataFrame:
    """dev처럼 정답 summary가 있는 데이터셋에서 요약 + 참조를 함께 저장."""
    dataset = DialogueSummaryDataset(
        df,
        tokenizer,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_type,
        is_train=False,
    )

    model.eval()
    outputs: List[Dict[str, Any]] = []

    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        fnames = [b["fname"] for b in batch]

        import torch

        input_ids = torch.tensor(input_ids).to(model.device)
        attention_mask = torch.tensor(attention_mask).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=decoder_max_len,
                num_beams=beam_size,
            )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for fname, pred in zip(fnames, preds):
            row = df.loc[df["fname"] == fname].iloc[0]
            outputs.append(
                {
                    "fname": fname,
                    "dialogue": str(row["dialogue"]),
                    "gold_summary": str(row.get("summary", "")),
                    "pred_summary": pred,
                }
            )

    return pd.DataFrame(outputs)
