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
    encoder_template: Optional[str],
    decoder_prefix: Optional[str],
    model_type: str,
    batch_size: int = 8,
    beam_size: int = 4,
    max_new_tokens: int = 80,
    min_length: int = 0,
    no_repeat_ngram_size: int = 3,
) -> pd.DataFrame:
    dataset = DialogueSummaryDataset(
        test_df,
        tokenizer,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=style_prompt,
        encoder_template=encoder_template,
        model_type=model_type,
        is_train=False,
    )

    model.eval()
    all_summaries: List[str] = []
    fnames: List[str] = []
    base_fnames: List[str] = []
    chunk_ids: List[int] = []
    row_indices: List[int] = []

    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        fnames.extend([b["fname"] for b in batch])
        base_fnames.extend([b.get("base_fname", b["fname"]) for b in batch])
        chunk_ids.extend([int(b.get("chunk_id", 0)) for b in batch])
        row_indices.extend([int(b.get("row_idx", idx)) for idx, b in enumerate(batch)])
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        import torch

        input_ids = torch.tensor(input_ids).to(model.device)
        attention_mask = torch.tensor(attention_mask).to(model.device)
        decoder_input_ids = None
        if decoder_prefix:
            prefix_ids = tokenizer(
                decoder_prefix,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(model.device)
            decoder_input_ids = prefix_ids.repeat(input_ids.size(0), 1)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        raw_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds = []
        for p in raw_preds:
            cleaned = p
            for key in ["최종 요약:", "수정된 최종 요약:", "요약:"]:
                if key in cleaned:
                    cleaned = cleaned.split(key)[-1].strip()
            if not cleaned:
                cleaned = p.strip()
            preds.append(cleaned)
        all_summaries.extend(preds)

    return pd.DataFrame(
        {
            "fname": fnames,
            "base_fname": base_fnames,
            "chunk_id": chunk_ids,
            "row_idx": row_indices,
            "summary": all_summaries,
        }
    )


def run_generation_with_references(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    df: pd.DataFrame,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    encoder_template: Optional[str],
    decoder_prefix: Optional[str],
    model_type: str,
    batch_size: int = 8,
    beam_size: int = 4,
    max_new_tokens: int = 80,
    min_length: int = 0,
    no_repeat_ngram_size: int = 3,
) -> pd.DataFrame:
    """dev처럼 정답 summary가 있는 데이터셋에서 요약 + 참조를 함께 저장."""
    dataset = DialogueSummaryDataset(
        df,
        tokenizer,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=style_prompt,
        encoder_template=encoder_template,
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
        base_fnames = [b.get("base_fname", b["fname"]) for b in batch]
        chunk_ids = [int(b.get("chunk_id", 0)) for b in batch]
        row_indices = [int(b.get("row_idx", idx)) for idx, b in enumerate(batch)]

        import torch

        input_ids = torch.tensor(input_ids).to(model.device)
        attention_mask = torch.tensor(attention_mask).to(model.device)
        decoder_input_ids = None
        if decoder_prefix:
            prefix_ids = tokenizer(
                decoder_prefix,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(model.device)
            decoder_input_ids = prefix_ids.repeat(input_ids.size(0), 1)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        raw_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds: List[str] = []
        for p in raw_preds:
            cleaned = p
            for key in ["최종 요약:", "수정된 최종 요약:", "요약:"]:
                if key in cleaned:
                    cleaned = cleaned.split(key)[-1].strip()
            if not cleaned:
                cleaned = p.strip()
            preds.append(cleaned)
        for fname, base_fname, chunk_id, row_idx, pred in zip(
            fnames, base_fnames, chunk_ids, row_indices, preds
        ):
            matching_rows = df.loc[
                (df.get("row_idx", df.index) == row_idx)
                & (df.get("chunk_id", 0) == chunk_id)
            ]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
            else:
                row = df.iloc[row_idx]
            outputs.append(
                {
                    "fname": fname,
                    "base_fname": base_fname,
                    "chunk_id": chunk_id,
                    "dialogue": str(row["dialogue"]),
                    "gold_summary": str(row.get("summary", "")),
                    "pred_summary": pred,
                }
            )

    return pd.DataFrame(outputs)
