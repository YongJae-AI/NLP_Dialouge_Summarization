import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


def split_sentences(text: str) -> List[str]:
    import re

    # 단순 문장 분할
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_similar(a: str, b: str, thresh: float = 0.8) -> bool:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    if not a_set or not b_set:
        return False
    jacc = len(a_set & b_set) / len(a_set | b_set)
    return jacc >= thresh


def dedup_sentences(sentences: List[str], max_keep: int = 3) -> List[str]:
    kept: List[str] = []
    for s in sentences:
        if any(is_similar(s, k) for k in kept):
            continue
        kept.append(s)
        if len(kept) >= max_keep:
            break
    return kept


def merge_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # fname 혹은 base_fname을 기준으로 묶는다.
    merged = []
    groups = {}
    for _, row in df.iterrows():
        fname = str(row["fname"])
        base = str(row.get("base_fname", fname))
        groups.setdefault(base, []).append(str(row["summary"]))

    for base, summaries in groups.items():
        sentences: List[str] = []
        for s in summaries:
            sentences.extend(split_sentences(s))
        sentences = [s for s in sentences if s]
        sentences = dedup_sentences(sentences, max_keep=3)
        merged_summary = " ".join(sentences)
        merged.append({"fname": base, "summary": merged_summary})
    return pd.DataFrame(merged)


def main():
    parser = argparse.ArgumentParser(description="Merge chunked predictions back to base fname.")
    parser.add_argument("--input", required=True, help="Prediction CSV with fname (chunked).")
    parser.add_argument("--output", required=True, help="Merged CSV output path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_df = merge_predictions(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Merged {len(out_df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
