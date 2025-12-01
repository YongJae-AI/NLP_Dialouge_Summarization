"""
NER 기반 이름 불일치 EDA 스크립트.

기능:
- data/train.csv, data/dev.csv를 읽어서 summary에 NER을 적용
- PERSON 엔티티만 추출 후, 해당 이름이 dialogue에 등장하는지 확인
- 요약에는 있지만 대화에는 전혀 없는 이름 리스트(unmatched_persons) 계산
- 통계 및 예시를 stdout에 출력
- train_ner_name_analysis.csv / dev_ner_name_analysis.csv 저장

주의:
- 원본 train/dev/test CSV는 절대 수정하지 않음
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from transformers import AutoTokenizer, pipeline


def extract_person_entities(text: str, ner_pipe, per_labels: Sequence[str]) -> List[str]:
    """
    Hugging Face NER pipeline 결과에서 PERSON 계열 엔티티 문자열만 추출합니다.

    - aggregation_strategy="simple" 이라고 가정합니다.
    - entity_group 이 per_labels 중 하나인 항목만 선택합니다.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    ents = ner_pipe(text)
    persons: List[str] = []
    for ent in ents:
        group = ent.get("entity_group") or ent.get("entity")
        if group in per_labels:
            word = ent.get("word", "").strip()
            if word:
                persons.append(word)
    # 중복 제거 + 정렬
    uniq = sorted(set(persons))
    return uniq


def compute_name_mismatch(df: pd.DataFrame, ner_pipe, n_samples: int | None = None) -> pd.DataFrame:
    """
    DataFrame에 대해 summary_persons, unmatched_persons, has_unmatched 컬럼을 추가합니다.
    """
    work_df = df.copy()
    if n_samples is not None and n_samples > 0:
        work_df = work_df.sample(min(n_samples, len(work_df)), random_state=42).reset_index(drop=True)

    per_labels = ["PER", "PERSON"]

    summary_persons_list: List[List[str]] = []
    unmatched_list: List[List[str]] = []
    has_unmatched_list: List[bool] = []

    for _, row in work_df.iterrows():
        dialogue = str(row.get("dialogue", ""))
        summary = str(row.get("summary", ""))

        persons = extract_person_entities(summary, ner_pipe, per_labels)
        unmatched = [name for name in persons if name and (name not in dialogue)]

        summary_persons_list.append(persons)
        unmatched_list.append(unmatched)
        has_unmatched_list.append(len(unmatched) > 0)

    work_df["summary_persons"] = summary_persons_list
    work_df["unmatched_persons"] = unmatched_list
    work_df["has_unmatched"] = has_unmatched_list
    return work_df


def run_split_analysis(
    split_name: str,
    path: Path,
    ner_pipe,
    n_samples: int | None,
    output_dir: Path,
) -> None:
    df = pd.read_csv(path)
    print(f"\n=== [{split_name.upper()}] NER 기반 이름 불일치 분석 ===")
    print(f"원본 샘플 수: {len(df)}")

    analyzed = compute_name_mismatch(df, ner_pipe, n_samples=n_samples)

    total = len(analyzed)
    num_unmatched = analyzed["has_unmatched"].sum()
    ratio = (num_unmatched / total * 100) if total > 0 else 0.0

    print(f"분석 대상 샘플 수: {total}")
    print(f"이름 불일치(has_unmatched=True) 샘플 수: {num_unmatched} ({ratio:.2f}%)")

    # 예시 몇 개 출력
    print("\n--- 불일치 예시 상위 5개 ---")
    examples = analyzed[analyzed["has_unmatched"]].head(5)
    for _, row in examples.iterrows():
        print("\n----------------------------------------")
        print("fname:", row.get("fname"))
        print("\n[DIALOGUE]\n", row.get("dialogue"))
        print("\n[SUMMARY]\n", row.get("summary"))
        print("\n[summary_persons]\n", row.get("summary_persons"))
        print("\n[unmatched_persons]\n", row.get("unmatched_persons"))

    # CSV 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split_name}_ner_name_analysis.csv"
    analyzed.to_csv(out_path, index=False)
    print(f"\n[{split_name}] 분석 결과 저장: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NER 기반 이름 불일치 EDA")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="train.csv, dev.csv가 위치한 디렉토리",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dslim/bert-base-NER",
        help="NER용 Hugging Face 모델 이름 (예: dslim/bert-base-NER)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="앞에서 n개만 샘플링해서 분석 (전체 사용 시 None 또는 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="분석 CSV를 저장할 디렉토리",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_path = data_dir / "train.csv"
    dev_path = data_dir / "dev.csv"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("train.csv 또는 dev.csv를 찾을 수 없습니다.")

    print("### NER 파이프라인 로딩 중 ###")
    # 영어 NER 기본값. 한국어 NER을 쓰고 싶다면 model_name만 교체하면 됩니다.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ner_pipe = pipeline(
        "token-classification",
        model=args.model_name,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    run_split_analysis("train", train_path, ner_pipe, args.n_samples, output_dir)
    run_split_analysis("dev", dev_path, ner_pipe, args.n_samples, output_dir)


if __name__ == "__main__":
    main()

