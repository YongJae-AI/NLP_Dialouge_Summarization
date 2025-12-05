# Dialogue Summarization 작업 메모 (절대경로 기준)

- 리포 루트: `/root/NLP_Dialouge_Summarization`

## 데이터/경진대회
- 대회 규칙/평가: ROUGE-1/2/L F1 평균, 평가 데이터 3개 참조 요약 사용.
- 데이터 위치: `/root/NLP_Dialouge_Summarization/data/data/` (`train.csv`, `dev.csv`, `test.csv`, `sample_submission.csv`).
- 외부 규정: DialogSum 기반 가중치/데이터 금지, 평가 데이터 학습 금지, 무료 API만 사용 (Solar 허용).

## 현재 모델 설정 확인용
- KoBART (파일: `/root/NLP_Dialouge_Summarization/config/config_kobart_style_prompt.yaml`):
  - 모델 `gogamza/kobart-base-v1`, enc 1024 / dec 80, lr `3e-5`, batch 8, epochs 12, fp16, eval/save 700스텝, early stop patience 6.
  - 스타일 프롬프트 적용: QA 템플릿(`질문: ... 대화: ... 답변:`).
  - 체크포인트 `checkpoints/kobart-style-prompt`, 출력 `prediction/`.
- KoT5 변형:
  - 기본(`config_kot5_base.yaml`): `ktoivola/mt5-small-finetuned-dialogsum-ko`, enc 768/dec 80, lr `3e-5`, batch 8, epochs 5, fp16.
  - 카카오브레인(`config_kot5_kakaobrain_base.yaml`): `wisenut-nlp-team/KoT5-base`, enc 896/dec 80, truncate_tail true, chunk_overlap true(128), lr `3e-4`, batch 4, grad_accum 2, epochs 5, fp16 false, eval/save 500, patience 3, QA 스타일 프롬프트 사용.
  - overlap 변형(`config_kot5_overlap.yaml`): enc 896/dec 80, truncate_tail false, chunk_overlap true(128), lr `3e-4`, batch 4, grad_accum 2, epochs 5, label_smoothing 0.1, fp16 false, eval/save 500, patience 3, QA 스타일 프롬프트 사용.

## 프롬프트 정책
- `style_prompt`가 설정된 모든 모델은 QA 스타일 템플릿으로 입력 구성: `질문: [스타일 가이드]\n\n대화:\n{dialogue}\n\n답변:` (`/root/NLP_Dialouge_Summarization/src/dialogsum/data.py`).

## Overlap 정책 (겹침/중복 제거)
- overlap 토큰 수: `chunk_overlap_tokens = 128` (config 값 그대로 사용).
- chunking: `fname`는 항상 원본 그대로 유지, `chunk_id`/`base_fname`/`row_idx`는 내부 관리용(`chunk_dialogues` 함수, `/root/NLP_Dialouge_Summarization/src/dialogsum/data.py`).
- 예측 병합: `_merge_predictions`에서 문장 분할 후 Jaccard 유사도(토큰 집합, 임계 0.8)로 중복 문장을 제거하고, base fname 당 1개 요약으로 병합. 병합 결과 행 수가 원본 base fname 개수와 다르면 에러를 발생시켜 확인하도록 함 (`/root/NLP_Dialouge_Summarization/code/full_pipeline.py`).
- 목적: enc 길이 한계를 넘는 대화를 턴 단위로 자르되, overlap으로 문맥을 보존하고, 병합 단계에서 중복 문장을 제거해 최종 제출행 수를 원본 test와 동일하게 유지.

## 제출 파일 규칙
- 출력 디렉토리: `/root/NLP_Dialouge_Summarization/prediction`.
- 파일명 규칙: 한국 시간 기준 `YYMMDD` + 순번(001~999) + `_실험명.csv` 예) `251205001_kobart-base-style_prompt.csv`.
- 제출 포맷: `fname`, `summary` 열을 갖는 CSV(샘플: `/root/NLP_Dialouge_Summarization/data/data/sample_submission.csv`). 현재 플레이스홀더: `/root/NLP_Dialouge_Summarization/prediction/251205001_placeholder.csv` (summary 공란).
- chunk 사용 시: 중간 예측은 chunk별로 저장해도 최종 제출 파일은 병합/중복 제거 후 원본 test 행 수와 동일해야 함. `fname`에 `_chunk` 접미어 금지.

## 경로 사용 원칙
- 코드 내 경로는 `dialogsum.utils.resolve_path`로 루트 기준 절대경로로 변환해 사용.
- 새로 추가하는 스크립트/명령에서도 절대경로 명시를 기본으로 함.

## 향후 작업
- decoder-only 모델(새 가상환경) 준비는 모델 지정 시 별도 환경 생성 예정.
