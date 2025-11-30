# Dialogue Summarization Experiment Log

이 파일은 주요 실험 설정과 결과를 사람이 읽기 좋게 요약해서 기록하는 곳입니다.
각 제출 CSV와 wandb run이 어떻게 연결되는지 최소한의 메타 정보를 남겨 주세요.

## 기록 예시

- 날짜: 2025-11-29
- 제출 파일: `prediction/2511292249_kobart-base-style_prompt_bs8.csv`
- 모델: `gogamza/kobart-base-v1`
- config: `config/config_kobart_style_prompt.yaml`
- batch size: 8
- encoder_max_len / decoder_max_len: 1024 / 80
- learning_rate: 3e-5
- epochs / early stopping: max 5, early stopping 없음 (예시)
- style_prompt: 기본 프롬프트
- special tokens: KoBART 기본 vocab/subword만 사용
- wandb: 사용 안 함
- wandb run: -
- 리더보드 점수(공개/비공개): 공개 XX.XX / 비공개 YY.YY
- 코멘트: 초기 기준선 실험

---

## 새 기록 추가 시 권장 포맷

- 날짜:
- 제출 파일:
- 모델:
- config:
- batch size:
- encoder_max_len / decoder_max_len:
- learning_rate:
- epochs / early stopping:
- style_prompt:
- special tokens:
- wandb: 사용/미사용
- wandb run:
- 리더보드 점수(공개/비공개):
- 코멘트:

