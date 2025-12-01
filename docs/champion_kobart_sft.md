# KoBART 챔피언 설정 메모

리더보드 최고 점수를 냈던 KoBART SFT 실험 기본값.

- 모델: `gogamza/kobart-base-v1` (사전학습-only)
- 인코더 최대 길이: 1024
- 디코더 최대 길이: 80
- 학습률: 3e-5
- 배치: 8 (3090 기준), fp16
- 에폭: 12 (early stopping 적용)
- Early stopping patience: 3 (기본값)
- 평가/저장 주기: 500 스텝
- 스타일 프롬프트: `"요약: 대화의 핵심만 간결하게 한두 문장으로 정리하시오."`
- 체크포인트: `checkpoints/kobart-style-prompt/`
- 제출 파일 예시: `prediction/2511292249_kobart-base-style_prompt_bs8.csv`
- wandb: project `Dialogue Summarization`, entity `aistages-nlp-3`, run_name prefix `kobart-base-style_prompt`

실험 변형 시 위 값을 기준으로 변경 사항을 기록하면 비교가 쉽습니다.
