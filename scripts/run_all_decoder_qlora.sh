#!/usr/bin/env bash
# 모든 decoder-only QLoRA 학습+추론을 순차 실행
# 사용 전: conda activate decoder-only-qlora, HF_TOKEN 환경변수 설정

set -euo pipefail

ROOT="/root/NLP_Dialouge_Summarization"
SCRIPT="${ROOT}/code/train_decoder_qlora.py"

CONFIGS=(
  "${ROOT}/config/config_hyperclovax_qlora.yaml"
  "${ROOT}/config/config_kanana_qlora.yaml"
  "${ROOT}/config/config_gemma_qlora.yaml"
  "${ROOT}/config/config_42dot_qlora.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running $cfg ==="
  python "${SCRIPT}" --config "${cfg}"
done
