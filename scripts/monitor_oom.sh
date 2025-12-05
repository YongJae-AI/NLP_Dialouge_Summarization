#!/usr/bin/env bash
# OOM 방지 모니터링 스크립트 (별도 터미널에서 실행)
# 사용: bash /root/NLP_Dialouge_Summarization/scripts/monitor_oom.sh

set -euo pipefail

INTERVAL="${INTERVAL:-5}"  # 초 단위 갱신 주기 (환경변수로 조정 가능)

while true; do
    clear
    echo "==== Resource Monitor ($(date '+%F %T')) ===="
    echo "[GPU]"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv
    else
        echo "nvidia-smi not found"
    fi

    echo
    echo "[CPU/RAM]"
    if command -v htop >/dev/null 2>&1; then
        echo "Tip: run htop in another terminal for live per-core view"
    fi
    free -h
    uptime

    echo
    echo "[Disk]"
    df -h /root

    echo
    echo "[PyTorch CUDA memory summary (if CUDA available)]"
    python - <<'PY' 2>/dev/null || echo "PyTorch not available in this shell/env"
import torch
if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
else:
    print("CUDA not available")
PY

    sleep "${INTERVAL}"
done
