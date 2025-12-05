#!/usr/bin/env bash
# OOM 방지 모니터링 스크립트 (별도 터미널에서 실행)
# 사용: bash /root/NLP_Dialouge_Summarization/scripts/monitor_oom.sh

set -euo pipefail

INTERVAL="${INTERVAL:-5}"  # 초 단위 갱신 주기 (환경변수로 조정 가능)

while true; do
    ts="$(date '+%F %T')"
    echo "[$ts] GPU mem/usage"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    else
        echo "nvidia-smi not found"
    fi

    echo "[$ts] CPU load / RAM"
    uptime | awk '{print "load:", $(NF-2), $(NF-1), $NF}'
    free -h | awk 'NR==2{print "ram:", $3 "/" $2}'

    echo "[$ts] Disk /root"
    df -h /root | awk 'NR==2{print "disk:", $3 "/" $2, "used"}'

    sleep "${INTERVAL}"
done
