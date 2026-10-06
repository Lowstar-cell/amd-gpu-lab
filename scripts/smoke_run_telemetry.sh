#!/usr/bin/env bash
# Smoke wrapper to run the telemetry trainer with a time budget (default 30 minutes)
set -euo pipefail

OUT_LOG=logs/telemetry_smoke_master.out
mkdir -p logs/telemetry_smoke

DATA_DIR=${1:-data/processed/get_data}
MINUTES=${2:-30}
PYTHON=${PYTHON:-python3}

echo "Starting telemetry smoke run: data=${DATA_DIR}, minutes=${MINUTES} -> log: ${OUT_LOG}"
nohup $PYTHON scripts/train_telemetry.py --data-dir "$DATA_DIR" --minutes "$MINUTES" --device auto > "$OUT_LOG" 2>&1 &
echo $! > logs/telemetry_smoke.pid
echo "Started PID $(cat logs/telemetry_smoke.pid)"
