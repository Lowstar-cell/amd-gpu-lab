#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run two sequential training runs using the .venv-rocm6.4 python
# Each run will use GPU (ROCm), 100000 epochs, write checkpoints and logs under ./logs/

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# venv python
VENV_PY="$ROOT_DIR/.venv-rocm6.4/bin/python"
# Use a dedicated generated data folder so existing SDK files don't confuse the dataset loader
DATA_DIR="$ROOT_DIR/data/generated_for_training"
LOG_ROOT="$ROOT_DIR/logs/rocm_runs"

mkdir -p "$LOG_ROOT"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: expected venv python at $VENV_PY not found or not executable." >&2
  exit 1
fi

# Ensure we have at least small sample data
# Only generate data if there aren't any CSVs in $DATA_DIR
shopt -s nullglob
csvs=($DATA_DIR/*.csv)
if [ ${#csvs[@]} -eq 0 ]; then
  echo "No CSV files found in $DATA_DIR — generating sample data (2 files, 200 samples each)"
  mkdir -p "$DATA_DIR"
  "$VENV_PY" "$ROOT_DIR/sample_data_gen.py" --out-dir "$DATA_DIR" --num-files 2 --samples-per-file 200 --max-seq-len 128
fi
shopt -u nullglob

# Determine epoch count: prefer env EPOCHS, fallback to first script arg, then default
EPOCHS=${EPOCHS:-${1:-100000}}

echo "Using EPOCHS=$EPOCHS for both runs"

run_one() {
  local idx="$1"
  local out_model="$ROOT_DIR/model_rocm_run${idx}.pt"
  local logdir="$LOG_ROOT/run${idx}"
  mkdir -p "$logdir"
  local out_log="$logdir/train.log"
  echo "Starting run $idx ($epochs epochs). Logs -> $out_log"

  # Run training (this will run for a long time)
  # EPOCHS may be specified in env or argument EPOCHS
  local epochs=${EPOCHS:-${2:-100000}}
  "$VENV_PY" "$ROOT_DIR/train.py" \
    --data-dir "$DATA_DIR" \
    --epochs "$epochs" \
    --batch-size 16 \
    --device gpu \
    --model-out "$out_model" \
    --logdir "$logdir" \
    --checkpoint-dir "$logdir/checkpoints" \
    --checkpoint-interval 5 \
    2>&1 | tee "$out_log"
}

run_one 1 "$EPOCHS"
echo "Run 1 finished — starting Run 2"
run_one 2 "$EPOCHS"

echo "Both runs finished — models: $ROOT_DIR/model_rocm_run1.pt and $ROOT_DIR/model_rocm_run2.pt"
