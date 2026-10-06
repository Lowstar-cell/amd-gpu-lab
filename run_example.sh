#!/usr/bin/env bash
set -euo pipefail

# Generate small sample dataset
python3 sample_data_gen.py --out-dir ./data --num-files 1 --samples-per-file 200

# Run training with device autodetection
python3 train.py --data-dir ./data --epochs 2 --batch-size 16 --device auto --model-out ./model.pt

echo "Done. Model saved to ./model.pt"