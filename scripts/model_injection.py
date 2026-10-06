#!/usr/bin/env python3
"""Load a saved checkpoint and run inference on a telemetry parquet (or the latest file in a folder).

The tool inspects the checkpoint to infer model input size/hidden size, builds a SmallLSTM,
loads weights, runs one forward pass on the most recent window and prints a JSON result
containing a score in range [0,1].

Usage:
  python3 scripts/model_injection.py --model logs/telemetry_smoke/model_epoch_last.pt --data data/processed/get_data
  python3 scripts/model_injection.py --model logs/online_train/model_online_epoch_1_step_1.pt --file data/processed/get_data/amd_gpu_logs.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path('.').resolve()))
from dataset_utils import TelemetryDataset


class SmallLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=1, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out_dim))

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        h = hn[-1]
        return self.fc(h).squeeze(-1)


def infer_checkpoint_info(state_dict: dict):
    # find lstm weight to infer layers/hidden/input
    for k, v in state_dict.items():
        if 'lstm.weight_ih_l0' in k:
            shape = v.shape
            hidden_times_4 = shape[0]
            input_size = shape[1]
            hidden = hidden_times_4 // 4
            return input_size, hidden
    raise RuntimeError('Unable to infer model structure from checkpoint')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--file', default=None, help='path to parquet/csv file to use for inference (optional)')
    parser.add_argument('--data', default='data/processed/get_data', help='folder to look for the most recent parquet file if --file omitted')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(json.dumps({'error': 'model not found', 'path': str(path)}))
        sys.exit(2)

    ck = torch.load(str(path), map_location='cpu')
    # checkpoint might be a state_dict or contain 'model_state'
    if isinstance(ck, dict) and 'model_state' in ck:
        state = ck['model_state']
    else:
        state = ck

    input_size, hidden = infer_checkpoint_info(state)
    # assume num_layers=1 unless there's a weight_ih_l1
    num_layers = 1
    for k in state.keys():
        if 'lstm.weight_ih_l1' in k:
            num_layers = 2

    model = SmallLSTM(input_size, hidden, num_layers=num_layers).to('cpu')
    try:
        model.load_state_dict(state)
    except Exception:
        # loading might require mapping key names if simple state-dict stored only weights
        # try to load partial
        model_state = {k: v for k, v in state.items() if k in model.state_dict()}
        model.load_state_dict(model_state, strict=False)

    # find file or most recent parquet in data dir
    if args.file:
        target = Path(args.file)
    else:
        data_dir = Path(args.data)
        files = sorted([p for p in data_dir.rglob('*.parquet')])
        target = files[-1] if files else None

    if target is None or not target.exists():
        print(json.dumps({'error': 'no data file found', 'used': str(target)}))
        sys.exit(2)

    # construct dataset with window_size that matches model input; assume 32 if not inferable
    window_size = 32
    ds = TelemetryDataset(str(target.parent), window_size=window_size, step=1)
    if len(ds) == 0:
        print(json.dumps({'error': 'no windows in data file', 'file': str(target)}))
        sys.exit(2)

    # use last window
    x, _, _ = ds[-1]
    # x is (window, features); we need (1, window, features)
    with torch.no_grad():
        inp = x.unsqueeze(0)
        out = model(inp)
        score = float(torch.sigmoid(out).item()) if torch.is_tensor(out) else float(out)

    print(json.dumps({'score': score, 'model': str(path), 'file': str(target)}))


if __name__ == '__main__':
    main()
