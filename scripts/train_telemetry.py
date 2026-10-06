#!/usr/bin/env python3
"""Small training script for telemetry sliding-window data.

This is a smoke-run trainer intended to run for a fixed time budget (minutes) or epochs.
It uses the TelemetryDataset in dataset_utils.

Usage examples:
  # quick smoke-run for 30 minutes (default)
  python3 scripts/train_telemetry.py --data-dir data/processed/get_data --minutes 30 --device auto

  # short test run for ~30 seconds
  python3 scripts/train_telemetry.py --data-dir data/processed/get_data --minutes 0.5 --device cpu

The trainer saves a small checkpoint model to `logs/telemetry_smoke/model_epoch_last.pt`.
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils import TelemetryDataset, telemetry_collate


class SmallLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=1, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out_dim))

    def forward(self, x):
        # x: (batch, seq, feat)
        out, (hn, cn) = self.lstm(x)
        h = hn[-1]
        return self.fc(h).squeeze(-1)


def get_device(pref: str):
    if pref == 'cpu':
        return torch.device('cpu')
    elif pref == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise RuntimeError('GPU requested but not available')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Dataset
    ds = TelemetryDataset(args.data_dir, window_size=args.window_size, step=args.step, feature_cols=None)
    if len(ds) == 0:
        raise RuntimeError('No windows found in dataset; run conversion first')

    # determine feature count
    dummy, _, _ = ds[0]
    n_features = dummy.shape[1]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=telemetry_collate)

    device = get_device(args.device)
    model = SmallLSTM(n_features, hidden=args.hidden, num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    logdir = Path(args.logdir)
    (logdir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    start = time.time()
    minutes = float(args.minutes) if args.minutes is not None else None
    time_budget = minutes * 60 if minutes else None

    epoch = 0
    while True:
        epoch += 1
        running_loss = 0.0
        n = 0
        for batch in loader:
            windows, labels, lengths = batch
            windows = windows.to(device)
            labels = labels.to(device)

            logits = model(windows)
            # if labels are 0/1 floats, we can use BCEWithLogits
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            n += 1

            # check time budget
            if time_budget and (time.time() - start) > time_budget:
                break

        avg_loss = running_loss / max(1, n)
        print(f"Epoch {epoch:04d}  avg_loss={avg_loss:.6f}  elapsed={int(time.time()-start)}s")

        # save checkpoint
        ckpt = logdir / 'checkpoints' / f'model_epoch_{epoch}.pt'
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, ckpt)

        # check time budget
        if time_budget and (time.time() - start) > time_budget:
            print('Time budget exhausted; finishing')
            break

        # stop at max epochs
        if args.epochs and epoch >= args.epochs:
            print('Reached max epochs; finishing')
            break

    # save final
    torch.save(model.state_dict(), logdir / 'model_epoch_last.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed/get_data')
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'gpu'])
    parser.add_argument('--minutes', type=float, default=30.0, help='time budget in minutes for the run')
    parser.add_argument('--epochs', type=int, default=None, help='max epochs (optional)')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--logdir', default='logs/telemetry_smoke')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
