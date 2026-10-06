#!/usr/bin/env python3
"""Online/fine-tune trainer that watches a processed data directory and incrementally
# fine-tunes a model when new Parquet files arrive.
#
# Design goals:
# - Use existing TelemetryDataset to consume Parquet windows
# - Avoid re-processing the same files (processed.json bookkeeping)
# - Small replay buffer (last N files) to reduce catastrophic forgetting
# - Save checkpoints per processing step
#
# Minimal operation: run continuously until SIGINT/SIGTERM; each poll cycle will process new files.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# ensure local imports resolve
sys.path.insert(0, str(Path('.').resolve()))

from dataset_utils import TelemetryDataset, telemetry_collate


class SmallLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, num_layers=1, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out_dim))

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        h = hn[-1]
        return self.fc(h).squeeze(-1)


def load_state_if_exists(model, opt, path):
    if path.exists():
        try:
            d = torch.load(path, map_location='cpu')
            if 'model_state' in d:
                model.load_state_dict(d['model_state'])
            else:
                model.load_state_dict(d)
            if 'opt_state' in d:
                opt.load_state_dict(d['opt_state'])
            print(f"Loaded model state from {path}")
        except Exception as e:
            print(f"Warning: failed to load state {path}: {e}")


def find_parquets(data_dir: Path):
    return sorted([p for p in data_dir.rglob('*.parquet') if p.is_file()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed/get_data')
    parser.add_argument('--out-dir', default='logs/online_train')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'gpu'])
    parser.add_argument('--poll-interval', type=float, default=10.0, help='seconds between checks for new files')
    parser.add_argument('--replay-size', type=int, default=5, help='number of previous files to use for replay')
    parser.add_argument('--epochs-per-file', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    processed_file = out_dir / 'processed.json'
    processed = set()
    if processed_file.exists():
        try:
            processed = set(json.loads(processed_file.read_text()))
        except Exception:
            processed = set()

    # initial scan - find existing parquets
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    seen = set(p.as_posix() for p in find_parquets(data_dir))
    # keep processed information across restarts
    print(f"Starting online trainer. initial files={len(seen)} processed={len(processed)}")

    # create a toy model using the first file to determine feature size if needed
    def build_model_from_sample():
        parq = next(iter(find_parquets(data_dir)), None)
        if parq is None:
            raise RuntimeError('No parquet files found in data-dir; add processed data first')
        # create a small dataset from that single file
        tmp_dir = out_dir / 'tmp_sample'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # copy sample to tmp dir
        shutil.copy(parq, tmp_dir / parq.name)
        ds = TelemetryDataset(str(tmp_dir), window_size=args.window_size, step=1)
        if len(ds) == 0:
            raise RuntimeError('sample file has no windows')
        x, label, l = ds[0]
        n_features = x.shape[1]
        shutil.rmtree(tmp_dir)
        return SmallLSTM(n_features, hidden=args.hidden, num_layers=args.num_layers)

    model = build_model_from_sample().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    running = True

    def _save_checkpoint(epoch, step):
        ckpt = out_dir / f'model_online_epoch_{epoch}_step_{step}.pt'
        torch.save({'epoch': epoch, 'step': step, 'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, ckpt)

    try:
        while running:
            all_files = find_parquets(data_dir)
            to_process = [p for p in all_files if p.as_posix() not in processed]

            if not to_process:
                # nothing new — wait
                time.sleep(args.poll_interval)
                continue

            # process each new file one by one
            for fpath in to_process:
                step += 1
                print(f"Processing new file: {fpath}")
                # prepare dataset from new file and optionally replay previous ones
                tmp = out_dir / f'tmp_{step}'
                tmp.mkdir(parents=True, exist_ok=True)
                # copy new file
                shutil.copy(fpath, tmp / fpath.name)

                # find replay files (most recent already processed)
                replay_files = sorted([Path(x) for x in processed])[-args.replay_size:]
                # copy replay into tmp as well
                for r in replay_files:
                    try:
                        rp = Path(r)
                        shutil.copy(rp, tmp / rp.name)
                    except Exception:
                        pass

                # create dataset and loader
                ds = TelemetryDataset(str(tmp), window_size=args.window_size, step=1)
                if len(ds) == 0:
                    print("No windows in file, skipping")
                    processed.add(fpath.as_posix())
                    processed_file.write_text(json.dumps(list(processed)))
                    shutil.rmtree(tmp)
                    continue

                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=telemetry_collate)

                # run a few fine-tuning epochs
                for epoch in range(args.epochs_per_file):
                    total_loss = 0.0
                    count = 0
                    for batch in loader:
                        model.train()
                        windows, labels, lengths = batch
                        windows = windows.to(device)
                        labels = labels.to(device)
                        logits = model(windows)
                        loss = criterion(logits, labels)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total_loss += float(loss.item())
                        count += 1
                    avg = total_loss / max(1, count)
                    print(f"Processed {fpath.name} epoch {epoch+1}/{args.epochs_per_file} avg_loss={avg:.6f}")

                # save checkpoint
                _save_checkpoint(step, step)

                # mark processed
                processed.add(fpath.as_posix())
                processed_file.write_text(json.dumps(list(processed)))

                # cleanup tmp
                shutil.rmtree(tmp)

            # short delay before next poll
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print('online trainer stopping (KeyboardInterrupt)')

    print('online trainer exiting')


if __name__ == '__main__':
    main()
