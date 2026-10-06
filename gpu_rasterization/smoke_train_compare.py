import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
import sys
# ensure project root is on sys.path so we can import top-level modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import LSTMClassifier


class H5RasterAsTokens(Dataset):
    """Memory-efficient HDF5-backed dataset.

    - Computes per-feature min/max in chunks (no full array in RAM)
    - Opens HDF5 file lazily per worker to avoid threading issues
    - Quantizes features into 10 bins -> token ids 2..11 (vocab size 12)
    """

    def __init__(self, path, target_id=0, stats_chunk=1_000_000):
        self.path = path
        self.target_id = int(target_id)
        # figure out total samples and feature dim without loading everything
        with h5py.File(self.path, 'r') as f:
            self.N = int(f.attrs.get('num_samples', f['labels'].shape[0]))
            ds = f['features']
            nfeat = ds.shape[1]

            # compute min/max per column in streaming chunks
            chunk = int(stats_chunk)
            min_ = np.full(nfeat, np.inf, dtype=np.float64)
            max_ = np.full(nfeat, -np.inf, dtype=np.float64)
            for start in range(0, self.N, chunk):
                end = min(self.N, start + chunk)
                block = ds[start:end]
                min_ = np.minimum(min_, block.min(axis=0))
                max_ = np.maximum(max_, block.max(axis=0))

        self.min_ = min_
        self.max_ = max_
        self.nfeat = nfeat
        # create 10 bins per feature -> tokens 0..9 -> map to token ids 2..11
        self.bins = [np.linspace(self.min_[i], self.max_[i], num=11)[1:-1] for i in range(self.nfeat)]

        # lazy file handle used per worker
        self._file = None

    def __len__(self):
        return self.N

    def __del__(self):
        try:
            if hasattr(self, '_file') and self._file is not None:
                self._file.close()
        except Exception:
            pass

    def __getitem__(self, idx):
        # open file per worker to avoid HDF5 threading issues
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
        feats = self._file['features'][idx]
        label = int(self._file['labels'][idx])
        # quantize
        tokens = []
        for i, val in enumerate(feats):
            b = self.bins[i]
            bin_idx = int(np.digitize(val, b))  # 0..10 -> clipped implicitly
            tok = 2 + bin_idx
            tokens.append(tok)
        tokens = torch.tensor(tokens, dtype=torch.long)
        binlabel = torch.tensor(1.0 if label == self.target_id else 0.0, dtype=torch.float32)
        length = torch.tensor(len(tokens), dtype=torch.long)
        return tokens, binlabel, length

def collate(batch):
    tokens, labels, lengths = zip(*batch)
    tokens = torch.stack(tokens)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return tokens, labels, lengths


def run_smoke(checkpoint_path, device, ds_path, target_id=0, batch_size=256, epochs=1):
    print('\n=== Smoke run: ', checkpoint_path, '===')
    ds = H5RasterAsTokens(ds_path, target_id=target_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    vocab_size = 12
    model = LSTMClassifier(vocab_size=vocab_size).to(device)

    state = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state:
        sd = state['model_state_dict']
    elif 'state_dict' in state:
        sd = state['state_dict']
    else:
        sd = state
    try:
        model.load_state_dict(sd)
        print('Loaded state dict into LSTMClassifier')
    except Exception as e:
        print('Failed to load state dict:', e)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    model.train()
    total_loss = 0.0
    steps = 0
    for epoch in range(epochs):
        for batch in loader:
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            opt.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

    end = time.time()
    avg_loss = total_loss / max(1, steps)
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mem_MB = peak_mem / 1024.0 / 1024.0
    else:
        peak_mem_MB = 0.0
    print('Done. avg_loss=%.6f steps=%d time=%.1fs peak_mem=%.1fMB' % (avg_loss, steps, end - start, peak_mem_MB))
    return {'checkpoint': checkpoint_path, 'avg_loss': avg_loss, 'steps': steps, 'time_s': end - start, 'peak_mem_MB': peak_mem_MB}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='gpu_rasterization/data/gaming_subset_50k.h5')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--require-gpu', action='store_true', help='Fail immediately if no GPU is available (no CPU fallback)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for full training')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint-dir', default='./checkpoints_raster')
    parser.add_argument('--mode', choices=['smoke','full'], default='full', help='smoke=1 epoch, full=use --epochs')
    args = parser.parse_args()

    # Resolve device strictly according to user's preference
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            if args.require_gpu:
                raise SystemExit('No CUDA/ROCm GPU available but --require-gpu was specified. Aborting.')
            else:
                print('Warning: CUDA requested but not available. Falling back to CPU.')
                device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device(args.device)

    # choose mode
    if args.mode == 'smoke':
        results = []
        for cp in args.checkpoints:
            res = run_smoke(cp, device, args.dataset, batch_size=args.batch_size, epochs=1)
            results.append(res)
    else:
        # full training loop: run training for each checkpoint sequentially and save checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        results = []
        for cp in args.checkpoints:
            print('\n=== Full training for', cp, '===')
            # reuse run_smoke as a single-training-epoch function by looping epochs
            ds = H5RasterAsTokens(args.dataset, target_id=0)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

            vocab_size = 12
            model = LSTMClassifier(vocab_size=vocab_size).to(device)

            state = torch.load(cp, map_location='cpu')
            if 'model_state_dict' in state:
                sd = state['model_state_dict']
            elif 'state_dict' in state:
                sd = state['state_dict']
            else:
                sd = state
            try:
                model.load_state_dict(sd)
                print('Loaded state dict into LSTMClassifier')
            except Exception as e:
                print('Failed to load state dict:', e)

            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.BCEWithLogitsLoss()

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)

            start = time.time()
            total_loss = 0.0
            steps = 0
            for epoch in range(args.epochs):
                model.train()
                for batch in loader:
                    x, y, lengths = batch
                    x = x.to(device)
                    y = y.to(device)
                    lengths = lengths.to(device)

                    opt.zero_grad()
                    logits = model(x, lengths)
                    loss = criterion(logits, y)
                    loss.backward()
                    opt.step()

                    total_loss += loss.item()
                    steps += 1
                # save checkpoint each epoch
                outp = os.path.join(args.checkpoint_dir, f"{os.path.basename(cp)}_epoch_{epoch+1}.pt")
                torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'best_val_loss': None}, outp)
                print(f'Saved checkpoint: {outp}')

            end = time.time()
            avg_loss = total_loss / max(1, steps)
            peak_mem_MB = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0 if device.type == 'cuda' else 0.0
            res = {'checkpoint': cp, 'avg_loss': avg_loss, 'steps': steps, 'time_s': end - start, 'peak_mem_MB': peak_mem_MB}
            results.append(res)

    print('\nSUMMARY')
    for r in results:
        print(r)
