import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class AMDLogDataset(Dataset):
    """
    Dataset expects CSV files in directory. Each CSV contains rows with columns:
    timestamp,event_type,severity,message,label

    The dataset groups rows into sequences per sample in the CSV file. For simplicity each row is treated as an event.

    This example assumes dataset CSVs are already grouped per sample (e.g., each row has a sample_id). If not, the sample generator creates sequences.
    """

    def __init__(self, data_dir: str, min_seq_len: int = 3):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        self.min_seq_len = min_seq_len

        # Read files and build token vocab for event_type
        self.samples = []  # list of (list(event_type_str), label)
        self._build_samples()
        self.token2id = self._build_vocab()

    def _build_samples(self):
        for f in self.files:
            df = pd.read_csv(f)
            # Expect columns: sample_id, timestamp, event_type, severity, message, label
            if 'sample_id' not in df.columns:
                # if no sample_id, treat the whole file as a single sequence
                if 'label' not in df.columns:
                    raise ValueError('No label column found in CSV')
                seq = df['event_type'].astype(str).tolist()
                label = int(df['label'].iloc[-1] if len(df)>0 else 0)
                self.samples.append((seq, label))
            else:
                # group by sample_id
                grouped = df.groupby('sample_id')
                for _, g in grouped:
                    seq = g['event_type'].astype(str).tolist()
                    label = int(g['label'].iloc[0]) if 'label' in g.columns else 0
                    if len(seq) >= self.min_seq_len:
                        self.samples.append((seq, label))

    def _build_vocab(self):
        token2id = {"<pad>": 0, "<unk>": 1}
        next_id = 2
        for seq, _ in self.samples:
            for t in seq:
                if t not in token2id:
                    token2id[t] = next_id
                    next_id += 1
        return token2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        tokens = [self.token2id.get(t, 1) for t in seq]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float32), len(tokens)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    # batch: list of (tokens, label, length)
    tokens, labels, lengths = zip(*batch)
    maxlen = max([t.shape[0] for t in tokens])
    padded = torch.zeros(len(tokens), maxlen, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded[i, :t.shape[0]] = t
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, labels, lengths


class TelemetryDataset(Dataset):
    """
    Loads per-file telemetry parquet/csv files and produces sliding windows of numeric features.

    Expects files with columns like: timestamp, device, temp_edge_C, gpu_use_pct, vram_used_B, ...
    Produces sequences of shape (window_size, n_features) and optional labels (if a 'label' column exists)
    """

    def __init__(self, data_dir: str, window_size: int = 32, step: int = 1, feature_cols: list | None = None):
        self.data_dir = data_dir
        self.window_size = int(window_size)
        self.step = int(step)
        self.files = glob.glob(os.path.join(data_dir, "*.parquet")) + glob.glob(os.path.join(data_dir, "*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No parquet/csv files found in {data_dir}")

        self.feature_cols = feature_cols
        self.samples = []  # list of (np.ndarray windows, label) or (window, None)
        self._build_samples()

    def _build_samples(self):
        for f in self.files:
            # read file with pandas
            try:
                if f.endswith('.parquet'):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
            except Exception:
                continue

            # If no explicit feature columns are provided, use numeric columns apart from timestamp/ingest/file_source
            if self.feature_cols is None:
                candidates = [c for c in df.columns if c not in ('timestamp', 'ingest_time', 'file_source', 'device')]
                # choose numeric ones
                numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
                self.feature_cols = list(numeric)

            if not self.feature_cols:
                # nothing to do
                continue

            arr = df[self.feature_cols].to_numpy()
            nrows = arr.shape[0]
            # create sliding windows
            for start in range(0, max(0, nrows - self.window_size + 1), self.step):
                window = arr[start:start + self.window_size]
                if window.shape[0] != self.window_size:
                    continue
                # ensure no NaNs or infs make it to the model - replace with zeroes
                # NaNs/infs commonly appear in telemetry when metrics are missing; use safe replacement
                import numpy as _np
                if not _np.isfinite(window).all():
                    window = _np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
                # try get label from the last row if 'label' exists in df
                label = None
                if 'label' in df.columns:
                    try:
                        label = float(df['label'].iloc[start + self.window_size - 1])
                    except Exception:
                        label = None
                self.samples.append((window.astype(float), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        win, label = self.samples[idx]
        tensor = torch.tensor(win, dtype=torch.float32)
        if label is None:
            return tensor, torch.tensor(0.0, dtype=torch.float32), self.window_size
        else:
            return tensor, torch.tensor(label, dtype=torch.float32), self.window_size


def telemetry_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Collate a batch of telemetry windows -> (batch, window_size, n_features), labels"""
    windows, labels, lengths = zip(*batch)
    windows = torch.stack(windows)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return windows, labels, lengths


class RasterDataset(Dataset):
    """
    Loads per-file rasterization data and returns one sequence per file.

    Each file should contain per-frame rows with numeric rasterization features. The dataset
    returns (tensor_of_frames, label, n_frames) where tensor_of_frames has shape
    (n_frames, n_features) and label is taken from `label` column on last row if available.
    """

    def __init__(self, data_dir: str, feature_cols: list | None = None):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, "*.parquet")) + glob.glob(os.path.join(data_dir, "*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No raster files found in {data_dir}")

        self.feature_cols = feature_cols
        self.samples = []  # list of (np.ndarray frames, label)
        self._build()

    def _build(self):
        for f in self.files:
            try:
                if f.endswith('.parquet'):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
            except Exception:
                continue

            # If no explicit feature columns are provided, choose numeric columns other than timestamp/labels
            if self.feature_cols is None:
                candidates = [c for c in df.columns if c not in ('timestamp', 'frame_id', 'label', 'workload')]
                numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
                self.feature_cols = list(numeric)

            if not self.feature_cols:
                continue

            arr = df[self.feature_cols].to_numpy()
            # replace NaNs/inf
            import numpy as _np
            if not _np.isfinite(arr).all():
                arr = _np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            label = None
            if 'label' in df.columns:
                try:
                    label = float(df['label'].iloc[-1])
                except Exception:
                    label = None

            self.samples.append((arr.astype(float), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        tensor = torch.tensor(frames, dtype=torch.float32)
        if label is None:
            return tensor, torch.tensor(0.0, dtype=torch.float32), tensor.shape[0]
        else:
            return tensor, torch.tensor(label, dtype=torch.float32), tensor.shape[0]


def raster_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    # variable-length frame sequences: return list of windows as padded tensor
    tensors, labels, lengths = zip(*batch)
    maxlen = max([t.shape[0] for t in tensors])
    nfeat = tensors[0].shape[1]
    padded = torch.zeros(len(tensors), maxlen, nfeat, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], :] = t
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, labels, lengths
