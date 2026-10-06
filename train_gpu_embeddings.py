#!/usr/bin/env python3
"""
GPU Performance Embedding Model Training

Nutzt die gesammelten GPU Daten zum Training eines Embedding Models.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GPUMetricsDataset(Dataset):
    """Dataset aus gesammelten GPU Metriken."""

    def __init__(self, data_files):
        self.samples = []

        # Load all data files
        for file_path in data_files:
            print(f"Loading: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)

                if isinstance(data, list):
                    # gpu_logs format
                    for item in data:
                        if 'data' in item:
                            self.samples.append(self.parse_rocm_csv(item['data']))
                else:
                    # gpu_metrics format
                    for item in data:
                        if 'raw_output' in item:
                            lines = item['raw_output'].split('\n')
                            if len(lines) >= 2:
                                self.samples.append(self.parse_rocm_csv(lines[1]))

        print(f"Loaded {len(self.samples)} samples")

    def parse_rocm_csv(self, csv_line):
        """Parse ROCm CSV line to feature vector."""
        if isinstance(csv_line, list):
            parts = csv_line
        else:
            parts = csv_line.split(',')

        # Extract features (skip device name)
        features = []
        for val in parts[1:]:
            try:
                if val == 'N/A' or val == '':
                    features.append(0.0)
                else:
                    features.append(float(val))
            except:
                features.append(0.0)

        # Pad to fixed size
        while len(features) < 8:
            features.append(0.0)

        return features[:8]  # Temp edge, junction, memory, power, use%, vram total, vram used, power2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features = torch.tensor(self.samples[idx], dtype=torch.float32)
        return features


class GPUEmbeddingModel(nn.Module):
    """
    GPU Performance Embedding Model

    Input: GPU metrics (8 features)
    Output: Embedding vector (64 dim)
    """

    def __init__(self, input_dim=8, embedding_dim=64, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, embedding_dim),
        )

        # Reconstruction head (autoencoder)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training'):
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward
        embedding, reconstruction = model(batch)

        # Reconstruction loss (autoencoder)
        loss = criterion(reconstruction, batch)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='gpu_embedding_model.pt')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("GPU PERFORMANCE EMBEDDING MODEL - TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Find all data files
    data_files = []

    # gpu_logs
    for f in Path('gpu_logs').glob('*.json'):
        data_files.append(f)

    # gpu_metrics from sd_gpu_data
    for f in Path('sd_gpu_data').glob('*/gpu_metrics.json'):
        data_files.append(f)

    # gpu_workload_data
    for f in Path('gpu_workload_data').glob('*/gpu_metrics.json'):
        data_files.append(f)

    if not data_files:
        print("No data files found!")
        return

    print(f"Found {len(data_files)} data files")

    # Create dataset
    dataset = GPUMetricsDataset(data_files)

    if len(dataset) == 0:
        print("No samples loaded!")
        return

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create model
    model = GPUEmbeddingModel(
        input_dim=8,
        embedding_dim=64,
        hidden_dim=128
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, args.output)
            print(f"  ✓ Saved best model")

    print()
    print("=" * 70)
    print("✓ Training complete!")
    print(f"✓ Best model saved to: {args.output}")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
