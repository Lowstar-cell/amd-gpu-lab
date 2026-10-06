"""
Integration: Bestehender Trainer mit GPU Rasterization
Erweitert train.py um 3D Rasterization Features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import bestehende Module
from model import SimpleModel  # Aus ../model.py
from dataset_utils import create_dataloaders  # Aus ../dataset_utils.py

# Import Rasterizer
from gpu_rasterize import AMDRasterizer, RasterizationEmbedding
from data_collection.collector import RasterizationDataCollector


class RasterizationAugmentedModel(nn.Module):
    """
    Erweitert bestehendes Model mit Rasterization Features
    """

    def __init__(self, base_model, raster_embedding_dim=128):
        super().__init__()
        self.base_model = base_model

        # Rasterization embedding
        self.raster_embedding = RasterizationEmbedding(
            input_dim=7,
            embedding_dim=raster_embedding_dim
        )

        # Fusion layer
        # Annahme: base_model hat hidden_dim
        base_dim = 128  # Adjust to actual base model
        self.fusion = nn.Sequential(
            nn.Linear(base_dim + raster_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, base_dim)
        )

    def forward(self, x, raster_features=None):
        # Base model forward
        base_out = self.base_model(x)

        if raster_features is not None:
            # Add rasterization features
            raster_emb = self.raster_embedding(raster_features)

            # Fuse
            combined = torch.cat([base_out, raster_emb], dim=-1)
            fused = self.fusion(combined)
            return fused
        else:
            return base_out


def train_with_rasterization(
    model,
    train_loader,
    val_loader,
    epochs=10,
    device='cuda',
    use_rasterization=True
):
    """
    Training Loop mit optionaler GPU Rasterization
    """

    # Setup rasterizer
    rasterizer = None
    collector = None

    if use_rasterization:
        rasterizer = AMDRasterizer(width=256, height=256)
        collector = RasterizationDataCollector(save_dir='./data')
        print("[Training] Rasterization features ENABLED")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        raster_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Optional: Generate rasterization features
            raster_features = None

            if use_rasterization and batch_idx % 10 == 0:  # Every 10 batches
                # Generate random 3D scene from batch
                batch_size = data.shape[0]

                # Create simple scene (in practice: use actual data)
                vertices = np.random.rand(5, 3, 3).astype(np.float32) * 256
                colors = np.random.rand(5, 3).astype(np.float32)
                mvp = np.eye(4, dtype=np.float32)

                # Rasterize
                _, training_data = rasterizer.rasterize_triangles(
                    vertices, colors, mvp, collect_data=True
                )

                if training_data:
                    # Collect for later analysis
                    collector.collect_batch(training_data, batch_idx=batch_idx)
                    raster_samples += 1

                    # Create feature tensor for model
                    # Simplified: use mean barycentric coords
                    bary = training_data['barycentric_coords']
                    feature_summary = np.array([
                        bary.mean(),
                        bary.std(),
                        training_data['depth'].mean(),
                        training_data['depth'].std(),
                        training_data['coverage'].mean(),
                        0.0, 0.0  # Padding
                    ], dtype=np.float32)

                    # Expand to batch
                    raster_features = torch.from_numpy(
                        np.tile(feature_summary, (batch_size, 1))
                    ).to(device)

            # Forward pass
            optimizer.zero_grad()

            if hasattr(model, 'raster_embedding'):
                output = model(data, raster_features)
            else:
                output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Raster samples: {raster_samples}")

        # Save rasterization data
        if use_rasterization and len(collector.features) > 0:
            collector.save_dataset(f'epoch_{epoch}_rasterization.h5')
            print(f"  → Saved {len(collector.features)} rasterization samples")
            collector.clear()

    print("\n[Training] Complete!")
    return model


def main():
    print("=" * 60)
    print("TRAINING WITH AMD GPU RASTERIZATION")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create base model
    try:
        base_model = SimpleModel()
        print("\n[OK] Loaded base model from model.py")
    except Exception as e:
        print(f"\n[INFO] Could not load base model: {e}")
        print("       Using dummy model for demo")

        # Dummy model
        base_model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    # Create augmented model
    model = RasterizationAugmentedModel(
        base_model=base_model,
        raster_embedding_dim=128
    )

    print(f"\n[OK] Created augmented model")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy dataloaders
    print("\n[INFO] Creating dummy dataloaders for demo")

    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10),
        torch.randn(1000, 10)
    )

    train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=False)

    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    trained_model = train_with_rasterization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        device=device,
        use_rasterization=True
    )

    # Save
    torch.save(trained_model.state_dict(), 'model_with_rasterization.pt')
    print(f"\n[OK] Model saved to: model_with_rasterization.pt")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)


if __name__ == '__main__':
    main()
