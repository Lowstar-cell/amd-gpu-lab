"""
Beispiel: Datensammlung für Training
Zeigt alle Methoden zum Sammeln von GPU Rasterization Features
"""

import numpy as np
import torch
from gpu_rasterize import AMDRasterizer
from data_collection.collector import RasterizationDataCollector

def methode_1_einfach():
    """Einfachste Methode: Direkt aus Rasterizer"""
    print("=" * 60)
    print("METHODE 1: Direkt aus Rasterizer")
    print("=" * 60)

    rasterizer = AMDRasterizer(width=256, height=256)

    # Erstelle Test-Scene
    vertices = np.random.rand(10, 3, 3).astype(np.float32) * 256
    colors = np.random.rand(10, 3).astype(np.float32)
    mvp = np.eye(4, dtype=np.float32)

    # Rasterize mit Datensammlung
    image, training_data = rasterizer.rasterize_triangles(
        vertices, colors, mvp,
        collect_data=True  # ← WICHTIG!
    )

    # Zugriff auf gesammelte Daten
    print(f"\nGesammelte Daten:")
    print(f"  Barycentric coords: {training_data['barycentric_coords'].shape}")
    print(f"  Depth buffer:       {training_data['depth'].shape}")
    print(f"  Coverage map:       {training_data['coverage'].shape}")

    # Direkt zu NumPy
    bary = training_data['barycentric_coords']
    depth = training_data['depth']
    coverage = training_data['coverage']

    # Feature-Vektor pro Pixel extrahieren
    H, W = coverage.shape
    pixels_rendered = coverage.sum()
    print(f"\n  Rendered pixels: {pixels_rendered} / {H*W}")

    return training_data


def methode_2_collector():
    """Mit DataCollector: Sammelt und speichert automatisch"""
    print("\n" + "=" * 60)
    print("METHODE 2: Mit DataCollector (Empfohlen)")
    print("=" * 60)

    rasterizer = AMDRasterizer(width=256, height=256)
    collector = RasterizationDataCollector(save_dir='./data')

    # Multiple Scenes rendern und sammeln
    num_scenes = 100
    print(f"\nSammle Daten aus {num_scenes} Scenes...")

    for i in range(num_scenes):
        # Neue zufällige Scene
        vertices = np.random.rand(20, 3, 3).astype(np.float32) * 256
        colors = np.random.rand(20, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float32)

        # Rasterize
        _, training_data = rasterizer.rasterize_triangles(
            vertices, colors, mvp, collect_data=True
        )

        # Automatisch sammeln!
        collector.collect_from_rasterization(
            training_data['barycentric_coords'],
            training_data['depth'],
            training_data['coverage']
        )

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_scenes} scenes")

    print(f"\n✓ Gesammelte Samples: {len(collector.features)}")

    # Speichern als HDF5
    filepath = collector.save_dataset('collected_data.h5')
    print(f"✓ Gespeichert: {filepath}")

    # Später laden:
    features, labels = collector.load_dataset('collected_data.h5')
    print(f"\n  Features shape: {features.shape}")
    print(f"  Labels shape:   {labels.shape}")

    return collector


def methode_3_pytorch_dataset():
    """Direkt als PyTorch Dataset für Training"""
    print("\n" + "=" * 60)
    print("METHODE 3: PyTorch Dataset für Training")
    print("=" * 60)

    rasterizer = AMDRasterizer(width=128, height=128)
    collector = RasterizationDataCollector(save_dir='./data')

    # Daten sammeln
    print("\nSammle Training-Daten...")
    for i in range(50):
        vertices = np.random.rand(15, 3, 3).astype(np.float32) * 128
        colors = np.random.rand(15, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float32)

        _, training_data = rasterizer.rasterize_triangles(
            vertices, colors, mvp, collect_data=True
        )

        collector.collect_from_rasterization(
            training_data['barycentric_coords'],
            training_data['depth'],
            training_data['coverage']
        )

    # Als PyTorch Dataset
    dataset = collector.get_pytorch_dataset()
    print(f"\n✓ PyTorch Dataset: {len(dataset)} samples")

    # DataLoader erstellen
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0
    )

    print(f"✓ DataLoader: {len(dataloader)} batches")

    # Beispiel: Ersten Batch anschauen
    features, labels = next(iter(dataloader))
    print(f"\n  Batch features: {features.shape}")
    print(f"  Batch labels:   {labels.shape}")
    print(f"  Device:         {features.device}")

    # Direkt für Training nutzbar!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        # ... dein Model Training hier ...

        if batch_idx >= 2:  # Demo: nur 3 Batches
            break

    print(f"\n✓ Training Loop funktioniert!")

    return dataset


def methode_4_integration_mit_bestehendem_training():
    """Integration in bestehende train.py"""
    print("\n" + "=" * 60)
    print("METHODE 4: Integration in bestehendes Training")
    print("=" * 60)

    print("""
# In deiner train.py:

from gpu_rasterization.gpu_rasterize import AMDRasterizer
from gpu_rasterization.data_collection.collector import RasterizationDataCollector

# Setup (einmalig vor Training Loop)
rasterizer = AMDRasterizer(width=256, height=256)
collector = RasterizationDataCollector(save_dir='./data/rasterization')

# In deinem Training Loop:
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):

        # Normales Training
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # ZUSÄTZLICH: Rasterization Features sammeln
        # (z.B. alle 10 Batches)
        if batch_idx % 10 == 0:
            # Erstelle 3D Scene aus deinen Daten
            # (oder verwende fixed test scene)
            vertices = create_scene_from_batch(data)
            colors = ...
            mvp = ...

            _, raster_features = rasterizer.rasterize_triangles(
                vertices, colors, mvp, collect_data=True
            )

            # Sammeln!
            collector.collect_from_rasterization(
                raster_features['barycentric_coords'],
                raster_features['depth'],
                raster_features['coverage']
            )

    # Nach jeder Epoche: Speichern
    collector.save_dataset(f'epoch_{epoch}_raster.h5')
    print(f"Epoch {epoch}: Collected {len(collector.features)} raster samples")
    collector.clear()  # Reset für nächste Epoche

# Nach Training: Alle gesammelten Daten nutzen
# für Embedding-Training, Analyse, etc.
    """)

    print("\n✓ Code-Beispiel oben zeigt Integration!")


def methode_5_streaming():
    """Streaming: Sammle während du renderst (große Datasets)"""
    print("\n" + "=" * 60)
    print("METHODE 5: Streaming für große Datasets")
    print("=" * 60)

    rasterizer = AMDRasterizer(width=512, height=512)
    collector = RasterizationDataCollector(save_dir='./data')

    # Große Anzahl Scenes
    total_scenes = 1000
    save_interval = 100  # Speichere alle 100 Scenes

    print(f"\nSammle {total_scenes} Scenes (Streaming Mode)...")

    for i in range(total_scenes):
        vertices = np.random.rand(25, 3, 3).astype(np.float32) * 512
        colors = np.random.rand(25, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float32)

        _, training_data = rasterizer.rasterize_triangles(
            vertices, colors, mvp, collect_data=True
        )

        collector.collect_from_rasterization(
            training_data['barycentric_coords'],
            training_data['depth'],
            training_data['coverage']
        )

        # Speichere periodisch (verhindert Memory-Overflow)
        if (i + 1) % save_interval == 0:
            filename = f'stream_part_{(i+1)//save_interval}.h5'
            collector.save_dataset(filename)
            print(f"  Saved part {(i+1)//save_interval}: {len(collector.features)} samples")
            collector.clear()  # Clear memory

    # Letzter Rest
    if len(collector.features) > 0:
        collector.save_dataset('stream_part_final.h5')

    print(f"\n✓ Streaming complete! Daten in ./data/stream_part_*.h5")

    # Später: Alle Parts laden und kombinieren
    print("\nLaden aller Parts:")
    import glob
    import h5py

    all_features = []
    all_labels = []

    for filepath in sorted(glob.glob('./data/stream_part_*.h5')):
        with h5py.File(filepath, 'r') as f:
            all_features.append(f['features'][:])
            all_labels.append(f['labels'][:])
        print(f"  Loaded: {filepath}")

    # Kombinieren
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    print(f"\n✓ Total: {combined_features.shape[0]} samples")


def main():
    print("\n" + "=" * 60)
    print("GPU RASTERIZATION - DATENSAMMLUNGS-METHODEN")
    print("=" * 60)

    # Alle Methoden demonstrieren
    methode_1_einfach()
    methode_2_collector()
    methode_3_pytorch_dataset()
    methode_4_integration_mit_bestehendem_training()

    # Methode 5 ist sehr lang - nur Code-Beispiel
    # methode_5_streaming()  # Auskommentiert für schnellen Test

    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print("""
Welche Methode wählen?

1. EINFACH & SCHNELL:
   → methode_1_einfach()
   → Für einzelne Renders, Experimente

2. EMPFOHLEN FÜR TRAINING:
   → methode_2_collector()
   → Automatisches Sammeln & Speichern
   → HDF5 Format (komprimiert)

3. PYTORCH INTEGRATION:
   → methode_3_pytorch_dataset()
   → Direkt in DataLoader
   → Sofort trainierbar

4. BESTEHENDE PIPELINE:
   → methode_4_integration...()
   → Erweitere train.py
   → Parallel zu normalem Training

5. GROSSE DATASETS:
   → methode_5_streaming()
   → Verhindert Memory-Overflow
   → Periodisches Speichern
    """)

    print("\n" + "=" * 60)
    print("FERTIG!")
    print("=" * 60)


if __name__ == '__main__':
    main()
