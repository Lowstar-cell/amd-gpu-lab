"""
EINFACHSTES Datensammlungs-Beispiel
Copy & Paste Ready!
"""

from gpu_rasterize import AMDRasterizer
from data_collection.collector import RasterizationDataCollector
import numpy as np

print("=" * 60)
print("EINFACHE DATENSAMMLUNG")
print("=" * 60)

# 1. Setup
print("\n[1] Setup...")
rasterizer = AMDRasterizer(width=256, height=256)
collector = RasterizationDataCollector(save_dir='./data')

# 2. Sammle Daten
print("\n[2] Sammle Daten aus 50 Scenes...")

samples_collected = 0

for scene_idx in range(50):
    # Erstelle zufällige Triangles im NDC-Raum [-1, 1]
    num_triangles = 10
    vertices = []

    for _ in range(num_triangles):
        # Drei Punkte im NDC Raum [-1, 1] x [-1, 1]
        # (wird automatisch zu Screen-Space transformiert)
        x1, y1 = np.random.rand(2) * 2 - 1  # Range: [-1, 1]
        x2, y2 = np.random.rand(2) * 2 - 1
        x3, y3 = np.random.rand(2) * 2 - 1

        # Depth: zufällig zwischen 0 und 1 (NDC depth)
        z = np.random.rand()

        triangle = np.array([
            [x1, y1, z],
            [x2, y2, z],
            [x3, y3, z]
        ], dtype=np.float32)

        vertices.append(triangle)

    vertices = np.array(vertices)

    # Zufällige Farben
    colors = np.random.rand(num_triangles, 3).astype(np.float32)

    # MVP Matrix (Identity = NDC coords direkt)
    mvp = np.eye(4, dtype=np.float32)

    # Rasterize mit Datensammlung
    image, features = rasterizer.rasterize_triangles(
        vertices,
        colors,
        mvp,
        collect_data=True  # <- WICHTIG!
    )

    # Sammeln
    if features is not None:
        collector.collect_from_rasterization(
            features['barycentric_coords'],
            features['depth'],
            features['coverage']
        )

        samples_collected += (features['coverage'].sum())

    # Progress
    if (scene_idx + 1) % 10 == 0:
        print(f"  Scenes: {scene_idx + 1}/50 | Samples: {len(collector.features)}")

# 3. Speichern
print(f"\n[3] Speichern...")
filepath = collector.save_dataset('simple_data.h5')

print(f"\n[OK] Fertig!")
print(f"  Gesammelte Samples: {len(collector.features)}")

if filepath:
    print(f"  Gespeichert in:     {filepath}")

    # 4. Laden (Test)
    print(f"\n[4] Test: Daten wieder laden...")
    features, labels = collector.load_dataset('simple_data.h5')

    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape:   {labels.shape}")
    if len(features) > 0:
        print(f"  Feature example: {features[0]}")
else:
    print(f"  WARNUNG: Keine Daten gesammelt!")

# 5. PyTorch Dataset (optional)
print(f"\n[5] Als PyTorch Dataset...")
dataset = collector.get_pytorch_dataset()
print(f"  Dataset size: {len(dataset)}")

if len(dataset) > 0:
    sample_features, sample_label = dataset[0]
    print(f"  Sample features: {sample_features.shape}")
    print(f"  Sample label:    {sample_label}")

print("\n" + "=" * 60)
print("SUCCESS! Datensammlung funktioniert!")
print("=" * 60)
print("""
Naechste Schritte:

1. In Training integrieren:
   - Copy Code in train.py
   - In Training-Loop einbauen

2. Mehr Daten sammeln:
   - Erhoehe Scenes auf 1000+
   - Mehr Triangles pro Scene
   - Hoehere Aufloesung

3. Mit echten Daten:
   - Ersetze random vertices mit echten 3D Models
   - Nutze reale Scenes

4. Training starten:
   dataset = collector.get_pytorch_dataset()
   dataloader = torch.utils.data.DataLoader(dataset, ...)
   # Train!
""")
