# Datensammlung - Quick Reference

## Schnellste Methode (Copy & Paste Ready!)

```python
from gpu_rasterization.gpu_rasterize import AMDRasterizer
from gpu_rasterization.data_collection.collector import RasterizationDataCollector
import numpy as np

# 1. Setup
rasterizer = AMDRasterizer(width=512, height=512)
collector = RasterizationDataCollector(save_dir='./data')

# 2. Daten sammeln (Loop über deine Scenes)
for i in range(100):  # 100 Scenes
    # Erstelle oder lade deine 3D Daten
    vertices = np.random.rand(20, 3, 3).astype(np.float32) * 512
    colors = np.random.rand(20, 3).astype(np.float32)

    # MVP Matrix (Identity für screen-space coords)
    mvp = np.eye(4, dtype=np.float32)

    # Rasterize + sammeln
    image, features = rasterizer.rasterize_triangles(
        vertices, colors, mvp,
        collect_data=True  # <- WICHTIG!
    )

    # Automatisch sammeln
    collector.collect_from_rasterization(
        features['barycentric_coords'],
        features['depth'],
        features['coverage']
    )

# 3. Speichern
collector.save_dataset('my_data.h5')
print(f"Gesammelt: {len(collector.features)} Samples")

# 4. Für Training laden
dataset = collector.get_pytorch_dataset()
# -> Direkt in torch.utils.data.DataLoader nutzbar!
```

## Was wird gesammelt?

Für **jedes gerenderte Pixel**:

```python
# Feature-Vektor (7 Werte):
[
    w0, w1, w2,      # Barycentric coordinates (3)
    depth,           # Z-Wert (1)
    screen_x,        # Normalisierte X-Position (1)
    screen_y,        # Normalisierte Y-Position (1)
    coverage         # 1.0 = gerendert, 0.0 = leer (1)
]

# Label:
triangle_id  # Welches Triangle ist sichtbar
```

## Integration in train.py

```python
# Am Anfang von train.py:
from gpu_rasterization.gpu_rasterize import AMDRasterizer
from gpu_rasterization.data_collection.collector import RasterizationDataCollector

rasterizer = AMDRasterizer(width=256, height=256)
collector = RasterizationDataCollector(save_dir='./data/rasterization')

# In deiner Training-Loop:
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):

        # Normales Training
        output = model(data)
        loss.backward()
        optimizer.step()

        # ZUSÄTZLICH: Alle 10 Batches Features sammeln
        if batch_idx % 10 == 0:
            # Erstelle 3D Scene (oder nutze fixed Scene)
            vertices = create_vertices()  # Deine Methode
            colors = create_colors()
            mvp = np.eye(4, dtype=np.float32)

            _, features = rasterizer.rasterize_triangles(
                vertices, colors, mvp, collect_data=True
            )

            collector.collect_batch(features)

    # Nach Epoche: Save
    collector.save_dataset(f'epoch_{epoch}.h5')
    collector.clear()
```

## Methoden im Detail

### Methode 1: Direkt nutzen

```python
rasterizer = AMDRasterizer(512, 512)

image, features = rasterizer.rasterize_triangles(
    vertices, colors, mvp, collect_data=True
)

# features ist ein dict:
bary = features['barycentric_coords']  # (H, W, N_tri, 3)
depth = features['depth']               # (H, W)
coverage = features['coverage']         # (H, W) bool

# Direkt verarbeiten:
rendered_pixels = np.where(coverage)
for y, x in zip(*rendered_pixels):
    pixel_features = bary[y, x, :, :]  # Alle triangles
    pixel_depth = depth[y, x]
    # ... deine Verarbeitung
```

### Methode 2: Mit Collector (Empfohlen)

```python
collector = RasterizationDataCollector(save_dir='./data')

# Viele Scenes sammeln
for scene in scenes:
    _, features = rasterizer.rasterize_triangles(...)
    collector.collect_from_rasterization(
        features['barycentric_coords'],
        features['depth'],
        features['coverage']
    )

# Einmal speichern
collector.save_dataset('all_data.h5')

# Später laden
features, labels = collector.load_dataset('all_data.h5')
# features: (N_samples, 7)
# labels: (N_samples,) - triangle IDs
```

### Methode 3: PyTorch DataLoader

```python
# Sammeln (wie oben)
collector.collect_from_rasterization(...)

# Als PyTorch Dataset
dataset = collector.get_pytorch_dataset()

# In DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

# Training
for features, labels in dataloader:
    features = features.cuda()  # (batch, 7)
    labels = labels.cuda()       # (batch,)

    # Dein Model
    embeddings = model(features)
    loss = criterion(embeddings, labels)
    loss.backward()
```

### Methode 4: Streaming (Große Datasets)

```python
collector = RasterizationDataCollector(save_dir='./data')

for i in range(10000):  # Viele Scenes
    _, features = rasterizer.rasterize_triangles(...)
    collector.collect_from_rasterization(...)

    # Alle 100 Scenes: Save & Clear
    if (i + 1) % 100 == 0:
        collector.save_dataset(f'part_{i//100}.h5')
        collector.clear()  # Memory freigeben!
```

## Wo landen die Daten?

Standard-Speicherort: `./data/*.h5` (HDF5 Format, komprimiert)

Struktur einer .h5 Datei:
```
my_data.h5
├── features  (N_samples, 7) float32
└── labels    (N_samples,)   int32
```

## Zugriff auf gespeicherte Daten

```python
import h5py
import numpy as np

# Direkt mit h5py
with h5py.File('./data/my_data.h5', 'r') as f:
    features = f['features'][:]    # NumPy array
    labels = f['labels'][:]

    print(f"Samples: {features.shape[0]}")
    print(f"Feature dim: {features.shape[1]}")

# Oder mit Collector
collector = RasterizationDataCollector(save_dir='./data')
features, labels = collector.load_dataset('my_data.h5')
```

## Beispiel: Kompletter Workflow

```python
from gpu_rasterization.gpu_rasterize import AMDRasterizer
from gpu_rasterization.data_collection.collector import RasterizationDataCollector
import torch
import numpy as np

# 1. SAMMELN
print("Phase 1: Daten sammeln...")
rasterizer = AMDRasterizer(256, 256)
collector = RasterizationDataCollector('./data')

for i in range(500):
    # Zufällige Scene
    vertices = np.random.rand(20, 3, 3).astype(np.float32) * 256
    colors = np.random.rand(20, 3).astype(np.float32)
    mvp = np.eye(4, dtype=np.float32)

    # Rasterize
    _, features = rasterizer.rasterize_triangles(
        vertices, colors, mvp, collect_data=True
    )

    # Sammeln
    collector.collect_from_rasterization(
        features['barycentric_coords'],
        features['depth'],
        features['coverage']
    )

    if i % 100 == 0:
        print(f"  Scene {i}/500")

# Save
collector.save_dataset('training_data.h5')
print(f"Gespeichert: {len(collector.features)} Samples")

# 2. TRAINING
print("\nPhase 2: Training...")
dataset = collector.get_pytorch_dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

# Model
model = torch.nn.Sequential(
    torch.nn.Linear(7, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64)
).cuda()

optimizer = torch.optim.Adam(model.parameters())

# Train
for epoch in range(5):
    for features, labels in dataloader:
        features = features.cuda()
        labels = labels.cuda()

        output = model(features)
        loss = torch.nn.functional.cross_entropy(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}: Done")

print("\nFertig!")
```

## Tipps

**Performance:**
- CPU-Modus ist langsam (~100ms pro Frame)
- Mit ROCm/PyTorch GPU: ~5-10ms
- Mit HIP Kernels: ~1-3ms
- Kompiliere HIP Kernels: `build.bat` (Windows) oder `build.sh` (Linux)

**Speicher:**
- 1000 Scenes @ 512x512 = ~500MB .h5 Datei (komprimiert)
- Nutze Streaming für >10k Scenes
- `collector.clear()` um RAM freizugeben

**Datenqualität:**
- Mehr Triangles = mehr Samples
- Höhere Auflösung = mehr Samples
- Coverage sollte >50% sein (sonst zu viele leere Pixel)

## Fehlerbehebung

**"Collected 0 samples"**
- Triangles sind außerhalb des Screens
- Checke MVP Matrix
- Nutze vertices im Bereich [0, width] x [0, height]

**"Memory Error"**
- Nutze Streaming-Methode
- Reduziere Resolution
- `collector.clear()` öfter aufrufen

**"Slow Performance"**
- Installiere PyTorch mit ROCm
- Kompiliere HIP Kernels
- Reduziere Scenes-Anzahl für Tests
