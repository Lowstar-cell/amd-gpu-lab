# Datensammlung mit Rasterisierung

## Übersicht

Es gibt jetzt **zwei Versionen** des Datensammlungs-Scripts:

1. **`daten-sammeln.py`** (Original)
   - Sammelt nur ROCm GPU Metriken
   - Speichert als CSV

2. **`daten-sammeln-mit-raster.py`** (Erweitert) ⭐ NEU
   - Sammelt ROCm GPU Metriken (CSV)
   - **+ Sammelt GPU Rasterisierungs-Features (HDF5)**
   - Beides parallel!

## Installation

```bash
# Haupt-Dependencies (sollten bereits installiert sein)
pip install h5py scipy numpy

# Falls noch nicht geschehen: PyTorch mit ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

## Verwendung

### Nur ROCm Metriken (wie bisher)
```bash
python daten-sammeln.py --interval 2.0
```

### ROCm Metriken + Rasterisierung
```bash
python daten-sammeln-mit-raster.py --interval 2.0 --with-raster
```

### Erweiterte Optionen
```bash
python daten-sammeln-mit-raster.py \
  --interval 2.0 \
  --with-raster \
  --raster-width 256 \
  --raster-height 256 \
  --raster-triangles 20 \
  --run-name "gaming_session_1"
```

## Parameter

- `--interval` - Sekunden zwischen Messungen (default: 2.0)
- `--with-raster` - Aktiviert Rasterisierungs-Datensammlung
- `--raster-width` - Rasterizer Breite (default: 256)
- `--raster-height` - Rasterizer Höhe (default: 256)
- `--raster-triangles` - Anzahl Dreiecke pro Scene (default: 20)
- `--run-name` - Name für diesen Run
- `--out-dir` - Output Verzeichnis (default: get_data)

## Output

Jeder Run erstellt ein Verzeichnis: `get_data/run_TIMESTAMP_ID/`

Enthält:
- `rocm_smi.csv` - GPU Metriken (immer)
- `rasterization_data.h5` - Rasterisierungs-Features (wenn --with-raster)
- `metadata.json` - Run Informationen
- `done` - Flag dass Run abgeschlossen ist

## HDF5 Format

Die Rasterisierungsdaten werden im HDF5 Format gespeichert:

```python
import h5py

with h5py.File('rasterization_data.h5', 'r') as f:
    features = f['features'][:]  # (N, 7) array
    labels = f['labels'][:]      # (N,) array

    # Feature columns:
    # [0-2]: Barycentric coords (w0, w1, w2)
    # [3]:   Depth
    # [4-5]: Screen position (normalized x, y)
    # [6]:   Coverage flag
```

## Integration in bestehende Pipelines

### Beispiel: Während des Gamings
```bash
# Terminal 1: Starte Game
steam

# Terminal 2: Sammle Daten während du spielst
python daten-sammeln-mit-raster.py --with-raster --interval 1.0 --run-name "doom_eternal_session"
```

### Beispiel: Kombiniere mehrere Runs
```python
from gpu_rasterization.combine_gaming_data import combine_h5_files

# Kombiniert alle .h5 Dateien
combine_h5_files('get_data/*/rasterization_data.h5', output='combined_gaming.h5')
```

## Verwendung der gesammelten Daten

```python
import h5py
import numpy as np

# Lade Daten
with h5py.File('get_data/run_XYZ/rasterization_data.h5', 'r') as f:
    features = f['features'][:]
    labels = f['labels'][:]

print(f"Gesammelte Samples: {features.shape[0]}")

# Für PyTorch Training
from gpu_rasterization.data_collection.collector import RasterizationDataCollector

collector = RasterizationDataCollector(save_dir='./data')
features, labels = collector.load_dataset('rasterization_data.h5')

dataset = collector.get_pytorch_dataset()
# ... weiter mit Training
```

## Troubleshooting

### "Module gpu_rasterization not found"
Das Script sucht nach dem gpu_rasterization Modul im Parent Directory.
Stelle sicher, dass du das Script aus dem `get_data/` Ordner ausführst:
```bash
cd get_data
python daten-sammeln-mit-raster.py --with-raster
```

### "rocm-smi not found"
ROCm ist nicht installiert oder nicht im PATH.
Das Script benötigt ROCm für GPU Metriken.

### "h5py not found"
Installiere h5py:
```bash
pip install h5py
```

## Vergleich zu vorher

**Vorher (nur daten-sammeln.py):**
- ✓ ROCm Metriken (GPU load, VRAM, temp, power)
- ✗ Keine Rasterisierungs-Features

**Jetzt (daten-sammeln-mit-raster.py):**
- ✓ ROCm Metriken (wie vorher)
- ✓ Rasterisierungs-Features (barycentric, depth, coverage)
- ✓ Beide in einem Run
- ✓ HDF5 Format (komprimiert, effizient)
- ✓ Direkt kompatibel mit bestehendem Training Code
