# AMD GPU Datensammler - Anleitung für Kollegen

## Schnellstart (3 Minuten)

### 1. Dateien kopieren
Du brauchst nur den `gpu_service` Ordner:
```
gpu_service/
├── __init__.py
├── distributed_collector.py   <-- Das ist das wichtige
├── embedding_service.py
└── client.py
```

### 2. Requirements installieren
```bash
pip install torch numpy h5py
```

### 3. Sammeln starten
```bash
# Mit deinem Namen (wird automatisch anonymisiert)
python gpu_service/distributed_collector.py --user "DeinName"

# Oder Demo-Modus zum Testen
python gpu_service/distributed_collector.py --user "DeinName" --demo
```

### 4. Daten exportieren & teilen
```bash
python gpu_service/distributed_collector.py --export
```

Die Export-Datei (`export_combined_*.h5`) an mich schicken!

---

## Was wird gesammelt?

- **Workload-Features**: Welche Operationen (matmul, conv2d, rasterize, etc.)
- **Performance**: Wie lange hat es gedauert (in ms)
- **GPU-Info**: Welche AMD GPU du hast
- **Strategie**: Welche Optimierung wurde verwendet

**Nicht gesammelt:**
- Deine echten Daten (nur Shapes wie "1024x1024")
- Persönliche Infos (Name wird gehasht)
- Nichts was nicht GPU-bezogen ist

---

## Integration in eigenen Code

```python
from gpu_service.distributed_collector import record, init_collector

# Einmal am Start
init_collector("DeinName")

# Nach jedem GPU-Workload
import time
start = time.perf_counter()

# ... dein GPU code ...

elapsed_ms = (time.perf_counter() - start) * 1000
record(
    operation="conv2d",
    input_shapes=[(batch, channels, height, width)],
    output_shape=(batch, out_channels, out_h, out_w),
    elapsed_ms=elapsed_ms
)
```

---

## Warum mitmachen?

Jeder gesammelte Datenpunkt hilft dem Model zu lernen:
- Welche Workloads auf AMD GPUs gut/schlecht laufen
- Welche Optimierungen bei welchen Patterns helfen
- Wie man näher an CUDA-Performance kommt

**Ziel: 70% CUDA-Performance auf AMD** - und mit deiner Hilfe kommen wir dahin schneller!

---

## Fragen?

Einfach melden - je mehr Daten von verschiedenen GPUs, desto besser wird das Model!
