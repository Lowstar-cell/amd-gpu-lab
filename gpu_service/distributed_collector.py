"""
Verteilter Datensammler - Für Kollegen die mitsammeln wollen

ANLEITUNG FÜR KOLLEGEN:
=======================

1. Repo klonen oder diese Datei + gpu_service Ordner kopieren

2. Requirements installieren:
   pip install torch numpy h5py

3. Script starten:
   python distributed_collector.py --user "KollegenName"

4. Gesammelte Daten exportieren:
   python distributed_collector.py --export

5. Die .h5 Datei an den Hauptentwickler schicken

Das wars! Jede AMD GPU Nutzung verbessert das Model für alle.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[WARNUNG] h5py nicht installiert - pip install h5py")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNUNG] torch nicht installiert")


@dataclass
class WorkloadRecord:
    """Ein aufgezeichneter Workload"""
    timestamp: float
    user_id: str
    gpu_name: str
    operation: str
    input_shapes: List[tuple]
    output_shape: tuple
    dtype: str
    elapsed_ms: float
    strategy_used: str
    features: List[float]
    success: bool
    error_msg: str = ""


class DistributedCollector:
    """
    Sammelt Workload-Daten für verteiltes Training.

    Features:
    - Automatische GPU-Erkennung
    - Anonymisierte User-IDs
    - Komprimierte Speicherung
    - Einfacher Export
    """

    def __init__(self, user_name: str = "anonymous", data_dir: str = "./collected_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Anonymisierte User-ID (Hash vom Namen)
        self.user_id = hashlib.sha256(user_name.encode()).hexdigest()[:12]
        self.user_name = user_name

        # GPU Info
        self.gpu_name = self._detect_gpu()

        # Collection buffer
        self.records: List[WorkloadRecord] = []
        self.lock = threading.Lock()

        # Stats
        self.session_start = time.time()
        self.total_collected = 0

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║         AMD GPU DATENSAMMLER - Verteiltes Lernen             ║
╠══════════════════════════════════════════════════════════════╣
║  User:     {user_name[:20]:<20}                              ║
║  User-ID:  {self.user_id:<12}                                ║
║  GPU:      {self.gpu_name[:30]:<30}                          ║
║  Daten:    {str(self.data_dir)[:40]:<40}      ║
╚══════════════════════════════════════════════════════════════╝
        """)

    def _detect_gpu(self) -> str:
        """Erkennt AMD GPU"""
        if not HAS_TORCH:
            return "unknown (torch not installed)"

        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                return name
            except:
                return "CUDA/ROCm GPU (name unknown)"
        return "CPU only"

    def record_workload(
        self,
        operation: str,
        input_shapes: List[tuple],
        output_shape: tuple,
        elapsed_ms: float,
        strategy_used: str = "default",
        dtype: str = "float32",
        success: bool = True,
        error_msg: str = "",
        features: Optional[np.ndarray] = None
    ):
        """
        Zeichnet einen Workload auf.

        Args:
            operation: Name der Operation (z.B. "rasterize", "matmul")
            input_shapes: Liste von Input-Shapes
            output_shape: Output-Shape
            elapsed_ms: Ausführungszeit in Millisekunden
            strategy_used: Welche Optimierung wurde verwendet
            dtype: Datentyp
            success: War die Ausführung erfolgreich?
            error_msg: Fehlermeldung falls nicht erfolgreich
            features: Optional vorberechnete Features
        """
        if features is None:
            features = self._extract_features(operation, input_shapes, output_shape, dtype)

        record = WorkloadRecord(
            timestamp=time.time(),
            user_id=self.user_id,
            gpu_name=self.gpu_name,
            operation=operation,
            input_shapes=[tuple(s) for s in input_shapes],
            output_shape=tuple(output_shape),
            dtype=dtype,
            elapsed_ms=elapsed_ms,
            strategy_used=strategy_used,
            features=features.tolist() if isinstance(features, np.ndarray) else list(features),
            success=success,
            error_msg=error_msg
        )

        with self.lock:
            self.records.append(record)
            self.total_collected += 1

        if self.total_collected % 100 == 0:
            print(f"[Collector] {self.total_collected} samples gesammelt")

    def _extract_features(
        self,
        operation: str,
        input_shapes: List[tuple],
        output_shape: tuple,
        dtype: str
    ) -> np.ndarray:
        """Extrahiert Features aus Workload"""
        features = []

        # Operation hash
        features.append(hash(operation) % 10000)

        # Input shapes (flatten, max 2 inputs mit je 4 dims)
        for i, shape in enumerate(input_shapes[:2]):
            flat = list(shape)[:4]
            features.extend(flat + [0] * (4 - len(flat)))
        # Padding falls weniger als 2 inputs
        while len(features) < 9:
            features.extend([0, 0, 0, 0])

        # Output shape
        out_flat = list(output_shape)[:4]
        features.extend(out_flat + [0] * (4 - len(out_flat)))

        # Dtype
        dtype_map = {"float32": 4, "float16": 2, "float64": 8, "int32": 4, "int64": 8}
        features.append(dtype_map.get(dtype, 4))

        # Total elements
        features.append(np.log10(np.prod(output_shape) + 1))

        # Compute intensity estimate
        total_input = sum(np.prod(s) for s in input_shapes)
        features.append(np.log10(total_input + 1))

        return np.array(features, dtype=np.float32)

    def save(self, filename: Optional[str] = None) -> str:
        """
        Speichert gesammelte Daten.

        Returns:
            Pfad zur gespeicherten Datei
        """
        if not self.records:
            print("[Collector] Keine Daten zum Speichern")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collected_{self.user_id}_{timestamp}.h5"

        filepath = self.data_dir / filename

        with self.lock:
            records_to_save = list(self.records)

        if HAS_H5PY:
            self._save_h5(filepath, records_to_save)
        else:
            # Fallback zu JSON
            filepath = filepath.with_suffix('.json')
            self._save_json(filepath, records_to_save)

        print(f"[Collector] {len(records_to_save)} samples gespeichert: {filepath}")
        return str(filepath)

    def _save_h5(self, filepath: Path, records: List[WorkloadRecord]):
        """Speichert als HDF5"""
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['user_id'] = self.user_id
            f.attrs['gpu_name'] = self.gpu_name
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['num_records'] = len(records)

            # Features
            features = np.array([r.features for r in records], dtype=np.float32)
            f.create_dataset('features', data=features, compression='gzip')

            # Performance
            elapsed = np.array([r.elapsed_ms for r in records], dtype=np.float32)
            f.create_dataset('elapsed_ms', data=elapsed, compression='gzip')

            # Strategies
            strategies = [r.strategy_used for r in records]
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('strategies', data=strategies, dtype=dt)

            # Operations
            operations = [r.operation for r in records]
            f.create_dataset('operations', data=operations, dtype=dt)

            # Success
            success = np.array([r.success for r in records], dtype=bool)
            f.create_dataset('success', data=success)

            # Timestamps
            timestamps = np.array([r.timestamp for r in records], dtype=np.float64)
            f.create_dataset('timestamps', data=timestamps)

    def _save_json(self, filepath: Path, records: List[WorkloadRecord]):
        """Fallback zu JSON"""
        data = {
            'metadata': {
                'user_id': self.user_id,
                'gpu_name': self.gpu_name,
                'created': datetime.now().isoformat(),
            },
            'records': [asdict(r) for r in records]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def clear(self):
        """Löscht Buffer"""
        with self.lock:
            self.records.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Statistiken"""
        elapsed = time.time() - self.session_start
        return {
            'user_id': self.user_id,
            'gpu_name': self.gpu_name,
            'total_collected': self.total_collected,
            'buffer_size': len(self.records),
            'session_minutes': elapsed / 60,
            'samples_per_minute': self.total_collected / (elapsed / 60) if elapsed > 60 else 0,
        }


# === Global Collector Instance ===
_collector: Optional[DistributedCollector] = None


def init_collector(user_name: str = "anonymous", data_dir: str = "./collected_data"):
    """Initialisiert den globalen Collector"""
    global _collector
    _collector = DistributedCollector(user_name, data_dir)
    return _collector


def get_collector() -> DistributedCollector:
    """Holt den globalen Collector"""
    global _collector
    if _collector is None:
        _collector = DistributedCollector()
    return _collector


def record(
    operation: str,
    input_shapes: List[tuple],
    output_shape: tuple,
    elapsed_ms: float,
    **kwargs
):
    """Convenience Funktion zum Aufzeichnen"""
    get_collector().record_workload(
        operation, input_shapes, output_shape, elapsed_ms, **kwargs
    )


# === Export Tools ===

def export_for_sharing(data_dir: str = "./collected_data", output_file: str = None) -> str:
    """
    Exportiert alle gesammelten Daten in eine Datei zum Teilen.

    Returns:
        Pfad zur Export-Datei
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[Export] Verzeichnis nicht gefunden: {data_dir}")
        return ""

    # Finde alle .h5 und .json Dateien
    files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.json"))
    if not files:
        print(f"[Export] Keine Daten gefunden in {data_dir}")
        return ""

    print(f"[Export] Gefunden: {len(files)} Dateien")

    # Kombiniere alle Daten
    all_features = []
    all_elapsed = []
    all_strategies = []
    all_operations = []
    metadata = []

    for f in files:
        try:
            if f.suffix == '.h5' and HAS_H5PY:
                with h5py.File(f, 'r') as h5:
                    all_features.append(h5['features'][:])
                    all_elapsed.append(h5['elapsed_ms'][:])
                    all_strategies.extend(h5['strategies'][:])
                    all_operations.extend(h5['operations'][:])
                    metadata.append({
                        'file': f.name,
                        'user_id': h5.attrs.get('user_id', 'unknown'),
                        'gpu_name': h5.attrs.get('gpu_name', 'unknown'),
                        'num_records': h5.attrs.get('num_records', 0),
                    })
            elif f.suffix == '.json':
                with open(f) as jf:
                    data = json.load(jf)
                    for r in data.get('records', []):
                        all_features.append(r['features'])
                        all_elapsed.append(r['elapsed_ms'])
                        all_strategies.append(r['strategy_used'])
                        all_operations.append(r['operation'])
        except Exception as e:
            print(f"[Export] Fehler bei {f}: {e}")

    if not all_features:
        print("[Export] Keine Daten zum Exportieren")
        return ""

    # Speichere kombinierte Datei
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"export_combined_{timestamp}.h5"

    output_path = data_dir / output_file

    if HAS_H5PY:
        features_arr = np.vstack(all_features) if isinstance(all_features[0], np.ndarray) else np.array(all_features)
        elapsed_arr = np.concatenate(all_elapsed) if isinstance(all_elapsed[0], np.ndarray) else np.array(all_elapsed)

        with h5py.File(output_path, 'w') as f:
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['num_records'] = len(features_arr)
            f.attrs['num_sources'] = len(files)
            f.attrs['sources'] = json.dumps(metadata)

            f.create_dataset('features', data=features_arr.astype(np.float32), compression='gzip')
            f.create_dataset('elapsed_ms', data=elapsed_arr.astype(np.float32), compression='gzip')

            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('strategies', data=[s if isinstance(s, str) else s.decode() for s in all_strategies], dtype=dt)
            f.create_dataset('operations', data=[o if isinstance(o, str) else o.decode() for o in all_operations], dtype=dt)

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPORT ERFOLGREICH                        ║
╠══════════════════════════════════════════════════════════════╣
║  Datei:   {str(output_path)[:45]:<45} ║
║  Records: {len(features_arr):<10}                                     ║
║  Quellen: {len(files):<10}                                     ║
╠══════════════════════════════════════════════════════════════╣
║  Diese Datei kannst du jetzt teilen!                         ║
╚══════════════════════════════════════════════════════════════╝
        """)

    return str(output_path)


# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="AMD GPU Datensammler")
    parser.add_argument('--user', '-u', default='anonymous', help='Dein Name (wird anonymisiert)')
    parser.add_argument('--data-dir', '-d', default='./collected_data', help='Datenverzeichnis')
    parser.add_argument('--export', '-e', action='store_true', help='Daten exportieren')
    parser.add_argument('--demo', action='store_true', help='Demo-Modus mit simulierten Daten')

    args = parser.parse_args()

    if args.export:
        export_for_sharing(args.data_dir)
        return

    collector = init_collector(args.user, args.data_dir)

    if args.demo:
        print("\n[DEMO] Simuliere Workloads...")
        for i in range(50):
            # Simuliere verschiedene Workloads
            ops = ["rasterize", "matmul", "conv2d", "reduce", "softmax"]
            op = np.random.choice(ops)

            input_shapes = [(np.random.randint(1, 1000), np.random.randint(1, 1000))
                           for _ in range(np.random.randint(1, 3))]
            output_shape = (np.random.randint(1, 1000), np.random.randint(1, 1000))

            elapsed = np.random.uniform(0.1, 50.0)
            strategy = np.random.choice(["default", "memory_coalesce", "wavefront_tune"])

            collector.record_workload(
                operation=op,
                input_shapes=input_shapes,
                output_shape=output_shape,
                elapsed_ms=elapsed,
                strategy_used=strategy
            )

            time.sleep(0.02)

        collector.save()
        print(f"\n[DEMO] Stats: {collector.get_stats()}")
        return

    # Interaktiver Modus
    print("""
[INFO] Collector läuft im Hintergrund.

Nutze in deinem Code:
    from gpu_service.distributed_collector import record

    # Nach jedem Workload:
    record("rasterize", [(256, 256)], (256, 256, 3), elapsed_ms=5.3)

Oder starte mit --demo für Testdaten.
Exportiere mit --export zum Teilen.
    """)

    try:
        while True:
            time.sleep(60)
            if collector.total_collected > 0 and collector.total_collected % 500 == 0:
                collector.save()
    except KeyboardInterrupt:
        print("\n[Collector] Speichere und beende...")
        collector.save()
        print(f"[Collector] Final Stats: {collector.get_stats()}")


if __name__ == "__main__":
    main()
