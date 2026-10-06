"""
Erweitertes Datensammlungs-Script:
- ROCm GPU Metriken (rocm-smi) -> CSV
- GPU Rasterisierungs-Features -> HDF5
"""

import argparse
import subprocess
import time
import csv
import os
import json
import socket
import uuid
import sys
from datetime import datetime
from pathlib import Path

# Import rasterization collector (falls verfügbar)
try:
    # Add parent directory to path to import gpu_rasterization modules
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    from gpu_rasterization.gpu_rasterize import AMDRasterizer
    from gpu_rasterization.data_collection.collector import RasterizationDataCollector
    import numpy as np
    RASTERIZATION_AVAILABLE = True
except ImportError as e:
    print(f"[INFO] Rasterization nicht verfügbar: {e}")
    print("       Nur ROCm Metriken werden gesammelt.")
    RASTERIZATION_AVAILABLE = False

# CLI / Konfiguration
parser = argparse.ArgumentParser(
    description='Sammle ROCm GPU Metriken und optional Rasterisierungs-Daten'
)
parser.add_argument('--interval', type=float, default=2.0, help='Seconds between measurements')
parser.add_argument('--out-dir', default='get_data', help='Top-level output folder for runs')
parser.add_argument('--run-name', default=None, help='Optional run name prefix')
parser.add_argument('--no-meta', action='store_true', help="Don't write metadata.json")
parser.add_argument('--with-raster', action='store_true', help='Sammle auch Rasterisierungs-Daten')
parser.add_argument('--raster-width', type=int, default=256, help='Rasterizer width (default: 256)')
parser.add_argument('--raster-height', type=int, default=256, help='Rasterizer height (default: 256)')
parser.add_argument('--raster-triangles', type=int, default=20, help='Number of triangles per scene (default: 20)')
args = parser.parse_args()

INTERVAL = float(args.interval)

# Build a safe run directory
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
uniq = uuid.uuid4().hex[:8]
run_id = args.run_name or f"run_{timestamp_str}_{uniq}"
RUN_DIR = os.path.join(args.out_dir, run_id)
os.makedirs(RUN_DIR, exist_ok=True)

# CSV filename for ROCm metrics
base_logfile = "rocm_smi.csv"
LOG_FILE = os.path.join(RUN_DIR, base_logfile)
if os.path.exists(LOG_FILE):
    for i in range(1, 1000):
        candidate = os.path.join(RUN_DIR, f"rocm_smi_{i}.csv")
        if not os.path.exists(candidate):
            LOG_FILE = candidate
            break

# Setup Rasterization collector (if enabled)
rasterizer = None
raster_collector = None
if args.with_raster:
    if not RASTERIZATION_AVAILABLE:
        print("[ERROR] --with-raster angegeben, aber Rasterization Module nicht verfügbar!")
        print("        Installiere zuerst: pip install h5py scipy")
        print("        Und stelle sicher, dass gpu_rasterization Module vorhanden sind.")
        sys.exit(1)

    print(f"[RASTER] Initialisiere AMDRasterizer ({args.raster_width}x{args.raster_height})...")
    try:
        rasterizer = AMDRasterizer(width=args.raster_width, height=args.raster_height)
        raster_collector = RasterizationDataCollector(save_dir=RUN_DIR)
        print("[RASTER] OK - Rasterisierungs-Datensammlung aktiviert")
    except Exception as e:
        print(f"[ERROR] Konnte Rasterizer nicht initialisieren: {e}")
        sys.exit(1)


def get_gpu_stats():
    """Sammle ROCm GPU Metriken via rocm-smi"""
    try:
        cmd = [
            "rocm-smi",
            "--showuse",      # GPU Auslastung
            "--showmeminfo", "vram",  # VRAM
            "--showtemp",     # Temperatur
            "--showpower",    # Stromverbrauch
            "--csv"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Fehler bei der Ausführung von rocm-smi: {result.stderr.strip()}")
            return None, None

        lines = result.stdout.strip().split(os.linesep)
        if len(lines) < 2:
            print(f"Unerwartete Ausgabe von rocm-smi. stdout: {result.stdout.strip()}")
            return None, None

        header = lines[0].split(',')
        data = lines[1].split(',')

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ["Timestamp"] + header, [timestamp] + data

    except FileNotFoundError:
        print("Fehler: 'rocm-smi' nicht gefunden. Ist ROCm installiert?")
        return None, None


def collect_rasterization_sample():
    """Sammle einen Rasterisierungs-Sample"""
    if rasterizer is None or raster_collector is None:
        return

    try:
        # Erstelle zufällige Test-Scene
        num_tris = args.raster_triangles
        vertices = np.random.rand(num_tris, 3, 3).astype(np.float32) * args.raster_width
        colors = np.random.rand(num_tris, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float32)

        # Rasterize mit Datensammlung
        _, training_data = rasterizer.rasterize_triangles(
            vertices, colors, mvp,
            collect_data=True
        )

        # Sammeln
        raster_collector.collect_from_rasterization(
            training_data['barycentric_coords'],
            training_data['depth'],
            training_data['coverage']
        )

    except Exception as e:
        print(f"[RASTER ERROR] {e}")


def _probe_rocm_smi_version():
    try:
        r = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            return r.stdout.strip()
        return r.stderr.strip() or None
    except Exception:
        return None


def _write_metadata(start_time, run_dir, interval, pid, raster_enabled):
    meta = {
        'run_id': os.path.basename(run_dir),
        'start_time': start_time.isoformat(),
        'interval_s': interval,
        'pid': pid,
        'host': socket.gethostname(),
        'rocm_smi_version': _probe_rocm_smi_version(),
        'rasterization_enabled': raster_enabled,
    }
    if raster_enabled:
        meta['raster_config'] = {
            'width': args.raster_width,
            'height': args.raster_height,
            'triangles_per_scene': args.raster_triangles
        }

    try:
        with open(os.path.join(run_dir, 'metadata.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
    except Exception:
        pass


def _mark_finished(rows_written: int):
    dpath = os.path.join(RUN_DIR, 'metadata.json')
    try:
        meta = {}
        if os.path.exists(dpath):
            with open(dpath, 'r') as fh:
                meta = json.load(fh)
        meta['end_time'] = datetime.now().isoformat()
        meta['rows'] = rows_written

        if raster_collector:
            meta['raster_samples'] = len(raster_collector.features)

        with open(dpath, 'w') as fh:
            json.dump(meta, fh, indent=2)

        open(os.path.join(RUN_DIR, 'done'), 'w').close()
    except Exception:
        pass


print("=" * 70)
print("ERWEITERTES DATENSAMMLUNGS-SCRIPT")
print("=" * 70)
print(f"Run directory: {RUN_DIR}")
print(f"ROCm CSV:      {LOG_FILE}")
if args.with_raster:
    print(f"Rasterization: AKTIVIERT ({args.raster_width}x{args.raster_height}, {args.raster_triangles} triangles)")
else:
    print(f"Rasterization: deaktiviert (nutze --with-raster zum Aktivieren)")
print(f"\nStrg+C zum Beenden")
print("=" * 70)

# Header nur schreiben, wenn Datei neu ist
file_exists = os.path.isfile(LOG_FILE)

# save metadata
start_time = datetime.now()
if not args.no_meta:
    _write_metadata(start_time, RUN_DIR, INTERVAL, os.getpid(), args.with_raster)

rows = 0
raster_samples_count = 0

try:
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        running = True

        def _handle_exit(signum, frame):
            nonlocal running
            running = False

        import signal
        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        while running:
            # Sammle ROCm Metriken
            header, data = get_gpu_stats()

            if header and data:
                if not file_exists:
                    writer.writerow(header)
                    file_exists = True

                writer.writerow(data)
                f.flush()
                rows += 1

                # Status ausgeben
                try:
                    header = [h.strip() for h in header]
                    load_index = header.index("GPU use (%)")
                    vram_candidates = [i for i, h in enumerate(header) if 'VRAM used' in h or 'VRAM Total Used' in h]
                    vram_index = vram_candidates[0] if vram_candidates else None

                    status = f"[{data[0]}] GPU: {data[load_index + 1].strip()}%"
                    if vram_index is not None:
                        status += f" | VRAM: {data[vram_index + 1].strip()}"

                    if args.with_raster:
                        status += f" | Raster: {len(raster_collector.features)} samples"

                    print(status)
                except (ValueError, IndexError):
                    print(f"Logged: {data[0]}")

            elif data is None:
                print("Fehler beim Abrufen der GPU-Statistiken. Das Skript wird beendet.")
                break

            # Sammle Rasterization Sample (falls aktiviert)
            if args.with_raster:
                collect_rasterization_sample()

            time.sleep(INTERVAL)

except KeyboardInterrupt:
    print('\n\nLogging beendet (KeyboardInterrupt).')
finally:
    _mark_finished(rows)
    print("\n" + "=" * 70)
    print("RUN ABGESCHLOSSEN")
    print("=" * 70)
    print(f"ROCm Metriken:  {rows} Zeilen -> {LOG_FILE}")

    if args.with_raster and raster_collector:
        if len(raster_collector.features) > 0:
            # Speichere Rasterization Daten
            h5_file = raster_collector.save_dataset('rasterization_data.h5')
            print(f"Raster Samples: {len(raster_collector.features)} -> {h5_file}")
        else:
            print(f"Raster Samples: 0 (keine Daten gesammelt)")

    print("=" * 70)
