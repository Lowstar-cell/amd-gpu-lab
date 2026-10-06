
import argparse
import subprocess
import time
import csv
import os
import json
import socket
import uuid
from datetime import datetime

# CLI / Konfiguration
parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=float, default=2.0, help='Seconds between measurements')
parser.add_argument('--out-dir', default='get_data', help='Top-level output folder for runs (default: get_data)')
parser.add_argument('--workload', default=None, help='Optional workload name / tag for this run')
parser.add_argument('--tags', default=None, help='Comma-separated freeform tags for this run')
parser.add_argument('--label', type=int, default=None, help='Optional integer label for this run (supervised)')
parser.add_argument('--feature-set', choices=['default', 'extended'], default='default', help='Which feature set to mark this run with')
parser.add_argument('--run-name', default=None, help='Optional run name prefix (if omitted a timestamp+UUID is used)')
parser.add_argument('--no-meta', action='store_true', help="Don't write metadata.json (for quick tests)")
args = parser.parse_args()

INTERVAL = float(args.interval)

# Build a safe run directory so we never overwrite existing runs
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
uniq = uuid.uuid4().hex[:8]
run_id = args.run_name or f"run_{timestamp_str}_{uniq}"
RUN_DIR = os.path.join(args.out_dir, run_id)
os.makedirs(RUN_DIR, exist_ok=True)

# CSV filename inside run dir - if present pick a non-clobbering name
base_logfile = "rocm_smi.csv"
LOG_FILE = os.path.join(RUN_DIR, base_logfile)
if os.path.exists(LOG_FILE):
    # append numeric suffix to avoid overwrite (unlikely due to unique run dir, but safe)
    for i in range(1, 1000):
        candidate = os.path.join(RUN_DIR, f"rocm_smi_{i}.csv")
        if not os.path.exists(candidate):
            LOG_FILE = candidate
            break

def get_gpu_stats():
    try:
        # Ruft rocm-smi im CSV-Modus auf
        # --show... flags definieren, was geloggt wird
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

        # rocm-smi gibt Header und Daten zurück
        header = lines[0].split(',')
        data = lines[1].split(',')
        
        # Zeitstempel hinzufügen
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ["Timestamp"] + header, [timestamp] + data

    except FileNotFoundError:
        print("Fehler: 'rocm-smi' nicht gefunden. Ist ROCm installiert?")
        return None, None

def _probe_rocm_smi_version():
    try:
        r = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            return r.stdout.strip()
        return r.stderr.strip() or None
    except Exception:
        return None


def _write_metadata(start_time, run_dir, interval, pid):
    meta = {
        'run_id': os.path.basename(run_dir),
        'start_time': start_time.isoformat(),
        'interval_s': interval,
        'pid': pid,
        'host': socket.gethostname(),
        'rocm_smi_version': _probe_rocm_smi_version(),
        # optional user-provided descriptors
        'workload': args.workload,
        'tags': args.tags.split(',') if args.tags else [],
        'label': args.label,
        'feature_set': args.feature_set,
    }
    try:
        with open(os.path.join(run_dir, 'metadata.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
    except Exception:
        pass


print(f"Starte Logging in run directory '{RUN_DIR}' -> file '{LOG_FILE}' (Strg+C zum Beenden)")

# Header nur schreiben, wenn Datei neu ist
file_exists = os.path.isfile(LOG_FILE)

# save metadata (if requested)
start_time = datetime.now()
if not args.no_meta:
    _write_metadata(start_time, RUN_DIR, INTERVAL, os.getpid())


def _mark_finished(rows_written: int):
    # update metadata to include end_time and rows
    dpath = os.path.join(RUN_DIR, 'metadata.json')
    try:
        meta = {}
        if os.path.exists(dpath):
            with open(dpath, 'r') as fh:
                meta = json.load(fh)
        meta['end_time'] = datetime.now().isoformat()
        meta['rows'] = rows_written
        with open(dpath, 'w') as fh:
            json.dump(meta, fh, indent=2)
        # create done flag
        open(os.path.join(RUN_DIR, 'done'), 'w').close()
    except Exception:
        pass


rows = 0
try:
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Install a simple signal-aware loop
        running = True

        def _handle_exit(signum, frame):
            nonlocal running
            running = False

        import signal
        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        while running:
            header, data = get_gpu_stats()

            if header and data:
                # Schreibe Header nur beim allerersten Mal
                if not file_exists:
                    writer.writerow(header)
                    file_exists = True

                writer.writerow(data)
                f.flush()  # Erzwingt Speichern
                rows += 1

                # Dynamisches Finden der Indizes für eine robustere Ausgabe
                try:
                    header = [h.strip() for h in header]
                    load_index = header.index("GPU use (%)")
                    # VRAM used might be expressed as 'VRAM used (%)' or 'VRAM Total Used Memory (B)'
                    vram_candidates = [i for i, h in enumerate(header) if 'VRAM used' in h or 'VRAM Total Used' in h]
                    vram_index = vram_candidates[0] if vram_candidates else None

                    # data-Indizes sind um 1 verschoben, wegen des Timestamps
                    if vram_index is not None:
                        print(f"Logged: {data[0]} | GPU Load: {data[load_index + 1].strip()}% | VRAM: {data[vram_index + 1].strip()}")
                    else:
                        print(f"Logged: {data[0]} | GPU Load: {data[load_index + 1].strip()}%")
                except (ValueError, IndexError):
                    # Fallback, wenn Spalten nicht gefunden werden
                    print(f"Logged: {data[0]}")

            elif data is None:
                print("Fehler beim Abrufen der GPU-Statistiken. Das Skript wird beendet.")
                break  # Beendet die Schleife bei einem Fehler

            time.sleep(INTERVAL)

except KeyboardInterrupt:
    print('\nLogging beendet (KeyboardInterrupt).')
finally:
    _mark_finished(rows)
    print(f"Run finished, wrote {rows} rows into {LOG_FILE}")