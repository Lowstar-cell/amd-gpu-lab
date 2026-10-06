"""
Standalone GPU Collector - Überwacht automatisch GPU-Nutzung

Dieses Script:
1. Erkennt die AMD GPU
2. Überwacht GPU-Auslastung
3. Sammelt Performance-Daten automatisch
4. Speichert in kompaktem Format
"""

import os
import sys
import time
import json
import struct
import ctypes
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

# === Minimale Dependencies - nur stdlib ===

class GPUMonitor:
    """Überwacht GPU ohne externe Dependencies"""

    def __init__(self):
        self.gpu_name = "Unknown AMD GPU"
        self.has_rocm_smi = False
        self._detect_gpu()

    def _detect_gpu(self):
        """Erkennt AMD GPU via rocm-smi oder WMI"""
        # Versuche rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'GPU' in line or 'Radeon' in line or 'AMD' in line:
                        self.gpu_name = line.strip()
                        self.has_rocm_smi = True
                        return
        except:
            pass

        # Windows: WMI
        if sys.platform == 'win32':
            try:
                import wmi
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if 'AMD' in gpu.Name or 'Radeon' in gpu.Name:
                        self.gpu_name = gpu.Name
                        return
            except:
                pass

            # Fallback: PowerShell
            try:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-WmiObject Win32_VideoController | Select-Object Name | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if isinstance(data, list):
                        for gpu in data:
                            name = gpu.get('Name', '')
                            if 'AMD' in name or 'Radeon' in name:
                                self.gpu_name = name
                                return
                    elif isinstance(data, dict):
                        name = data.get('Name', '')
                        if 'AMD' in name or 'Radeon' in name:
                            self.gpu_name = name
            except:
                pass

    def get_utilization(self) -> Dict[str, float]:
        """Holt GPU-Auslastung"""
        stats = {'gpu_util': 0.0, 'mem_util': 0.0, 'temp': 0.0}

        if self.has_rocm_smi:
            try:
                # GPU Utilization
                result = subprocess.run(
                    ["rocm-smi", "--showuse"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if '%' in line:
                        try:
                            val = float(line.split('%')[0].split()[-1])
                            stats['gpu_util'] = val
                        except:
                            pass

                # Memory
                result = subprocess.run(
                    ["rocm-smi", "--showmemuse"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if '%' in line:
                        try:
                            val = float(line.split('%')[0].split()[-1])
                            stats['mem_util'] = val
                        except:
                            pass

                # Temperature
                result = subprocess.run(
                    ["rocm-smi", "--showtemp"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'c' in line.lower() or 'C' in line:
                        try:
                            # Find number before 'c' or 'C'
                            parts = line.replace('°', ' ').replace('C', ' ').replace('c', ' ').split()
                            for p in parts:
                                try:
                                    val = float(p)
                                    if 20 < val < 120:  # Reasonable temp range
                                        stats['temp'] = val
                                        break
                                except:
                                    pass
                        except:
                            pass
            except:
                pass

        return stats


class SimpleCollector:
    """Einfacher Datensammler ohne externe Dependencies"""

    def __init__(self, user_name: str = "anonymous"):
        self.user_name = user_name
        # Simple hash
        self.user_id = hex(hash(user_name) & 0xFFFFFFFF)[2:]
        self.gpu = GPUMonitor()

        self.data_dir = Path("./amd_collected_data")
        self.data_dir.mkdir(exist_ok=True)

        self.samples: List[Dict] = []
        self.start_time = time.time()

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           AMD GPU COLLECTOR - Standalone Edition             ║
╠══════════════════════════════════════════════════════════════╣
║  User:     {user_name[:30]:<30}                ║
║  GPU:      {self.gpu.gpu_name[:40]:<40}║
╚══════════════════════════════════════════════════════════════╝
        """)

    def collect_sample(self):
        """Sammelt einen Datenpunkt"""
        stats = self.gpu.get_utilization()

        sample = {
            'timestamp': time.time(),
            'user_id': self.user_id,
            'gpu_name': self.gpu.gpu_name,
            'gpu_util': stats['gpu_util'],
            'mem_util': stats['mem_util'],
            'temp': stats['temp'],
        }
        self.samples.append(sample)

    def save(self) -> str:
        """Speichert Daten als JSON"""
        if not self.samples:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_data_{self.user_id}_{timestamp}.json"
        filepath = self.data_dir / filename

        data = {
            'metadata': {
                'user_id': self.user_id,
                'gpu_name': self.gpu.gpu_name,
                'created': datetime.now().isoformat(),
                'num_samples': len(self.samples),
                'duration_minutes': (time.time() - self.start_time) / 60,
            },
            'samples': self.samples
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Saved] {len(self.samples)} samples -> {filepath}")
        return str(filepath)

    def export_all(self) -> str:
        """Exportiert alle gesammelten Daten in eine Datei"""
        all_files = list(self.data_dir.glob("gpu_data_*.json"))
        if not all_files:
            print("[Export] Keine Daten gefunden")
            return ""

        all_samples = []
        metadata_list = []

        for f in all_files:
            try:
                with open(f) as jf:
                    data = json.load(jf)
                    all_samples.extend(data.get('samples', []))
                    metadata_list.append(data.get('metadata', {}))
            except Exception as e:
                print(f"[Export] Fehler bei {f}: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = self.data_dir / f"EXPORT_combined_{timestamp}.json"

        export_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'num_samples': len(all_samples),
                'num_sources': len(all_files),
                'sources': metadata_list,
            },
            'samples': all_samples
        }

        with open(export_file, 'w') as f:
            json.dump(export_data, f)

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPORT FERTIG!                            ║
╠══════════════════════════════════════════════════════════════╣
║  Datei:    {str(export_file.name)[:40]:<40}  ║
║  Samples:  {len(all_samples):<10}                                    ║
║                                                              ║
║  >>> Diese Datei bitte teilen! <<<                           ║
╚══════════════════════════════════════════════════════════════╝
        """)

        return str(export_file)

    def run_continuous(self, interval_seconds: float = 1.0, save_every: int = 300):
        """Läuft kontinuierlich und sammelt Daten"""
        print(f"\n[Running] Sammle alle {interval_seconds}s, speichere alle {save_every} samples")
        print("[Running] Drücke Ctrl+C zum Beenden\n")

        count = 0
        try:
            while True:
                self.collect_sample()
                count += 1

                if count % 10 == 0:
                    stats = self.gpu.get_utilization()
                    print(f"\r[{count}] GPU: {stats['gpu_util']:.0f}% | Mem: {stats['mem_util']:.0f}% | Temp: {stats['temp']:.0f}°C    ", end='')

                if count % save_every == 0:
                    print()  # Newline
                    self.save()
                    self.samples = []  # Clear buffer

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n[Stopping] Speichere letzte Daten...")
            self.save()
            print(f"[Done] Total: {count} samples in {(time.time()-self.start_time)/60:.1f} Minuten")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        AMD GPU DATA COLLECTOR                            ║
    ║        Hilf mit, AMD GPUs schneller zu machen!           ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Einfache Argumente
    args = sys.argv[1:]

    user_name = "anonymous"
    export_mode = False

    for i, arg in enumerate(args):
        if arg in ['--user', '-u'] and i + 1 < len(args):
            user_name = args[i + 1]
        elif arg in ['--export', '-e']:
            export_mode = True

    collector = SimpleCollector(user_name)

    if export_mode:
        collector.export_all()
        return

    # Interaktives Menu
    print("""
Optionen:
  [1] Daten sammeln (läuft im Hintergrund)
  [2] Gesammelte Daten exportieren
  [3] Beenden

""")

    try:
        choice = input("Auswahl [1]: ").strip() or "1"

        if choice == "1":
            collector.run_continuous()
        elif choice == "2":
            collector.export_all()
        else:
            print("Beendet.")

    except KeyboardInterrupt:
        print("\nBeendet.")


if __name__ == "__main__":
    main()
