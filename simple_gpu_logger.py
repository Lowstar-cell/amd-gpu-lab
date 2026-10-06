#!/usr/bin/env python3
"""
Einfacher GPU Logger - läuft im Hintergrund und sammelt rocm-smi Daten.
"""

import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# Config
INTERVAL = 0.2  # Sekunden zwischen Messungen
OUTPUT_DIR = Path("gpu_logs")

# Create output dir
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"gpu_log_{timestamp}.json"

print(f"GPU Logger gestartet")
print(f"Speichert nach: {output_file}")
print(f"Interval: {INTERVAL}s")
print(f"Drücke Ctrl+C zum Stoppen")
print("-" * 60)

data = []

try:
    while True:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--csv"],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                data.append({
                    'timestamp': time.time(),
                    'datetime': datetime.now().isoformat(),
                    'header': lines[0],
                    'data': lines[1]
                })

                # Live output
                if len(data) % 10 == 0:
                    print(f"[{len(data):5d}] {datetime.now().strftime('%H:%M:%S')}")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print(f"\n\nStopping... Gesammelt: {len(data)} samples")

finally:
    # Save
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Gespeichert: {output_file}")
    print(f"✓ Samples: {len(data)}")
    print(f"✓ Dauer: {len(data) * INTERVAL:.1f}s")
