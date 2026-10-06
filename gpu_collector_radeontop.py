#!/usr/bin/env python3
"""GPU Collector mit radeontop - liest beide GPUs aus. Speichert alle 100 Samples."""

import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path

INTERVAL = 0.5
SAVE_EVERY = 100
OUTPUT_DIR = Path("gpu_logs")
OUTPUT_DIR.mkdir(exist_ok=True)

BUSES = ["03", "0f"]

def read_gpu(bus):
    try:
        result = subprocess.run(
            ["radeontop", "-b", bus, "-d", "-", "-l", "1"],
            capture_output=True, text=True, timeout=2
        )
        line = result.stdout.strip().split('\n')[-1]
        gpu = re.search(r'gpu (\d+\.\d+)%', line)
        vram = re.search(r'vram (\d+\.\d+)%', line)
        sclk = re.search(r'sclk (\d+\.\d+)%', line)
        return {
            'bus': bus,
            'gpu': float(gpu.group(1)) if gpu else 0,
            'vram': float(vram.group(1)) if vram else 0,
            'sclk': float(sclk.group(1)) if sclk else 0,
        }
    except:
        return {'bus': bus, 'gpu': 0, 'vram': 0, 'sclk': 0}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"gpu_radeontop_{timestamp}.json"

print(f"GPU Collector - Output: {output_file}")
print(f"Speichert alle {SAVE_EVERY} Samples")

data = []
while True:
    sample = {'t': time.time(), 'gpus': [read_gpu(b) for b in BUSES]}
    data.append(sample)

    g0 = sample['gpus'][0]['gpu']
    g1 = sample['gpus'][1]['gpu']
    print(f"\r[{len(data):5d}] GPU-03: {g0:5.1f}% | GPU-0f: {g1:5.1f}%", end='', flush=True)

    if len(data) % SAVE_EVERY == 0:
        with open(output_file, 'w') as f:
            json.dump(data, f)
        print(f" [saved]", end='', flush=True)

    time.sleep(INTERVAL)
