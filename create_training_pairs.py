#!/usr/bin/env python3
"""Erstellt CUDA→HIP Training-Paare mit hipify-clang."""

import json
import subprocess
import tempfile
from pathlib import Path

INPUT_FILE = Path("cuda_training_data/cuda_code_samples.json")
OUTPUT_FILE = Path("cuda_training_data/training_pairs.json")

def hipify_code(cuda_code):
    """Konvertiert CUDA-Code zu HIP via hipify-perl (regex-basiert, toleriert Snippets)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(cuda_code)
        cuda_file = f.name

    try:
        result = subprocess.run(
            ['/opt/rocm-7.1.1/bin/hipify-perl', cuda_file],
            capture_output=True, text=True, timeout=10
        )
        Path(cuda_file).unlink()
        if result.stdout and result.stdout.strip():
            return result.stdout
    except:
        try:
            Path(cuda_file).unlink()
        except:
            pass
    return None

print("Loading CUDA samples...")
with open(INPUT_FILE) as f:
    samples = json.load(f)

print(f"Found {len(samples)} samples")
print("Converting with hipify-clang...")

pairs = []
for i, sample in enumerate(samples):
    cuda_code = sample.get('code', sample.get('cuda', ''))
    if not cuda_code:
        continue

    hip_code = hipify_code(cuda_code)
    if hip_code and hip_code != cuda_code:
        pairs.append({
            'cuda': cuda_code,
            'hip': hip_code,
            'source': sample.get('source', 'unknown')
        })

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(samples)}] {len(pairs)} pairs")

print(f"\nSaving {len(pairs)} pairs to {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(pairs, f, indent=2)

print("Done!")
