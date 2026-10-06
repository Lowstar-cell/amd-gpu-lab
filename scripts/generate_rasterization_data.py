#!/usr/bin/env python3
"""Generate synthetic rasterization/frame-level data for training/benchmarks.

Creates multiple files (CSV or Parquet) under an output run directory. Each file simulates
a sequence of frames with frame-level rasterization metrics useful to train models.

Usage:
  python3 scripts/generate_rasterization_data.py --out-dir data/raster_samples --num-files 3 --frames 200
"""
import argparse
from pathlib import Path
import random
import csv
from datetime import datetime, timezone
import json

try:
    import pandas as pd
except Exception:
    pd = None


def make_frame(frame_id, workload='normal'):
    # base values
    base_primitives = 1000
    if workload == 'heavy':
        primitives = int(base_primitives * random.uniform(2.0, 6.0))
    elif workload == 'light':
        primitives = int(base_primitives * random.uniform(0.2, 0.9))
    else:
        primitives = int(base_primitives * random.uniform(0.8, 1.4))

    vertices = primitives * random.randint(3, 6)
    shader_inv = int(primitives * random.uniform(8, 35))
    rasterized_fragments = int(primitives * random.uniform(0.8, 5.0))
    samples_covered = int(rasterized_fragments * random.uniform(0.6, 1.0))
    z_culls = int(rasterized_fragments * random.uniform(0.0, 0.4))
    bandwidth = int(1e6 + rasterized_fragments * random.uniform(20, 300))
    gpu_temp = round(35 + random.uniform(0, 45), 2)
    # label: 0 normal, 1 heavy/anomalous by workload param
    label = 1 if workload == 'heavy' else 0

    # return dict
    return {
        'frame_id': frame_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'workload': workload,
        'primitives': primitives,
        'vertices': vertices,
        'shader_invocations': shader_inv,
        'rasterized_fragments': rasterized_fragments,
        'samples_covered': samples_covered,
        'z_culls': z_culls,
        'bandwidth_bytes': bandwidth,
        'gpu_temp_C': gpu_temp,
        'label': label,
    }


def generate_file(out_path: Path, frames: int = 1000, heavy_ratio: float = 0.05, seed: int | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        random.seed(seed)

    rows = []
    for i in range(frames):
        w = 'heavy' if random.random() < heavy_ratio else 'normal'
        rows.append(make_frame(i, workload=w))

    if out_path.suffix == '.parquet' and pd is not None:
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
    else:
        # write CSV
        with open(out_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='data/raster_samples', help='output directory for run')
    parser.add_argument('--num-files', type=int, default=3)
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--heavy-ratio', type=float, default=0.05)
    parser.add_argument('--format', choices=['csv', 'parquet'], default='csv')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    # make a unique run directory
    run_dir = out_root / datetime.utcnow().strftime('run_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(args.num_files):
        fname = f'raster_{i}.{args.format}'
        out_file = run_dir / fname
        nrows = generate_file(out_file, frames=args.frames, heavy_ratio=args.heavy_ratio, seed=(args.seed or None))
        manifest.append({'file': str(out_file), 'rows': nrows})
        print(f'Wrote {out_file} ({nrows} frames)')

    # write manifest/meta
    meta = {
        'run_dir': str(run_dir),
        'num_files': args.num_files,
        'frames_per_file': args.frames,
        'heavy_ratio': args.heavy_ratio,
        'format': args.format,
        'seed': args.seed,
        'files': manifest,
    }
    with open(run_dir / 'manifest.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    print(f'Run manifest written to {run_dir}/manifest.json')


if __name__ == '__main__':
    main()
