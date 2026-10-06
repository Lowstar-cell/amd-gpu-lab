#!/usr/bin/env python3
"""
Compute per-counter statistics from a directory of collected GPA CSVs and their .meta.json files.

Outputs a stats JSON with min/max/mean/std/count for each counter index found.
"""
import argparse
import os
import json
import glob
import csv
from collections import defaultdict
import math


def compute_stats(collected_dir, out='stats.json', method='zscore'):
    collected_dir = os.path.abspath(collected_dir)
    csv_files = sorted(glob.glob(os.path.join(collected_dir, 'gpa_rps_output_*.csv')))
    if not csv_files:
        raise FileNotFoundError('No collected CSVs in ' + collected_dir)

    # Data structure: device_id -> counter_index -> list of values
    counter_values = defaultdict(lambda: defaultdict(list))
    global_values = defaultdict(list)

    for csv_path in csv_files:
        # read header to find which columns map to counters
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # determine device id from corresponding meta JSON if present
                meta_path = csv_path + '.meta.json'
                device_id = 'global'
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                            device_id = str(meta.get('device_id', 'global'))
                    except Exception:
                        device_id = 'global'
                for k, v in row.items():
                    if k.startswith('counter_'):
                        try:
                            idx = int(k.split('_')[1])
                            val = float(v) if v != '' else float('nan')
                        except Exception:
                            continue
                        if math.isfinite(val):
                            counter_values[device_id][idx].append(val)
                            global_values[idx].append(val)

    # compute stats
    stats = {"collected_dir": collected_dir, "devices": {}, "global": {}}

    # Compute stats per-device
    for dev, counters in counter_values.items():
        stats['devices'][dev] = {}
        for idx, vals in counters.items():
            n = len(vals)
        if n == 0:
            continue
            s = sum(vals)
            mean = s / n
            var = sum((x - mean) ** 2 for x in vals) / n
            std = math.sqrt(var)
            # median & IQR
            sorted_vals = sorted(vals)
            median = sorted_vals[n//2] if n > 0 else 0.0
            q1 = sorted_vals[max(0, n//4)]
            q3 = sorted_vals[min(n-1, (3*n)//4)]
            iqr = q3 - q1
            stats['devices'][dev][str(idx)] = {"count": n, "min": min(vals), "max": max(vals), "mean": mean, "std": std, "median": median, "q1": q1, "q3": q3, "iqr": iqr}

    # Compute global stats as well
    for idx, vals in global_values.items():
        n = len(vals)
        if n == 0:
            continue
        s = sum(vals)
        mean = s / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(var)
        sorted_vals = sorted(vals)
        median = sorted_vals[n//2] if n > 0 else 0.0
        q1 = sorted_vals[max(0, n//4)]
        q3 = sorted_vals[min(n-1, (3*n)//4)]
        iqr = q3 - q1
        stats['global'][str(idx)] = {"count": n, "min": min(vals), "max": max(vals), "mean": mean, "std": std, "median": median, "q1": q1, "q3": q3, "iqr": iqr}

    out_path = os.path.abspath(out)
    with open(out_path, 'w', encoding='utf-8') as of:
        json.dump(stats, of, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('collected_dir')
    parser.add_argument('--out', default=None)
    parser.add_argument('--method', choices=['zscore','robust'], default='zscore', help='Normalization method to use for stats (zscore=mean/std, robust=median/IQR)')
    args = parser.parse_args()
    out = args.out or os.path.join(args.collected_dir, 'stats.json')
    path = compute_stats(args.collected_dir, out, method=args.method)
    print('Wrote stats to', path)


if __name__ == '__main__':
    raise SystemExit(main())
