#!/usr/bin/env python3
"""
Collect ADLX device/display info and write CSV with device metadata.

This script attempts to import ADLXPybind; if not available it will try to
parse existing sample output files under `data/ADLX-main/Samples/`.

Usage:
  python scripts/collect_adlx_info.py --out devices.csv

"""
import argparse
import csv
import glob
import os
import sys

def parse_sample_output_file(path):
    # Very simple parser that extracts lists of properties from the sample output
    # This is a best-effort fallback only
    info = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read()
        # Extract name line 'Name: ...'
        for line in data.splitlines():
            if line.strip().startswith('Name:'):
                info['name'] = line.split(':', 1)[1].strip()
            if line.strip().startswith('Resolution:'):
                kv = line.split(':', 1)[1].strip()
                info['resolution'] = kv
            if line.strip().startswith('Refresh rate:'):
                info['refresh_rate'] = line.split(':', 1)[1].strip()
    except Exception:
        pass
    return info


def collect_from_samples(sample_dir):
    out = []
    for root, dirs, files in os.walk(sample_dir):
        for f in files:
            if f.endswith('.txt') or f.endswith('.log'):
                p = os.path.join(root, f)
                info = parse_sample_output_file(p)
                if info:
                    info['source'] = p
                    out.append(info)
    return out


def collect_via_adlx():
    try:
        import ADLXPybind as ADLX
    except Exception:
        return None
    info = []
    try:
        adlxHelper = ADLX.ADLXHelper()
        ret = adlxHelper.Initialize()
        if ret != ADLX.ADLX_RESULT.ADLX_OK:
            return None
        system = adlxHelper.GetSystemServices()
        displayService = system.GetDisplaysServices()
        disList = displayService.GetDisplays()
        for display in disList:
            if display is None:
                continue
            item = {
                'name': display.name(),
                'type': display.type(),
                'connector': display.connectType(),
                'resolution': '{}x{}'.format(*display.resolution()),
                'refresh_rate': display.RefreshRate(),
                'unique_id': display.UniqueId()
            }
            info.append(item)
        adlxHelper.Terminate()
    except Exception:
        return None
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/device_info.csv')
    parser.add_argument('--sample-dir', default='data/ADLX-main/Samples')
    args = parser.parse_args()

    # Try ADLX first
    adlx_info = collect_via_adlx()
    if adlx_info:
        rows = adlx_info
    else:
        rows = collect_from_samples(args.sample_dir)
    if not rows:
        print('No device info found. Please build ADLXPybind and ensure ADLX is installed.')
        sys.exit(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'Wrote {len(rows)} entries to {args.out}')


if __name__ == '__main__':
    main()
