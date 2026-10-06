#!/usr/bin/env python3
"""Convert CSV files in get_data/ into normalized Parquet files for training.

The script tries to be robust against CSVs with or without headers and common rocm-smi layouts.

Usage:
  python scripts/convert_get_data_to_parquet.py --input-dir get_data --out-dir data/processed/get_data
"""
import argparse
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Typical rocm-smi CSV header we observed
DEFAULT_HEADER = [
    "Timestamp",
    "device",
    "Temperature (Sensor edge) (C)",
    "Average Graphics Package Power (W)",
    "GPU use (%)",
    "VRAM Total Memory (B)",
    "VRAM Total Used Memory (B)",
    "Temperature (Sensor junction) (C)",
    "Temperature (Sensor memory) (C)",
]

CANONICAL_MAP = {
    "Timestamp": "timestamp",
    "device": "device",
    "Temperature (Sensor edge) (C)": "temp_edge_C",
    "Average Graphics Package Power (W)": "avg_graphics_power_W",
    "GPU use (%)": "gpu_use_pct",
    "VRAM Total Memory (B)": "vram_total_B",
    "VRAM Total Used Memory (B)": "vram_used_B",
    "Temperature (Sensor junction) (C)": "temp_junction_C",
    "Temperature (Sensor memory) (C)": "temp_memory_C",
}


def normalize_dataframe(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    # Replace common N/A strings
    df = df.replace({"N/A": pd.NA, "": pd.NA})

    # Strip header whitespace
    df.columns = [c.strip() for c in df.columns]

    # Ensure expected columns exist; if not, attempt fallbacks
    for expected in CANONICAL_MAP.keys():
        if expected not in df.columns:
            # no-op; we'll just leave it missing and fill with NaN later
            continue

    # Rename to canonical names where possible
    rename_map = {k: v for k, v in CANONICAL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Convert types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "gpu_use_pct" in df.columns:
        df["gpu_use_pct"] = pd.to_numeric(df["gpu_use_pct"], errors="coerce")
    for c in ("temp_edge_C", "temp_junction_C", "temp_memory_C", "avg_graphics_power_W"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("vram_total_B", "vram_used_B"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add provenance
    df["file_source"] = os.path.basename(source_file)
    df["ingest_time"] = pd.Timestamp(datetime.utcnow())

    return df


def convert_dir(input_dir: str, out_dir: str, overwrite: bool = False) -> int:
    p = Path(input_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # traverse CSVs recursively so we can process per-run directories like get_data/run_*/
    csv_files = sorted([f for f in p.rglob("*.csv") if f.is_file()])
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0

    written = 0
    for fpath in csv_files:
        # preserve relative path of CSV inside input dir
        try:
            rel = fpath.relative_to(p)
        except Exception:
            rel = Path(fpath.name)
        target = out / rel.with_suffix('.parquet')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            print(f"Skipping {fpath.name}, {target.name} already exists (use --overwrite to force)")
            continue

        # Try reading the file with header; if first row looks like a timestamp, read without header
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                first_line = fh.readline().strip()

            # Heuristic: if first line contains the word 'Timestamp' or 'device' treat it as header
            use_header = True
            if not ("Timestamp" in first_line or "device" in first_line or "GPU use" in first_line):
                use_header = False

            if use_header:
                df = pd.read_csv(fpath)
            else:
                # read with fallback names
                df = pd.read_csv(fpath, header=None, names=[c.strip() for c in DEFAULT_HEADER])

        except Exception as ex:
            print(f"Failed to read {fpath}: {ex}")
            continue

        df = normalize_dataframe(df, fpath)

        # Write to parquet
        try:
            df.to_parquet(target, index=False)
            written += 1
            print(f"Wrote {target}  (rows={len(df)})")
        except Exception as ex:
            print(f"Failed to write parquet for {fpath}: {ex}")

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="get_data", help="where CSVs live (default: get_data)")
    parser.add_argument("--out-dir", default="data/processed/get_data", help="out dir for parquet files")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing parquet files")

    args = parser.parse_args()

    n = convert_dir(args.input_dir, args.out_dir, overwrite=args.overwrite)
    print(f"Converted {n} files")


if __name__ == "__main__":
    main()
