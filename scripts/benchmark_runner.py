#!/usr/bin/env python3
"""Simple benchmark runner to orchestrate workloads and collect telemetry.

This is a lightweight prototype to run workload A and B repeatedly while collecting system/GPU
telemetry using the existing collector scripts. It stores per-trial artifacts and a small summary.

Usage:
  python3 scripts/benchmark_runner.py --workload-cmd "python3 my_workload.py" --trials 3

If you don't have a workload, use --synthetic to use the raster generator as the workload.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ensure local imports available
sys.path.insert(0, str(Path('.').resolve()))

from scripts.convert_get_data_to_parquet import convert_dir


def start_collector(trial_dir: Path, interval: float = 1.0):
    # run the linux collector Python script in background if on non-Windows,
    # on Windows users should run the PowerShell collector separately or we can spawn it.
    python = shutil_which_python()
    cmd = [python, 'get_data/daten-sammeln.py', '--out-dir', str(trial_dir / 'collector'), '--interval', str(interval)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def stop_process(proc: subprocess.Popen, timeout: float = 5.0):
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def shutil_which_python():
    # prefer venv-rocm6.4 python if available
    venv = Path('.venv-rocm6.4/bin/python3')
    if venv.exists():
        return str(venv)
    return sys.executable


def run_workload(cmd: str, duration: float | None = None, out_dir: Path | None = None):
    # run the provided shell command. If duration is given, run with timeout.
    start = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=duration)
        rc = r.returncode
        out = r.stdout
        err = r.stderr
    except subprocess.TimeoutExpired as e:
        rc = -1
        out = e.stdout or ''
        err = e.stderr or ''
    elapsed = time.time() - start
    if out_dir is not None:
        (out_dir / 'workload.stdout.txt').write_text(out)
        (out_dir / 'workload.stderr.txt').write_text(err)
    return rc, elapsed


def summarize_parquets(folder: Path):
    # quick stats: mean gpu_use_pct across all parquet files under folder
    try:
        import pandas as pd
    except Exception:
        return {}

    files = list(folder.rglob('*.parquet'))
    if not files:
        return {}

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    stats = {}
    if 'gpu_use_pct' in df.columns:
        stats['gpu_use_mean'] = float(df['gpu_use_pct'].dropna().mean())
        stats['gpu_use_std'] = float(df['gpu_use_pct'].dropna().std())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload-cmd', type=str, default=None, help='Shell command to run as workload')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--out-dir', type=str, default='bench_runs')
    parser.add_argument('--collector-interval', type=float, default=1.0)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic rasterization generator as workload')
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--model-ckpt', type=str, default=None, help='optional model checkpoint to use for injection')
    parser.add_argument('--workload-template', type=str, default=None, help='template for workload cmd when using model, use {score} and {trial_dir}')
    parser.add_argument('--duration', type=float, default=None, help='Timeout in seconds for workload')

    args = parser.parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    for t in range(args.trials):
        trial_dir = out_root / f'trial_{t+1}_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}'
        trial_dir.mkdir(parents=True, exist_ok=True)

        # start collector
        print(f"Trial {t+1}/{args.trials} starting collector -> {trial_dir}")
        collector_proc = None
        if sys.platform.startswith('win'):
            print('Windows detected - please start get_data/collect_windows.ps1 manually in a separate shell for full telemetry collection.')
        else:
            collector_proc = start_collector(trial_dir, interval=args.collector_interval)

        # run workload (synthetic or provided)
        if args.synthetic:
            workload_cmd = f"python3 scripts/generate_rasterization_data.py --out-dir {str(trial_dir)} --num-files 1 --frames {args.frames} --format csv"
        elif args.workload_cmd:
            workload_cmd = args.workload_cmd
        else:
            # default: short sleep to simulate a run
            workload_cmd = f"sleep {int(args.duration or 1)}"

        # If a model checkpoint and workload template are provided, run model_injection to get a score
        if args.model_ckpt and args.workload_template:
            try:
                mi_cmd = [shutil_which_python(), 'scripts/model_injection.py', '--model', args.model_ckpt, '--data', str(trial_dir / 'processed')]
                res = subprocess.run(mi_cmd, capture_output=True, text=True, timeout=10)
                out = res.stdout.strip() or res.stderr.strip()
                parsed = json.loads(out)
                score = parsed.get('score', 0.0)
                # format the template
                workload_cmd = args.workload_template.format(score=score, trial_dir=str(trial_dir))
                print(f"Model injection produced score={score:.4f}, workload_cmd='{workload_cmd}'")
            except Exception as e:
                print('Model injection failed:', e)

        rc, elapsed = run_workload(workload_cmd, duration=args.duration, out_dir=trial_dir)
        print(f"Workload finished (rc={rc}) elapsed={elapsed:.2f}s")

        # stop collector
        if collector_proc:
            stop_process(collector_proc)

        # convert collected CSVs if present
        converted_dir = trial_dir / 'processed'
        converted_dir.mkdir(parents=True, exist_ok=True)
        converted = convert_dir(str(trial_dir), str(converted_dir), overwrite=True)

        # summarize collected parquets
        stats = summarize_parquets(converted_dir)

        results.append({'trial': t+1, 'rc': rc, 'elapsed_s': elapsed, 'converted_files': converted, 'stats': stats, 'trial_dir': str(trial_dir)})

        # small cooldown
        time.sleep(1.0)

    # write summary
    (out_root / 'summary.json').write_text(json.dumps(results, indent=2))
    print('All trials finished. Summary written to', out_root / 'summary.json')


if __name__ == '__main__':
    main()
