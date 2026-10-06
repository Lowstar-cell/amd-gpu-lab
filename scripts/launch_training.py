#!/usr/bin/env python3
"""Cross-platform launcher for collector + telemetry trainer.

This script provides a single interface to start collection and training on
both Linux and Windows. It tries to be defensive: it will start the collector
when requested and the trainer in background, write PID and simple metadata.

Examples:
  # Linux (default): starts daten-sammeln.py collector and trainer
  python3 scripts/launch_training.py --minutes 30

  # Windows (pass venv python if desired)
  python3 scripts/launch_training.py --minutes 30 --venv-path ".\\.venv-win\\Scripts\\python.exe"
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scripts.orchestrator import manager as mgr


## core helpers are provided by scripts.orchestrator.manager (mgr)


def start_collector(interval: float, out_dir: str, background: bool) -> subprocess.Popen | None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return mgr.start_collector(interval, out_dir, background)


    return mgr.start_trainer(python_path, minutes, data_dir, device=device, logdir=logdir, detached=detached)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=30.0)
    parser.add_argument('--data-dir', default='data/processed/get_data')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--venv-path', default=None)
    parser.add_argument('--collector-interval', type=float, default=2.0)
    parser.add_argument('--collector-outdir', default='get_data')
    parser.add_argument('--no-collector', action='store_true')
    parser.add_argument('--background', action='store_true', help='Start collector/trainer in background (non-blocking)')
    parser.add_argument('--action', choices=['start', 'stop', 'status'], default='start', help='Action to perform (start default)')
    args = parser.parse_args(argv)

    python_exec = find_python_executable(args.venv_path)
    print('Using python:', python_exec)

    collector_proc = None
    # control helpers are provided by orchestrator.manager (mgr)

    pidfile = Path('logs/telemetry_smoke/telemetry_smoke.pid')

    if args.action == 'stop':
        # Stop both collector and trainer if PID files exist
        coll_pidfile = Path('logs/telemetry_collector.pid')
        coll_pid = mgr.read_pid(coll_pidfile) if coll_pidfile.exists() else None
        trainer_pid = mgr.read_pid(pidfile) if pidfile.exists() else None

        results = []
        if coll_pid:
            print(f'Stopping collector PID {coll_pid}')
            ok = mgr.stop_pid(coll_pid)
            results.append(('collector', ok))
            if coll_pidfile.exists():
                try:
                    coll_pidfile.unlink()
                except Exception:
                    pass
        else:
            print('No collector PID file found')

        if trainer_pid:
            print(f'Stopping trainer PID {trainer_pid}')
            ok2 = mgr.stop_pid(trainer_pid)
            results.append(('trainer', ok2))
            if pidfile.exists():
                try:
                    pidfile.unlink()
                except Exception:
                    pass
        else:
            print('No trainer PID file found')

        # report final status
        if all(r for (_, r) in results) or not results:
            print('Stop completed')
            return 0
        else:
            print('Some stop operations failed')
            return 2

    if args.action == 'status':
        pid = mgr.read_pid(pidfile)
        if not pid:
            print('No PID file found')
            return 1
        alive = mgr.is_process_alive(pid)
        print(f'PID {pid} alive={alive}')
        return 0 if alive else 3

    if not args.no_collector:
        collector_proc = start_collector(args.collector_interval, args.collector_outdir, background=args.background)

    # start trainer
    trainer_proc = start_trainer(python_exec, args.minutes, args.data_dir, args.device, logdir='logs/telemetry_smoke', detached=args.background)

    if args.background:
        print('Launched collector and trainer in background. Use logs/telemetry_smoke to inspect progress.')
        return 0

    # otherwise wait for trainer to finish (collector runs until killed)
    try:
        trainer_proc.wait()
        print('Trainer finished with code', trainer_proc.returncode)
        return trainer_proc.returncode
    except KeyboardInterrupt:
        print('Interrupted — stopping collector and trainer')
        try:
            if collector_proc and collector_proc.poll() is None:
                collector_proc.terminate()
        except Exception:
            pass
        try:
            trainer_proc.terminate()
        except Exception:
            pass
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
