"""Process management helpers for telemetry collection and training.

This module exports helper functions to start / stop / inspect collector and trainer
processes. It is intended to be imported by CLI wrappers (e.g., launch_training.py)
so the program can be extended or tested easier.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def find_python_executable(venv_path: Optional[str]) -> str:
    if venv_path:
        p = Path(venv_path)
        if p.exists():
            return str(p)
        cand = p / 'Scripts' / 'python.exe'
        if cand.exists():
            return str(cand)
    for candidate in ['.venv-win/Scripts/python.exe', '.venv/Scripts/python.exe', '.venv/bin/python', 'venv/bin/python']:
        p = Path(candidate)
        if p.exists():
            return str(p)
    return sys.executable


def _write_pid_and_meta(kind: str, pid: int, info: dict):
    Path('logs').mkdir(parents=True, exist_ok=True)
    pidfile = Path(f'logs/telemetry_{kind}.pid')
    pidfile.write_text(str(pid), encoding='utf8')
    meta = {'pid': pid, 'start_time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    meta.update(info)
    Path(f'logs/telemetry_{kind}_run.json').write_text(json.dumps(meta, indent=2), encoding='utf8')


def start_collector(interval: float, out_dir: str, background: bool) -> Optional[subprocess.Popen]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        cmd = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'get_data/collect_windows.ps1', '-Interval', str(interval), '-OutDir', out_dir]
        if background:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _write_pid_and_meta('collector', proc.pid, {'out_dir': out_dir, 'interval': interval})
            return proc
        subprocess.check_call(cmd)
        return None
    else:
        cmd = [sys.executable, 'get_data/daten-sammeln.py', '--interval', str(interval), '--out-dir', out_dir]
        if background:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _write_pid_and_meta('collector', proc.pid, {'out_dir': out_dir, 'interval': interval})
            return proc
        subprocess.check_call(cmd)
        return None


def start_trainer(python_path: str, minutes: float, data_dir: str, device: str, logdir: str, detached: bool = False) -> subprocess.Popen:
    Path(logdir).mkdir(parents=True, exist_ok=True)
    out = Path(logdir) / 'telemetry_smoke_master.out'
    args = [str(python_path), 'scripts/train_telemetry.py', '--data-dir', data_dir, '--minutes', str(minutes), '--device', device, '--logdir', logdir]

    creationflags = 0
    if detached and os.name == 'nt':
        creationflags = getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)

    with open(out, 'a', encoding='utf8') as fh:
        if creationflags:
            proc = subprocess.Popen(args, stdout=fh, stderr=fh, creationflags=creationflags)
        else:
            proc = subprocess.Popen(args, stdout=fh, stderr=fh)

    _write_pid_and_meta('smoke', proc.pid, {'args': args})
    return proc


def read_pid(pidfile: Path) -> Optional[int]:
    try:
        p = pidfile.read_text(encoding='utf8').strip()
        return int(p)
    except Exception:
        return None


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def stop_pid(pid: int) -> bool:
    try:
        if os.name == 'nt':
            subprocess.check_call(['taskkill', '/PID', str(pid), '/F'])
        else:
            os.kill(pid, 15)
    except Exception:
        try:
            if os.name != 'nt':
                os.kill(pid, 9)
        except Exception:
            return False
    return True
