#!/usr/bin/env python3
"""Small GUI to control the telemetry collector and trainer.

This uses Tkinter (standard library) so there are no extra dependencies.
The GUI provides controls to start/stop/status collector & trainer and includes
a Dark Mode toggle to improve usability.

Run: python3 scripts/launch_gui.py
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from scripts.orchestrator import manager as mgr


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Telemetry Orchestrator')
        self.geometry('700x480')
        self.resizable(True, True)
        self._dark = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        # top toolbar
        toolbar = ttk.Frame(self, padding=(6, 6))
        toolbar.pack(fill='x')

        self.minutes_var = tk.DoubleVar(value=30.0)
        ttk.Label(toolbar, text='Minutes:').pack(side='left')
        ttk.Entry(toolbar, width=8, textvariable=self.minutes_var).pack(side='left', padx=(4, 8))

        self.interval_var = tk.DoubleVar(value=2.0)
        ttk.Label(toolbar, text='Collector interval(s):').pack(side='left')
        ttk.Entry(toolbar, width=6, textvariable=self.interval_var).pack(side='left', padx=(4, 8))

        ttk.Label(toolbar, text='Device:').pack(side='left')
        self.device_var = tk.StringVar(value='auto')
        ttk.Combobox(toolbar, values=['auto', 'cpu', 'gpu'], width=6, textvariable=self.device_var, state='readonly').pack(side='left', padx=(4, 8))

        ttk.Button(toolbar, text='Start (background)', command=self._start_background).pack(side='left', padx=6)
        ttk.Button(toolbar, text='Start (blocking)', command=self._start_blocking).pack(side='left')
        ttk.Button(toolbar, text='Stop', command=self._stop).pack(side='left', padx=6)
        ttk.Button(toolbar, text='Status', command=self._status).pack(side='left')

        # venv and data directory
        cfg = ttk.Frame(self, padding=(6, 6))
        cfg.pack(fill='x')
        ttk.Label(cfg, text='Python (venv path, optional)').pack(anchor='w')
        self.venv_var = tk.StringVar(value='')
        vframe = ttk.Frame(cfg)
        vframe.pack(fill='x')
        ttk.Entry(vframe, textvariable=self.venv_var).pack(side='left', fill='x', expand=True)
        ttk.Button(vframe, text='Browse', command=self._browse_venv).pack(side='left', padx=6)

        ttk.Label(cfg, text='Data dir (processed):').pack(anchor='w', pady=(6, 0))
        self.datadir_var = tk.StringVar(value='data/processed/get_data')
        dframe = ttk.Frame(cfg)
        dframe.pack(fill='x')
        ttk.Entry(dframe, textvariable=self.datadir_var).pack(side='left', fill='x', expand=True)
        ttk.Button(dframe, text='Browse', command=self._browse_datadir).pack(side='left', padx=6)

        # logs and output
        logs_frame = ttk.Frame(self, padding=(6, 6))
        logs_frame.pack(fill='both', expand=True)

        left = ttk.Frame(logs_frame)
        left.pack(side='left', fill='both', expand=True)
        ttk.Label(left, text='Status / Messages').pack(anchor='w')
        self.log_text = tk.Text(left, height=14, wrap='word')
        self.log_text.pack(fill='both', expand=True)

        right = ttk.Frame(logs_frame, width=240)
        right.pack(side='left', fill='y', padx=(6, 0))
        ttk.Label(right, text='Run info').pack(anchor='w')
        self.info_text = tk.Text(right, width=36, height=10, wrap='word')
        self.info_text.pack(fill='y')

        # bottom controls
        bottom = ttk.Frame(self, padding=(6, 6))
        bottom.pack(fill='x')
        ttk.Checkbutton(bottom, text='Dark Mode', variable=self._dark, command=self._apply_theme).pack(side='left')
        ttk.Button(bottom, text='Refresh', command=self._refresh_status).pack(side='right')

        self._apply_theme()
        self._refresh_status()

    def _apply_theme(self):
        dark = self._dark.get()
        bg = '#2b2b2b' if dark else '#f6f6f6'
        fg = '#e6e6e6' if dark else '#111111'
        self.configure(bg=bg)
        self.log_text.configure(bg=bg, fg=fg, insertbackground=fg)
        self.info_text.configure(bg=bg, fg=fg, insertbackground=fg)

    def _append_log(self, txt: str):
        self.log_text.insert('end', f"[{time.strftime('%H:%M:%S')}] {txt}\n")
        self.log_text.see('end')

    def _browse_venv(self):
        p = filedialog.askopenfilename(title='Pick python.exe (venv) or python binary')
        if p:
            self.venv_var.set(p)

    def _browse_datadir(self):
        p = filedialog.askdirectory(title='Select processed data directory')
        if p:
            self.datadir_var.set(p)

    def _start_background(self):
        self._append_log('Starting collector+trainer (background)')
        try:
            mgr.start_collector(float(self.interval_var.get()), self.datadir_var.get(), background=True)
            mgr.start_trainer(mgr.find_python_executable(self.venv_var.get() or None), float(self.minutes_var.get()), self.datadir_var.get(), device=self.device_var.get(), logdir='logs/telemetry_smoke', detached=True)
            self._append_log('Launched (background).')
            self._refresh_status()
        except Exception as e:
            messagebox.showerror('Start failed', str(e))

    def _start_blocking(self):
        self._append_log('Starting trainer (blocking) — collector will run in background by default')
        try:
            # start collector in background
            mgr.start_collector(float(self.interval_var.get()), self.datadir_var.get(), background=True)

            python = mgr.find_python_executable(self.venv_var.get() or None)

            # start trainer blocking in new thread to keep GUI responsive
            def run_trainer():
                proc = mgr.start_trainer(python, float(self.minutes_var.get()), self.datadir_var.get(), device=self.device_var.get(), logdir='logs/telemetry_smoke', detached=False)
                rc = proc.wait()
                self._append_log(f'Trainer finished rc={rc}')

            t = threading.Thread(target=run_trainer, daemon=True)
            t.start()
            self._append_log('Started trainer in blocking mode (GUI remains responsive)')
        except Exception as e:
            messagebox.showerror('Start failed', str(e))

    def _stop(self):
        self._append_log('Stopping collector and trainer')
        try:
            # use manager stop functions
            coll_pidfile = Path('logs/telemetry_collector.pid')
            trainer_pidfile = Path('logs/telemetry_smoke/telemetry_smoke.pid')

            coll_pid = mgr.read_pid(coll_pidfile) if coll_pidfile.exists() else None
            trainer_pid = mgr.read_pid(trainer_pidfile) if trainer_pidfile.exists() else None

            ok = True
            if coll_pid:
                ok = ok and mgr.stop_pid(coll_pid)
                self._append_log(f'stopped collector pid={coll_pid}')
            if trainer_pid:
                ok = ok and mgr.stop_pid(trainer_pid)
                self._append_log(f'stopped trainer pid={trainer_pid}')

            if ok:
                self._append_log('Stop completed')
            else:
                self._append_log('Stop encountered errors (see logs)')
        except Exception as e:
            messagebox.showerror('Stop failed', str(e))

    def _status(self):
        self._append_log('Checking status...')
        self._refresh_status()

    def _refresh_status(self):
        # read pid files and check status
        info_lines = []
        coll_pidfile = Path('logs/telemetry_collector.pid')
        trainer_pidfile = Path('logs/telemetry_smoke/telemetry_smoke.pid')

        if coll_pidfile.exists():
            pid = mgr.read_pid(coll_pidfile)
            info_lines.append(f'Collector PID: {pid}  alive={mgr.is_process_alive(pid) if pid else False}')
        else:
            info_lines.append('Collector: not running')

        if trainer_pidfile.exists():
            pid = mgr.read_pid(trainer_pidfile)
            info_lines.append(f'Trainer PID: {pid}  alive={mgr.is_process_alive(pid) if pid else False}')
        else:
            info_lines.append('Trainer: not running')

        # show last lines of the trainer log if present
        last_log = ''
        logpath = Path('logs/telemetry_smoke/telemetry_smoke_master.out')
        if logpath.exists():
            try:
                with logpath.open('r', encoding='utf8') as fh:
                    lines = fh.read().splitlines()[-20:]
                    last_log = '\n'.join(lines)
            except Exception:
                last_log = '[could not read log]'
        else:
            last_log = '[no trainer log yet]'

        self.info_text.delete('1.0', 'end')
        self.info_text.insert('end', '\n'.join(info_lines))

        self.log_text.insert('end', '\n--- log tail ---\n')
        self.log_text.insert('end', last_log + '\n')
        self.log_text.see('end')


def run_gui():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    run_gui()
