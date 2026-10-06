# Start/Stop GPU Data Collection (Windows)

Quick scripts to start and stop the Windows GPU collector (`collect_windows.ps1`).

Files:
- `start_collect.bat` — wrapper to start the collector. Usage:
  - `start_collect.bat [interval] [outdir] [runname] [bg]`
  - Examples:
    - `start_collect.bat 0.5`  (start in foreground, Interval 0.5s, OutDir `get_data`)
    - `start_collect.bat 0.5 .\get_data myrun bg` (start in new window)

- `stop_collect.bat` — stops the collector. Usage:
  - `stop_collect.bat [run-dir]`
  - If no run-dir is provided it will stop the collector in the newest run directory under `get_data`.

Notes:
- `start_collect.bat` calls PowerShell with `-ExecutionPolicy Bypass` and passes the given arguments to `collect_windows.ps1`.
- `bg` argument causes the collector to be launched in a new window (`start "AmdGpuCollector" ...`). Without `bg` it runs in the current console (Ctrl+C to stop).
- `stop_collect.bat` looks for the `collector.pid` file written by the collector and uses that PID to stop the process. If the collector was terminated uncleanly the `done` marker may be missing; that is expected.

- `start_collect.ps1` — New robust wrapper that optionally enforces GPU availability and sets an env var so rasterizers can mark CPU fallbacks. Usage:
  - `./start_collect.ps1 -Interval 0.5 -OutDir .\get_data -RunName myrun`  (runs in foreground and prints run directory)
  - `./start_collect.ps1 -Interval 0.5 -OutDir .\get_data -RunName myrun -Background`  (starts collector in background and prints run directory)
  - `./start_collect.ps1 -RequireGpu -Interval 0.5 -RunName myrun` (abort if no GPU detected; writes `metadata.json` with `cpu_fallback=true`)

If you want the wrapper to return structured JSON (run_dir, pid, log_paths) for automation I can add a `-Json` flag; tell me whether you want that.
