@echo off
REM start.bat - Helper to start GPU data collection on Windows (or ROCm via --rocm)
REM Usage examples:
REM   start.bat                -> runs the Windows PowerShell collector (get_data\collect_windows.ps1)
REM   start.bat --interval 1   -> passes --interval 1 to the Windows PowerShell collector
REM   start.bat --rocm --interval 2 -> runs the ROCm (Python) collector (get_data\daten-sammeln.py) with interval 2

SETLOCAL ENABLEDELAYEDEXPANSION

if "%1"=="" goto run_windows

if /I "%1"=="--help" goto usage
if /I "%1"=="/?" goto usage

if /I "%1"=="status" goto do_status
if /I "%1"=="monitor" goto do_monitor

REM detect if user requested the ROCm/python collector (flag --rocm)
set USE_ROCM=0
set ARGS=

for %%A in (%*) do (
  if /I "%%A"=="--rocm" (
    set USE_ROCM=1
  ) else (
    if defined ARGS (
      set ARGS=!ARGS! %%A
    ) else (
      set ARGS=%%A
    )
  )
)

if "%USE_ROCM%"=="1" (
  echo Running ROCm collector (python get_data\daten-sammeln.py) with args: %ARGS%
  python "%~dp0get_data\daten-sammeln.py" %ARGS%
  exit /b %ERRORLEVEL%
)

:run_windows
echo Running Windows collector (PowerShell get_data\collect_windows.ps1) with args: %*
powershell -ExecutionPolicy Bypass -File "%~dp0get_data\collect_windows.ps1" %*
exit /b %ERRORLEVEL%

:do_status
echo Showing status for run (status_run.ps1): %*
powershell -ExecutionPolicy Bypass -File "%~dp0get_data\status_run.ps1" %*
exit /b %ERRORLEVEL%

:do_monitor
echo Monitoring run (monitor_run.ps1): %*
powershell -ExecutionPolicy Bypass -File "%~dp0get_data\monitor_run.ps1" %*
exit /b %ERRORLEVEL%

:usage
echo Start data collection — Windows PowerShell and (optionally) ROCm python collector
echo.
echo Usage:
echo   start.bat [--rocm] [collector-args]
echo   start.bat status [--run <path>] [--lines N]   -> show run status and tail sample from latest or specified run
echo   start.bat monitor [--run <path>] [--lines N]  -> tail live log and press any key to stop collector (graceful)
echo.
echo Options:
echo   --rocm        Run the ROCm python collector ("get_data\daten-sammeln.py") instead of the PowerShell collector
echo   collector-args Arguments forwarded to the underlying collector: e.g. --interval 1 --out-dir .\get_data --run-name myrun
exit /b 0
