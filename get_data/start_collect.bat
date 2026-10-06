@echo off
REM start_collect.bat
REM Usage: start_collect.bat [interval] [outdir] [runname] [bg]
REM Example: start_collect.bat 0.5 .\get_data myrun bg

setlocal
set INTERVAL=%1




















exit /b 0endlocal)  %PS_CMD%  echo Running collector in foreground. Press Ctrl+C to stop.) else (  echo Collector started in a new window (background).  start "AmdGpuCollector" %PS_CMD%if /I "%BG%"=="bg" (if not "%RUNNAME%"=="" set PS_CMD=%PS_CMD% -RunName "%RUNNAME%"set PS_CMD=PowerShell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -Interval %INTERVAL% -OutDir "%OUTDIR%"set PS_SCRIPT=%~dp0collect_windows.ps1set BG=%4set RUNNAME=%3if "%OUTDIR%"=="" set OUTDIR=get_dataset OUTDIR=%2nif "%INTERVAL%"=="" set INTERVAL=1