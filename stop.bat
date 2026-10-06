@echo off
REM stop.bat - Stop a running Windows GPU collector started from this repo

SETLOCAL ENABLEDELAYEDEXPANSION

if "%1"=="" (
  echo Stopping most recent run under get_data
  powershell -ExecutionPolicy Bypass -File "%~dp0get_data\stop_windows_collector.ps1"
  exit /b %ERRORLEVEL%
)

REM forward any arguments to PowerShell helper
powershell -ExecutionPolicy Bypass -File "%~dp0get_data\stop_windows_collector.ps1" %*
exit /b %ERRORLEVEL%
