@echo off
chcp 65001 >nul
title AMD GPU Collector

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║           AMD GPU DATA COLLECTOR - Auto Setup                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Prüfe ob Python vorhanden
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Python gefunden
    goto :run_python
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Python gefunden (py launcher)
    set PYTHON=py
    goto :run_python
)

:: Kein Python - nutze PowerShell Alternative
echo [INFO] Python nicht gefunden - nutze PowerShell Version
goto :run_powershell

:run_python
echo.
set /p USERNAME="Dein Name (für anonyme ID): "
if "%USERNAME%"=="" set USERNAME=anonymous

python gpu_service/standalone_collector.py --user "%USERNAME%"
goto :end

:run_powershell
echo.
echo Starte PowerShell-Version...
powershell -ExecutionPolicy Bypass -File "gpu_service/collector_nopython.ps1"
goto :end

:end
echo.
pause
