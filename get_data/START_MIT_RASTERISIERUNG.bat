@echo off
REM Startet die erweiterte Datensammlung mit Rasterisierung
REM Sammelt ROCm Metriken + GPU Rasterisierungs-Features

echo ========================================
echo ERWEITERTE DATENSAMMLUNG
echo ========================================
echo.
echo Sammelt:
echo  - ROCm GPU Metriken (rocm-smi)
echo  - GPU Rasterisierungs-Features (HDF5)
echo.
echo Druecke Strg+C zum Beenden
echo ========================================
echo.

REM Aktiviere Virtual Environment falls vorhanden
if exist ..\venv\Scripts\activate.bat (
    echo [INFO] Aktiviere venv...
    call ..\venv\Scripts\activate.bat
)

if exist ..\.venv\Scripts\activate.bat (
    echo [INFO] Aktiviere .venv...
    call ..\.venv\Scripts\activate.bat
)

REM Starte Script mit Rasterisierung
python daten-sammeln-mit-raster.py ^
    --interval 2.0 ^
    --with-raster ^
    --raster-width 256 ^
    --raster-height 256 ^
    --raster-triangles 20

echo.
echo ========================================
echo Datensammlung beendet
echo ========================================
pause
