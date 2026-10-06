@echo off
REM Startet nur ROCm Metriken Sammlung (Original, ohne Rasterisierung)

echo ========================================
echo ROCm METRIKEN SAMMLUNG
echo ========================================
echo.
echo Sammelt nur ROCm GPU Metriken (CSV)
echo Keine Rasterisierung
echo.
echo Druecke Strg+C zum Beenden
echo ========================================
echo.

REM Aktiviere Virtual Environment falls vorhanden
if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
)

if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
)

REM Starte Original Script (OHNE Rasterisierung)
python daten-sammeln.py --interval 2.0

echo.
echo ========================================
echo Datensammlung beendet
echo ========================================
pause
