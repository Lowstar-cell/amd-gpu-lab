@echo off
REM Quick Test der Rasterisierungs-Datensammlung
REM Nutzt kleines Fenster und kurze Intervalle

echo ========================================
echo QUICK TEST - Rasterisierung
echo ========================================
echo.
echo Test-Konfiguration:
echo  - Kleines Fenster (128x128)
echo  - Wenig Triangles (10)
echo  - Schnelles Interval (0.5s)
echo.
echo Druecke Strg+C nach ein paar Sekunden
echo ========================================
echo.

REM Aktiviere Virtual Environment
if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
) else if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
)

REM Starte mit Test-Config
python daten-sammeln-mit-raster.py ^
    --interval 0.5 ^
    --with-raster ^
    --raster-width 128 ^
    --raster-height 128 ^
    --raster-triangles 10 ^
    --run-name "quick_test"

echo.
echo ========================================
echo Test abgeschlossen
echo ========================================
echo.
echo Pruefe Output in get_data/quick_test_*
echo.
pause
