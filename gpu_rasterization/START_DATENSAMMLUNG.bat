@echo off
echo ======================================
echo GPU DATENSAMMLUNG - GAMING MODE
echo ======================================
echo.
echo Dieses Script sammelt im Hintergrund
echo Trainingsdaten waehrend du zockst!
echo.
echo WICHTIG:
echo - Lass dieses Fenster offen
echo - Laeuft 1 Stunde
echo - Speichert alle paar Minuten
echo - Minimaler Performance-Impact
echo.
echo ======================================
echo.

cd /d "%~dp0"
python collect_while_gaming.py

echo.
echo Datensammlung beendet!
pause
