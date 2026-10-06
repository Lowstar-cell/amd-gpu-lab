@echo off
REM GPU Training Launch - using current Python 3.12.9
echo 🚀 Starting GPU Training...
echo ================================

echo.
echo 1️⃣  Converting run_* data to Parquet...
python scripts/convert_get_data_to_parquet.py
if errorlevel 1 echo ⚠️  Conversion warning
echo ✅ Data ready

echo.
echo 2️⃣  Launching GPU-only training (300 min)...
echo 📊 Logs: logs\telemetry_smoke\telemetry_smoke_master.out
echo.

python scripts/train_telemetry.py --device gpu --minutes 300 --data-dir data/processed/get_data

echo.
echo ✅ Training complete!
