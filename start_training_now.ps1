# Fast GPU Training Launch Script
# Activates venv, converts run_* data, starts GPU-only training

Write-Host "🚀 Starting GPU-only Training..." -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Setup ROCm venv Python path
Write-Host "1️⃣  Setting up ROCm venv (.venv-rocm6.4)..." -ForegroundColor Yellow
$pythonExe = ".\.venv-rocm6.4\bin\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = ".\.venv-rocm6.4\bin\python"
}
if (-not (Test-Path $pythonExe)) {
    Write-Host "❌ ROCm Python not found!" -ForegroundColor Red
    ls -la ".\.venv-rocm6.4\bin\" | grep python
    exit 1
}
Write-Host "✅ Found ROCm Python at: $pythonExe`n" -ForegroundColor Green

# 2. Convert run_* data to Parquet
Write-Host "2️⃣  Converting run_* telemetry to Parquet..." -ForegroundColor Yellow
& $pythonExe scripts/convert_get_data_to_parquet.py 2>&1 | Select-String -Pattern "Wrote|Converted" -ErrorAction SilentlyContinue
Write-Host "✅ Data conversion done`n" -ForegroundColor Green

# 3. Start GPU-only training (300 minutes = 5 hours)
Write-Host "3️⃣  Launching GPU-only training (no CPU fallback)..." -ForegroundColor Yellow
Write-Host "📊 Watch logs: cat logs\telemetry_smoke\telemetry_smoke_master.out" -ForegroundColor Cyan
Write-Host "⏹️  Stop: Ctrl+C or delete logs\telemetry_smoke\telemetry_smoke.pid`n" -ForegroundColor Cyan

& $pythonExe scripts/train_telemetry.py --device gpu --minutes 300 --data-dir data/processed/get_data

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "✅ Training launched!" -ForegroundColor Green
