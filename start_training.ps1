# GPU Training Launch Script - ROCm
Write-Host "🚀 Starting GPU Training (ROCm)..." -ForegroundColor Cyan

$pythonExe = ".\.venv-rocm6.4\bin\python"

# 1. Check Python
Write-Host "`n1️⃣  Checking ROCm Python..." -ForegroundColor Yellow
if (-not (Test-Path $pythonExe)) {
    Write-Host "❌ Python not found at $pythonExe" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Found: $pythonExe" -ForegroundColor Green

# 2. Convert data
Write-Host "`n2️⃣  Converting run_* data to Parquet..." -ForegroundColor Yellow
$convertCmd = "$pythonExe scripts/convert_get_data_to_parquet.py"
Invoke-Expression $convertCmd | Select-String "Wrote|Converted|Skipping"
Write-Host "✅ Data ready" -ForegroundColor Green

# 3. Train
Write-Host "`n3️⃣  Starting GPU training (300 min)..." -ForegroundColor Yellow
$trainCmd = "$pythonExe scripts/train_telemetry.py --device gpu --minutes 300 --data-dir data/processed/get_data"
Write-Host "📊 Logs: logs\telemetry_smoke\telemetry_smoke_master.out" -ForegroundColor Cyan
Invoke-Expression $trainCmd

Write-Host "`n✅ Training complete!" -ForegroundColor Green
