<#
Windows helper to run the telemetry trainer in a background process with logging.

Usage (PowerShell):
  # use python from a venv (preferred) or system python
  powershell -ExecutionPolicy Bypass -File .\scripts\train_telemetry_windows.ps1 -Minutes 30 -DataDir .\data\processed\get_data -Device gpu

Options:
  -Minutes <float>    time budget in minutes (default 30)
  -DataDir <string>   processed data directory (default data/processed/get_data)
  -Device <string>    cpu|gpu|auto (default auto)
  -VenvPath <string>  optional venv path, e.g. .venv-win\Scripts\python.exe
#>

param(
  [float]$Minutes = 30.0,
  [string]$DataDir = "data/processed/get_data",
  [string]$Device = "auto",
  [string]$VenvPath = $null
)

function Find-PythonExecutable {
  param([string]$venv)
  if ($venv) {
    if (Test-Path $venv) { return $venv }
    # allow passing venv folder, look for Scripts\python.exe
    $cand = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $cand) { return $cand }
  }

  # prefer local .venv-win, fallback to system python
  $localVenv = Join-Path (Get-Location) '.venv-win\Scripts\python.exe'
  if (Test-Path $localVenv) { return $localVenv }

  # also allow common windows venv names
  $localVenv2 = Join-Path (Get-Location) '.venv\Scripts\python.exe'
  if (Test-Path $localVenv2) { return $localVenv2 }

  # finally rely on PATH python
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) { return $py.Source }

  throw "No python executable found. Create or pass a venv path using -VenvPath"
}


$python = Find-PythonExecutable -venv $VenvPath

# quick validation: ensure python runs and prints a short version line (non-fatal)
try {
  $ver = & $python -c "import sys; print('PY:' + sys.version.split()[0])" 2>$null
  if ($ver) { Write-Host "Using python -> $ver" }
} catch {
  Write-Host "Warning: python executable returned an error during validation: $_"
}

$logdir = Join-Path (Get-Location) 'logs\telemetry_smoke'
New-Item -ItemType Directory -Path $logdir -Force | Out-Null
$out = Join-Path $logdir 'telemetry_smoke_master.out'
$pidfile = Join-Path $logdir 'telemetry_smoke.pid'

$args = @()
$args += '--data-dir'; $args += $DataDir
$args += '--minutes'; $args += [string]$Minutes
$args += '--device'; $args += $Device
$args += '--logdir'; $args += $logdir

Write-Host "Starting training with: $python $($args -join ' ')"
# log the full invocation line so users can inspect it in the log file
"Command: $python scripts/train_telemetry.py $($args -join ' ')" | Out-File -FilePath $out -Append -Encoding utf8

# Start-Process returns process; capture it and write PID
try {
  $argList = @('scripts/train_telemetry.py') + $args
  $proc = Start-Process -FilePath $python -ArgumentList $argList -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $out -PassThru
} catch {
  Write-Error "Failed to start process: $_"
  exit 2
}

if ($proc -and $proc.Id) {
  $proc.Id | Out-File -FilePath $pidfile -Encoding ascii
  # also write a short run-start metadata file
  $meta = [ordered]@{
    pid = $proc.Id
    start_time = (Get-Date).ToString('o')
    args = $args -join ' '
  }
  $meta | ConvertTo-Json -Depth 2 | Out-File -FilePath (Join-Path $logdir 'telemetry_smoke_run.json') -Encoding utf8
  Write-Host "Started trainer PID $($proc.Id), logs -> $out"
} else {
  Write-Error "Failed to start trainer process"
}
