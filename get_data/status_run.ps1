<#
Show status and quick preview for a run directory under get_data.

Usage:
  powershell -ExecutionPolicy Bypass -File .\get_data\status_run.ps1
  powershell -ExecutionPolicy Bypass -File .\get_data\status_run.ps1 -RunDir .\get_data\run_2025-xx -Lines 40
#>

param(
  [string] $OutDir = "get_data",
  [string] $RunDir = $null,
  [int] $Lines = 20
)

function Find-MostRecentRunDir {
  param($base)
  if (-not (Test-Path $base)) { return $null }
  # prefer directories containing gpu_metrics.csv or rocm_smi
  $dirs = Get-ChildItem -Path $base -Directory | Sort-Object LastWriteTime -Descending
  foreach ($d in $dirs) {
    if (Test-Path (Join-Path $d.FullName 'gpu_metrics.csv') -or Test-Path (Join-Path $d.FullName 'rocm_smi.csv') -or Test-Path (Join-Path $d.FullName 'metadata.json') -or Test-Path (Join-Path $d.FullName 'collector.pid')) { return $d.FullName }
  }
  return $null
}

if (-not $RunDir) {
  $RunDir = Find-MostRecentRunDir -base $OutDir
  if (-not $RunDir) {
    Write-Error "No run dir found under '$OutDir'"
    exit 2
  }
}

# resolve relative paths relative to script location for robustness
if (-not ([System.IO.Path]::IsPathRooted($RunDir))) {
  # join relative run paths against the repository root (parent of this script's folder)
  $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
  $repoRoot = Split-Path -Parent $scriptRoot
  $abs = Join-Path $repoRoot $RunDir
  try { $rp = Resolve-Path -Path $abs -ErrorAction Stop; $RunDir = $rp.ProviderPath } catch { $RunDir = (Get-Item -Path $abs -ErrorAction SilentlyContinue).FullName }
}

Write-Host "Run dir: $RunDir"

$metaFile = Join-Path $RunDir 'metadata.json'
if (Test-Path $metaFile) {
  try { $meta = Get-Content $metaFile -Raw | ConvertFrom-Json -ErrorAction Stop; Write-Host "metadata:"; $meta | ConvertTo-Json -Depth 4 | Write-Host } catch { Write-Warning "Failed to parse metadata.json" }
} else {
  Write-Host "metadata.json: (not present)"
}

# PID from collector.pid or metadata.json
$pidfile = Join-Path $RunDir 'collector.pid'
$runPid = $null
if (Test-Path $pidfile) {
  $runPid = (Get-Content $pidfile -ErrorAction SilentlyContinue | Select-Object -First 1) -as [int]
  Write-Host "collector.pid present: $runPid"
} elseif (Test-Path $metaFile) {
  try { $m = Get-Content $metaFile -Raw | ConvertFrom-Json -ErrorAction Stop; if ($m.pid) { $runPid = $m.pid -as [int]; Write-Host "metadata.pid: $runPid" } } catch { }
}

if ($runPid) {
  try {
    $proc = Get-Process -Id $runPid -ErrorAction Stop
    Write-Host "Process running: $($proc.ProcessName) (Id: $($proc.Id), StartTime: $($proc.StartTime))"
  } catch {
    Write-Host "No running process with pid $runPid"
  }
} else {
  Write-Host "No PID found for this run."
}

$done = Join-Path $RunDir 'done'
Write-Host "Done file present: $(Test-Path $done)"

# Determine log file
$log = Join-Path $RunDir 'gpu_metrics.csv'
if (-not (Test-Path $log)) {
  # try rocm names
  $cand = Get-ChildItem -Path $RunDir -Filter 'rocm_smi*.csv' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($cand) { $log = $cand.FullName }
}

if (Test-Path $log) {
  Write-Host "Showing last $Lines lines from: $log`n"
  try { Get-Content -Path $log -Tail $Lines | ForEach-Object { Write-Host $_ } } catch { Write-Warning "Could not read log file: $_" }
} else {
  Write-Host "No known log file (gpu_metrics.csv / rocm_smi*.csv) found in run dir."
}

exit 0
