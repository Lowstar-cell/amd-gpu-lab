<#
Tail a run's log and allow clean stop via keypress.

Usage:
  powershell -ExecutionPolicy Bypass -File .\get_data\monitor_run.ps1
  powershell -ExecutionPolicy Bypass -File .\get_data\monitor_run.ps1 -RunDir .\get_data\run_2025-xx -Lines 40
#>

param(
  [string] $OutDir = "get_data",
  [string] $RunDir = $null,
  [int] $Lines = 20,
  [int] $WaitForDoneSeconds = 20
)

function Find-MostRecentRunDir {
  param($base)
  if (-not (Test-Path $base)) { return $null }
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

Write-Host "Monitoring run dir: $RunDir"

# select log file
$log = Join-Path $RunDir 'gpu_metrics.csv'
if (-not (Test-Path $log)) {
  $cand = Get-ChildItem -Path $RunDir -Filter 'rocm_smi*.csv' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($cand) { $log = $cand.FullName }
}

if (-not (Test-Path $log)) { Write-Error "No log file found to tail in $RunDir"; exit 3 }

Write-Host "Tailing log: $log (last $Lines lines). Press any key to stop the collector gracefully."

# start a background job that tails the file
$job = Start-Job -ScriptBlock { param($p,$l) Get-Content -Path $p -Tail $l -Wait } -ArgumentList $log,$Lines

try {
  while ($true) {
    # stream any available output
    # receive new output from the tail job (do not keep so we don't re-print the same lines repeatedly)
    $out = Receive-Job -Job $job
    if ($out) { $out | ForEach-Object { Write-Host $_ } }

    if ([console]::KeyAvailable) {
      $k = [console]::ReadKey($true)
      Write-Host "Key pressed: $($k.KeyChar) - initiating graceful stop..."
      break
    }

    Start-Sleep -Milliseconds 200
  }

  # call stop helper and wait for completion marker
  $stopScript = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'stop_windows_collector.ps1'
  if (-not (Test-Path $stopScript)) { $stopScript = Join-Path $RunDir '..\stop_windows_collector.ps1' }
  if (-not (Test-Path $stopScript)) { Write-Warning "stop_windows_collector.ps1 not found; attempting to create done marker only" }

  if (Test-Path $stopScript) {
    Write-Host "Calling stop helper for run: $RunDir"
    powershell -ExecutionPolicy Bypass -File $stopScript -RunDir $RunDir
  } else {
    $done = Join-Path $RunDir 'done'
    New-Item -ItemType File -Path $done -Force | Out-Null
  }

  # wait for done marker (up to WaitForDoneSeconds)
  $timeout = [DateTime]::UtcNow.AddSeconds($WaitForDoneSeconds)
  $doneFile = Join-Path $RunDir 'done'
  while (-not (Test-Path $doneFile) -and (Get-Date) -lt $timeout) { Start-Sleep -Milliseconds 200 }
  if (Test-Path $doneFile) { Write-Host "Run finished (done marker present)." } else { Write-Warning "Done marker not observed within $WaitForDoneSeconds seconds." }

} finally {
  if ($job) {
    try { Stop-Job -Job $job -Force -ErrorAction SilentlyContinue } catch { }
    try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch { }
  }
}

exit 0
