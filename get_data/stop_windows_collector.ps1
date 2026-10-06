<#
Graceful stop helper for a running collect_windows.ps1 run.

This reads the `collector.pid` in the selected run directory, attempts to stop
the process and waits for it to exit. If the collector doesn't create a `done`
file, this script will create one and update metadata.json with an end_time.

Usage:
  # Stop the most recent run found under get_data
  powershell -ExecutionPolicy Bypass -File .\get_data\stop_windows_collector.ps1

  # Stop a specific run dir
  powershell -ExecutionPolicy Bypass -File .\get_data\stop_windows_collector.ps1 -RunDir .\get_data\run_1234

  # Force immediate termination (if standard stop fails)
  powershell -ExecutionPolicy Bypass -File .\get_data\stop_windows_collector.ps1 -Force -WaitTimeout 10
#>

param(
  [string] $OutDir = "get_data",
  [string] $RunDir = $null,
  [int] $WaitTimeout = 15, # seconds to wait for the process to exit
  [switch] $Force
)

function Find-MostRecentRunDir {
  param($base)
  if (-not (Test-Path $base)) { return $null }
  $dirs = Get-ChildItem -Path $base -Directory | Sort-Object LastWriteTime -Descending
  foreach ($d in $dirs) {
    if (Test-Path (Join-Path $d.FullName 'collector.pid')) { return $d.FullName }
  }
  return $null
}

if (-not $RunDir) {
  $RunDir = Find-MostRecentRunDir -base $OutDir
  if (-not $RunDir) {
    Write-Error "No run directory with collector.pid found under '$OutDir'. Provide -RunDir or check $OutDir."
    exit 2
  }
}

Write-Host "Stopping collector in run dir: $RunDir"

$pidfile = Join-Path $RunDir 'collector.pid'
$targetPid = $null
  if (Test-Path $pidfile) {
  $raw = Get-Content -Path $pidfile -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $raw) { Write-Error "Could not read PID from $pidfile"; exit 4 }
  $targetPid = $raw -as [int]
} else {
  # Fallback: try pid from metadata.json (e.g. python collector writes pid there)
  $metaFile = Join-Path $RunDir 'metadata.json'
  if (Test-Path $metaFile) {
    try {
      $m = Get-Content $metaFile -Raw | ConvertFrom-Json -ErrorAction Stop
      if ($m.pid) { $targetPid = $m.pid -as [int] }
    } catch {
      # ignore parse errors
    }
  }
}

if (-not $targetPid) { Write-Error "Could not find a valid PID (collector.pid or metadata.json) in run dir: '$RunDir'"; exit 5 }

Write-Host "PID: $targetPid"

  try {
    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
} catch { $proc = $null }

if (-not $proc) {
  Write-Host "Process $targetPid is not running. Ensuring run is tidied up."
} else {
  Write-Host "Found process: $($proc.ProcessName) (StartTime=$($proc.StartTime))"

  # Try to stop gracefully: ask the process to close main window (rare for console), then fallback to Stop-Process
  $stopped = $false

  try {
    Write-Host "Attempting Stop-Process (no force) to allow graceful exit..."
    Stop-Process -Id $targetPid -ErrorAction Stop
    $stopped = $true
  } catch {
    Write-Warning "Stop-Process without force failed: $_"
  }

  if (-not $stopped -and $Force.IsPresent) {
    try {
      Write-Host "Force-stopping process $targetPid."
      Stop-Process -Id $targetPid -Force -ErrorAction Stop
      $stopped = $true
    } catch {
      Write-Error "Failed to force-stop process $targetPid: $_"
    }
  }

  if ($stopped) {
    # wait for process to exit
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $WaitTimeout) {
      Start-Sleep -Milliseconds 200
      if (-not (Get-Process -Id $targetPid -ErrorAction SilentlyContinue)) { break }
    }
    if (Get-Process -Id $targetPid -ErrorAction SilentlyContinue) {
      Write-Warning "Process $targetPid still running after $WaitTimeout seconds. Consider rerunning with -Force or check manually."
    } else {
      Write-Host "Process $targetPid exited."
    }
  } else {
    Write-Warning "Failed to stop process $targetPid (no force requested or Stop-Process failed)."
  }
}

# Ensure `done` exists and metadata is updated
$done = Join-Path $RunDir 'done'
if (-not (Test-Path $done)) {
  Write-Host "Creating done marker and updating metadata"
  try {
    New-Item -ItemType File -Path $done -Force | Out-Null
  } catch { Write-Warning "Could not write done marker: $_" }
}

$metaFile = Join-Path $RunDir 'metadata.json'
if (Test-Path $metaFile) {
  try {
    $meta = Get-Content $metaFile -Raw | ConvertFrom-Json -ErrorAction Stop
    $meta.end_time = (Get-Date).ToString('o')
    $meta.stopped_by = $env:USERNAME
    $meta | ConvertTo-Json -Depth 4 | Out-File -FilePath $metaFile -Encoding utf8
    Write-Host "metadata.json updated with end_time."
  } catch {
    Write-Warning "Failed to update metadata.json: $_"
  }
} else {
  # create a minimal metadata if missing
  try {
    $meta = @{ run_id = Split-Path -Leaf $RunDir; end_time = (Get-Date).ToString('o'); stopped_by = $env:USERNAME }
    $meta | ConvertTo-Json -Depth 3 | Out-File -FilePath $metaFile -Encoding utf8
    Write-Host "Created metadata.json with end_time."
  } catch {
    Write-Warning "Failed to write metadata.json: $_"
  }
}

Write-Host "Stop helper finished for run: $RunDir"
