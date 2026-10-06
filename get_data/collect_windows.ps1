<#
PowerShell GPU telemetry collector for Windows

This script samples available GPU performance counters (if supported) and
writes CSVs into a per-run folder under the provided output dir (default: get_data).

Usage examples (PowerShell):
  # basic
  powershell -ExecutionPolicy Bypass -File .\get_data\collect_windows.ps1 -Interval 1 -OutDir .\get_data

  # with run name
  powershell -ExecutionPolicy Bypass -File .\get_data\collect_windows.ps1 -Interval 0.5 -OutDir .\get_data -RunName mybench01

This script is intentionally defensive: it will try several collection methods and
fall back to writing at least a timestamp + host record if more detail is not available.
#>

param(
  [float] $Interval = 2.0,
  [string] $OutDir = "get_data",
  [string] $RunName = $null,
  [switch] $NoMeta
)

function Get-UniqueRunDir {
  param($outdir, $runName)
  $ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
  $id = [guid]::NewGuid().ToString().Substring(0,8)
  if ($runName) { $name = "$runName-$ts-$id" } else { $name = "run_$ts_$id" }
  $path = Join-Path $outdir $name
  New-Item -ItemType Directory -Path $path -Force | Out-Null
  return $path
}

try {
  $RunDir = Get-UniqueRunDir -outdir $OutDir -runName $RunName
} catch {
  Write-Error "Failed to create run dir: $_"
  exit 1
}

$LogFile = Join-Path $RunDir "gpu_metrics.csv"

function Write-Meta {
  param($path, $interval)
  $meta = [ordered]@{
    run_id = Split-Path -Leaf $path
    start_time = (Get-Date).ToString('o')
    host = $env:COMPUTERNAME
    interval_s = $interval
  }
  $meta | ConvertTo-Json -Depth 3 | Out-File -FilePath (Join-Path $path 'metadata.json') -Encoding utf8
}

if (-not $NoMeta) { Write-Meta -path $RunDir -interval $Interval }

Write-Host "Starting Windows GPU collector -> $LogFile  (Interval = $Interval s)"

# write pid file for the collector so users can stop it or inspect it
$pidfile = Join-Path $RunDir 'collector.pid'
try {
  $PID | Out-File -FilePath $pidfile -Encoding ascii
} catch { }

# header: Timestamp, source, metric_name, instance, value
if (-not (Test-Path $LogFile)) { "Timestamp,source,metric,instance,value" | Out-File -FilePath $LogFile -Encoding utf8 }

function Sample-GPUMetrics {
  # Try ADL/ADLX commandline if available in repo or on PATH (not guaranteed)
  try {
    if (Get-Command adlx -ErrorAction SilentlyContinue) {
      $out = adlx --query --json 2>$null
      if ($out) {
        return @{ source = 'adlx'; payload = $out }
      }
    }
  } catch { }

  # Try to use performance counters for GPU Engine utilization
  try {
    $counters = Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -MaxSamples 1 -ErrorAction SilentlyContinue
    if ($counters) { return @{ source='perf'; payload=$counters } }
  } catch { }

  # Fallback: query basic video controller info
  try {
    $v = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name,DriverVersion,AdapterRAM
    if ($v) { return @{ source='cim'; payload=$v } }
  } catch { }

  return @{ source='none'; payload = $null }
}

$rows = 0
try {
  while ($true) {
    $ts = (Get-Date).ToString('o')
    $sample = Sample-GPUMetrics
    if ($sample.source -eq 'perf') {
      # parse counter sample 
      $vals = $sample.payload.CounterSamples
      foreach ($s in $vals) {
        $inst = $s.InstanceName -replace ',', ';'
        $metric = $s.Path -replace '\\','/'
        $val = $s.CookedValue
        "$ts,perf,$metric,$inst,$val" | Out-File -FilePath $LogFile -Append -Encoding utf8
        $rows += 1
      }
    } elseif ($sample.source -eq 'cim') {
      foreach ($card in $sample.payload) {
        $name = $card.Name -replace ',', ';'
        $drv = $card.DriverVersion -replace ',', ';'
        $ram = $card.AdapterRAM
        "$ts,cim,Name,,$name" | Out-File -FilePath $LogFile -Append -Encoding utf8
        "$ts,cim,Driver,,""$drv""" | Out-File -FilePath $LogFile -Append -Encoding utf8
        "$ts,cim,AdapterRAM,,$ram" | Out-File -FilePath $LogFile -Append -Encoding utf8
        $rows += 3
      }
    } elseif ($sample.source -eq 'adlx') {
      # store raw adlx JSON lines
      "$ts,adlx,json,," + ($sample.payload -replace '\n',' ' ) | Out-File -FilePath $LogFile -Append -Encoding utf8
      $rows += 1
    } else {
      # minimal record
      "$ts,none,info,,no_data" | Out-File -FilePath $LogFile -Append -Encoding utf8
      $rows += 1
    }

    Start-Sleep -Seconds $Interval
  }
} catch [System.Exception] {
  Write-Host "Collector stopping: $_"
} finally {
  # write done marker
  (Join-Path $RunDir 'done') | Out-File -FilePath (Join-Path $RunDir 'done') -Encoding utf8
  "Finished - wrote $rows rows" | Out-Host
}
