<#
Start the GPU collector with optional GPU enforcement. Prints the created run directory path (for CI).
Usage:
  .\start_collect_gpu.ps1 -Interval 0.5 -RunName myrun -RequireGpu -Background -ShowConsole
#>
param(
  [float]$Interval = 1.0,
  [string]$OutDir = "get_data",
  [string]$RunName = $null,
  [switch]$Background,
  [switch]$NoMeta,
  [switch]$ShowConsole,
  [switch]$RequireGpu
)

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }
$ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$id = [guid]::NewGuid().ToString().Substring(0,8)
if ($RunName) { $name = "$RunName-$ts-$id" } else { $name = "run_$ts-$id" }
$RunDir = Join-Path $OutDir $name
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

$scriptPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) 'collect_windows.ps1'
$checkScript = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) 'check_gpu.ps1'

if ($RequireGpu) {
  try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $checkScript -Verbose:$false 2>$null
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "No GPU detected by probe. Aborting run and writing cpu_fallback metadata."
      $meta = [ordered]@{
        run_id = Split-Path -Leaf $RunDir
        start_time = (Get-Date).ToString('o')
        cpu_fallback = $true
        fallback_reason = 'gpu_required_but_not_found'
      }
      $meta | ConvertTo-Json -Depth 3 | Out-File -FilePath (Join-Path $RunDir 'metadata.json') -Encoding utf8
      exit 2
    }
  } catch {
    Write-Warning "GPU probe failed: $_"
    $meta = [ordered]@{
      run_id = Split-Path -Leaf $RunDir
      start_time = (Get-Date).ToString('o')
      cpu_fallback = $true
      fallback_reason = 'gpu_probe_failed'
    }
    $meta | ConvertTo-Json -Depth 3 | Out-File -FilePath (Join-Path $RunDir 'metadata.json') -Encoding utf8
    exit 2
  }
}

# set env var so rasterizer can write fallback marker
$env:RASTERIZER_RUN_DIR = $RunDir

# build args
$argList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$scriptPath,'-Interval',$Interval,'-OutDir',$OutDir,'-RunDir',$RunDir)
if ($RunName) { $argList += @('-RunName',$RunName) }
if ($NoMeta) { $argList += '-NoMeta' }
if ($ShowConsole) { $argList += '-ShowConsole' }

# print run dir and start
Write-Output $RunDir
if ($Background) {
  $childCmd = "-NoProfile -ExecutionPolicy Bypass -Command \"`$env:RASTERIZER_RUN_DIR='$RunDir'; & '$scriptPath' -Interval $Interval -OutDir '$OutDir' -RunDir '$RunDir'\""
  Start-Process -FilePath powershell -ArgumentList $childCmd -WindowStyle Minimized | Out-Null
} else {
  $env:RASTERIZER_RUN_DIR = $RunDir
  & powershell @argList
}
