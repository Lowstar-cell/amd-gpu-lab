<#
Start the GPU collector and print the created run directory path.
Usage examples:
  .\start_collect.ps1 -Interval 0.5 -OutDir .\get_data -RunName myrun         # foreground, prints run dir and runs collector
  .\start_collect.ps1 -Interval 0.5 -OutDir .\get_data -RunName myrun -Background # start in background and print run dir

By default the script pre-creates the run directory using the same naming scheme as the collector and then launches `collect_windows.ps1` with `-RunDir` so the collector uses the pre-created directory.
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

# Ensure outdir exists
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

n# replicate collector naming logic to create deterministic run dir
$ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$id = [guid]::NewGuid().ToString().Substring(0,8)
if ($RunName) { $name = "$RunName-$ts-$id" } else { $name = "run_$ts-$id" }
$RunDir = Join-Path $OutDir $name
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

n# locate collector script (same folder)
$scriptPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) 'collect_windows.ps1'
$checkScript = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) 'check_gpu.ps1'

n# If RequireGpu: run probe and abort if not found (write metadata.json with cpu_fallback)
if ($RequireGpu) {
  try {
    $probeOut = & powershell -NoProfile -ExecutionPolicy Bypass -File $checkScript -Verbose:$false 2>$null
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

# set env var for rasterizer / data collection code to write fallback metadata if needed
n$env:RASTERIZER_RUN_DIR = $RunDir

n# build argument list to pass to a powershell process
$argList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$scriptPath,'-Interval',$Interval,'-OutDir',$OutDir,'-RunDir',$RunDir)
if ($RunName) { $argList += @('-RunName',$RunName) }
if ($NoMeta) { $argList += '-NoMeta' }
if ($ShowConsole) { $argList += '-ShowConsole' }

# print run directory for automation/CI to consume
Write-Output $RunDir

nif ($Background) {
  # Start a child powershell that sets the same env var inside the child to enable rasterizer fallback marking
  $childCmd = "-NoProfile -ExecutionPolicy Bypass -Command \"`$env:RASTERIZER_RUN_DIR='$RunDir'; & '$scriptPath' -Interval $Interval -OutDir '$OutDir' -RunDir '$RunDir'\""
  Start-Process -FilePath powershell -ArgumentList $childCmd -WindowStyle Minimized | Out-Null
} else {
  # For foreground, ensure env var is set in our process so child inherits it
  $env:RASTERIZER_RUN_DIR = $RunDir
  & powershell @argList
}



















}  & powershell @argList} else {  Start-Process -FilePath powershell -ArgumentList $argList -WindowStyle Minimized | Out-Null
nif ($Background) {
n# print run directory for automation/CI to consume
nWrite-Output $RunDirif ($NoMeta) { $argList += '-NoMeta' }if ($RunName) { $argList += @('-RunName',$RunName) }
n# build argument list to pass to a powershell process
n$argList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$scriptPath,'-Interval',$Interval,'-OutDir',$OutDir,'-RunDir',$RunDir)$scriptPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) 'collect_windows.ps1'
n# locate collector script (same folder)New-Item -ItemType Directory -Path $RunDir -Force | Out-Null$RunDir = Join-Path $OutDir $nameif ($RunName) { $name = "$RunName-$ts-$id" } else { $name = "run_$ts_$id" }$id = [guid]::NewGuid().ToString().Substring(0,8)$ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
n# replicate collector naming logic to create deterministic run dirif (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }n# Ensure outdir exists