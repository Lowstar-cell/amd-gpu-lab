<#
Start the GPU collector and print the created run directory path.
This wrapper supports optional GPU requirement enforcement (-RequireGpu) and will set an env var so the rasterizer can mark CPU fallback.

Usage examples:
  .\start_collect_wrapper.ps1 -Interval 0.5 -OutDir .\get_data -RunName myrun         # foreground, prints run dir and runs collector
  .\start_collect_wrapper.ps1 -Interval 0.5 -OutDir .\get_data -RunName myrun -Background # start in background and print run dir
  .\start_collect_wrapper.ps1 -RequireGpu -Interval 0.5 -RunName myrun  # abort if no GPU detected

Options:
  -Interval <float>     Sampling interval in seconds (fractions allowed)
  -OutDir <string>      Output base directory (default: get_data)
  -RunName <string>     Optional run name prefix
  -Background           Start the collector in a new (minimized) window
  -NoMeta               Do not write metadata.json
  -ShowConsole          Forward collected lines to the console for progress display
  -RequireGpu           Probe for OpenCL/GPU and abort if not present (writes metadata.json cpu_fallback)
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























}  & powershell @argList} else {  Start-Process -FilePath powershell -ArgumentList $argList -WindowStyle Minimized | Out-Null
nif ($Background) {Write-Output $RunDir
n# print run directory for automation/CI to consumeif ($ShowConsole) { $argList += '-ShowConsole' }if ($NoMeta) { $argList += '-NoMeta' }if ($RunName) { $argList += @('-RunName',$RunName) }$argList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$scriptPath,'-Interval',$Interval,'-OutDir',$OutDir,'-RunDir',$RunDir)
n# build argument list to pass to a powershell process$scriptPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) 'collect_windows.ps1'
n# locate collector script (same folder)New-Item -ItemType Directory -Path $RunDir -Force | Out-Null$RunDir = Join-Path $OutDir $nameif ($RunName) { $name = "$RunName-$ts-$id" } else { $name = "run_$ts-$id" }$id = [guid]::NewGuid().ToString().Substring(0,8)$ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
n# replicate collector naming logic to create deterministic run dirif (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }n# Ensure outdir exists