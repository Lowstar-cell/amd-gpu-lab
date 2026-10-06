<#
Check for an OpenCL GPU device (best-effort).
Returns exit code 0 if a GPU device usable by OpenCL is found, non-zero otherwise.
Emits JSON to stdout with fields: gpu (bool), reason (string).
#>
param(
  [switch] $Verbose
)

function Write-JsonAndExit($gpu, $reason) {
  $obj = [ordered]@{
    gpu = $gpu
    reason = $reason
  }
  $obj | ConvertTo-Json -Depth 3
  if ($gpu) { exit 0 } else { exit 2 }
}

# 1) Try clinfo if available
if (Get-Command clinfo -ErrorAction SilentlyContinue) {
  try {
    $out = clinfo 2>$null
    if ($out -and ($out -match "Device Type\s*:\s*GPU" -or $out -match "GPU device")) {
      Write-JsonAndExit $true 'clinfo_gpu_present'
    }
  } catch {
    if ($Verbose) { Write-Warning "clinfo failed: $_" }
  }
}

# 2) Try python + pyopencl
if (Get-Command python -ErrorAction SilentlyContinue) {
  try {
    $py = @'
import sys
try:
    import pyopencl as cl
    devs = []
    for p in cl.get_platforms():
        for d in p.get_devices():
            devs.append((p.name, d.type))
    if len(devs) == 0:
        print('NO_OPENCL')
    else:
        # print device types
        for p in cl.get_platforms():
            for d in p.get_devices():
                try:
                    t = str(d.type)
                except:
                    t = ''
                print(p.name + ':' + t)
except Exception as e:
    print('PY_ERR:' + str(e))
'@
    $out = python - <<<'PY'
$py
PY
    if ($out -and $out -notmatch '^PY_ERR:' -and $out -notmatch 'NO_OPENCL') {
      if ($out -match '2' -or $out -match 'GPU') {
        Write-JsonAndExit $true 'pyopencl_gpu_present'
      }
    }
  } catch {
    if ($Verbose) { Write-Warning "python/pyopencl probe failed: $_" }
  }
}

# 3) WMI fallback: check for video controllers (not a guarantee for OpenCL)
try {
  $v = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name,AdapterRAM
  if ($v) {
    foreach ($card in $v) {
      if ($card.Name -and $card.AdapterRAM -gt 0) {
        Write-JsonAndExit $true 'wmi_gpu_present'
      }
    }
  }
} catch {
  if ($Verbose) { Write-Warning "WMI probe failed: $_" }
}

# nothing found
Write-JsonAndExit $false 'no_gpu_detected'
