# AMD GPU Collector - PowerShell Version (kein Python nötig!)
#
# Nutzung: Rechtsklick -> Mit PowerShell ausführen
# Oder: powershell -ExecutionPolicy Bypass -File collector_nopython.ps1

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           AMD GPU DATA COLLECTOR - PowerShell Edition        ║" -ForegroundColor Cyan
Write-Host "║           Kein Python nötig!                                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# === Config ===
$DataDir = "./amd_collected_data"
$IntervalSeconds = 2
$SaveEvery = 150  # Alle 5 Minuten bei 2s Interval

# Erstelle Datenverzeichnis
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

# === GPU Erkennung ===
function Get-AMDGpu {
    $gpus = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" }
    if ($gpus) {
        return $gpus[0].Name
    }
    return "Unknown GPU"
}

# === GPU Stats (via WMI/rocm-smi) ===
function Get-GPUStats {
    $stats = @{
        gpu_util = 0
        mem_util = 0
        temp = 0
        power = 0
    }

    # Versuche rocm-smi
    try {
        $rocmPath = Get-Command rocm-smi -ErrorAction SilentlyContinue
        if ($rocmPath) {
            # GPU Utilization
            $output = & rocm-smi --showuse 2>$null
            foreach ($line in $output) {
                if ($line -match "(\d+)%") {
                    $stats.gpu_util = [int]$Matches[1]
                    break
                }
            }

            # Memory
            $output = & rocm-smi --showmemuse 2>$null
            foreach ($line in $output) {
                if ($line -match "(\d+)%") {
                    $stats.mem_util = [int]$Matches[1]
                    break
                }
            }

            # Temperature
            $output = & rocm-smi --showtemp 2>$null
            foreach ($line in $output) {
                if ($line -match "(\d+\.?\d*)") {
                    $val = [double]$Matches[1]
                    if ($val -gt 20 -and $val -lt 120) {
                        $stats.temp = $val
                        break
                    }
                }
            }
        }
    } catch {
        # rocm-smi nicht verfügbar
    }

    # Fallback: Performance Counter (wenn verfügbar)
    if ($stats.gpu_util -eq 0) {
        try {
            $counter = Get-Counter "\GPU Engine(*engtype_3D)\Utilization Percentage" -ErrorAction SilentlyContinue
            if ($counter) {
                $val = ($counter.CounterSamples | Measure-Object -Property CookedValue -Average).Average
                $stats.gpu_util = [math]::Round($val, 1)
            }
        } catch { }
    }

    return $stats
}

# === User ID ===
$UserName = Read-Host "Dein Name (wird anonymisiert) [anonymous]"
if ([string]::IsNullOrWhiteSpace($UserName)) {
    $UserName = "anonymous"
}

# Simple hash
$md5 = New-Object -TypeName System.Security.Cryptography.MD5CryptoServiceProvider
$utf8 = New-Object -TypeName System.Text.UTF8Encoding
$hash = [System.BitConverter]::ToString($md5.ComputeHash($utf8.GetBytes($UserName))).Replace("-","").Substring(0,12)
$UserId = $hash.ToLower()

$GpuName = Get-AMDGpu

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  User:     $($UserName.PadRight(40))      ║" -ForegroundColor Green
Write-Host "║  User-ID:  $($UserId.PadRight(40))      ║" -ForegroundColor Green
Write-Host "║  GPU:      $($GpuName.Substring(0, [Math]::Min(40, $GpuName.Length)).PadRight(40))      ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# === Menu ===
Write-Host "Optionen:"
Write-Host "  [1] Daten sammeln (Ctrl+C zum Stoppen)"
Write-Host "  [2] Daten exportieren"
Write-Host "  [3] Beenden"
Write-Host ""

$choice = Read-Host "Auswahl [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

switch ($choice) {
    "2" {
        # Export
        Write-Host ""
        Write-Host "[Export] Suche Dateien in $DataDir..."

        $files = Get-ChildItem -Path $DataDir -Filter "*.json" -ErrorAction SilentlyContinue
        if (-not $files) {
            Write-Host "[Export] Keine Daten gefunden!" -ForegroundColor Yellow
            exit
        }

        $allSamples = @()
        foreach ($file in $files) {
            try {
                $data = Get-Content $file.FullName | ConvertFrom-Json
                $allSamples += $data.samples
            } catch { }
        }

        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $exportFile = "$DataDir/EXPORT_combined_$timestamp.json"

        $exportData = @{
            metadata = @{
                created = (Get-Date).ToString("o")
                num_samples = $allSamples.Count
                num_sources = $files.Count
                user_id = $UserId
                gpu_name = $GpuName
            }
            samples = $allSamples
        }

        $exportData | ConvertTo-Json -Depth 10 | Out-File $exportFile -Encoding UTF8

        Write-Host ""
        Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
        Write-Host "║                    EXPORT FERTIG!                            ║" -ForegroundColor Green
        Write-Host "║  $($exportFile.PadRight(50))      ║" -ForegroundColor Green
        Write-Host "║  Samples: $($allSamples.Count.ToString().PadRight(10))                                   ║" -ForegroundColor Green
        Write-Host "║                                                              ║" -ForegroundColor Green
        Write-Host "║  >>> Diese Datei bitte teilen! <<<                           ║" -ForegroundColor Yellow
        Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
        exit
    }
    "3" {
        Write-Host "Beendet."
        exit
    }
}

# === Sammeln ===
Write-Host ""
Write-Host "[Running] Sammle alle ${IntervalSeconds}s, speichere alle $SaveEvery samples" -ForegroundColor Cyan
Write-Host "[Running] Druecke Ctrl+C zum Beenden" -ForegroundColor Cyan
Write-Host ""

$samples = @()
$count = 0
$startTime = Get-Date

try {
    while ($true) {
        $stats = Get-GPUStats

        $sample = @{
            timestamp = [DateTimeOffset]::Now.ToUnixTimeSeconds()
            user_id = $UserId
            gpu_name = $GpuName
            gpu_util = $stats.gpu_util
            mem_util = $stats.mem_util
            temp = $stats.temp
        }
        $samples += $sample
        $count++

        # Status alle 10 Samples
        if ($count % 10 -eq 0) {
            Write-Host "`r[$count] GPU: $($stats.gpu_util)% | Mem: $($stats.mem_util)% | Temp: $($stats.temp)C    " -NoNewline
        }

        # Speichern
        if ($count % $SaveEvery -eq 0) {
            Write-Host ""
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $filename = "$DataDir/gpu_data_${UserId}_${timestamp}.json"

            $data = @{
                metadata = @{
                    user_id = $UserId
                    gpu_name = $GpuName
                    created = (Get-Date).ToString("o")
                    num_samples = $samples.Count
                }
                samples = $samples
            }

            $data | ConvertTo-Json -Depth 5 | Out-File $filename -Encoding UTF8
            Write-Host "[Saved] $($samples.Count) samples -> $filename" -ForegroundColor Green
            $samples = @()
        }

        Start-Sleep -Seconds $IntervalSeconds
    }
} catch {
    # Ctrl+C
}

# Finale Speicherung
if ($samples.Count -gt 0) {
    Write-Host ""
    Write-Host "[Stopping] Speichere letzte Daten..." -ForegroundColor Yellow

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $filename = "$DataDir/gpu_data_${UserId}_${timestamp}_final.json"

    $data = @{
        metadata = @{
            user_id = $UserId
            gpu_name = $GpuName
            created = (Get-Date).ToString("o")
            num_samples = $samples.Count
        }
        samples = $samples
    }

    $data | ConvertTo-Json -Depth 5 | Out-File $filename -Encoding UTF8
    Write-Host "[Saved] $($samples.Count) samples -> $filename" -ForegroundColor Green
}

$duration = ((Get-Date) - $startTime).TotalMinutes
Write-Host ""
Write-Host "[Done] Total: $count samples in $([math]::Round($duration, 1)) Minuten" -ForegroundColor Cyan
Write-Host ""
Write-Host "Exportiere mit Option [2] und teile die EXPORT_*.json Datei!" -ForegroundColor Yellow
