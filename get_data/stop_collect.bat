@echo off
REM stop_collect.bat
REM Usage: stop_collect.bat [run-dir]
REM If no run-dir is provided, the script stops the newest run folder in %~dp0

if "%1"=="" (
  REM find newest run dir and stop collector based on its pid file
  PowerShell -NoProfile -ExecutionPolicy Bypass -Command "Try { $d = Get-ChildItem '%~dp0' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if (-not $d) { Write-Output \"No run directories found in %~dp0\"; exit 0}; $pidfile = Join-Path $d.FullName 'collector.pid'; if (Test-Path $pidfile) { $pid = Get-Content $pidfile; Stop-Process -Id $pid -Force; Write-Output \"Stopped PID $pid (run: $($d.Name))\" } else { Write-Output \"No pid file found in newest run dir: $($d.Name)\" } } Catch { Write-Error $_ }"
) else (
  PowerShell -NoProfile -ExecutionPolicy Bypass -Command "Try { $pidfile = Join-Path (Resolve-Path '%~1') 'collector.pid'; if (Test-Path $pidfile) { $pid = Get-Content $pidfile; Stop-Process -Id $pid -Force; Write-Output \"Stopped PID $pid\" } else { Write-Output \"No pid file found in %~1\" } } Catch { Write-Error $_ }"
)

exit /b 0
