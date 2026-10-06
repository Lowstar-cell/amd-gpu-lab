@echo off
REM run_model.bat - Run a saved PyTorch model on Windows (CPU)
REM Usage:
REM   run_model.bat --model path\to\model.pt --sample 1,2,3,4
REM   run_model.bat --model path\to\model.pt --sample-file sample_ids.txt

SETLOCAL ENABLEDELAYEDEXPANSION

if "%1"=="" (
  echo Usage: run_model.bat --model path\to\model.pt [--sample "1,2,3,4"] [--sample-file file]
  exit /b 1
)

python "%~dp0scripts\infer_model.py" %*
exit /b %ERRORLEVEL%
