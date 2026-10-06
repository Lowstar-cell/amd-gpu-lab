@echo off
REM Wrapper to run the GPU-only embedding scheduler
setlocal

REM Usage: run_embedding.bat --onnx path\to\model.onnx [--input-shape 1 3 224 224]
python -u "%~dp0\scripts\embedding_scheduler_gpu.py" %*
endlocal
