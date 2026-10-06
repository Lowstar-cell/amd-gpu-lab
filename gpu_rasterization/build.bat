@echo off
REM Build Script für AMD GPU Rasterizer (Windows)
REM Kompiliert HIP Kernels mit ROCm on Windows

echo ===================================
echo AMD GPU Rasterizer Build Script
echo ===================================

REM Detect ROCm installation
if exist "C:\Program Files\AMD\ROCm" (
    set ROCM_PATH=C:\Program Files\AMD\ROCm
    echo [OK] Found ROCm at %ROCM_PATH%
) else if defined HIP_PATH (
    set ROCM_PATH=%HIP_PATH%
    echo [OK] Using ROCm at %ROCM_PATH%
) else (
    echo [WARNING] ROCm not found!
    echo Install ROCm for Windows from: https://rocmdocs.amd.com/
    goto :skip_hip
)

REM Add ROCm to PATH
set PATH=%ROCM_PATH%\bin;%PATH%

REM Check for hipcc
where hipcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] hipcc not found in PATH
    goto :skip_hip
)

REM Create build directory
if not exist "build" mkdir build

echo.
echo [1/2] Building HIP Kernels...

REM Detect GPU (default to gfx1030 = RDNA2)
set GPU_ARCH=gfx1030

REM Try to detect actual GPU
hipinfo >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=2" %%a in ('hipinfo ^| findstr "gcnArchName"') do set GPU_ARCH=%%a
    echo [OK] Detected GPU: %GPU_ARCH%
) else (
    echo [INFO] Using default GPU: %GPU_ARCH%
)

REM Compile HIP kernel
hipcc.exe -fPIC --shared ^
    --offload-arch=%GPU_ARCH% ^
    -O3 ^
    kernels\hip\rasterizer.hip ^
    -o build\rasterizer_hip.dll

if %ERRORLEVEL% EQU 0 (
    echo [OK] HIP kernel compiled: build\rasterizer_hip.dll
) else (
    echo [ERROR] HIP compilation failed
)

:skip_hip

echo.
echo [2/2] Validating OpenCL Kernels...
REM OpenCL kernels compile at runtime - just check file exists
if exist "kernels\opencl\rasterizer.cl" (
    echo [OK] OpenCL kernel found
) else (
    echo [ERROR] OpenCL kernel missing
)

echo.
echo ===================================
echo Build Summary:
echo ===================================
dir build 2>nul || echo No builds generated

echo.
echo Next steps:
echo   1. Run: python gpu_rasterization\example_rasterize.py
echo   2. Check: gpu_rasterization\rasterizer_output.png
echo.

if exist "build\rasterizer_hip.dll" (
    echo For native HIP performance, add to PATH:
    echo   set PATH=%cd%\build;%%PATH%%
)

echo.
pause
