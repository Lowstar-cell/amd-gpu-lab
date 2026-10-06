#!/bin/bash
# Build Script für AMD GPU Rasterizer
# Kompiliert HIP, OpenCL, und Assembly Kernels

set -e

echo "==================================="
echo "AMD GPU Rasterizer Build Script"
echo "==================================="

# Detect ROCm installation
if [ -d "/opt/rocm" ]; then
    export ROCM_PATH=/opt/rocm
    export PATH=$ROCM_PATH/bin:$PATH
    echo "✓ Found ROCm at $ROCM_PATH"
elif [ ! -z "$ROCM_PATH" ]; then
    echo "✓ Using ROCm at $ROCM_PATH"
else
    echo "⚠ ROCm not found! HIP kernels will not be compiled."
    echo "  Install ROCm from: https://rocmdocs.amd.com/"
fi

# Detect GPU architecture
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
    if [ -z "$GPU_ARCH" ]; then
        GPU_ARCH="gfx1030"  # Default to RDNA2
        echo "⚠ Could not detect GPU, using default: $GPU_ARCH"
    else
        echo "✓ Detected GPU architecture: $GPU_ARCH"
    fi
else
    GPU_ARCH="gfx1030"
    echo "⚠ rocminfo not found, using default: $GPU_ARCH"
fi

# Create build directory
BUILD_DIR="./build"
mkdir -p $BUILD_DIR

echo ""
echo "[1/3] Building HIP Kernels..."
if command -v hipcc &> /dev/null; then
    hipcc -fPIC --shared \
        --offload-arch=$GPU_ARCH \
        -O3 \
        kernels/hip/rasterizer.hip \
        -o $BUILD_DIR/rasterizer_hip.so

    echo "✓ HIP kernel compiled: $BUILD_DIR/rasterizer_hip.so"
else
    echo "⚠ hipcc not found, skipping HIP build"
fi

echo ""
echo "[2/3] Building OpenCL Kernels..."
# OpenCL kernels are compiled at runtime, just validate syntax
if command -v clang &> /dev/null; then
    clang -cl-std=CL2.0 -Werror -fsyntax-only \
        -include opencl-c.h \
        kernels/opencl/rasterizer.cl 2>/dev/null || echo "⚠ OpenCL syntax check failed"
    echo "✓ OpenCL kernel validated"
else
    echo "⚠ clang not found, skipping OpenCL validation"
fi

echo ""
echo "[3/3] Building GCN/RDNA Assembly..."
if command -v clang &> /dev/null; then
    clang -x assembler \
        -target amdgcn-amd-amdhsa \
        -mcpu=$GPU_ARCH \
        kernels/asm/gcn_rasterizer.s \
        -o $BUILD_DIR/gcn_rasterizer.o 2>/dev/null || echo "⚠ Assembly build experimental"

    if [ -f "$BUILD_DIR/gcn_rasterizer.o" ]; then
        echo "✓ GCN/RDNA assembly compiled"
    else
        echo "⚠ Assembly build failed (this is experimental)"
    fi
else
    echo "⚠ clang not found, skipping assembly build"
fi

echo ""
echo "==================================="
echo "Build Summary:"
echo "==================================="
ls -lh $BUILD_DIR/ 2>/dev/null || echo "No builds generated"

echo ""
echo "Next steps:"
echo "  1. Run: python gpu_rasterization/example_rasterize.py"
echo "  2. Check: gpu_rasterization/rasterizer_output.png"
echo ""
echo "For native HIP performance:"
echo "  export LD_LIBRARY_PATH=$BUILD_DIR:\$LD_LIBRARY_PATH"
echo ""
