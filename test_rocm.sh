#!/bin/bash
# AMD GPU Test Script - Auf Linux ausführen!

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              AMD ROCm TEST                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check ROCm
echo "1. Checking ROCm..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
    echo "[OK] ROCm found"
else
    echo "[ERROR] rocm-smi not found - is ROCm installed?"
    exit 1
fi

echo ""

# Activate venv if exists
if [ -d ".venv-rocm6.4" ]; then
    echo "2. Activating ROCm venv..."
    source .venv-rocm6.4/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check PyTorch
echo ""
echo "3. Checking PyTorch..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm/CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Run full test
echo ""
echo "4. Running full test..."
python3 test_rocm_full.py
