#!/bin/bash
# Quick Start für CUDA→HIP Translator PoC

set -e

echo "=========================================="
echo "CUDA → HIP Translator - Proof of Concept"
echo "=========================================="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Install dependencies
echo "1. Installing dependencies..."
pip install -q torch

# Check if training data exists
if [ ! -f "cuda_translator/training_data.json" ]; then
    echo "Error: training_data.json not found"
    exit 1
fi

# Train model
echo
echo "2. Training model (100 epochs, ~2-5 min)..."
python3 cuda_translator/train_translator.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --eval-interval 20 \
    --model-out cuda_translator/translator.pt

# Test translations
echo
echo "3. Testing translations..."
echo

test_examples=(
    "cudaMalloc(&ptr, size);"
    "int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    "cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);"
    "__syncthreads();"
    "cudaDeviceSynchronize();"
)

for cuda_code in "${test_examples[@]}"; do
    echo "CUDA: $cuda_code"
    python3 cuda_translator/translate.py "$cuda_code"
    echo
done

echo "=========================================="
echo "PoC Complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Try interactive mode: python3 cuda_translator/translate.py --interactive"
echo "  2. Translate a file: python3 cuda_translator/translate.py --input kernel.cu --output kernel.hip"
echo "  3. Add more training examples to training_data.json"
echo "  4. Improve model architecture (try Transformer)"
echo
