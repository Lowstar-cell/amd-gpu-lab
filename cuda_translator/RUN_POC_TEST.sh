#!/bin/bash
# Kompletter Test des CUDA→HIP Translator PoC

set -e  # Exit on error

echo "======================================================================"
echo "CUDA → HIP Translator - Proof of Concept Test"
echo "======================================================================"
echo

# Schritt 1: PyTorch Installation prüfen/installieren
echo "Schritt 1/5: PyTorch Installation prüfen..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch bereits installiert"
    python3 -c "import torch; print(f'  Version: {torch.__version__}')"
else
    echo "  ⚠ PyTorch nicht gefunden. Installiere CPU Version..."
    pip install -q torch
    echo "  ✓ PyTorch installiert"
fi
echo

# Schritt 2: Komponenten testen
echo "Schritt 2/5: Komponenten testen..."
python3 cuda_translator/test_poc.py
echo

# Schritt 3: Model trainieren (Quick Training mit weniger Epochs)
echo "Schritt 3/5: Model trainieren (50 Epochs für schnellen Test)..."
python3 cuda_translator/train_translator.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.001 \
    --eval-interval 10 \
    --model-out cuda_translator/translator.pt

echo
echo "  ✓ Training abgeschlossen!"
echo

# Schritt 4: Übersetzungen testen
echo "Schritt 4/5: Test-Übersetzungen..."
echo "----------------------------------------------------------------------"

test_cases=(
    "cudaMalloc(&ptr, size);"
    "cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);"
    "int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    "cudaFree(ptr);"
    "__syncthreads();"
)

for cuda_code in "${test_cases[@]}"; do
    echo "CUDA: $cuda_code"
    python3 cuda_translator/translate.py "$cuda_code"
    echo
done

echo "----------------------------------------------------------------------"
echo

# Schritt 5: Evaluation
echo "Schritt 5/5: Model Evaluation..."
python3 cuda_translator/evaluate.py

echo
echo "======================================================================"
echo "✓ PoC Test abgeschlossen!"
echo "======================================================================"
echo
echo "Nächste Schritte:"
echo "  • Interaktiver Modus: python3 cuda_translator/translate.py --interactive"
echo "  • Datei übersetzen: python3 cuda_translator/translate.py -i file.cu -o file.hip"
echo "  • Mehr trainieren: python3 cuda_translator/train_translator.py --epochs 200"
echo
