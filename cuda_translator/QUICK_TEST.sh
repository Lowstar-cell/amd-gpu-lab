#!/bin/bash
# Schneller Test - nur Setup und Komponenten (kein Training)

echo "======================================================================"
echo "CUDA → HIP Translator - Quick Test (kein Training)"
echo "======================================================================"
echo

# PyTorch check
echo "1. PyTorch Check..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch OK"
else
    echo "  ⚠ PyTorch fehlt - installiere..."
    pip install -q torch
    echo "  ✓ Installiert"
fi
echo

# Komponenten Test
echo "2. Komponenten Test..."
python3 cuda_translator/test_poc.py
echo

echo "======================================================================"
echo "✓ Quick Test abgeschlossen!"
echo "======================================================================"
echo
echo "Für vollständigen Test (mit Training):"
echo "  bash cuda_translator/RUN_POC_TEST.sh"
echo
