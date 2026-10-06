#!/bin/bash
# AMD ML Environment Setup Script
# Source dieses Script um die Umgebung korrekt zu konfigurieren
#
# Verwendung:
#   source amd_ml_env.sh
#   python3 your_script.py
#

# ROCm Pfade
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# AOTriton experimentelle Features aktivieren (für SDPA Flash/MemEff Attention)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# HSA (Heterogeneous System Architecture) für AMD GPUs
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Für gfx1100 (RX 7900 XTX)

# GPU Auswahl (falls nicht schon in .bashrc gesetzt)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}  # Default: GPU 1

# Python venv aktivieren
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/.venv-rocm6.4" ]; then
    source "$SCRIPT_DIR/.venv-rocm6.4/bin/activate"
    echo "✅ AMD ML Environment aktiviert"
    echo "   ROCm: $(rocm-smi --version 2>/dev/null | head -1 || echo 'N/A')"
    echo "   PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
    echo "   GPU: $(rocm-smi --showproductname 2>/dev/null | grep -i radeon | head -1 || echo 'N/A')"
else
    echo "⚠️  venv nicht gefunden: $SCRIPT_DIR/.venv-rocm6.4"
    echo "   Erstelle mit: python3 -m venv .venv-rocm6.4"
fi
