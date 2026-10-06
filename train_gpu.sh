#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

/home/user/Dokumente/Projekte/AmdGpudrive/.venv-rocm6.4/bin/python3 \
    /home/user/Dokumente/Projekte/AmdGpudrive/cuda_translator/train_translator.py \
    --data /home/user/Dokumente/Projekte/AmdGpudrive/cuda_translator/training_data_combined.json \
    --epochs 500 \
    --batch-size 32 \
    --model-out /home/user/Dokumente/Projekte/AmdGpudrive/cuda_translator/translator_v2.pt
