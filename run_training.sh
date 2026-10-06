#!/bin/bash
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export CPLUS_INCLUDE_PATH=/opt/rocm-7.1.1/include

cd /home/user/Dokumente/Projekte/AmdGpudrive
exec .venv-rocm6.4/bin/python3 cuda_translator/train_translator.py \
    --data cuda_translator/training_data_short.json \
    --epochs 500 \
    --batch-size 16 \
    --model-out cuda_translator/translator_v2.pt
