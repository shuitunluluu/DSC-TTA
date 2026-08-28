#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./dataset}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python run_DSC_TTA.py \
  --config configs/vit \
  --log-path ./log/cd_vit \
  --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
  --backbone "ViT-B/16" \
  --data-root "${DATA_ROOT}"
