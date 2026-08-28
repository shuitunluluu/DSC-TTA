#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./dataset}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python run_DSC_TTA.py \
  --config configs/rn \
  --log-path ./log/cd_rn50 \
  --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
  --backbone "RN50" \
  --data-root "${DATA_ROOT}"
