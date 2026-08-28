#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./dataset}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python run_DSC_TTA.py \
  --config configs/vit \
  --log-path ./log/ood_vit \
  --datasets I/A/R/S/V \
  --backbone "ViT-B/16" \
  --data-root "${DATA_ROOT}"
