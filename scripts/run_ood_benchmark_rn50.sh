#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./dataset}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python run_DSC_TTA.py \
  --config configs/rn \
  --log-path ./log/ood_rn50 \
  --datasets I/A/R/S/V \
  --backbone "RN50" \
  --data-root "${DATA_ROOT}"
