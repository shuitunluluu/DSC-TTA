# DSC-TTA: Dual-Space Collaborative Test-Time Adaptation for Vision-Language Models

This repository contains the implementation of **DSC-TTA**, a training-free online test-time adaptation method for CLIP. DSC-TTA uses two complementary branches:

- A **cross-modal adaptation branch** that estimates class visual centers from the incoming test stream and updates dynamic text prototypes online.
- A **visual induction branch** that constructs a spectral reprojection space, maintains online class visual prototypes in that space, and supplies complementary image-image scores.

The two branch outputs are adaptively fused at the logit level. The procedure requires no parameter update and no backpropagation during testing.

Class-level text descriptions for the CD and OOD benchmarks are embedded in `utils.py`, so no external prompt directory is required.

## Environment

The implementation is provided as a standalone PyTorch project.

```bash
conda create -n dsctta python=3.7
conda activate dsctta

# The reported runs use PyTorch 1.12.1 and CUDA 11.3.
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

## Datasets

Prepare the cross-dataset and OOD benchmarks under one root directory. Detailed preparation instructions are provided in [`docs/DATASETS.md`](docs/DATASETS.md).

A typical layout is:

```text
$DATA/
├── imagenet/
├── imagenet-adversarial/
├── imagenet-rendition/
├── imagenet-sketch/
├── imagenetv2/
├── caltech-101/
├── dtd/
├── eurosat/
├── fgvc_aircraft/
├── food-101/
├── oxford_flowers/
├── oxford_pets/
├── stanford_cars/
├── sun397/
└── ucf101/
```

## Repository Layout

```text
DSC-TTA/
├── run_DSC_TTA.py        # Main evaluation entry point
├── reproject.py          # Spectral filtering and reprojected feature utilities
├── utils.py              # Data loading and CLIP inference helpers
├── configs/
│   ├── vit/              # ViT-B/16 settings
│   └── rn/               # RN50 settings
├── datasets/             # Dataset wrappers
├── scripts/              # Benchmark commands
└── docs/DATASETS.md      # Dataset preparation
```

## Configuration

Each YAML file contains dataset-specific hyperparameters:

| Key | Description |
| --- | --- |
| `text_alpha` | Base step size for updating dynamic text prototypes toward estimated visual centers. |
| `trust_kappa` | Rate at which dynamic text prototypes enter the reliable cross-modal classifier. |
| `filter_top` | Number of leading singular directions removed during spectral filtering. |
| `filter_bottom` | Number of trailing singular directions removed during spectral filtering. |
| `fusion_rho` | Growth rate of the visual-induction fusion weight. |
| `fusion_eta` | Maximum visual-induction fusion weight. |

## Running DSC-TTA

Set `DATA_ROOT` to the folder containing the prepared datasets. `CUDA_VISIBLE_DEVICES` defaults to `0` and can be overridden.

### ViT-B/16

Cross-dataset benchmark:

```bash
DATA_ROOT=/path/to/datasets bash scripts/run_cd_benchmark_vit.sh
```

OOD benchmark:

```bash
DATA_ROOT=/path/to/datasets bash scripts/run_ood_benchmark_vit.sh
```

### RN50

Cross-dataset benchmark:

```bash
DATA_ROOT=/path/to/datasets bash scripts/run_cd_benchmark_rn50.sh
```

OOD benchmark:

```bash
DATA_ROOT=/path/to/datasets bash scripts/run_ood_benchmark_rn50.sh
```

### Direct Invocation

```bash
CUDA_VISIBLE_DEVICES=0 python run_DSC_TTA.py \
  --config configs/vit \
  --log-path ./log \
  --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
  --backbone "ViT-B/16" \
  --data-root /path/to/datasets
```

## Reproducibility note

This repository accompanies the submitted manuscript. All experiments use
the online, training-free test-time adaptation protocol described above.
