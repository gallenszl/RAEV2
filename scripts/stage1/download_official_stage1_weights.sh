#!/bin/bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}

mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" pretrained_models

INCLUDES=(
  "encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  "encoders/dino/dino_vit_small_patch8_224.pth"
  "stage1/general/dinov3l-k7/**"
  "stage1/general/dinov3l-k23/**"
  "stage1/imagenet/dinov3l-k7/**"
  "stage1/imagenet/dinov3l-k23/**"
)

for include in "${INCLUDES[@]}"; do
  echo
  echo "=== Downloading ${include} ==="
  "${UV_BIN}" run hf download nyu-visionx/RAEv2-models \
    --exclude ".gitattributes" \
    --include "${include}" \
    --local-dir pretrained_models
done

find pretrained_models -type f -printf "%p %s bytes\n" | sort
