#!/bin/bash
#SBATCH --job-name=raev2-512-fluffy-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/training/dinov3l-k23-imagenet-512-fluffy5-dynamic-smoke.yaml}
RESULTS_DIR=${RESULTS_DIR:-ckpts/stage1-512-fluffy5-dynamic-smoke}
DATA_DIR=${DATA_DIR:-/scratch/zs3325/datasets/imagenet}
FLUFFY_DIR=${FLUFFY_DIR:-/scratch/zs3325/datasets/FluffyElephant}
WARMSTART_DECODER=${WARMSTART_DECODER:-${REPO_DIR}/pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt}
DISC_CKPT=${DISC_CKPT:-${REPO_DIR}/pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}
export DINOV3_CKPT_DIR=${DINOV3_CKPT_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3}
export DINOV3_REPO_DIR=${DINOV3_REPO_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3_repo}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3l-k23-imagenet-512-fluffy5-dynamic-smoke}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "${RESULTS_DIR}"

if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
  echo "Missing ImageNet train/val directories under ${DATA_DIR}" >&2
  exit 2
fi
if [[ ! -d "${FLUFFY_DIR}" ]]; then
  echo "Missing FluffyElephant directory: ${FLUFFY_DIR}" >&2
  exit 2
fi
if [[ ! -f "${WARMSTART_DECODER}" ]]; then
  echo "Missing warm-start 512 decoder: ${WARMSTART_DECODER}" >&2
  exit 2
fi
if [[ ! -f "${DISC_CKPT}" ]]; then
  echo "Missing DINO discriminator checkpoint: ${DISC_CKPT}" >&2
  exit 2
fi

"${UV_BIN}" run python - <<'PYCONFIG'
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/stage1/training/dinov3l-k23-imagenet-512-fluffy5-dynamic-smoke.yaml')
print('[Config] global_batch_size=', cfg.training.global_batch_size)
print('[Config] grad_accum_steps=', cfg.training.grad_accum_steps)
print('[Config] virtual_epoch_steps=', cfg.training.virtual_epoch_steps)
print('[Config] fluffy_expected_samples=', cfg.dataset.concat[1].expected_samples)
print('[Config] expected_mixed_len=', 1281167 + cfg.dataset.concat[1].expected_samples * cfg.dataset.concat[1].views_per_sample)
PYCONFIG

nvidia-smi
"${UV_BIN}" run torchrun --standalone --nproc_per_node=1   src/train_stage1.py   --config "${CONFIG}"   --results-dir "${RESULTS_DIR}"   --precision bf16
