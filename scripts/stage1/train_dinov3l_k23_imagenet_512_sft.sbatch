#!/bin/bash
#SBATCH --job-name=raev2-512-sft
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/training/dinov3l-k23-imagenet-512-sft.yaml}
RESULTS_DIR=${RESULTS_DIR:-ckpts/stage1-512-sft}
ENABLE_WANDB=${ENABLE_WANDB:-1}
DATA_DIR=${DATA_DIR:-/scratch/zs3325/datasets/imagenet}
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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_ENTITY=${WANDB_ENTITY:-szlgallen-peking-university}
export WANDB_PROJECT=${WANDB_PROJECT:-raev2-stage1-512}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3l-k23-imagenet-512-sft-mb24-ga2}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "${RESULTS_DIR}"

if [[ ! -d "${DATA_DIR}/train" ]]; then
  echo "Missing ImageNet train directory: ${DATA_DIR}/train" >&2
  exit 2
fi
if [[ ! -d "${DATA_DIR}/val" ]]; then
  echo "Missing ImageNet val directory: ${DATA_DIR}/val" >&2
  exit 2
fi
if [[ ! -f "${DISC_CKPT}" ]]; then
  echo "Missing DINO discriminator checkpoint: ${DISC_CKPT}" >&2
  exit 2
fi

WANDB_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  if [[ -z "${WANDB_KEY:-}" ]]; then
    echo "ENABLE_WANDB=1 requires WANDB_KEY in the environment." >&2
    exit 2
  fi
  WANDB_ARGS=(--wandb)
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Config: ${CONFIG}"
echo "Results: ${RESULTS_DIR}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Commit: $(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${UV_BIN}" run python - <<'PYCONFIG'
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/stage1/training/dinov3l-k23-imagenet-512-sft.yaml')
print('[Config] global_batch_size=', cfg.training.global_batch_size)
print('[Config] grad_accum_steps=', cfg.training.grad_accum_steps)
print('[Config] expected_micro_batch_4gpu=', cfg.training.global_batch_size // (4 * cfg.training.grad_accum_steps))
print('[Config] epochs=', cfg.training.epochs)
print('[Config] disc_upd_start=', cfg.gan.loss.disc_upd_start)
print('[Config] disc_start=', cfg.gan.loss.disc_start)
PYCONFIG

"${UV_BIN}" run torchrun --standalone --nproc_per_node=4 \
  src/train_stage1.py \
  --config "${CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16 \
  "${WANDB_ARGS[@]}"
