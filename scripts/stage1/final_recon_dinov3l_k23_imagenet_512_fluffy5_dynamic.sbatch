#!/bin/bash
#SBATCH --job-name=raev2-512-fluffy-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/sampling/dinov3l-k23-imagenet-512-fluffy5-dynamic.yaml}
NUM_VIZ=${NUM_VIZ:-64}
ENABLE_WANDB=${ENABLE_WANDB:-0}

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
export WANDB_ENTITY=${WANDB_ENTITY:-szlgallen-peking-university}
export WANDB_PROJECT=${WANDB_PROJECT:-raev2-stage1-512}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3l-k23-imagenet-512-fluffy5-dynamic-final-eval}

mkdir -p /scratch/zs3325/runs "${TMPDIR}"

if [[ ! -f "pretrained_models/stage1/imagenet-512-fluffy5-dynamic/dinov3l-k23/decoder.pt" ]]; then
  echo "Missing mixed decoder." >&2
  exit 2
fi
if [[ ! -f "pretrained_models/stage1/imagenet-512-fluffy5-dynamic/dinov3l-k23/stats.pt" ]]; then
  echo "Missing mixed stats." >&2
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
echo "Config: ${CONFIG}"
echo "Experiment: ${EXPERIMENT_NAME}"
nvidia-smi

"${UV_BIN}" run torchrun --standalone --nproc_per_node=1   src/offline_eval_stage1.py   --config "${CONFIG}"   --precision bf16   --num-viz "${NUM_VIZ}"   "${WANDB_ARGS[@]}"
