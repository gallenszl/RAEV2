#!/bin/bash
#SBATCH --job-name=raev2-512-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/training/dinov3l-k23-imagenet-512-smoke.yaml}
RESULTS_DIR=${RESULTS_DIR:-ckpts/stage1-512-smoke}
DATA_DIR=${DATA_DIR:-/scratch/zs3325/datasets/imagenet}
DISC_CKPT=${DISC_CKPT:-${REPO_DIR}/pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}
export DINOV3_CKPT_DIR=${DINOV3_CKPT_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3}
export DINOV3_REPO_DIR=${DINOV3_REPO_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3_repo}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3l-k23-imagenet-512-smoke}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "${RESULTS_DIR}"

if [[ ! -d "${DATA_DIR}/train" ]]; then
  echo "Missing ImageNet train directory: ${DATA_DIR}/train" >&2
  exit 2
fi
if [[ ! -f "${DISC_CKPT}" ]]; then
  echo "Missing DINO discriminator checkpoint: ${DISC_CKPT}" >&2
  echo "Run: scripts/stage1/download_official_stage1_weights.sh" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Config: ${CONFIG}"
echo "Results: ${RESULTS_DIR}"
echo "Commit: $(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${UV_BIN}" run python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count()); assert torch.cuda.is_available()"

"${UV_BIN}" run torchrun --standalone --nproc_per_node=1 \
  src/train_stage1.py \
  --config "${CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16
