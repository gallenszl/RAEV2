#!/bin/bash
#SBATCH --job-name=raev2-vggt-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/training/vggt-mls-k4-objaverse-512-smoke.yaml}
RESULTS_DIR=${RESULTS_DIR:-ckpts/stage1-vggt-mls-k4-objaverse-512-smoke}
OFFICIAL_VGGT_REPO=${OFFICIAL_VGGT_REPO:-/home/zs3325/code/vggt-official}
DISC_CKPT=${DISC_CKPT:-${REPO_DIR}/pretrained_models/encoders/dino/dino_vit_small_patch8_224.pth}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${OFFICIAL_VGGT_REPO}:${PYTHONPATH:-}"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-vggt-mls-k4-objaverse-512-smoke}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "${RESULTS_DIR}"

if [[ ! -d "${OFFICIAL_VGGT_REPO}/vggt" ]]; then
  echo "Missing official VGGT repo: ${OFFICIAL_VGGT_REPO}" >&2
  exit 2
fi
if [[ ! -f "${DISC_CKPT}" ]]; then
  echo "Missing DINO discriminator checkpoint: ${DISC_CKPT}" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Official VGGT: ${OFFICIAL_VGGT_REPO}"
echo "Config: ${CONFIG}"
echo "Results: ${RESULTS_DIR}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Commit: $(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

export CONFIG
"${UV_BIN}" run python - <<'PYCONFIG'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["CONFIG"])
world_size = 1
scene_micro = cfg.training.global_batch_size // (world_size * cfg.training.grad_accum_steps)
views = cfg.dataset.views_per_scene
print("[Smoke Config] global_batch_size_scenes=", cfg.training.global_batch_size)
print("[Smoke Config] views_per_scene=", views)
print("[Smoke Config] grad_accum_steps=", cfg.training.grad_accum_steps)
print("[Smoke Config] scene_micro_batch_per_gpu=", scene_micro)
print("[Smoke Config] image_micro_batch_per_gpu=", scene_micro * views)
print("[Smoke Config] global_batch_size_images=", cfg.training.global_batch_size * views)
PYCONFIG

"${UV_BIN}" run torchrun --standalone --nproc_per_node=1 \
  src/train_stage1.py \
  --config "${CONFIG}" \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16
