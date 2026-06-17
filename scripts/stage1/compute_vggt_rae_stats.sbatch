#!/bin/bash
#SBATCH --job-name=raev2-vggt-stats
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16.yaml}
CKPT=${CKPT:-ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/checkpoints/ep-0000016.pt}
OUTPUT_PATH=${OUTPUT_PATH:-ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/stats.pt}
OFFICIAL_VGGT_REPO=${OFFICIAL_VGGT_REPO:-/home/zs3325/code/vggt-official}
BATCH_SIZE=${BATCH_SIZE:-6}
NUM_WORKERS=${NUM_WORKERS:-8}
IMAGE_SIZE=${IMAGE_SIZE:-512}
NUM_SAMPLES=${NUM_SAMPLES:-}
DATASET_EPOCH=${DATASET_EPOCH:-0}
NPROC=${NPROC:-4}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${OFFICIAL_VGGT_REPO}:${REPO_DIR}/src:${PYTHONPATH:-}"

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "$(dirname "${OUTPUT_PATH}")"

if [[ ! -d "${OFFICIAL_VGGT_REPO}/vggt" ]]; then
  echo "Missing official VGGT repo: ${OFFICIAL_VGGT_REPO}" >&2
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Official VGGT: ${OFFICIAL_VGGT_REPO}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT}"
echo "Output stats: ${OUTPUT_PATH}"
echo "Scene batch size per GPU: ${BATCH_SIZE}"
echo "Num workers per GPU: ${NUM_WORKERS}"
echo "Image size: ${IMAGE_SIZE}"
echo "Num scenes: ${NUM_SAMPLES:-all}"
echo "Dataset epoch: ${DATASET_EPOCH}"
echo "NPROC: ${NPROC}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

cmd=(
  "${UV_BIN}" run torchrun --standalone --nproc_per_node="${NPROC}"
  scripts/stage1/compute_vggt_rae_stats.py
  --config "${CONFIG}"
  --ckpt "${CKPT}"
  --output-path "${OUTPUT_PATH}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --image-size "${IMAGE_SIZE}"
  --dataset-epoch "${DATASET_EPOCH}"
  --use-ema
)
if [[ -n "${NUM_SAMPLES}" ]]; then
  cmd+=(--num-samples "${NUM_SAMPLES}")
fi

"${cmd[@]}"

"${UV_BIN}" run python - "${OUTPUT_PATH}" <<'PYCHECK'
import sys
import torch

path = sys.argv[1]
stats = torch.load(path, map_location="cpu")
mean = stats["mean"]
var = stats["var"]
print("[Stats check] path=", path)
print("[Stats check] mean_shape=", tuple(mean.shape), "var_shape=", tuple(var.shape))
print("[Stats check] mean_range=", float(mean.min()), float(mean.max()))
print("[Stats check] var_range=", float(var.min()), float(var.max()))
print("[Stats check] metadata=", stats.get("metadata", {}))
if tuple(mean.shape) != (2048, 32, 32) or tuple(var.shape) != (2048, 32, 32):
    raise SystemExit("Unexpected VGGT RAE pre-projection feature stats shape")
if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
    raise SystemExit("Stats contain non-finite values")
PYCHECK
