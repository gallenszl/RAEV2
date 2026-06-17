#!/bin/bash
#SBATCH --job-name=raev2-vggt-gso-infer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16.yaml}
CHECKPOINT=${CHECKPOINT:-ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/checkpoints/ep-0000016.pt}
OUT_DIR=${OUT_DIR:-/scratch/zs3325/runs/raev2-vggt-gso-final-inference-indexed}
OFFICIAL_VGGT_REPO=${OFFICIAL_VGGT_REPO:-/home/zs3325/code/vggt-official}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${OFFICIAL_VGGT_REPO}:${REPO_DIR}/src:${PYTHONPATH:-}"

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "${OUT_DIR}"

if [[ ! -d "${OFFICIAL_VGGT_REPO}/vggt" ]]; then
  echo "Missing official VGGT repo: ${OFFICIAL_VGGT_REPO}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Out dir: ${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${UV_BIN}" run python scripts/stage1/infer_vggt_gso_fixed.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --precision bf16 \
  --save-overview
