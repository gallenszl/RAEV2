#!/bin/bash
#SBATCH --job-name=raev2-512-mbprobe
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
CONFIG=${CONFIG:-configs/stage1/training/dinov3l-k23-imagenet-512-sft.yaml}
BATCHES=${BATCHES:-"4 8 12 16 24 32"}
PROBE_STEPS=${PROBE_STEPS:-20}
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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" /scratch/zs3325/runs/raev2-512-mbprobe

if [[ ! -d "${DATA_DIR}/train" ]]; then
  echo "Missing ImageNet train directory: ${DATA_DIR}/train" >&2
  exit 2
fi
if [[ ! -f "${DISC_CKPT}" ]]; then
  echo "Missing DINO discriminator checkpoint: ${DISC_CKPT}" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG}"
echo "Batches: ${BATCHES}"
echo "Probe steps: ${PROBE_STEPS}"
nvidia-smi

for batch in ${BATCHES}; do
  echo "[Probe] Starting batch ${batch}"
  log_file="/scratch/zs3325/runs/raev2-512-mbprobe/${SLURM_JOB_ID:-manual}-mb${batch}.log"
  set +e
  "${UV_BIN}" run python scripts/stage1/probe_stage1_microbatch.py \
    --config "${CONFIG}" \
    --batch-size "${batch}" \
    --steps "${PROBE_STEPS}" \
    --precision bf16 \
    2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "[Probe] Batch ${batch} failed with status ${status}; stopping probe."
    exit 0
  fi
  echo "[Probe] Batch ${batch} passed."
done

echo "[Probe] All requested batches passed."
