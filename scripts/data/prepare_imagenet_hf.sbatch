#!/bin/bash
#SBATCH --job-name=imagenet-hf-prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
RAW_DIR=${RAW_DIR:-/scratch/zs3325/datasets/imagenet_hf_raw}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/zs3325/datasets/imagenet}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN must be exported before sbatch. It will not be printed by this script." >&2
  exit 2
fi

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}

mkdir -p /scratch/zs3325/runs "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}" "${RAW_DIR}" "${OUTPUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Raw dir: ${RAW_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"

"${UV_BIN}" run python scripts/data/prepare_imagenet_hf.py \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-workers "${NUM_WORKERS}"
