#!/bin/bash
#SBATCH --job-name=raev2-stage1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

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
export DINOV3_CKPT_DIR=${DINOV3_CKPT_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3}
export DINOV3_REPO_DIR=${DINOV3_REPO_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3_repo}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" results/stage1/inference

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Commit: $(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${UV_BIN}" run python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count()); assert torch.cuda.is_available(), 'CUDA is required for RAEv2 Stage1 inference'"

CONFIGS=(
  "dinov3l-k7-general:configs/stage1/sampling/dinov3l-k7-general.yaml"
  "dinov3l-k23-general:configs/stage1/sampling/dinov3l-k23-general.yaml"
  "dinov3l-k7-imagenet:configs/stage1/sampling/dinov3l-k7-imagenet.yaml"
  "dinov3l-k23-imagenet:configs/stage1/sampling/dinov3l-k23-imagenet.yaml"
)

for entry in "${CONFIGS[@]}"; do
  name=${entry%%:*}
  config=${entry#*:}
  out_dir="results/stage1/inference/${name}"
  mkdir -p "${out_dir}"

  echo
  echo "=== Running ${name} ==="
  "${UV_BIN}" run python scripts/stage1/sample.py \
    --config "${config}" \
    --image assets/samples/sample_1.png \
    --output-dir "${out_dir}" \
    --stem sample_1 \
    --device cuda
done

echo
echo "Generated PNG files:"
find results/stage1/inference -maxdepth 2 -type f -name "*.png" -printf "%p %s bytes\n" | sort
