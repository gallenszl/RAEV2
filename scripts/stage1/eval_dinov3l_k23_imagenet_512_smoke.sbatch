#!/bin/bash
#SBATCH --job-name=raev2-512-smokeeval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
CONFIG=${CONFIG:-configs/stage1/sampling/dinov3l-k23-imagenet-512-smoke-eval.yaml}
CHECKPOINTS=${CHECKPOINTS:-"ckpts/stage1-512-smoke/dinov3l-k23-imagenet-512-smoke/checkpoints/ep-0000000.pt ckpts/stage1-512-smoke/dinov3l-k23-imagenet-512-smoke/checkpoints/ep-0000001.pt"}
ENABLE_WANDB=${ENABLE_WANDB:-1}
DATA_DIR=${DATA_DIR:-/scratch/zs3325/datasets/imagenet}

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

mkdir -p /scratch/zs3325/runs "${TMPDIR}"

if [[ ! -d "${DATA_DIR}/val" ]]; then
  echo "Missing ImageNet val directory: ${DATA_DIR}/val" >&2
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
echo "Checkpoints: ${CHECKPOINTS}"
nvidia-smi

for ckpt in ${CHECKPOINTS}; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
  ckpt_name=$(basename "${ckpt%.pt}")
  tmp_config="${TMPDIR}/raev2-512-smoke-eval-${SLURM_JOB_ID:-manual}-${ckpt_name}.yaml"
  "${UV_BIN}" run python - "${CONFIG}" "${ckpt}" "${tmp_config}" <<'PYCONFIG'
import sys
from omegaconf import OmegaConf
base_config, ckpt, out_config = sys.argv[1:4]
cfg = OmegaConf.load(base_config)
cfg.stage_1.ckpt = ckpt
OmegaConf.save(cfg, out_config)
PYCONFIG
  export EXPERIMENT_NAME="dinov3l-k23-imagenet-512-smoke-${ckpt_name}-eval2048"
  echo "Evaluating ${ckpt} as ${EXPERIMENT_NAME}"
  "${UV_BIN}" run torchrun --standalone --nproc_per_node=1 \
    src/offline_eval_stage1.py \
    --config "${tmp_config}" \
    --precision bf16 \
    --num-viz 64 \
    "${WANDB_ARGS[@]}"
done
