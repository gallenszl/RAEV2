#!/bin/bash
#SBATCH --job-name=raev2-512-finaleval
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
CONFIG=${CONFIG:-configs/stage1/sampling/dinov3l-k23-imagenet-512.yaml}
DATA_ROOT=${DATA_ROOT:-/scratch/zs3325/datasets/imagenet}
NUM_SAMPLES=${NUM_SAMPLES:-2048}
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

mkdir -p /scratch/zs3325/runs "${TMPDIR}"

if [[ ! -d "${DATA_ROOT}/val" ]]; then
  echo "Missing ImageNet val directory: ${DATA_ROOT}/val" >&2
  exit 2
fi
if [[ ! -f "pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt" ]]; then
  echo "Missing exported 512 decoder." >&2
  exit 2
fi
if [[ ! -f "pretrained_models/stage1/imagenet-512/dinov3l-k23/stats.pt" ]]; then
  echo "Missing 512 stats." >&2
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

tmp_config="${TMPDIR}/raev2-512-final-eval-${SLURM_JOB_ID:-manual}.yaml"
"${UV_BIN}" run python - "${CONFIG}" "${tmp_config}" "${DATA_ROOT}" "${NUM_SAMPLES}" <<'PYCONFIG'
import sys
from omegaconf import OmegaConf

base_config, out_config, data_root, num_samples = sys.argv[1:5]
cfg = OmegaConf.load(base_config)
cfg.eval.datasets.imagenet.data_dir = data_root
if int(num_samples) > 0:
    cfg.eval.datasets.imagenet.num_samples = int(num_samples)
elif "num_samples" in cfg.eval.datasets.imagenet:
    del cfg.eval.datasets.imagenet.num_samples
OmegaConf.save(cfg, out_config)
print("[Eval config]", out_config)
print("[Eval config] num_samples=", cfg.eval.datasets.imagenet.get("num_samples", "all"))
print("[Eval config] decoder=", cfg.stage_1.params.pretrained_decoder_path)
print("[Eval config] stats=", cfg.stage_1.params.normalization_stat_path)
PYCONFIG

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3l-k23-imagenet-512-sft-final-ema-eval${NUM_SAMPLES}}

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Config: ${tmp_config}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${UV_BIN}" run torchrun --standalone --nproc_per_node=1 \
  src/offline_eval_stage1.py \
  --config "${tmp_config}" \
  --precision bf16 \
  --num-viz "${NUM_VIZ}" \
  "${WANDB_ARGS[@]}"
