#!/bin/bash
#SBATCH --job-name=raev2-512-sanity
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
BASE_CONFIG=${BASE_CONFIG:-configs/stage1/training/dinov3l-k23-imagenet-512-sft.yaml}
MICRO_BATCH=${MICRO_BATCH:-4}
SANITY_STEPS=${SANITY_STEPS:-200}
ENABLE_WANDB=${ENABLE_WANDB:-1}
RESULTS_DIR=${RESULTS_DIR:-ckpts/stage1-512-sanity}
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
export WANDB_ENTITY=${WANDB_ENTITY:-szlgallen-peking-university}
export WANDB_PROJECT=${WANDB_PROJECT:-raev2-stage1-512}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-dinov3l-k23-imagenet-512-sanity-mb${MICRO_BATCH}}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "${RESULTS_DIR}"

if [[ ! -d "${DATA_DIR}/train" ]]; then
  echo "Missing ImageNet train directory: ${DATA_DIR}/train" >&2
  exit 2
fi
if [[ ! -f "${DISC_CKPT}" ]]; then
  echo "Missing DINO discriminator checkpoint: ${DISC_CKPT}" >&2
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

tmp_config="${TMPDIR}/raev2-512-sanity-${SLURM_JOB_ID:-manual}.yaml"
"${UV_BIN}" run python - "${BASE_CONFIG}" "${tmp_config}" "${MICRO_BATCH}" "${SANITY_STEPS}" <<'PYCONFIG'
import sys
from omegaconf import OmegaConf
base_config, out_config, micro_batch, steps = sys.argv[1:5]
cfg = OmegaConf.load(base_config)
cfg.training.global_batch_size = int(micro_batch)
cfg.training.grad_accum_steps = 1
cfg.training.epochs = 1
cfg.training.virtual_epoch_steps = int(steps)
cfg.training.log_interval = 10
cfg.training.sample_every = 100
cfg.training.scheduler.decay_end_epoch = 1
cfg.gan.scheduler.decay_end_epoch = 1
cfg.gan.loss.disc_upd_start = 0
cfg.gan.loss.disc_start = 0
cfg.eval.eval_interval = 100
cfg.eval.datasets.imagenet.num_samples = 2048
OmegaConf.save(cfg, out_config)
PYCONFIG

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Config: ${tmp_config}"
echo "Micro batch: ${MICRO_BATCH}"
echo "Sanity steps: ${SANITY_STEPS}"
nvidia-smi

"${UV_BIN}" run torchrun --standalone --nproc_per_node=1 \
  src/train_stage1.py \
  --config "${tmp_config}" \
  --results-dir "${RESULTS_DIR}" \
  --precision bf16 \
  "${WANDB_ARGS[@]}"
