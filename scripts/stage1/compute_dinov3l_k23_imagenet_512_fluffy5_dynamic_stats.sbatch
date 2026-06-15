#!/bin/bash
#SBATCH --job-name=raev2-512-fluffy-stats
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/zs3325/code/RAEv2}
UV_BIN=${UV_BIN:-/home/zs3325/.local/bin/uv}
BASE_CONFIG=${BASE_CONFIG:-configs/stage1/training/dinov3l-k23-imagenet-512-fluffy5-dynamic-sft.yaml}
DECODER_PATH=${DECODER_PATH:-pretrained_models/stage1/imagenet-512-fluffy5-dynamic/dinov3l-k23/decoder.pt}
OUTPUT_PATH=${OUTPUT_PATH:-pretrained_models/stage1/imagenet-512-fluffy5-dynamic/dinov3l-k23/stats.pt}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
IMAGE_SIZE=${IMAGE_SIZE:-512}
DATASET_EPOCH=${DATASET_EPOCH:-0}
NUM_SAMPLES=${NUM_SAMPLES:-}
NPROC=${NPROC:-4}

cd "${REPO_DIR}"

export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/zs3325/uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/scratch/zs3325/pip-cache}
export HF_HOME=${HF_HOME:-/scratch/zs3325/hf}
export TORCH_HOME=${TORCH_HOME:-/scratch/zs3325/torch}
export TMPDIR=${TMPDIR:-/scratch/zs3325/tmp}
export DINOV3_CKPT_DIR=${DINOV3_CKPT_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3}
export DINOV3_REPO_DIR=${DINOV3_REPO_DIR:-${REPO_DIR}/pretrained_models/encoders/dinov3_repo}
export MPLBACKEND=${MPLBACKEND:-Agg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p /scratch/zs3325/runs "${TMPDIR}" "$(dirname "${OUTPUT_PATH}")"

if [[ ! -f "${DECODER_PATH}" ]]; then
  echo "Missing exported mixed EMA decoder: ${DECODER_PATH}" >&2
  exit 2
fi

tmp_config="${TMPDIR}/raev2-512-fluffy-stats-${SLURM_JOB_ID:-manual}.yaml"
"${UV_BIN}" run python - "${BASE_CONFIG}" "${DECODER_PATH}" "${tmp_config}" <<'PYCONFIG'
import sys
from omegaconf import OmegaConf
base_config, decoder_path, out_config = sys.argv[1:4]
cfg = OmegaConf.load(base_config)
cfg.stage_1.params.pretrained_decoder_path = decoder_path
cfg.stage_1.params.normalization_stat_path = None
OmegaConf.save(cfg, out_config)
print('[Stats config]', out_config)
print('[Stats config] decoder=', cfg.stage_1.params.pretrained_decoder_path)
PYCONFIG

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Config: ${tmp_config}"
echo "Output stats: ${OUTPUT_PATH}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Dataset epoch: ${DATASET_EPOCH}"
echo "Num samples: ${NUM_SAMPLES:-all}"
echo "NPROC: ${NPROC}"
nvidia-smi

cmd=(
  "${UV_BIN}" run torchrun --standalone --nproc_per_node="${NPROC}"
  scripts/stage1/compute_encoder_stats.py
  --config "${tmp_config}"
  --use-config-dataset
  --dataset-epoch "${DATASET_EPOCH}"
  --image-size "${IMAGE_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --output-path "${OUTPUT_PATH}"
)
if [[ -n "${NUM_SAMPLES}" ]]; then
  cmd+=(--num-samples "${NUM_SAMPLES}")
fi
"${cmd[@]}"

"${UV_BIN}" run python - "${OUTPUT_PATH}" <<'PYCHECK'
import sys
import torch
path = sys.argv[1]
stats = torch.load(path, map_location='cpu')
mean = stats['mean']
var = stats['var']
print('[Stats check] path=', path)
print('[Stats check] mean_shape=', tuple(mean.shape), 'var_shape=', tuple(var.shape))
print('[Stats check] mean_range=', float(mean.min()), float(mean.max()))
print('[Stats check] var_range=', float(var.min()), float(var.max()))
if tuple(mean.shape) != (1024, 32, 32) or tuple(var.shape) != (1024, 32, 32):
    raise SystemExit('Unexpected 512 stats shape')
if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
    raise SystemExit('Stats contain non-finite values')
PYCHECK
