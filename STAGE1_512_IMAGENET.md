# RAEv2 Stage1 512x512 ImageNet 适配说明

本文档总结本仓库为了训练和使用 512x512 `dinov3l-k23-imagenet` Stage1 RAE 做过的改动、当前可用脚本，以及本轮训练后已经得到的产物。

## 目标范围

- 目标模型：Stage1 RAE, `dinov3l-k23-imagenet`, 512x512。
- 输入图片分辨率：`512x512`。
- encoder latent：`[B, 1024, 32, 32]`。
- encoder：冻结 DINOv3-L/16。
- decoder：从官方 256 ImageNet decoder warm-start，再 SFT 到 512。
- 本轮只处理 Stage1 RAE；Stage2 尚未适配。

## 512 适配做了什么

### 数据与 transform

新增 ImageNet `ImageFolder` 数据入口，支持：

- `dataset.type: imagefolder`
- `data_dir: /scratch/zs3325/datasets/imagenet`
- `split: train` 或 `split: val`
- `condition_type: label` 或 `text`

相关代码：

- `src/data/unified_dataloader.py`
- `src/configs/shared.py`
- `src/stage1/utils.py`

512 transform 当前设定：

- train：`Resize(768, bicubic) + RandomCrop(512) + ToTensor`
- eval/stats：`Resize(768, bicubic) + CenterCrop(512) + ToTensor`

ImageNet HF 下载与 ImageFolder 转换脚本：

- `scripts/data/prepare_imagenet_hf.py`
- `scripts/data/prepare_imagenet_hf.sbatch`

HF token 只从环境变量 `HF_TOKEN` 读取，不写入代码或日志。

### 512 配置

新增配置：

- 正式 512 SFT：`configs/stage1/training/dinov3l-k23-imagenet-512-sft.yaml`
- 512 smoke：`configs/stage1/training/dinov3l-k23-imagenet-512-smoke.yaml`
- 512 base/full：`configs/stage1/training/dinov3l-k23-imagenet-512.yaml`
- 512 sampling：`configs/stage1/sampling/dinov3l-k23-imagenet-512.yaml`
- 512 smoke eval：`configs/stage1/sampling/dinov3l-k23-imagenet-512-smoke-eval.yaml`

关键参数：

- `stage_1.params.resolution: 512`
- `training.image_size: 512`
- 512 decoder/stat 路径：
  - `pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt`
  - `pretrained_models/stage1/imagenet-512/dinov3l-k23/stats.pt`

### 256 到 512 decoder warm-start

相关代码：

- `src/stage1/rae.py`

行为：

- 从官方 256 checkpoint 加载：
  `pretrained_models/stage1/imagenet/dinov3l-k23/decoder.pt`
- shape-safe load。
- 只允许跳过 `decoder_pos_embed`。
- 如果除 `decoder_pos_embed` 外还有 missing key、unexpected key 或 shape mismatch，直接报错。
- 512 的 `decoder_pos_embed` 重新生成，形状为 `[1, 1025, 1152]`，其中 `1025 = 32*32 + cls`。

### Stage1 grad accumulation

相关代码：

- `src/train_stage1.py`
- `src/stage1/engine.py`

新增真实 `grad_accum_steps` 支持：

- micro-batch = `global_batch_size / (world_size * grad_accum_steps)`
- optimizer/scheduler/EMA/global_step 只在 accumulation boundary 更新。
- 非 boundary micro-step 使用 DDP `no_sync()`。
- discriminator optimizer/scheduler 同样按 accumulation boundary 更新。
- W&B/log 中记录 micro batch、effective batch、grad accumulation 和显存峰值。

### Eval 与 W&B

相关代码：

- `src/offline_eval_stage1.py`
- `src/eval/reconstruction.py`
- `src/utils/wandb_utils.py`

主要改动：

- offline eval 支持 `num_samples`，可以只评估 val 前若干张图。
- reconstruction eval 在没有 reference NPZ 时，会用当前输入图片作为 reference 计算 PSNR/SSIM/LPIPS。
- eval 会保存 original/reconstruction grid。
- W&B 通过环境变量认证：
  - `WANDB_KEY`
  - `WANDB_ENTITY`
  - `WANDB_PROJECT`
- 代码不会写入或提交 W&B token。

### Encoder stats

相关脚本：

- `scripts/stage1/compute_encoder_stats.py`
- `scripts/stage1/compute_dinov3l_k23_imagenet_512_stats.sbatch`

512 stats 的含义：

- 对 ImageNet train split 的图片过冻结 DINOv3-L encoder。
- 对 latent `[B, 1024, 32, 32]` 按样本维度计算 `mean` 和 `var`。
- 输出 `stats.pt`，下游 latent normalization 需要它。

## 训练脚本

### 准备官方权重

```bash
scripts/stage1/download_official_stage1_weights.sh
```

用途：

- 下载官方 Stage1 decoder/stats。
- 准备 DINOv3/DINO discriminator 所需权重。

### 准备 ImageNet 数据

```bash
export HF_TOKEN=...
sbatch scripts/data/prepare_imagenet_hf.sbatch
```

输出数据目录：

```text
/scratch/zs3325/datasets/imagenet/train
/scratch/zs3325/datasets/imagenet/val
```

目录格式：

```text
train/0000 ... train/0999
val/0000 ... val/0999
```

### Smoke training

```bash
sbatch scripts/stage1/train_dinov3l_k23_imagenet_512_smoke.sbatch
```

用途：

- 单卡 H200 快速验证 dataloader、encoder latent、decoder reconstruction 和 checkpoint 保存。

### Micro-batch probe

```bash
sbatch scripts/stage1/probe_dinov3l_k23_imagenet_512_microbatch.sbatch
```

用途：

- 单卡测试不同 micro batch 的显存、step time 和 OOM 情况。
- 本轮最终选择 micro batch `24`。

### Sanity run

```bash
MICRO_BATCH=24 sbatch scripts/stage1/train_dinov3l_k23_imagenet_512_sanity.sbatch
```

用途：

- 单卡短步数检查 checkpoint、resume、W&B、eval/sample 是否正常。

### 正式 SFT 训练

```bash
export WANDB_KEY=...
export WANDB_ENTITY=szlgallen-peking-university
export WANDB_PROJECT=raev2-stage1-512
sbatch scripts/stage1/train_dinov3l_k23_imagenet_512_sft.sbatch
```

本轮正式训练 setting：

- GPU：4 卡 H200。
- micro batch：`24/GPU`。
- `grad_accum_steps: 2`。
- effective batch：`192`。
- epochs：`5`。
- decoder/generator LR：`5e-5 -> 5e-6`。
- discriminator LR：保持官方 ImageNet setting，`2e-4 -> 2e-5`。
- EMA：`0.9978`。
- `lpips_start: 0`。
- `disc_upd_start: 1`。
- `disc_start: 3`。

训练阶段含义：

- epoch 0：decoder reconstruction + LPIPS。
- epoch 1-2：训练 discriminator，但 generator 不吃 GAN loss。
- epoch 3-4：generator 加入 GAN loss。

## 导出、stats 和推理验收脚本

### 导出 EMA decoder

```bash
/home/zs3325/.local/bin/uv run python scripts/stage1/extract_decoder.py \
  --config configs/stage1/training/dinov3l-k23-imagenet-512-sft.yaml \
  --ckpt ckpts/stage1-512-sft/dinov3l-k23-imagenet-512-sft-mb24-ga2/checkpoints/ep-0000005.pt \
  --use-ema \
  --out pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt
```

输出：

```text
pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt
```

### 重算 512 stats

```bash
sbatch scripts/stage1/compute_dinov3l_k23_imagenet_512_stats.sbatch
```

默认设置：

- 4 卡 H200。
- `batch_size=32/GPU`。
- ImageNet train full split。
- 输出：

```text
pretrained_models/stage1/imagenet-512/dinov3l-k23/stats.pt
```

当前状态：

- Slurm job：`18300`
- 状态：pending
- 预计开始：`2026-06-05 18:28:00 UTC`

### Final 512 reconstruction 验收

```bash
sbatch --dependency=afterok:18300 scripts/stage1/final_recon_dinov3l_k23_imagenet_512.sbatch
```

默认设置：

- 单卡 H200。
- 使用 `configs/stage1/sampling/dinov3l-k23-imagenet-512.yaml`。
- val 前 `2048` 张。
- metrics：PSNR / SSIM / LPIPS。
- 保存 original/reconstruction grid。

当前状态：

- Slurm job：`18302`
- 状态：pending
- 依赖：`afterok:18300`

### 单图 reconstruction / sampling

stats 完成后，可使用 512 sampling config：

```bash
/home/zs3325/.local/bin/uv run python scripts/stage1/sample.py \
  --config configs/stage1/sampling/dinov3l-k23-imagenet-512.yaml \
  --image /path/to/image.JPEG \
  --out-dir results/stage1/inference/dinov3l-k23-imagenet-512
```

## 当前训练完成后已有产物

### 完整训练 checkpoint

路径：

```text
ckpts/stage1-512-sft/dinov3l-k23-imagenet-512-sft-mb24-ga2/checkpoints/
```

已有 checkpoint：

```text
ep-0000000.pt
ep-0000001.pt
ep-0000002.pt
ep-0000003.pt
ep-0000004.pt
ep-0000005.pt
```

最终训练 checkpoint：

```text
ckpts/stage1-512-sft/dinov3l-k23-imagenet-512-sft-mb24-ga2/checkpoints/ep-0000005.pt
```

用途：

- 继续训练 / resume。
- 重新导出 EMA decoder。
- 包含 model、EMA、optimizer、scheduler、discriminator 等训练状态。

下游推理通常不需要完整 checkpoint。

### 已导出的 EMA decoder

路径：

```text
pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt
```

当前状态：

- 已生成。
- 文件大小约 `1.6G`。
- decoder state dict keys：`456`。
- `decoder_pos_embed` 形状：`[1, 1025, 1152]`。

用途：

- 下游把 512 latent 解码回 RGB 图。
- Stage2 sampling / reconstruction 验收需要它。

### 训练期间 eval 指标

CSV：

```text
experiments/jas/evals/stage1/dinov3l-k23-imagenet-512-sft-mb24-ga2_ema_imagenet.csv
```

当前记录显示：

- best PSNR/SSIM 在 step `20000`：
  - PSNR `33.2204`
  - SSIM `0.9207`
  - LPIPS `0.0743`
- best LPIPS 在 step `32500`：
  - PSNR `33.0650`
  - SSIM `0.9172`
  - LPIPS `0.0737`

### 尚在生成/等待的产物

512 stats：

```text
pretrained_models/stage1/imagenet-512/dinov3l-k23/stats.pt
```

当前尚未生成，正在等待 Slurm job `18300` 运行。

final reconstruction eval 输出：

```text
results/stage1/eval/
```

当前尚未生成，等待 job `18302` 在 stats 成功后自动运行。

## 下游实际需要哪些文件

最小使用集合：

```text
configs/stage1/sampling/dinov3l-k23-imagenet-512.yaml
configs/decoder/ViTXL
pretrained_models/stage1/imagenet-512/dinov3l-k23/decoder.pt
pretrained_models/stage1/imagenet-512/dinov3l-k23/stats.pt
pretrained_models/encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
pretrained_models/encoders/dinov3_repo
```

各自作用：

- `decoder.pt`：EMA decoder，用于 latent -> image。
- `stats.pt`：512 latent mean/var，用于 normalization/denormalization。
- DINOv3-L 权重和 repo：用于 image -> latent。
- 512 sampling config：把 encoder、decoder、stats 和 resolution 组织起来。
- `configs/decoder/ViTXL`：decoder 网络结构配置。

通常不需要：

- `ep-0000005.pt`：除非继续训练或重新导出。
- discriminator 权重：推理和 Stage2 sampling 不需要。
- optimizer/scheduler：只对 resume 训练有用。
- Slurm/W&B/eval logs：只用于实验记录和排错。

## Git 提交状态

512 适配代码已经提交并推送：

```text
commit 4980039 Add ImageNet 512 Stage1 training support
remote git@github.com:gallenszl/RAEV2.git
branch main
```

注意：checkpoint、decoder、stats、ImageNet 数据、W&B 日志和 eval 结果没有提交到 Git。
