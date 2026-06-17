# RAEv2 Stage1 VGGT Encoder 实验说明

本文档总结本仓库当前将 RAEv2 Stage1 encoder 从 DINOv3 替换为 official VGGT encoder 的实验设计、代码改动、训练配置、已跑任务和可视化产物。

## 目标范围

- 目标：基于 RAEv2 Stage1 训练流程，pretrain 一个可接收 VGGT multiview MLS feature 的 RAE-style decoder。
- encoder：official VGGT Aggregator，冻结、eval，不使用 RnG-fa3 的魔改 VGGT。
- decoder：保持 RAEv2 现有 `GeneralDecoder` 结构，512 输出，patch size 16。
- 数据：使用 RnG 当前 Objaverse 40K scene list；每个 scene 固定训练输入 `V=4` views。
- eval：使用 GSO fixed 4-view list 看重建效果。
- 训练语义：保持 RAEv2 原生 Stage1 语义，target 是 `[0,1]` pixel，decoder 直接输出 `[0,1]` pixel，loss 直接比较 `recon` 和 `target`。
- RnG 侧后续负责适配这个 decoder；本仓库训练时不做 ImageNet denorm 或 RnG 特殊 pixel-space 改写。

## 核心数据流

Dataset 输出 scene-level multiview batch：

```text
images_512: [B, 4, 3, 512, 512], range [0,1]
scene_ids:   [B]
view_ids:    [B, 4]
```

VGGT encoder 输入前先 resize 到 448：

```text
images_512 -> images_448
images_448: [B, 4, 3, 448, 448], range [0,1]
```

VGGT wrapper 不手动做 ImageNet normalization；official VGGT Aggregator 内部做对应 mean/std normalization。

Official VGGT Aggregator 输出：

```text
aggregated_tokens_list[layer]: [B, 4, 1029, 2048]
patch_start_idx = 5
```

其中：

```text
1029 = 1 camera token + 4 register tokens + 1024 patch tokens
1024 = 32 * 32, because 448 / 14 = 32
```

MLS feature 当前使用 `K=4`，层选择和 RnG/DPT head 对齐：

```text
selected layers = [4, 11, 17, 23]

patch_tokens_l = aggregated_tokens_list[l][:, :, 5:, :]
               = [B, 4, 1024, 2048]

stacked = [4, B, 4, 1024, 2048]
mls = stacked.mean(dim=0) = [B, 4, 1024, 2048]

final_mean = stacked[-1].mean(dim=2, keepdim=True)
           = [B, 4, 1, 2048]

vggt_features = mls + final_mean
              = [B, 4, 1024, 2048]
```

RAE decoder path：

```text
vggt_features: [B, 4, 1024, 2048]
     flatten -> [B*4, 1024, 2048]

projection: Linear(2048 -> decoder_latent_dim)

z_rae:  [B*4, 1024, decoder_latent_dim]
recon:  [B*4, 3, 512, 512], range [0,1]
target: images_512.flatten(0,1) = [B*4, 3, 512, 512]
```

Loss 保持 RAEv2 当前逻辑：

```text
rec_loss = abs(recon - target).mean()

real_normed  = target * 2 - 1
recon_normed = recon  * 2 - 1
```

LPIPS/GAN/discriminator 继续处理 flattened single-image batch，不感知 scene/view 结构。

## 主要代码改动

### VGGT-backed Stage1 model

新增 `stage1.VGGTImageRAE`：

- 加载 official VGGT weights：`/home/zs3325/models/VGGT-1B/model.pt`。
- VGGT encoder 全程 frozen/eval。
- 强制输入为 `[B, 4, 3, H, W]`。
- MLS layers 固定为 `[4, 11, 17, 23]`。
- trainable 部分是 `Linear(2048 -> decoder_latent_dim) + GeneralDecoder`。
- forward 返回 flattened reconstruction `[B*4, 3, 512, 512]`。
- checkpoint 只保存 trainable state，避免保存 frozen VGGT 1B 权重。

VGGT loader 和 RnG-fa3 对齐的细节：

- `patch_embed.pos_embed` resize 使用 `F.interpolate(..., mode="bilinear")`，不显式传 `align_corners`。
- 补齐 `patch_embed.patch_embed.proj.weight` 的 bilinear resize 逻辑；当前 patch size 都是 14，所以实际是 no-op。
- official VGGT 518 pos embed 会 resize 到当前 448 对应的 pos embed：
  `[1, 1370, 1024] -> [1, 1025, 1024]`。

### Multiview dataset

新增 Objaverse/GSO multiview dataset：

- 训练 list：`/home/zs3325/code/RnG-fa3/data/objaverse_v1_in_lvis_25v.txt`
- 训练 root：`/scratch/zs3325/datasets/FluffyElephant`
- GSO root：`/scratch/zs3325/datasets/FluffyElephant/gso_render_rv`
- GSO split：`/home/zs3325/code/RnG-fa3/data/gso_subset64.txt`
- GSO fixed view list：`configs/eval/gso_4view_fixed_seed0.json`
- 每个 scene 采 4 个 RGB view；RGBA 会 white composite alpha。
- 训练时 epoch-level view sampling seed 会变化。
- Dataloader batch 的 `B` 维是 scene batch，同 batch item 来自不同 scene。

### Train/eval 适配

- Stage1 train/eval 支持 5D image batch。
- 5D batch 在 model forward 前保持 `[B,4,...]`。
- loss 和 metrics 前 flatten target 到 `[B*4,3,512,512]`。
- 原有 4D single-image RAEv2 config 保持不变。
- Eval 支持 GSO fixed multiview dataset，并保存 original/reconstruction/paired grid。
- 额外新增 GSO inference helper，按 scene/view 拆图保存，避免只得到一张大拼图。

## Batch 与训练设置

沿用之前 RAEv2 512 SFT 的 image-level effective batch：

```text
previous setting:
  global_batch_size_images = 192
  world_size = 4
  grad_accum_steps = 2
  image_micro_batch_per_gpu = 192 / (4 * 2) = 24
```

Multiview setting：

```text
views_per_scene = 4
scene_micro_batch_per_gpu = 6
image_micro_batch_per_gpu = 6 * 4 = 24

world_size = 4
grad_accum_steps = 2

global_batch_size_scenes = 6 * 4 * 2 = 48 scenes/update
global_batch_size_images = 48 * 4 = 192 images/update
```

每张 GPU 每次 forward 的输入：

```text
images_512: [6, 4, 3, 512, 512]
images_448: [6, 4, 3, 448, 448]
flattened target/recon: [24, 3, 512, 512]
```

Scratch schedule：

- `epochs: 16`
- generator scheduler：`warmup_epochs: 1`, `decay_end_epoch: 16`
- discriminator scheduler：`warmup_epochs: 1`, `decay_end_epoch: 16`
- GAN timing：`disc_start: 8`, `disc_upd_start: 6`

## 当前配置

VGGT 训练相关配置：

- `configs/stage1/training/vggt-mls-k4-objaverse-512.yaml`
- `configs/stage1/training/vggt-mls-k4-objaverse-512-smoke.yaml`
- `configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16.yaml`
- `configs/stage1/training/vggt-mls-k4-objaverse-512-smoke-dinov3l-k7dec.yaml`
- `configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16-dinov3l-k7dec.yaml`

Slurm 脚本：

- smoke：`scripts/stage1/train_vggt_mls_k4_objaverse_512_smoke.sbatch`
- formal 4xH200：`scripts/stage1/train_vggt_mls_k4_objaverse_512.sbatch`
- GSO final inference：`scripts/stage1/infer_vggt_gso_fixed.sbatch`

Export：

- `scripts/stage1/export_vggt_rae_decoder.py`
- 可导出 `rae_decoder.pt` 或 `projection_plus_decoder.pt`。

Stats：

- `scripts/stage1/compute_vggt_rae_stats.py`
- `scripts/stage1/compute_vggt_rae_stats.sbatch`
- 当前 RnG-fa3 需要的 `stats.pt` 统计对象是 projection 前的 VGGT MLS feature，不是 RAE projection 后 latent。
- 统计 shape 是 `[2048, 32, 32]`，对应 `vggt_features: [B,V,1024,2048]` 展成 `[B*V,2048,32,32]`。
- RnG 侧使用该 `stats.pt` 时，normalization/denormalization 应该放在 `Linear(2048 -> decoder_latent_dim)` 之前；如果放在 projection 后，会和 `[768,32,32]` 或 `[1024,32,32]` latent shape 不匹配。

## 两条 decoder 初始化实验

### Scratch16 / 768 latent run

配置：

```text
configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16.yaml
```

Decoder 初始化：

```text
/home/zs3325/models/RAE-collections/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt
decoder_latent_dim = 768
projection = Linear(2048 -> 768)
```

训练任务：

```text
Slurm job: 20726
State: COMPLETED, ExitCode: 0:0
Start: 2026-06-16 16:47:37
End:   2026-06-17 06:06:32
Elapsed: 13:18:55
Node: gpu2
```

W&B：

```text
https://wandb.ai/szlgallen-peking-university/raev2-vggt-stage1-512/runs/37045789
```

Final checkpoint：

```text
/home/zs3325/code/RAEv2/ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/checkpoints/ep-0000016.pt
```

Pre-projection feature stats：

```text
/home/zs3325/code/RAEv2/ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/stats.pt
mean: [2048, 32, 32]
var:  [2048, 32, 32]
semantics: vggt_features before Linear(2048 -> 768)
```

Exported EMA weights：

```text
/home/zs3325/code/RAEv2/ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/exports/decoder.pt
/home/zs3325/code/RAEv2/ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/exports/projection_plus_decoder.pt
```

关键 shape：

```text
decoder.decoder_embed.weight: [1152, 768]
projection.weight:            [768, 2048]
projection.bias:              [768]
```

Final training log：

```text
/scratch/zs3325/runs/raev2-vggt-512-20726.out
/scratch/zs3325/runs/raev2-vggt-512-20726.err
```

### dinov3l-k7 decoder init / 1024 latent run

配置：

```text
configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16-dinov3l-k7dec.yaml
```

Decoder 初始化：

```text
pretrained_models/stage1/general/dinov3l-k7/decoder.pt
decoder_latent_dim = 1024
projection = Linear(2048 -> 1024)
```

这里必须使用 `decoder_latent_dim=1024`，因为该 decoder checkpoint 的关键 shape 是：

```text
decoder_embed.weight: [1152, 1024]
decoder_pos_embed:    [1, 257, 1152]
```

512 训练时只跳过 `decoder_pos_embed`，其余 decoder 权重可加载：

```text
Skipped shape-mismatched decoder keys: ['decoder_pos_embed']
Loaded 455/456 decoder tensors
```

注意：不使用同目录下的 `stats.pt`。该 stats 对应 DINOv3 latent normalization，不适用于当前 VGGT projected features。

这个注意事项只针对 official `nyu-visionx/RAEv2-models` 里的 DINOv3 `stats.pt`。本实验自己生成的 VGGT `stats.pt` 是 projection 前 `[2048,32,32]` feature stats，可供 RnG-fa3 在 VGGT feature 侧使用。

Smoke：

```text
Slurm job: 20803
State: COMPLETED, ExitCode: 0:0
Elapsed: 00:01:17
```

Formal 4xH200 training：

```text
Slurm job: 20805
State at doc time: RUNNING
Start: 2026-06-17 11:17:53
Node: gpu1
```

W&B：

```text
https://wandb.ai/szlgallen-peking-university/raev2-vggt-stage1-512/runs/16971412
```

Logs：

```text
/scratch/zs3325/runs/raev2-vggt-512-20805.out
/scratch/zs3325/runs/raev2-vggt-512-20805.err
```

当前 run 已确认：

```text
scene_micro_batch_per_gpu = 6
image_micro_batch_per_gpu = 24
global_batch_size_images = 192
```

Step 2500 GSO eval 记录：

```text
psnr: 25.052797
ssim: 0.895343
lpips: 0.259286
```

## GSO fixed eval 与 inference 可视化

Fixed view list：

```text
configs/eval/gso_4view_fixed_seed0.json
```

Scratch16 / 768 latent final checkpoint 的拆图 inference：

```text
Slurm job: 20796
State: COMPLETED, ExitCode: 0:0
Elapsed: 00:02:18
```

输出目录：

```text
/scratch/zs3325/runs/raev2-vggt-gso-final-inference-indexed
```

Manifest：

```text
/scratch/zs3325/runs/raev2-vggt-gso-final-inference-indexed/manifest.json
```

输出内容：

- `64` 个 scene 目录。
- `4` views/scene。
- 总计 `844` 个文件。
- 每个 scene 目录包含：
  - `view_XXX_orig.png`
  - `view_XXX_recon.png`
  - `view_XXX_pair.png`
  - `paired_grid.png`
- 总览图目录：
  `/scratch/zs3325/runs/raev2-vggt-gso-final-inference-indexed/grids`
- 包含：
  - `original_all.png`
  - `reconstructed_all.png`
  - `paired_all.png`
  - `paired_page_000.png` 到 `paired_page_007.png`

目录名前加了 scene index，例如：

```text
scenes/0000_3D_Dollhouse_Sofa
scenes/0001_3D_Dollhouse_Swing
```

这是因为 GSO split 中有重复 scene name，单用 scene name 会覆盖。

## 常用命令

### Decoder load smoke

```bash
PYTHONPATH=/home/zs3325/code/RAEv2/src /home/zs3325/.local/bin/uv run python - <<'PY'
from omegaconf import OmegaConf
from stage1.rae import _load_decoder

cfg = OmegaConf.load("configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16-dinov3l-k7dec.yaml")
_load_decoder(
    cfg.stage_1.params.decoder_config_path,
    cfg.stage_1.params.decoder_latent_dim,
    cfg.stage_1.params.decoder_patch_size,
    (cfg.stage_1.params.output_resolution // cfg.stage_1.params.decoder_patch_size) ** 2,
    cfg.stage_1.params.pretrained_decoder_path,
)
PY
```

预期输出：

```text
Skipped shape-mismatched decoder keys: ['decoder_pos_embed']
Loaded 455/456 decoder tensors
```

### 单卡 smoke

```bash
CONFIG=configs/stage1/training/vggt-mls-k4-objaverse-512-smoke-dinov3l-k7dec.yaml \
RESULTS_DIR=ckpts/stage1-vggt-mls-k4-objaverse-512-smoke-dinov3l-k7dec \
EXPERIMENT_NAME=vggt-mls-k4-objaverse-512-smoke-dinov3l-k7dec \
sbatch scripts/stage1/train_vggt_mls_k4_objaverse_512_smoke.sbatch
```

### 4 卡正式训练

不要默认使用 priority QOS。

W&B token 只作为临时环境变量传入，不写进脚本、不提交到 git：

```bash
CONFIG=configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16-dinov3l-k7dec.yaml \
RESULTS_DIR=ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16-dinov3l-k7dec \
EXPERIMENT_NAME=vggt-mls-k4-objaverse-512-scratch16-dinov3l-k7dec-mb6v4-ga2 \
ENABLE_WANDB=1 \
WANDB_KEY=... \
sbatch scripts/stage1/train_vggt_mls_k4_objaverse_512.sbatch
```

### GSO final inference

```bash
sbatch scripts/stage1/infer_vggt_gso_fixed.sbatch
```

默认使用：

```text
CONFIG=configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16.yaml
CHECKPOINT=ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/checkpoints/ep-0000016.pt
OUT_DIR=/scratch/zs3325/runs/raev2-vggt-gso-final-inference-indexed
```

### Compute VGGT pre-projection stats

```bash
sbatch scripts/stage1/compute_vggt_rae_stats.sbatch
```

默认输出：

```text
ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/stats.pt
```

完成记录：

```text
Slurm job: 20818
State: COMPLETED
Elapsed: 00:26:01
GPUs: 4x H200
Samples processed: 178112 image-views
mean/var shape: [2048,32,32]
```

### Export scratch16 EMA weights

```bash
PYTHONPATH=/home/zs3325/code/vggt-official:/home/zs3325/code/RAEv2/src \
/home/zs3325/.local/bin/uv run python scripts/stage1/export_vggt_rae_decoder.py \
  --config configs/stage1/training/vggt-mls-k4-objaverse-512-scratch16.yaml \
  --ckpt ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/checkpoints/ep-0000016.pt \
  --use-ema \
  --decoder-out ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/exports/decoder.pt \
  --projection-decoder-out ckpts/stage1-vggt-mls-k4-objaverse-512-scratch16/vggt-mls-k4-objaverse-512-scratch16-mb6v4-ga2/exports/projection_plus_decoder.pt
```

## Hugging Face 上传与使用指南

目标仓库：

```text
https://huggingface.co/szlgallen/RnG_model
```

本次 scratch16 / 768 latent run 建议上传到：

```text
pretrained_models/stage1/objaverse-512/vggt-mls-k4-scratch16/
```

文件清单：

```text
checkpoints/ep-0000016.pt              full RAEv2 training checkpoint, EMA/model/optimizer/disc/scheduler all included
stats.pt                              VGGT pre-projection feature stats, mean/var [2048,32,32]
exports/decoder.pt                    EMA GeneralDecoder-only state_dict, latent dim 768
exports/projection_plus_decoder.pt    EMA projection + GeneralDecoder state_dict
config.yaml                           exact training config copied from experiment dir
log.txt                               training log copied from experiment dir
README.md                             artifact-level usage guide
```

RnG-fa3 推荐使用：

```text
projection_plus_decoder.pt + stats.pt
```

数据流：

```text
official VGGT MLS feature:
  vggt_features [B,V,1024,2048]
      -> flatten [B*V,1024,2048]
      -> normalize with stats.pt after reshaping stats to [1,1024,2048]
      -> projection Linear(2048 -> 768)
      -> RAE GeneralDecoder
      -> recon [B*V,3,512,512], range [0,1]
```

`stats.pt` 里的 `mean/var` 是 `[2048,32,32]`。RnG 读取时如果沿用 RAEv2 原有 `[C,H,W] -> [1,H*W,C]` reshape 逻辑，会得到 `[1,1024,2048]`，这正好对应 projection 前的 token feature。不要把这份 stats 放到 projection 后使用。

如果只想替换 decoder 而保留 RnG 侧自己的 projection，需要加载：

```text
exports/decoder.pt
```

此时 RnG 侧 projection 输出必须是 `768` 维，并且仍需在 projection 前用本实验 `stats.pt` 处理 VGGT feature。

## 复现检查清单

- Dataset smoke：
  - batch shape `[6,4,3,512,512]`
  - value range `[0,1]`
  - `B` 维 scene name unique
- Model smoke：
  - VGGT layer tokens `[6,4,1029,2048]`
  - MLS feature `[6,4,1024,2048]`
  - `z_rae [24,1024,decoder_latent_dim]`
  - `recon [24,3,512,512]`
  - `target [24,3,512,512]`
- Training smoke：
  - VGGT 无 grad
  - projection/decoder/discriminator 有 grad
  - `decoder_pos_embed` 是 warm-start 中唯一允许跳过的 decoder key
- 4 GPU startup：
  - `scene_micro_batch_per_gpu = 6`
  - `image_micro_batch_per_gpu = 24`
  - `global_batch_size_images = 192`
- GSO eval：
  - fixed view list 跨 run 一致
  - 输出 PSNR/SSIM/LPIPS
  - 保存 grouped original/recon visualization

## 注意事项

- 不要把 W&B token、HF token 或其他 credential 写进脚本、配置或日志。
- 当前 VGGT encoder checkpoint 不保存到 Stage1 checkpoint；只保存 projection/decoder trainable state 和 metadata。
- 使用 dinov3l-k7 decoder init 时，必须保持 `decoder_latent_dim=1024`。
- 使用 scratch16 / 768 latent checkpoint 时，对应 decoder latent dim 是 768；不要和 1024 run 混用 checkpoint。
- scratch16 上传包里的 `stats.pt` 是 VGGT pre-projection `[2048,32,32]` stats，和 DINOv3 RAE latent stats 不是同一种语义。
- RnG 侧使用导出的 decoder 时，需要按本实验 pixel convention 适配，并关闭旧的 ImageNet denorm 假设。
