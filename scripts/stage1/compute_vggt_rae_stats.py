#!/usr/bin/env python3
"""Compute normalization stats for pre-projection VGGTImageRAE MLS features."""

from __future__ import annotations

import argparse
import sys
from math import sqrt
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from compute_encoder_stats import (  # noqa: E402
    WelfordAggregator,
    cleanup_distributed,
    create_config_dataloader,
    setup_distributed,
)
from utils.model_utils import instantiate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="VGGTImageRAE stage-1 training config.")
    parser.add_argument("--ckpt", default=None, help="Optional Stage-1 checkpoint path recorded in metadata.")
    parser.add_argument("--output-path", required=True, help="Where to save stats.pt.")
    parser.add_argument("--batch-size", type=int, default=6, help="Scene batch size per GPU.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=None, help="Optional number of scenes.")
    parser.add_argument("--dataset-epoch", type=int, default=0, help="View-sampling epoch for multiview data.")
    parser.add_argument("--use-ema", action="store_true", help="Recorded in metadata only; pre-projection stats do not load trainable state.")
    return parser.parse_args()


def _checkpoint_metadata(ckpt_path: str | None, use_ema: bool) -> dict | None:
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "state_key": "ema" if use_ema else "model",
        "stage1_trainable_only": bool(ckpt.get("stage1_trainable_only", False)),
        "stage1_metadata": ckpt.get("stage1_metadata"),
    }


@torch.no_grad()
def encode_vggt_features(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return pre-projection VGGT MLS features as [B*V, C, H, W]."""
    images = images.to(device, non_blocking=True)
    features = model.encode_multiview(images)
    b, v, n, c = features.shape
    h = w = int(sqrt(n))
    if h * w != n:
        raise ValueError(f"Expected square patch tokens, got N={n}")
    return features.reshape(b * v, n, c).transpose(1, 2).reshape(b * v, c, h, w)


def main() -> None:
    args = parse_args()
    rank, world_size, device, is_distributed = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    cfg = OmegaConf.load(args.config)
    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Config: {args.config}")
        print(f"Checkpoint metadata source: {args.ckpt or 'none'}")
        print(f"Output stats: {args.output_path}")
        print("Stats target: pre-projection VGGT MLS features")

    model = instantiate_from_config(cfg.stage_1).to(device)
    ckpt_meta = _checkpoint_metadata(args.ckpt, use_ema=args.use_ema)
    model.eval()

    feature_dim = int(model.encoder_hidden_size)
    base_patches = int(model.base_patches)
    h = w = int(sqrt(base_patches))
    expected_shape = (feature_dim, h, w)
    if rank == 0:
        print(f"Feature dim: {feature_dim}")
        print(f"Base patches: {base_patches}")
        print(f"Expected stat shape: {expected_shape}")
        print(f"Checkpoint metadata: {ckpt_meta}")

    loader, total_scenes = create_config_dataloader(cfg, args, rank, world_size, is_distributed)
    views_per_scene = int(cfg.dataset.get("views_per_scene", 1))
    if rank == 0:
        print(f"Total scenes: {total_scenes}")
        print(f"Views per scene: {views_per_scene}")
        print(f"Expected image-view samples: {total_scenes * views_per_scene}")
        print(f"Batches per GPU: {len(loader)}")
        print(f"Scene batch size per GPU: {args.batch_size}")

    aggregator = WelfordAggregator(expected_shape, device=device)
    pbar = tqdm(total=len(loader), desc="Processing", disable=rank != 0)
    for images, _ in loader:
        z = encode_vggt_features(model, images, device)
        aggregator.update_batch(z)
        pbar.update(1)
    pbar.close()

    if is_distributed:
        if rank == 0:
            print(f"Aggregating statistics across {world_size} GPUs...")
        aggregator.all_reduce()

    mean, var = aggregator.finalize()
    if rank == 0:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "mean": mean.cpu(),
            "var": var.cpu(),
            "metadata": {
                "type": "VGGTImageRAE_pre_projection_feature_stats",
                "config": args.config,
                "ckpt": args.ckpt,
                "use_ema": bool(args.use_ema),
                "dataset_epoch": int(args.dataset_epoch),
                "num_scenes": int(total_scenes),
                "views_per_scene": int(views_per_scene),
                "num_image_view_samples": int(aggregator.n),
                "feature_shape": list(expected_shape),
                "feature_semantics": "vggt_features before Linear(2048 -> decoder_latent_dim)",
                "checkpoint": ckpt_meta,
            },
        }
        torch.save(stats, output_path)
        print("Statistics computed:")
        print(f"  Mean shape: {tuple(mean.shape)}")
        print(f"  Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
        print(f"  Var shape: {tuple(var.shape)}")
        print(f"  Var range: [{var.min():.6f}, {var.max():.6f}]")
        print(f"  Samples processed: {aggregator.n}")
        print(f"Saved statistics to {output_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
