#!/usr/bin/env python
"""Run fixed-list GSO reconstruction inference for VGGTImageRAE checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data.multiview_dataset import GSOMultiviewFixedDataset
from stage1 import VGGTImageRAE
from utils.train_utils import get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Stage-1 config with VGGTImageRAE params and GSO eval dataset.")
    parser.add_argument("--checkpoint", required=True, help="Final stage-1 checkpoint, e.g. ep-0000016.pt.")
    parser.add_argument("--out-dir", required=True, help="Directory to write per-scene reconstruction images.")
    parser.add_argument("--batch-size", type=int, default=4, help="Scene batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-scenes", type=int, default=None, help="Optional scene limit for quick checks.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--save-overview", action="store_true", help="Also save original/recon/paired overview grids.")
    return parser.parse_args()


def _load_ema_trainable(model: VGGTImageRAE, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("ema", checkpoint.get("model"))
    if state is None:
        raise KeyError(f"Checkpoint has neither 'ema' nor 'model': {checkpoint_path}")
    if bool(checkpoint.get("stage1_trainable_only", False)) and hasattr(model, "load_trainable_state_dict"):
        model.load_trainable_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "stage1_trainable_only": checkpoint.get("stage1_trainable_only", False),
        "stage1_metadata": checkpoint.get("stage1_metadata"),
    }


def _sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _save_scene_views(
    out_dir: Path,
    scene_key: str,
    scene_name: str,
    view_ids: torch.Tensor,
    originals: torch.Tensor,
    recons: torch.Tensor,
) -> None:
    scene_dir = out_dir / "scenes" / scene_key
    scene_dir.mkdir(parents=True, exist_ok=True)
    paired = torch.stack([originals.cpu(), recons.cpu()], dim=1).reshape(-1, *originals.shape[1:])
    save_image(make_grid(paired, nrow=2), scene_dir / "paired_grid.png")
    for idx, view_id in enumerate(view_ids.tolist()):
        save_image(originals[idx].cpu(), scene_dir / f"view_{int(view_id):03d}_orig.png")
        save_image(recons[idx].cpu(), scene_dir / f"view_{int(view_id):03d}_recon.png")
        pair = make_grid(torch.stack([originals[idx].cpu(), recons[idx].cpu()], dim=0), nrow=2)
        save_image(pair, scene_dir / f"view_{int(view_id):03d}_pair.png")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT inference requires a CUDA GPU.")

    cfg = OmegaConf.load(args.config)
    eval_cfg = cfg.eval.datasets.gso
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    model = VGGTImageRAE(**OmegaConf.to_container(cfg.stage_1.params, resolve=True)).to(device)
    ckpt_meta = _load_ema_trainable(model, args.checkpoint)
    model.eval()

    dataset = GSOMultiviewFixedDataset(
        root=str(eval_cfg.root),
        split_file=str(eval_cfg.split_file),
        fixed_view_list_path=str(eval_cfg.fixed_view_list_path),
        image_size=int(cfg.training.image_size),
        views_per_scene=int(eval_cfg.views_per_scene),
        total_views=int(eval_cfg.total_views),
    )
    num_scenes = len(dataset) if args.num_scenes is None else min(int(args.num_scenes), len(dataset))
    dataset.scenes = dataset.scenes[:num_scenes]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    autocast_kwargs = get_autocast_kwargs(args)

    all_originals = []
    all_recons = []
    manifest = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "out_dir": str(out_dir),
        "num_scenes": num_scenes,
        "views_per_scene": int(eval_cfg.views_per_scene),
        "fixed_view_list_path": str(eval_cfg.fixed_view_list_path),
        "checkpoint_metadata": ckpt_meta,
        "scenes": [],
    }

    for images, meta in loader:
        images = images.to(device, non_blocking=True)
        with torch.inference_mode(), torch.cuda.amp.autocast(**autocast_kwargs):
            recon = model(images).clamp(0, 1)
        b, v = images.shape[:2]
        targets = images.reshape(b * v, *images.shape[2:]).detach().cpu()
        recons = recon.detach().cpu()
        if args.save_overview:
            all_originals.append(targets)
            all_recons.append(recons)

        scene_names = list(meta["scene_name"])
        scene_indices = meta["scene_idx"].cpu()
        view_ids = meta["view_ids"].cpu()
        recons_by_scene = recons.reshape(b, v, *recons.shape[1:])
        targets_by_scene = targets.reshape(b, v, *targets.shape[1:])
        for batch_idx, scene_name in enumerate(scene_names):
            scene_idx = int(scene_indices[batch_idx].item())
            scene_key = f"{scene_idx:04d}_{_sanitize_name(scene_name)}"
            scene_view_ids = view_ids[batch_idx]
            _save_scene_views(
                out_dir,
                scene_key,
                scene_name,
                scene_view_ids,
                targets_by_scene[batch_idx],
                recons_by_scene[batch_idx],
            )
            manifest["scenes"].append({
                "scene_idx": scene_idx,
                "scene_name": scene_name,
                "view_ids": [int(x) for x in scene_view_ids.tolist()],
                "scene_dir": str(out_dir / "scenes" / scene_key),
            })

    if args.save_overview and all_originals:
        grid_dir = out_dir / "grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        originals = torch.cat(all_originals, dim=0)
        recons = torch.cat(all_recons, dim=0)
        nrow = int(eval_cfg.views_per_scene) * 2
        paired = torch.stack([originals, recons], dim=1).reshape(-1, *originals.shape[1:])
        save_image(make_grid(originals, nrow=int(eval_cfg.views_per_scene)), grid_dir / "original_all.png")
        save_image(make_grid(recons, nrow=int(eval_cfg.views_per_scene)), grid_dir / "reconstructed_all.png")
        save_image(make_grid(paired, nrow=nrow), grid_dir / "paired_all.png")
        for start in range(0, originals.shape[0], int(eval_cfg.views_per_scene) * 8):
            chunk = paired[start * 2:(start + int(eval_cfg.views_per_scene) * 8) * 2]
            page = start // (int(eval_cfg.views_per_scene) * 8)
            save_image(make_grid(chunk, nrow=nrow), grid_dir / f"paired_page_{page:03d}.png")
        manifest["overview"] = {
            "original_all": str(grid_dir / "original_all.png"),
            "reconstructed_all": str(grid_dir / "reconstructed_all.png"),
            "paired_all": str(grid_dir / "paired_all.png"),
            "paired_pages": math.ceil(originals.shape[0] / (int(eval_cfg.views_per_scene) * 8)),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {num_scenes} scenes to {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
