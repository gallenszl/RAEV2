#!/usr/bin/env python
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline evaluation script for Stage-1 VAE/RAE models.

Reconstructs images using a pre-trained stage-1 model and computes metrics
(PSNR, SSIM, rFID). This is the offline counterpart to train_stage1.py,
analogous to how offline_eval.py relates to train.py for stage-2.

Supports:
- Multiple eval datasets through unified dataloader
- Multiple reconstruction metrics per dataset: psnr, ssim, rfid
- EMA checkpoint loading via config.stage_1.ckpt
"""

import argparse
import dataclasses
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from configs.stage1 import Stage1Config
from eval import evaluate_reconstruction_distributed
from eval.datasets import normalize_eval_datasets, prepare_eval_datasets
from stage1 import RAE
from utils import wandb_utils
from utils.logging import save_eval_to_csv
from utils.model_utils import instantiate_from_config
from utils.train_utils import get_autocast_kwargs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _flatten_image_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:
        b, v, c, h, w = images.shape
        return images.reshape(b * v, c, h, w)
    if images.ndim == 4:
        return images
    raise ValueError(f"Expected 4D or 5D image batch, got shape {tuple(images.shape)}")


def main(args):
    """Run offline reconstruction evaluation with distributed execution."""
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires at least one GPU.")

    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Initialize distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    # Setup autocast
    autocast_kwargs = get_autocast_kwargs(args)

    config: Stage1Config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage1Config), OmegaConf.load(args.config)))

    # Set seed
    seed = config.training.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #########################################################
    # Model setup
    #########################################################
    rae: RAE = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    if rank == 0:
        logger.info(f"  Model parameters: {sum(p.numel() for p in rae.parameters())/1e6:.2f}M")

    # ============================================================
    # Eval datasets setup
    # ============================================================
    global_batch_size = config.training.global_batch_size or (config.training.batch_size * world_size)
    assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    batch_size = global_batch_size // world_size

    assert config.eval is not None, "eval section is required in config"
    eval_datasets_config = normalize_eval_datasets(config.eval.datasets)
    eval_datasets = prepare_eval_datasets(
        eval_datasets_config,
        image_size=config.training.image_size,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        rank=rank,
        world_size=world_size,
    )
    eval_dir = config.eval.eval_dir

    experiment_name = os.environ.get("EXPERIMENT_NAME")
    assert experiment_name is not None, "Please set the EXPERIMENT_NAME environment variable."

    if args.wandb and rank == 0:
        entity = os.environ["WANDB_ENTITY"]
        project = os.environ["WANDB_PROJECT"]
        wandb_utils.initialize(args, entity, experiment_name, project)
        import wandb
        wandb.config.update({
            "config": dataclasses.asdict(config),
            "eval/experiment_name": experiment_name,
        }, allow_val_change=True)

    global_step = 0

    # ============================================================
    # Run evaluation for each dataset
    # ============================================================
    for ds_name, ds_info in eval_datasets.items():
        eval_n = min(ds_info.num_samples or len(ds_info.dataset), len(ds_info.dataset))
        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on {ds_name}...")
            logger.info(f"  Samples: {eval_n} / {len(ds_info.dataset)}")
            logger.info(f"  Metrics: {ds_info.metrics}")
            logger.info(f"  Reference: {ds_info.reference_npz}")
            logger.info(f"{'='*60}")

            num_viz = min(args.num_viz, eval_n)
            if num_viz > 0:
                images = torch.stack([ds_info.dataset[i][0] for i in range(num_viz)]).to(device)
                with torch.inference_mode(), torch.cuda.amp.autocast(**autocast_kwargs):
                    recon = rae(images).clamp(0, 1)
                image_targets = _flatten_image_batch(images)
                original_grid = make_grid(image_targets.cpu().float(), nrow=8)
                recon_grid = make_grid(recon.cpu().float(), nrow=8)
                paired_grid = None
                if images.ndim == 5:
                    paired = torch.stack([image_targets, recon], dim=1).reshape(-1, *image_targets.shape[1:])
                    paired_grid = make_grid(paired.cpu().float(), nrow=images.shape[1] * 2)
                grid_dir = Path(eval_dir) / "grids"
                grid_dir.mkdir(parents=True, exist_ok=True)
                save_image(original_grid, grid_dir / f"{experiment_name}_{ds_name}_original.png")
                save_image(recon_grid, grid_dir / f"{experiment_name}_{ds_name}_reconstructed.png")
                if paired_grid is not None:
                    save_image(paired_grid, grid_dir / f"{experiment_name}_{ds_name}_paired.png")
                if args.wandb:
                    images_to_log = {
                        f"eval/{ds_name}/original": original_grid,
                        f"eval/{ds_name}/reconstructed": recon_grid,
                    }
                    if paired_grid is not None:
                        images_to_log[f"eval/{ds_name}/paired"] = paired_grid
                    wandb_utils.log_images(images_to_log, step=global_step)

        eval_stats = evaluate_reconstruction_distributed(
            rae, ds_info.dataset, eval_n,
            rank=rank, world_size=world_size, device=device,
            batch_size=batch_size, experiment_dir=experiment_name,
            global_step=global_step, autocast_kwargs=autocast_kwargs,
            reference_npz_path=ds_info.reference_npz,
            shared_tmpdir=config.dataset.shared_tmpdir,
            metrics_to_compute=ds_info.metrics,
        )
        if eval_stats is not None and rank == 0:
            save_eval_to_csv(experiment_name, f"ema_{ds_name}", global_step, eval_stats, eval_dir)
            if args.wandb:
                wandb_utils.log({f"eval/{k}_{ds_name}": v for k, v in eval_stats.items()}, step=global_step)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if args.wandb:
            import wandb
            wandb.finish()
        logger.info("\nOffline Stage-1 evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation for Stage-1 VAE/RAE models")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Compute precision")
    parser.add_argument("--wandb", action="store_true", help="Use W&B for eval logging.")
    parser.add_argument("--num-viz", type=int, default=64, help="Number of reconstructions to save/log.")
    args = parser.parse_args()
    main(args)
