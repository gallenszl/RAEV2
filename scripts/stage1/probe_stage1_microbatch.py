#!/usr/bin/env python
"""Single-GPU Stage1 512 micro-batch memory probe."""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from omegaconf import OmegaConf

from configs import Stage1Config
from data import prepare_unified_dataloader
from stage1.disc import LPIPS, build_discriminator, calculate_adaptive_weight, select_gan_losses
from stage1.utils import validate_stage1_config
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer
from utils.train_utils import get_autocast_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Stage1 512 micro-batch size on one GPU.")
    parser.add_argument("--config", required=True, help="Stage1 config to probe.")
    parser.add_argument("--batch-size", type=int, required=True, help="Single-GPU micro batch size.")
    parser.add_argument("--steps", type=int, default=20, help="Number of optimizer steps to run.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Micro-batch probe requires a CUDA GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage1Config), OmegaConf.load(args.config)))
    config.training.global_batch_size = args.batch_size
    config.training.grad_accum_steps = 1
    config.training.virtual_epoch_steps = args.steps
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    validate_stage1_config(config)

    torch.manual_seed(config.training.global_seed)
    torch.cuda.manual_seed_all(config.training.global_seed)

    dataloader_result = prepare_unified_dataloader(
        config=dataclasses.asdict(config.dataset),
        image_size=config.training.image_size,
        batch_size=args.batch_size,
        num_workers=config.training.num_workers,
        rank=0,
        world_size=1,
        shuffle=True,
        virtual_epoch_steps=args.steps,
    )
    dataloader = dataloader_result.loader

    print(f"[Probe] batch_size={args.batch_size} steps={args.steps} image_size={config.training.image_size}")
    print("[Probe] Building RAE...")
    rae = instantiate_from_config(config.stage_1).to(device)
    rae.encoder.eval()
    rae.decoder.train()
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)

    print("[Probe] Building discriminator and LPIPS...")
    discriminator, disc_aug = build_discriminator(config.gan.arch, device, config.gan.augment)
    lpips_model = LPIPS().to(device).eval()

    optimizer, _ = build_optimizer(rae.decoder.parameters(), config.training.optimizer)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, _ = build_optimizer(disc_params, config.gan.optimizer)
    disc_loss_fn, gen_loss_fn = select_gan_losses(config.gan.loss.disc_loss, config.gan.loss.gen_loss)
    last_layer = rae.decoder.decoder_pred.weight
    autocast_kwargs = get_autocast_kwargs(args)

    step_times = []
    final_stats = {}
    torch.cuda.reset_peak_memory_stats(device)
    start_all = time.perf_counter()

    try:
        for step, (images, _) in enumerate(dataloader):
            if step >= args.steps:
                break
            step_start = time.perf_counter()
            images = images.to(device, non_blocking=True)
            real_normed = images * 2.0 - 1.0

            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            for param in disc_params:
                param.requires_grad_(False)
            with torch.cuda.amp.autocast(**autocast_kwargs):
                recon = rae(images)
                recon_normed = recon * 2.0 - 1.0
                rec_loss = (recon - images).abs().mean()
                lpips_loss = lpips_model(real_normed, recon_normed)
                recon_total = rec_loss + config.gan.loss.perceptual_weight * lpips_loss
                fake_aug = disc_aug.aug(recon_normed)
                logits_fake, _ = discriminator(fake_aug, None)
                gan_loss = gen_loss_fn(logits_fake)
            adaptive_weight = calculate_adaptive_weight(
                recon_total, gan_loss, last_layer, config.gan.loss.max_d_weight
            )
            total_loss = recon_total + config.gan.loss.disc_weight * adaptive_weight * gan_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite generator loss at step {step}: {total_loss.item()}")
            total_loss.backward()
            optimizer.step()

            for param in disc_params:
                param.requires_grad_(True)
            discriminator.train()
            disc_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(**autocast_kwargs):
                with torch.no_grad():
                    recon_disc = rae(images)
                    recon_disc_normed = recon_disc * 2.0 - 1.0
                fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                fake_input = disc_aug.aug(fake_detached)
                real_input = disc_aug.aug(real_normed)
                logits_fake, logits_real = discriminator(fake_input, real_input)
                d_loss = disc_loss_fn(logits_real, logits_fake)
                accuracy = (logits_real > logits_fake).float().mean()
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"non-finite discriminator loss at step {step}: {d_loss.item()}")
            d_loss.backward()
            disc_optimizer.step()

            torch.cuda.synchronize(device)
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            final_stats = {
                "loss_total": float(total_loss.detach().cpu()),
                "loss_recon": float(rec_loss.detach().cpu()),
                "loss_lpips": float(lpips_loss.detach().cpu()),
                "loss_gan": float(gan_loss.detach().cpu()),
                "loss_disc": float(d_loss.detach().cpu()),
                "disc_accuracy": float(accuracy.detach().cpu()),
            }
            print(
                f"[Probe] step={step:03d} time={step_time:.3f}s "
                f"loss={final_stats['loss_total']:.4f} disc={final_stats['loss_disc']:.4f}"
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"[Probe] OOM batch_size={args.batch_size}", file=sys.stderr)
        return 70
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            print(f"[Probe] OOM batch_size={args.batch_size}: {exc}", file=sys.stderr)
            return 70
        raise

    elapsed = time.perf_counter() - start_all
    if len(step_times) != args.steps:
        print(f"[Probe] only completed {len(step_times)} / {args.steps} steps", file=sys.stderr)
        return 72

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    mean_step = sum(step_times) / len(step_times)
    print("[Probe] PASS")
    print(f"[Probe] batch_size={args.batch_size}")
    print(f"[Probe] steps={len(step_times)}")
    print(f"[Probe] peak_memory_allocated_gb={peak_mem_gb:.3f}")
    print(f"[Probe] mean_step_time_sec={mean_step:.3f}")
    print(f"[Probe] elapsed_sec={elapsed:.3f}")
    for key, value in final_stats.items():
        if not math.isfinite(value):
            print(f"[Probe] non-finite {key}={value}", file=sys.stderr)
            return 71
        print(f"[Probe] final_{key}={value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
