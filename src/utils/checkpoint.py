"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def _stage1_model_state(module: torch.nn.Module) -> tuple[dict, bool]:
    if hasattr(module, "trainable_state_dict"):
        return module.trainable_state_dict(), True
    return module.state_dict(), False


def _load_stage1_model_state(module: torch.nn.Module, state: dict, trainable_only: bool) -> None:
    if trainable_only and hasattr(module, "load_trainable_state_dict"):
        module.load_trainable_state_dict(state, strict=True)
    else:
        module.load_state_dict(state)


def _stage1_metadata(module: torch.nn.Module):
    if hasattr(module, "checkpoint_metadata"):
        return module.checkpoint_metadata()
    return None


def save_stage1_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 1 training checkpoint (model + discriminator)."""
    model_state, model_trainable_only = _stage1_model_state(model.module)
    ema_state, ema_trainable_only = _stage1_model_state(ema_model)
    trainable_only = model_trainable_only or ema_trainable_only
    state = {
        "step": step,
        "epoch": epoch,
        "model": model_state,
        "ema": ema_state,
        "stage1_trainable_only": trainable_only,
        "stage1_metadata": _stage1_metadata(model.module),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_stage1_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 1 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    trainable_only = bool(checkpoint.get("stage1_trainable_only", False))
    _load_stage1_model_state(model.module, checkpoint["model"], trainable_only)
    _load_stage1_model_state(ema_model, checkpoint["ema"], trainable_only)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 2 training checkpoint."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
]
