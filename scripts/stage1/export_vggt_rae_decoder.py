#!/usr/bin/env python3
"""Export decoder weights from a VGGTImageRAE stage-1 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.model_utils import instantiate_from_config


def _select_state(ckpt: dict, use_ema: bool) -> dict:
    key = "ema" if use_ema else "model"
    if key not in ckpt:
        raise KeyError(f"Checkpoint has no {key!r} entry")
    return ckpt[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VGGTImageRAE decoder/projection weights.")
    parser.add_argument("--config", required=True, help="Stage-1 YAML config used to instantiate the model.")
    parser.add_argument("--ckpt", required=True, help="Stage-1 training checkpoint.")
    parser.add_argument("--use-ema", action="store_true", help="Export EMA weights instead of training weights.")
    parser.add_argument("--decoder-out", required=True, help="Output path for GeneralDecoder state_dict.")
    parser.add_argument("--projection-decoder-out", default=None, help="Optional output for projection.* + decoder.*.")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.stage_1)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = _select_state(ckpt, args.use_ema)

    if hasattr(model, "load_trainable_state_dict"):
        model.load_trainable_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)

    decoder_state = model.decoder.state_dict()
    projection_decoder_state = {}
    if hasattr(model, "projection"):
        projection_decoder_state.update({f"projection.{k}": v for k, v in model.projection.state_dict().items()})
    projection_decoder_state.update({f"decoder.{k}": v for k, v in decoder_state.items()})

    if args.dtype != "fp32":
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        decoder_state = {
            k: (v.to(dtype) if torch.is_floating_point(v) else v)
            for k, v in decoder_state.items()
        }
        projection_decoder_state = {
            k: (v.to(dtype) if torch.is_floating_point(v) else v)
            for k, v in projection_decoder_state.items()
        }

    decoder_out = Path(args.decoder_out)
    decoder_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder_state, decoder_out)
    print(f"Saved decoder to {decoder_out}")

    if args.projection_decoder_out:
        out = Path(args.projection_decoder_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(projection_decoder_state, out)
        print(f"Saved projection+decoder to {out}")


if __name__ == "__main__":
    main()
