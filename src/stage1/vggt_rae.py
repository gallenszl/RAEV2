"""VGGT-backed RAE stage-1 model for multiview decoder pretraining."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import GeneralDecoder
from .rae import _load_decoder


def _prepend_repo_to_path(repo_path: str | None) -> None:
    if not repo_path:
        return
    repo = str(Path(repo_path).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _resize_abs_pos_embed(pos_embed: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if tuple(pos_embed.shape) == tuple(target_shape):
        return pos_embed
    if pos_embed.ndim != 3 or len(target_shape) != 3:
        raise ValueError(f"Cannot resize pos_embed shape {tuple(pos_embed.shape)} to {tuple(target_shape)}")
    cls_embed = pos_embed[:, :1]
    patch_embed = pos_embed[:, 1:]
    target_patches = target_shape[1] - 1
    old_hw = int(math.sqrt(patch_embed.shape[1]))
    new_hw = int(math.sqrt(target_patches))
    if old_hw * old_hw != patch_embed.shape[1] or new_hw * new_hw != target_patches:
        raise ValueError(f"Non-square pos_embed resize: {tuple(pos_embed.shape)} -> {tuple(target_shape)}")
    patch_embed = patch_embed.reshape(1, old_hw, old_hw, pos_embed.shape[-1]).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(new_hw, new_hw), mode="bilinear")
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, pos_embed.shape[-1])
    return torch.cat([cls_embed, patch_embed], dim=1)


def _extract_aggregator_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    """Return an Aggregator-compatible state_dict from common VGGT checkpoint layouts."""
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported VGGT checkpoint type: {type(checkpoint)}")

    state_dict = {}
    for key, value in checkpoint.items():
        norm_key = key
        changed = True
        while changed:
            old_key = norm_key
            for prefix in ("module.", "_orig_mod."):
                if norm_key.startswith(prefix):
                    norm_key = norm_key[len(prefix):]
            changed = norm_key != old_key
        state_dict[norm_key] = value

    aggregator_prefix = "aggregator."
    aggregator_keys = [k for k in state_dict if k.startswith(aggregator_prefix)]
    if aggregator_keys:
        return {
            k[len(aggregator_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(aggregator_prefix)
        }
    return dict(state_dict)


class VGGTImageRAE(nn.Module):
    """Frozen official VGGT image-view encoder plus RAEv2 GeneralDecoder.

    The model keeps RAEv2's pixel convention: the decoder output is directly
    supervised in [0, 1] image space. VGGT input images are also [0, 1]; the
    official VGGT Aggregator performs its own ImageNet mean/std normalization.
    """

    def __init__(
        self,
        vggt_repo_path: str = "/home/zs3325/code/vggt-official",
        vggt_ckpt_path: str = "/home/zs3325/models/VGGT-1B/model.pt",
        encoder_resolution: int = 448,
        output_resolution: int = 512,
        views_per_scene: int = 4,
        mls_layers: Sequence[int] = (4, 11, 17, 23),
        vggt_embed_dim: int = 1024,
        decoder_latent_dim: int = 768,
        decoder_config_path: str = "configs/decoder/ViTXL",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: str | None = None,
        noise_tau: float = 0.8,
    ):
        super().__init__()
        _prepend_repo_to_path(vggt_repo_path)
        from vggt.models.aggregator import Aggregator

        self.vggt_repo_path = vggt_repo_path
        self.vggt_ckpt_path = vggt_ckpt_path
        self.encoder_resolution = int(encoder_resolution)
        self.output_resolution = int(output_resolution)
        self.views_per_scene = int(views_per_scene)
        self.mls_layers = tuple(int(i) for i in mls_layers)
        self.vggt_embed_dim = int(vggt_embed_dim)
        self.encoder_hidden_size = 2 * self.vggt_embed_dim
        self.latent_dim = int(decoder_latent_dim)
        self.decoder_patch_size = int(decoder_patch_size)
        self.noise_tau = float(noise_tau)
        self.base_patches = (self.output_resolution // self.decoder_patch_size) ** 2
        self.patch_size = 14
        self.hidden_size = self.latent_dim

        self.encoder = Aggregator(
            img_size=self.encoder_resolution,
            patch_size=self.patch_size,
            embed_dim=self.vggt_embed_dim,
            cached_layer_indices=self.mls_layers,
        )
        self._load_vggt_encoder(vggt_ckpt_path)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.projection = nn.Linear(self.encoder_hidden_size, self.latent_dim)
        self.decoder: GeneralDecoder = _load_decoder(
            decoder_config_path,
            self.latent_dim,
            self.decoder_patch_size,
            self.base_patches,
            pretrained_decoder_path,
        )
        print(
            "VGGTImageRAE: "
            f"views={self.views_per_scene}, encoder_res={self.encoder_resolution}, "
            f"output_res={self.output_resolution}, mls_layers={self.mls_layers}, "
            f"vggt_dim={self.encoder_hidden_size}, latent_dim={self.latent_dim}"
        )

    def _load_vggt_encoder(self, ckpt_path: str) -> None:
        if not ckpt_path:
            raise ValueError("vggt_ckpt_path must be provided")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_aggregator_state_dict(checkpoint)
        patch_embed_weight_key = "patch_embed.patch_embed.proj.weight"
        if patch_embed_weight_key in state_dict:
            patch_embed_weight = state_dict[patch_embed_weight_key]
            if patch_embed_weight.shape[-1] != self.patch_size:
                new_patch_embed_weight = F.interpolate(
                    patch_embed_weight,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                )
                state_dict[patch_embed_weight_key] = new_patch_embed_weight
                print(
                    "VGGTImageRAE: resized official VGGT patch_embed.patch_embed.proj.weight "
                    f"from {tuple(patch_embed_weight.shape)} to {tuple(new_patch_embed_weight.shape)}"
                )
        current_state = self.encoder.state_dict()
        for key, value in list(state_dict.items()):
            if key in current_state and tuple(value.shape) != tuple(current_state[key].shape):
                if key == "patch_embed.pos_embed":
                    state_dict[key] = _resize_abs_pos_embed(value, current_state[key].shape)
                    print(
                        "VGGTImageRAE: resized official VGGT patch_embed.pos_embed "
                        f"from {tuple(value.shape)} to {tuple(state_dict[key].shape)}"
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected official VGGT Aggregator shape mismatch for {key}: "
                        f"checkpoint {tuple(value.shape)} vs model {tuple(current_state[key].shape)}"
                    )
        keys = self.encoder.load_state_dict(state_dict, strict=False)
        if keys.missing_keys:
            raise RuntimeError(
                "Missing keys when loading official VGGT Aggregator: "
                + ", ".join(keys.missing_keys[:20])
            )
        if keys.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys when loading official VGGT Aggregator: "
                + ", ".join(keys.unexpected_keys[:20])
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def configure_stage1_training(self) -> None:
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.projection.train()
        self.decoder.train()
        self.projection.requires_grad_(True)
        self.decoder.requires_grad_(True)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.projection.parameters()
        yield from self.decoder.parameters()

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for key, value in self.projection.state_dict().items():
            state[f"projection.{key}"] = value
        for key, value in self.decoder.state_dict().items():
            state[f"decoder.{key}"] = value
        return state

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        projection_state = {
            k[len("projection."):]: v
            for k, v in state_dict.items()
            if k.startswith("projection.")
        }
        decoder_state = {
            k[len("decoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
        if not projection_state and not decoder_state:
            # Fallback for a full model state_dict.
            projection_state = {
                k[len("projection."):]: v
                for k, v in state_dict.items()
                if k.startswith("projection.")
            }
            decoder_state = {
                k[len("decoder."):]: v
                for k, v in state_dict.items()
                if k.startswith("decoder.")
            }
        proj_keys = self.projection.load_state_dict(projection_state, strict=strict)
        dec_keys = self.decoder.load_state_dict(decoder_state, strict=strict)
        return proj_keys, dec_keys

    def make_ema_model(self) -> "VGGTImageRAEEMA":
        return VGGTImageRAEEMA(self)

    @torch.no_grad()
    def update_ema_model(self, ema_model: "VGGTImageRAEEMA", decay: float) -> None:
        for ema_param, param in zip(ema_model.projection.parameters(), self.projection.parameters()):
            ema_param.mul_(decay).add_(param.data, alpha=1 - decay)
        for ema_param, param in zip(ema_model.decoder.parameters(), self.decoder.parameters()):
            ema_param.mul_(decay).add_(param.data, alpha=1 - decay)

    def checkpoint_metadata(self) -> dict:
        return {
            "type": "VGGTImageRAE",
            "vggt_repo_path": self.vggt_repo_path,
            "vggt_ckpt_path": self.vggt_ckpt_path,
            "encoder_resolution": self.encoder_resolution,
            "output_resolution": self.output_resolution,
            "views_per_scene": self.views_per_scene,
            "mls_layers": list(self.mls_layers),
            "pixel_convention": "decoder_outputs_0_1_pixels",
        }

    def _check_input(self, images: torch.Tensor) -> None:
        if images.ndim != 5:
            raise ValueError(f"VGGTImageRAE expects [B,V,3,H,W], got {tuple(images.shape)}")
        if images.shape[1] != self.views_per_scene:
            raise ValueError(
                f"Expected V={self.views_per_scene} views per scene, got {images.shape[1]}"
            )
        if images.shape[2] != 3:
            raise ValueError(f"Expected RGB images, got {images.shape[2]} channels")

    def _resize_for_encoder(self, images: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = images.shape
        if h == self.encoder_resolution and w == self.encoder_resolution:
            return images
        flat = images.reshape(b * v, c, h, w)
        flat = F.interpolate(
            flat,
            size=(self.encoder_resolution, self.encoder_resolution),
            mode="bicubic",
            align_corners=False,
        )
        return flat.reshape(b, v, c, self.encoder_resolution, self.encoder_resolution)

    @torch.no_grad()
    def encode_multiview(self, images: torch.Tensor) -> torch.Tensor:
        self._check_input(images)
        images = self._resize_for_encoder(images)
        aggregated_tokens_list, patch_start_idx = self.encoder(images)

        selected = []
        for layer_idx in self.mls_layers:
            layer_tokens = aggregated_tokens_list[layer_idx]
            if layer_tokens is None:
                raise RuntimeError(f"VGGT layer {layer_idx} was not cached by the official Aggregator.")
            selected.append(layer_tokens[:, :, patch_start_idx:, :])

        stacked = torch.stack(selected, dim=0)
        mls = stacked.mean(dim=0)
        final_mean = stacked[-1].mean(dim=2, keepdim=True)
        return mls + final_mean

    def noising(self, z: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((z.size(0), 1, 1), device=z.device, dtype=z.dtype)
        return z + noise_sigma * torch.randn_like(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        output = self.decoder(z, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)

    def forward(self, images: torch.Tensor, return_latent: bool = False):
        features = self.encode_multiview(images)
        b, v, n, c = features.shape
        z = features.reshape(b * v, n, c)
        z = self.projection(z)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon


class VGGTImageRAEEMA(nn.Module):
    """EMA copy that shares the frozen VGGT encoder with the training model."""

    def __init__(self, source: VGGTImageRAE):
        super().__init__()
        self.__dict__["encoder"] = source.encoder
        self.vggt_repo_path = source.vggt_repo_path
        self.vggt_ckpt_path = source.vggt_ckpt_path
        self.encoder_resolution = source.encoder_resolution
        self.output_resolution = source.output_resolution
        self.views_per_scene = source.views_per_scene
        self.mls_layers = source.mls_layers
        self.vggt_embed_dim = source.vggt_embed_dim
        self.encoder_hidden_size = source.encoder_hidden_size
        self.latent_dim = source.latent_dim
        self.decoder_patch_size = source.decoder_patch_size
        self.base_patches = source.base_patches
        self.patch_size = source.patch_size
        self.hidden_size = source.hidden_size
        self.noise_tau = 0.0
        self.projection = copy.deepcopy(source.projection)
        self.decoder = copy.deepcopy(source.decoder)
        self.requires_grad_(False)
        self.eval()

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for key, value in self.projection.state_dict().items():
            state[f"projection.{key}"] = value
        for key, value in self.decoder.state_dict().items():
            state[f"decoder.{key}"] = value
        return state

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        projection_state = {
            k[len("projection."):]: v
            for k, v in state_dict.items()
            if k.startswith("projection.")
        }
        decoder_state = {
            k[len("decoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
        proj_keys = self.projection.load_state_dict(projection_state, strict=strict)
        dec_keys = self.decoder.load_state_dict(decoder_state, strict=strict)
        return proj_keys, dec_keys

    def _resize_for_encoder(self, images: torch.Tensor) -> torch.Tensor:
        return VGGTImageRAE._resize_for_encoder(self, images)

    def _check_input(self, images: torch.Tensor) -> None:
        return VGGTImageRAE._check_input(self, images)

    @torch.no_grad()
    def encode_multiview(self, images: torch.Tensor) -> torch.Tensor:
        return VGGTImageRAE.encode_multiview(self, images)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        output = self.decoder(z, drop_cls_token=False).logits
        return self.decoder.unpatchify(output)

    def forward(self, images: torch.Tensor, return_latent: bool = False):
        features = self.encode_multiview(images)
        b, v, n, c = features.shape
        z = self.projection(features.reshape(b * v, n, c))
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon
