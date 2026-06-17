"""Scene-level multiview datasets for VGGT-backed RAE stage-1 training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _load_rgb_white_bg(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    return image.convert("RGB")


def _build_square_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


class ObjaverseMultiviewDataset(Dataset):
    """Return one scene with a fixed number of randomly sampled RGB views."""

    def __init__(
        self,
        root: str,
        list_path: str,
        image_size: int = 512,
        views_per_scene: int = 4,
        total_views: int = 25,
        seed: int = 20260616,
        transform=None,
    ):
        self.root = Path(root)
        self.list_path = Path(list_path)
        self.image_size = int(image_size)
        self.views_per_scene = int(views_per_scene)
        self.total_views = int(total_views)
        self.seed = int(seed)
        self.transform = transform or _build_square_transform(self.image_size)
        self._epoch = 0

        if self.views_per_scene < 1:
            raise ValueError("views_per_scene must be >= 1")
        if self.views_per_scene > self.total_views:
            raise ValueError("views_per_scene cannot exceed total_views")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Multiview root not found: {self.root}")
        if not self.list_path.is_file():
            raise FileNotFoundError(f"Multiview list not found: {self.list_path}")

        with self.list_path.open("r") as f:
            self.scenes = [line.strip() for line in f if line.strip()]
        if not self.scenes:
            raise ValueError(f"No scenes found in {self.list_path}")

    def __len__(self) -> int:
        return len(self.scenes)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _sample_view_ids(self, idx: int) -> list[int]:
        rng = random.Random(self.seed + self._epoch * 1_000_003 + idx)
        return sorted(rng.sample(range(self.total_views), self.views_per_scene))

    def _load_views(self, scene_name: str, view_ids: list[int]) -> torch.Tensor:
        images = []
        scene_dir = self.root / scene_name
        for view_id in view_ids:
            image_path = scene_dir / f"{view_id:03d}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing RGB view: {image_path}")
            image = _load_rgb_white_bg(image_path)
            images.append(self.transform(image))
        return torch.stack(images, dim=0)

    def __getitem__(self, idx: int):
        scene_name = self.scenes[idx]
        view_ids = self._sample_view_ids(idx)
        images = self._load_views(scene_name, view_ids)
        meta = {
            "scene_name": scene_name,
            "scene_idx": idx,
            "view_ids": torch.tensor(view_ids, dtype=torch.long),
        }
        return images, meta


class GSOMultiviewFixedDataset(ObjaverseMultiviewDataset):
    """GSO eval dataset using a committed fixed scene->view list."""

    def __init__(
        self,
        root: str,
        split_file: str,
        fixed_view_list_path: Optional[str] = None,
        image_size: int = 512,
        views_per_scene: int = 4,
        total_views: int = 25,
        seed: int = 0,
        transform=None,
    ):
        super().__init__(
            root=root,
            list_path=split_file,
            image_size=image_size,
            views_per_scene=views_per_scene,
            total_views=total_views,
            seed=seed,
            transform=transform,
        )
        self.fixed_views_by_scene: dict[str, list[int]] = {}
        if fixed_view_list_path:
            path = Path(fixed_view_list_path)
            if not path.is_file():
                raise FileNotFoundError(f"Fixed GSO view list not found: {path}")
            data = json.loads(path.read_text())
            scenes = data.get("scenes", data)
            self.fixed_views_by_scene = {
                str(scene): [int(v) for v in views]
                for scene, views in scenes.items()
            }

    def _sample_view_ids(self, idx: int) -> list[int]:
        scene_name = self.scenes[idx]
        if scene_name in self.fixed_views_by_scene:
            view_ids = self.fixed_views_by_scene[scene_name]
            if len(view_ids) != self.views_per_scene:
                raise ValueError(
                    f"Fixed GSO view list for {scene_name} has {len(view_ids)} views; "
                    f"expected {self.views_per_scene}"
                )
            return view_ids
        # Deterministic fallback keeps eval fixed even if a scene is not listed.
        rng = random.Random(self.seed + idx)
        return sorted(rng.sample(range(self.total_views), self.views_per_scene))
