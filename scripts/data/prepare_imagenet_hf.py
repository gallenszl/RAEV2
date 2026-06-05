#!/usr/bin/env python3
"""Download ILSVRC/imagenet-1k parquet shards and convert to ImageFolder.

The Hugging Face token must be supplied via the HF_TOKEN environment variable.
The token is never printed or written by this script.
"""

from __future__ import annotations

import argparse
import io
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image

TRAIN_COUNT = 1_281_167
VAL_COUNT = 50_000
NUM_CLASSES = 1000
SPLIT_TO_PREFIX = {"train": "train", "val": "validation"}
SPLIT_TO_DIR = {"train": "train", "val": "val"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1k ImageFolder data from Hugging Face parquet shards.")
    parser.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--raw-dir", type=Path, default=Path("/scratch/zs3325/datasets/imagenet_hf_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("/scratch/zs3325/datasets/imagenet"))
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--sample-checks", type=int, default=1000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment. It will not be printed or written.")
    return token


def download_shards(repo_id: str, raw_dir: Path, splits: Iterable[str]) -> None:
    from huggingface_hub import snapshot_download

    token = require_hf_token()
    allow_patterns = []
    for split in splits:
        prefix = SPLIT_TO_PREFIX[split]
        allow_patterns.append(f"data/{prefix}-*.parquet")

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=str(raw_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=["*.lock", "README.md", ".gitattributes"],
    )


def list_shards(raw_dir: Path, split: str) -> list[Path]:
    prefix = SPLIT_TO_PREFIX[split]
    shards = sorted((raw_dir / "data").glob(f"{prefix}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split} parquet shards found under {raw_dir / 'data'}")
    return shards


def shard_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def build_offsets(shards: list[Path]) -> list[int]:
    offsets = []
    total = 0
    for shard in shards:
        offsets.append(total)
        total += shard_num_rows(shard)
    return offsets


def image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def image_bytes(value) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is None:
            raise ValueError("Image parquet cell has no bytes field")
        return data
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"Unsupported image cell type: {type(value)!r}")


def write_jpeg(data: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp_path.write_bytes(data)
        if not image_is_readable(tmp_path):
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                image.save(tmp_path, format="JPEG", quality=95)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def process_shard(args_tuple) -> tuple[str, str, int, int, int]:
    import pyarrow.parquet as pq

    split, shard_path, output_dir, base_index = args_tuple
    shard_path = Path(shard_path)
    output_dir = Path(output_dir)
    split_dir = SPLIT_TO_DIR[split]
    written = 0
    skipped = 0
    failed = 0
    row_base = 0

    parquet = pq.ParquetFile(shard_path)
    for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
        images = batch.column(0).to_pylist()
        labels = batch.column(1).to_pylist()
        for row_idx, (img_value, label_value) in enumerate(zip(images, labels)):
            global_index = base_index + row_base + row_idx
            label = int(label_value)
            if label < 0 or label >= NUM_CLASSES:
                raise ValueError(f"Label out of range in {shard_path}: {label}")
            out_path = output_dir / split_dir / f"{label:04d}" / f"{split_dir}-{global_index:08d}.JPEG"
            if out_path.exists() and image_is_readable(out_path):
                skipped += 1
                continue
            try:
                write_jpeg(image_bytes(img_value), out_path)
                written += 1
            except Exception as exc:  # Keep the long job moving, but report failures.
                failed += 1
                print(f"FAILED {shard_path} row={row_base + row_idx} label={label}: {exc}", flush=True)
        row_base += len(images)

    return split, shard_path.name, written, skipped, failed


def convert_split(raw_dir: Path, output_dir: Path, split: str, num_workers: int) -> None:
    shards = list_shards(raw_dir, split)
    offsets = build_offsets(shards)
    tasks = [(split, str(shard), str(output_dir), offset) for shard, offset in zip(shards, offsets)]
    print(f"Converting {split}: {len(shards)} shards with {num_workers} workers", flush=True)
    totals = {"written": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_shard, task) for task in tasks]
        for future in as_completed(futures):
            split_name, shard_name, written, skipped, failed = future.result()
            totals["written"] += written
            totals["skipped"] += skipped
            totals["failed"] += failed
            print(
                f"{split_name} {shard_name}: written={written} skipped={skipped} failed={failed} "
                f"totals={totals}",
                flush=True,
            )
    if totals["failed"]:
        raise RuntimeError(f"{split} conversion had {totals['failed']} failed images")


def count_split(output_dir: Path, split: str) -> tuple[int, int, list[Path]]:
    split_dir = output_dir / SPLIT_TO_DIR[split]
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir()) if split_dir.exists() else []
    files = []
    total = 0
    for class_dir in class_dirs:
        class_files = list(class_dir.glob("*.JPEG"))
        total += len(class_files)
        files.extend(class_files[:2])
    return len(class_dirs), total, files


def validate(output_dir: Path, splits: Iterable[str], sample_checks: int) -> None:
    expected = {"train": TRAIN_COUNT, "val": VAL_COUNT}
    rng = random.Random(0)
    for split in splits:
        split_dir = output_dir / SPLIT_TO_DIR[split]
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if len(class_dirs) != NUM_CLASSES:
            raise RuntimeError(f"{split}: expected {NUM_CLASSES} class dirs, found {len(class_dirs)}")
        total = 0
        candidates = []
        for class_dir in class_dirs:
            files = list(class_dir.glob("*.JPEG"))
            total += len(files)
            if files:
                candidates.append(rng.choice(files))
        if total != expected[split]:
            raise RuntimeError(f"{split}: expected {expected[split]} images, found {total}")
        checks = min(sample_checks, len(candidates))
        for image_path in rng.sample(candidates, checks):
            if not image_is_readable(image_path):
                raise RuntimeError(f"Unreadable image: {image_path}")
        print(f"Validated {split}: classes={len(class_dirs)} images={total} sample_checks={checks}", flush=True)


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.validate_only and not args.skip_download:
        download_shards(args.repo_id, args.raw_dir, args.splits)
    if not args.validate_only and not args.skip_convert:
        for split in args.splits:
            convert_split(args.raw_dir, args.output_dir, split, args.num_workers)
    validate(args.output_dir, args.splits, args.sample_checks)


if __name__ == "__main__":
    main()
