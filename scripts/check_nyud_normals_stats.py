#!/usr/bin/env python3
"""Inspect NYUDv2 normals labels for encoding/range/normalization issues."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


IGNORE_VALUE = 255


def iter_normal_files(root: Path, split: str | None) -> Iterable[Path]:
    if split:
        split_dir = root / split / "normal"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing normals directory: {split_dir}")
        yield from sorted(split_dir.glob("*"))
        return

    candidates = []
    for candidate in ("train", "val", "test"):
        normal_dir = root / candidate / "normal"
        if normal_dir.is_dir():
            candidates.append(normal_dir)

    if candidates:
        for normal_dir in candidates:
            yield from sorted(normal_dir.glob("*"))
        return

    legacy_dir = root / "normals"
    if legacy_dir.is_dir():
        yield from sorted(legacy_dir.glob("*"))
        return

    raise FileNotFoundError(
        "Could not find normals directories. Expected <root>/<split>/normal or <root>/normals."
    )


def load_normal(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(float)
    with Image.open(path) as img:
        return np.array(img).astype(float)


def compute_stats(arr: np.ndarray) -> Tuple[float, float, float, float, float]:
    flat = arr.reshape(-1)
    return float(np.min(flat)), float(np.max(flat)), float(np.mean(flat)), float(np.std(flat)), float(arr.dtype == np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check NYUDv2 normals encoding and normalization")
    parser.add_argument("--root", required=True, type=Path, help="NYUDv2 dataset root")
    parser.add_argument("--split", default=None, help="Optional split (train/val/test)")
    parser.add_argument("--sample", type=int, default=200, help="Number of files to sample")
    args = parser.parse_args()

    normal_files = [p for p in iter_normal_files(args.root, args.split) if p.is_file()]
    if not normal_files:
        raise FileNotFoundError("No normal files found in the specified directory.")

    sample_files = normal_files[: args.sample]

    mins, maxs, means, stds = [], [], [], []
    unit_norm_ratio = []
    ignore_ratio = []
    enc_0_255 = 0
    enc_minus1_1 = 0

    for path in sample_files:
        arr = load_normal(path)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 normals in {path}, got {arr.shape}")

        min_v, max_v, mean_v, std_v, _ = compute_stats(arr)
        mins.append(min_v)
        maxs.append(max_v)
        means.append(mean_v)
        stds.append(std_v)

        if min_v >= 0 and max_v <= 255:
            enc_0_255 += 1
        if min_v >= -1.1 and max_v <= 1.1:
            enc_minus1_1 += 1

        ignore_mask = np.all(arr == IGNORE_VALUE, axis=2)
        ignore_ratio.append(float(np.mean(ignore_mask)))

        norms = np.linalg.norm(arr, axis=2)
        unit_norm_ratio.append(float(np.mean((norms > 0.9) & (norms < 1.1))))

    print(f"Scanned {len(sample_files)} / {len(normal_files)} normal files")
    print(f"Value range: min={min(mins):.4f}, max={max(maxs):.4f}")
    print(f"Mean value: {np.mean(means):.4f} ± {np.mean(stds):.4f}")
    print(f"Files within [0,255] range: {enc_0_255}/{len(sample_files)}")
    print(f"Files within [-1,1] range: {enc_minus1_1}/{len(sample_files)}")
    print(f"Ignore ratio (all channels == 255): {np.mean(ignore_ratio):.4f}")
    print(f"Unit-norm ratio (0.9-1.1): {np.mean(unit_norm_ratio):.4f}")


if __name__ == "__main__":
    main()
