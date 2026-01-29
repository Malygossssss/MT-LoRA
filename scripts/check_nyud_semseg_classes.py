#!/usr/bin/env python3
"""Verify unique semantic labels in an NYUDv2-style dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


IGNORE_VALUES = {-1, 255}


def iter_label_files(root: Path, split: str | None) -> Iterable[Path]:
    if split:
        split_dir = root / split / "label"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {split_dir}")
        yield from sorted(split_dir.glob("*"))
        return

    # Try to infer split folders
    candidates = []
    for candidate in ("train", "val", "test"):
        label_dir = root / candidate / "label"
        if label_dir.is_dir():
            candidates.append(label_dir)

    if candidates:
        for label_dir in candidates:
            yield from sorted(label_dir.glob("*"))
        return

    # Legacy layout fallback
    legacy_dir = root / "segmentation"
    if legacy_dir.is_dir():
        yield from sorted(legacy_dir.glob("*"))
        return

    raise FileNotFoundError(
        "Could not find label directories. Expected <root>/<split>/label or <root>/segmentation."
    )


def load_label(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.squeeze(arr, axis=2)
        if arr.ndim != 2:
            raise ValueError(f"Expected HxW label array in {path}, got shape {arr.shape}")
        return arr
    with Image.open(path) as img:
        return np.array(img)


def normalize_labels(arr: np.ndarray) -> np.ndarray:
    """Match NYUD_MT._load_semseg normalization: ignore 0 and ignore values, then -1."""
    arr = arr.astype(float)
    ignore_mask = (arr < 0) | (arr == 255)
    arr[ignore_mask] = 256
    arr[arr == 0] = 256
    return arr - 1


def collect_unique(labels: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    raw_values = set()
    normalized_values = set()
    for path in labels:
        if not path.is_file():
            continue
        arr = load_label(path)
        raw_values.update(np.unique(arr).tolist())
        norm = normalize_labels(arr)
        normalized_values.update(np.unique(norm).tolist())
    return np.array(sorted(raw_values)), np.array(sorted(normalized_values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Count unique NYUDv2 semantic labels")
    parser.add_argument("--root", required=True, type=Path, help="NYUDv2 dataset root")
    parser.add_argument("--split", default=None, help="Optional split (train/val/test)")
    args = parser.parse_args()

    label_files = list(iter_label_files(args.root, args.split))
    if not label_files:
        raise FileNotFoundError("No label files found in the specified directory.")

    raw_values, normalized_values = collect_unique(label_files)

    raw_ignore_filtered = [v for v in raw_values if v not in IGNORE_VALUES]
    norm_ignore_filtered = [v for v in normalized_values if v not in (-1, 255)]

    print(f"Found {len(label_files)} label files")
    print(f"Raw unique values: {raw_values}")
    print(f"Raw class count (excluding {sorted(IGNORE_VALUES)}): {len(raw_ignore_filtered)}")
    print("---")
    print(f"Normalized unique values: {normalized_values}")
    print("Normalized class count (excluding ignore): "
          f"{len(norm_ignore_filtered)}")


if __name__ == "__main__":
    main()