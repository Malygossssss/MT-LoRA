#!/usr/bin/env python3
"""Diagnose NYUDv2 semantic label issues for 13-class setups."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from evaluation.eval_semseg import NYU_CATEGORY_NAMES
except Exception:  # noqa: BLE001
    NYU_CATEGORY_NAMES = []


def iter_label_files(root: Path, split: str | None) -> Iterable[Path]:
    if split:
        split_dir = root / split / "label"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {split_dir}")
        yield from sorted(split_dir.glob("*"))
        return

    candidates = []
    for candidate in ("train", "val", "test"):
        label_dir = root / candidate / "label"
        if label_dir.is_dir():
            candidates.append(label_dir)

    if candidates:
        for label_dir in candidates:
            yield from sorted(label_dir.glob("*"))
        return

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check NYUDv2 semseg labels for class range, ignore usage, and mapping issues."
    )
    parser.add_argument("--root", required=True, type=Path, help="NYUDv2 dataset root")
    parser.add_argument("--split", default=None, help="Optional split (train/val/test)")
    parser.add_argument("--n-classes", type=int, default=13, help="Expected number of classes")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignore label value")
    parser.add_argument("--sample", type=int, default=200, help="Number of files to sample")
    args = parser.parse_args()

    label_files = [p for p in iter_label_files(args.root, args.split) if p.is_file()]
    if not label_files:
        raise FileNotFoundError("No label files found in the specified directory.")

    sample_files = label_files[: args.sample]
    n_classes = args.n_classes

    total_pixels = 0
    ignore_pixels = 0
    invalid_pixels = 0
    label_counts = np.zeros(n_classes, dtype=np.int64)
    invalid_values = set()
    min_label = None
    max_label = None

    for path in sample_files:
        labels = load_label(path).astype(int)
        flat = labels.reshape(-1)

        ignore_mask = flat == args.ignore_index
        valid_mask = (flat >= 0) & (flat < n_classes)
        invalid_mask = ~(valid_mask | ignore_mask)

        total_pixels += flat.size
        ignore_pixels += int(np.sum(ignore_mask))
        invalid_pixels += int(np.sum(invalid_mask))

        valid_vals = flat[valid_mask]
        if valid_vals.size:
            cur_min = int(valid_vals.min())
            cur_max = int(valid_vals.max())
            min_label = cur_min if min_label is None else min(min_label, cur_min)
            max_label = cur_max if max_label is None else max(max_label, cur_max)
            label_counts += np.bincount(valid_vals, minlength=n_classes)

        if np.any(invalid_mask):
            invalid_values.update(np.unique(flat[invalid_mask]).tolist())

    ignore_frac = ignore_pixels / max(total_pixels, 1)
    invalid_frac = invalid_pixels / max(total_pixels, 1)

    print("=== NYUD Semseg Diagnostics ===")
    print(f"Scanned {len(sample_files)} / {len(label_files)} label files")
    print(f"Expected classes: {n_classes} (valid labels 0..{n_classes - 1})")
    print(f"Valid min/max: {min_label} / {max_label}")
    print(f"Ignore pixels: {ignore_pixels} ({ignore_frac:.4%})")
    print(f"Invalid pixels: {invalid_pixels} ({invalid_frac:.4%})")
    if invalid_values:
        preview = sorted(invalid_values)
        preview_str = preview[:20]
        suffix = "..." if len(preview) > 20 else ""
        print(f"Invalid label values (preview): {preview_str}{suffix}")

    if total_pixels > 0:
        label_freq = label_counts / max(label_counts.sum(), 1)
        top_idx = int(np.argmax(label_freq))
        print(f"Most frequent label: {top_idx} ({label_freq[top_idx]:.2%})")
        if label_freq[0] > 0.5:
            print("Warning: label 0 dominates (>50%). It may be background and should be ignored.")

    if NYU_CATEGORY_NAMES:
        max_name_count = min(n_classes, len(NYU_CATEGORY_NAMES))
        print("Class name preview (NYU list):")
        for idx, name in enumerate(NYU_CATEGORY_NAMES[:max_name_count]):
            print(f"  {idx}: {name}")
        if n_classes > len(NYU_CATEGORY_NAMES):
            print("Warning: expected classes exceed NYU category list length.")
    else:
        print("Note: NYU_CATEGORY_NAMES could not be loaded; mapping preview skipped.")


if __name__ == "__main__":
    main()