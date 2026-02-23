#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import scipy.io as sio
except Exception:
    sio = None


def _collect_files(folder: Path, ext: str) -> Dict[str, Path]:
    return {p.stem: p for p in folder.glob(f"*.{ext}")}


def _load_split_ids(split: str, gt_dir: Path, dataset_root: Path | None) -> set[str]:
    if split == "all":
        return set()

    root = dataset_root if dataset_root is not None else gt_dir.parent
    split_file = root / "gt_sets" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Please set --dataset-root correctly."
        )

    ids = set()
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if line:
            ids.add(line)
    return ids


def _load_depth(path: Path, pred_ext: str, pred_key: str) -> np.ndarray:
    if pred_ext == "npy":
        arr = np.load(path)
    elif pred_ext == "mat":
        if sio is None:
            raise RuntimeError("scipy is required to load .mat predictions")
        mat = sio.loadmat(path)
        if pred_key not in mat:
            raise KeyError(f"Key '{pred_key}' not found in {path}")
        arr = mat[pred_key]
    else:
        raise ValueError(f"Unsupported extension: {pred_ext}")
    return np.asarray(arr, dtype=np.float64)


def _valid_mask(gt: np.ndarray) -> np.ndarray:
    return np.isfinite(gt) & (gt != 0) & (gt != 255)


def _resize_to_shape(arr: np.ndarray, out_hw: Tuple[int, int], mode: str) -> np.ndarray:
    h, w = out_hw
    if arr.shape == (h, w):
        return arr

    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor, size=(h, w), mode="nearest")
    else:
        resized = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy()


def _metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    m = _valid_mask(gt) & np.isfinite(pred)
    if m.sum() == 0:
        return {"n_valid": 0}

    p = np.clip(pred[m], eps, None)
    g = np.clip(gt[m], eps, None)
    sq = (g - p) ** 2

    return {
        "n_valid": int(m.sum()),
        "rmse": float(np.sqrt(np.mean(sq))),
        "log_rmse": float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2))),
        "log10_rmse": float(np.sqrt(np.mean((np.log10(g) - np.log10(p)) ** 2))),
        "wrong_log_style": float(np.sqrt(np.mean(np.log((g - p) ** 2 + eps)))),
        "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)),
        "gt_min": float(np.min(g)),
        "gt_max": float(np.max(g)),
        "scale_ratio_median_pred_over_gt": float(np.median(p / g)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def _sample_vs_pixel_rmse(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    all_sq_sum, all_n = 0.0, 0
    per_sample = []
    for pred, gt in pairs:
        m = _valid_mask(gt) & np.isfinite(pred)
        if m.sum() == 0:
            continue
        sq = (gt[m] - pred[m]) ** 2
        all_sq_sum += float(np.sum(sq))
        all_n += int(m.sum())
        per_sample.append(math.sqrt(float(np.mean(sq))))
    return math.sqrt(all_sq_sum / max(all_n, 1)), float(np.mean(per_sample)) if per_sample else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--pred-ext", choices=["npy", "mat"], default="npy")
    ap.add_argument("--pred-key", default="depth")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--gt-scale", type=float, default=1.0,
                    help="Multiply GT depth by this factor before evaluation (e.g., 0.001 for mm->m)")
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                    help="Only audit IDs from this NYUD split (read from <dataset_root>/gt_sets/<split>.txt)")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="Dataset root containing gt_sets/. Defaults to parent of --gt-dir")
    ap.add_argument("--resize-gt-to-pred", action="store_true",
                    help="Resize GT depth to prediction spatial size before metric computation")
    ap.add_argument("--resize-mode", choices=["nearest", "bilinear"], default="bilinear",
                    help="Interpolation mode used when --resize-gt-to-pred is enabled")
    args = ap.parse_args()

    gt_files = _collect_files(args.gt_dir, "npy")
    pred_files = _collect_files(args.pred_dir, args.pred_ext)

    split_ids = _load_split_ids(args.split, args.gt_dir, args.dataset_root)
    if split_ids:
        gt_files = {k: v for k, v in gt_files.items() if k in split_ids}
        pred_files = {k: v for k, v in pred_files.items() if k in split_ids}

    gt_ids, pred_ids = set(gt_files), set(pred_files)
    common = sorted(gt_ids & pred_ids)
    missing_pred = sorted(gt_ids - pred_ids)
    extra_pred = sorted(pred_ids - gt_ids)
    if args.max_samples > 0:
        common = common[: args.max_samples]

    print("=== Pairing / split check ===")
    print(f"GT files: {len(gt_ids)}")
    print(f"Pred files: {len(pred_ids)}")
    print(f"Paired files: {len(common)}")
    print(f"Missing preds: {len(missing_pred)}")
    print(f"Extra preds: {len(extra_pred)}")
    if missing_pred:
        print("First missing preds:", missing_pred[:10])
    if extra_pred:
        print("First extra preds:", extra_pred[:10])
    if not common:
        return

    pairs, agg_pred, agg_gt = [], [], []
    shape_mismatch = 0
    invalid_pred_pixels = 0
    total_pred_pixels = 0

    for sid in common:
        gt = np.load(gt_files[sid]).astype(np.float64)
        if args.gt_scale != 1.0:
            gt = gt * args.gt_scale
        pred = _load_depth(pred_files[sid], args.pred_ext, args.pred_key)
        if gt.shape != pred.shape:
            if args.resize_gt_to_pred:
                gt = _resize_to_shape(gt, pred.shape, mode=args.resize_mode).astype(np.float64)
            else:
                shape_mismatch += 1
                continue
        pairs.append((pred, gt))
        m = _valid_mask(gt) & np.isfinite(pred)
        if m.any():
            agg_pred.append(pred[m])
            agg_gt.append(gt[m])
        total_pred_pixels += pred.size
        invalid_pred_pixels += int((~np.isfinite(pred)).sum())

    if not pairs:
        print("No shape-aligned pairs to evaluate.")
        return

    agg_pred_v = np.concatenate(agg_pred)
    agg_gt_v = np.concatenate(agg_gt)
    summary = _metrics(agg_pred_v, agg_gt_v)
    pixel_rmse, sample_rmse = _sample_vs_pixel_rmse(pairs)

    print("\n=== Metric definition check ===")
    print(f"RMSE (pixel-weighted): {summary['rmse']:.6f}")
    print(f"RMSE (sample-averaged): {sample_rmse:.6f}")
    print(f"Difference (sample - pixel): {sample_rmse - pixel_rmse:.6f}")
    print(f"log_RMSE (natural log): {summary['log_rmse']:.6f}")
    print(f"log10_RMSE: {summary['log10_rmse']:.6f}")
    print(f"Wrong-style log metric: {summary['wrong_log_style']:.6f}")

    print("\n=== Value / scale sanity ===")
    print(f"GT range (valid): [{summary['gt_min']:.6f}, {summary['gt_max']:.6f}]")
    print(f"Pred range (valid): [{summary['pred_min']:.6f}, {summary['pred_max']:.6f}]")
    print(f"Median(pred/gt): {summary['scale_ratio_median_pred_over_gt']:.6f}")
    print(f"Non-finite pred ratio: {invalid_pred_pixels / max(total_pred_pixels,1):.6%}")
    print(f"Shape mismatches: {shape_mismatch}")

    corr_depth = _pearson(agg_pred_v, agg_gt_v)
    corr_inv = _pearson(agg_pred_v, 1.0 / np.clip(agg_gt_v, 1e-9, None))
    print("\n=== Depth vs inverse-depth hypothesis ===")
    print(f"corr(pred, depth): {corr_depth:.6f}")
    print(f"corr(pred, 1/depth): {corr_inv:.6f}")
    if np.isfinite(corr_depth) and np.isfinite(corr_inv) and corr_inv > corr_depth + 0.1:
        print("[WARN] prediction seems closer to inverse-depth/disparity.")
    if summary["pred_max"] <= 1.0 and summary["gt_max"] > 2.0:
        print("[WARN] pred appears clamped to [0,1] while gt range is much larger.")


if __name__ == "__main__":
    main()
