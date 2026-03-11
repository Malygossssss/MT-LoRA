import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import disk


DEFAULT_THRESHOLDS = np.linspace(1.0, 0.0, 101)
DEFAULT_TOLERANCE_RATIO = 0.0075


def _to_path(path_like):
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _ensure_2d_float(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    return arr.astype(np.float32)


def _ensure_3d_gt(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array, got shape {arr.shape}")
    return arr.astype(np.float32)


def _normalize_prob_map(arr):
    arr = _ensure_2d_float(arr)
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _normalize_gt_stack(arr):
    arr = _ensure_3d_gt(arr)
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = arr / 255.0
    return (arr >= 0.5).astype(np.uint8)


def _prediction_candidates(pred_root, image_id):
    pred_root = _to_path(pred_root)
    direct = [pred_root / f"{image_id}.npy", pred_root / f"{image_id}.png"]
    nested = [pred_root / "edge" / f"{image_id}.npy", pred_root / "edge" / f"{image_id}.png"]
    return nested + direct


def _gt_candidates(gt_root, image_id):
    gt_root = _to_path(gt_root)
    edge_eval = gt_root / "edge_eval"
    edge = gt_root / "edge"
    return {
        "edge_eval": [edge_eval / f"{image_id}.npz", edge_eval / f"{image_id}.npy"],
        "edge": [edge / f"{image_id}.npy", edge / f"{image_id}.png"],
    }


def load_prediction_map(path_like):
    path = _to_path(path_like)
    if not path.is_file():
        raise FileNotFoundError(f"prediction file not found: {path}")
    if path.suffix.lower() == ".npy":
        return _normalize_prob_map(np.load(path))
    if path.suffix.lower() == ".png":
        return _normalize_prob_map(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    raise ValueError(f"unsupported prediction format: {path.suffix}")


def resolve_prediction_path(pred_root, image_id):
    for candidate in _prediction_candidates(pred_root, image_id):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing prediction for {image_id} under {pred_root}")


def load_ground_truth_stack(gt_root, image_id):
    candidates = _gt_candidates(gt_root, image_id)

    for candidate in candidates["edge_eval"]:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".npz":
            data = np.load(candidate)
            if "gts" in data:
                return _normalize_gt_stack(data["gts"])
            first_key = next(iter(data.files))
            return _normalize_gt_stack(data[first_key])
        return _normalize_gt_stack(np.load(candidate))

    for candidate in candidates["edge"]:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".npy":
            return _normalize_gt_stack(np.load(candidate))
        return _normalize_gt_stack(cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED))

    raise FileNotFoundError(f"missing edge ground truth for {image_id} under {gt_root}")


def has_edge_ground_truth(gt_root):
    gt_root = _to_path(gt_root)
    return (gt_root / "edge").is_dir() or (gt_root / "edge_eval").is_dir()


def collect_prediction_ids(pred_root):
    pred_root = _to_path(pred_root)
    edge_root = pred_root / "edge"
    search_roots = [edge_root] if edge_root.is_dir() else [pred_root]

    image_ids = set()
    for root in search_roots:
        for suffix in ("*.npy", "*.png"):
            for path in root.glob(suffix):
                image_ids.add(path.stem)
    return sorted(image_ids)


def _resize_prediction(prob_map, shape):
    if prob_map.shape == shape:
        return prob_map
    return cv2.resize(prob_map, shape[::-1], interpolation=cv2.INTER_LINEAR)


def _make_kernel(shape, tolerance_ratio):
    diag = float(np.hypot(shape[0], shape[1]))
    radius = max(1, int(round(diag * tolerance_ratio)))
    footprint = disk(radius).astype(np.uint8)
    if footprint.ndim != 2 or footprint.size == 0:
        footprint = np.ones((1, 1), dtype=np.uint8)
    return footprint


def _f1(precision, recall):
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def _boundary_stats(pred_mask, gt_mask, kernel):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    pred_count = int(pred_mask.sum())
    gt_count = int(gt_mask.sum())

    if pred_count == 0 and gt_count == 0:
        return {
            "tp_pred": 0,
            "pred_count": 0,
            "tp_gt": 0,
            "gt_count": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    gt_dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_dilated = cv2.dilate(pred_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    tp_pred = int(np.logical_and(pred_mask, gt_dilated).sum())
    tp_gt = int(np.logical_and(gt_mask, pred_dilated).sum())

    precision = tp_pred / pred_count if pred_count > 0 else 0.0
    recall = tp_gt / gt_count if gt_count > 0 else 1.0

    return {
        "tp_pred": tp_pred,
        "pred_count": pred_count,
        "tp_gt": tp_gt,
        "gt_count": gt_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _best_stats_for_threshold(prob_map, gt_stack, threshold, kernel):
    pred_mask = prob_map >= threshold
    best = None
    for gt_mask in gt_stack:
        stats = _boundary_stats(pred_mask, gt_mask, kernel)
        if best is None or stats["f1"] > best["f1"]:
            best = stats
    return best


def evaluate_edge_predictions(predictions, ground_truths, thresholds=None, tolerance_ratio=DEFAULT_TOLERANCE_RATIO):
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    thresholds = [float(t) for t in thresholds]

    image_ids = sorted(predictions.keys())
    if not image_ids:
        raise ValueError("no edge predictions were provided")

    curves = []
    per_image_best = []

    for threshold in thresholds:
        total_tp_pred = 0
        total_pred = 0
        total_tp_gt = 0
        total_gt = 0

        for image_id in image_ids:
            prob_map = predictions[image_id]
            gt_stack = ground_truths[image_id]
            gt_shape = tuple(gt_stack.shape[-2:])
            prob_map = _resize_prediction(prob_map, gt_shape)
            kernel = _make_kernel(gt_shape, tolerance_ratio)
            stats = _best_stats_for_threshold(prob_map, gt_stack, threshold, kernel)
            total_tp_pred += stats["tp_pred"]
            total_pred += stats["pred_count"]
            total_tp_gt += stats["tp_gt"]
            total_gt += stats["gt_count"]

        precision = total_tp_pred / total_pred if total_pred > 0 else 0.0
        recall = total_tp_gt / total_gt if total_gt > 0 else 1.0
        curves.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )

    for image_id in image_ids:
        prob_map = predictions[image_id]
        gt_stack = ground_truths[image_id]
        gt_shape = tuple(gt_stack.shape[-2:])
        prob_map = _resize_prediction(prob_map, gt_shape)
        kernel = _make_kernel(gt_shape, tolerance_ratio)
        best = {"threshold": None, "f1": -1.0}
        for threshold in thresholds:
            stats = _best_stats_for_threshold(prob_map, gt_stack, threshold, kernel)
            if stats["f1"] > best["f1"]:
                best = {"threshold": threshold, "f1": stats["f1"]}
        per_image_best.append(best)

    precisions = np.array([row["precision"] for row in curves], dtype=np.float64)
    recalls = np.array([row["recall"] for row in curves], dtype=np.float64)
    f1_scores = np.array([row["f1"] for row in curves], dtype=np.float64)
    best_idx = int(np.argmax(f1_scores))

    ap_precisions = precisions.copy()
    ap_recalls = recalls.copy()
    order = np.argsort(ap_recalls)
    ap_precisions = ap_precisions[order]
    ap_recalls = ap_recalls[order]
    ap_precisions = np.concatenate(([ap_precisions[0]], ap_precisions, [0.0]))
    ap_recalls = np.concatenate(([0.0], ap_recalls, [1.0]))
    for idx in range(len(ap_precisions) - 2, -1, -1):
        ap_precisions[idx] = max(ap_precisions[idx], ap_precisions[idx + 1])
    ap = float(np.sum((ap_recalls[1:] - ap_recalls[:-1]) * ap_precisions[1:]))

    return {
        "odsF": float(f1_scores[best_idx]),
        "oisF": float(np.mean([row["f1"] for row in per_image_best])),
        "ap": ap,
        "best_threshold": float(curves[best_idx]["threshold"]),
        "num_images": len(image_ids),
        "thresholds": [row["threshold"] for row in curves],
        "precision": [row["precision"] for row in curves],
        "recall": [row["recall"] for row in curves],
        "f1": [row["f1"] for row in curves],
    }


def evaluate_edge_directory(pred_root, gt_root, image_ids=None, thresholds=None, tolerance_ratio=DEFAULT_TOLERANCE_RATIO):
    pred_root = _to_path(pred_root)
    gt_root = _to_path(gt_root)
    if image_ids is None:
        image_ids = collect_prediction_ids(pred_root)
    if not image_ids:
        raise ValueError(f"no edge predictions found under {pred_root}")

    predictions = {}
    ground_truths = {}
    for image_id in image_ids:
        predictions[image_id] = load_prediction_map(resolve_prediction_path(pred_root, image_id))
        ground_truths[image_id] = load_ground_truth_stack(gt_root, image_id)

    return evaluate_edge_predictions(
        predictions=predictions,
        ground_truths=ground_truths,
        thresholds=thresholds,
        tolerance_ratio=tolerance_ratio,
    )


def write_pr_csv(path_like, edge_results):
    path = _to_path(path_like)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "precision", "recall", "f1"])
        for threshold, precision, recall, f1_score in zip(
            edge_results["thresholds"],
            edge_results["precision"],
            edge_results["recall"],
            edge_results["f1"],
        ):
            writer.writerow([threshold, precision, recall, f1_score])
