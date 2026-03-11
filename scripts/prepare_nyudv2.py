#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.morphology import thin


SUPPORTED_EDGE_FORMATS = ("auto", "png", "npy", "npz")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare NYUDv2 data to match MT-LoRA expected structure."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source NYUDv2 root (contains image/depth/normal/seg40/etc.).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination root for MT-LoRA-compatible NYUDv2 structure.",
    )
    parser.add_argument(
        "--splits-mat",
        type=Path,
        default=None,
        help="Optional splits.mat path. If provided, indices determine train/val/test splits.",
    )
    parser.add_argument(
        "--val-from",
        choices=["test", "train"],
        default="test",
        help="If splits.mat does not contain val, which split to copy for val.",
    )
    parser.add_argument(
        "--edge-source",
        choices=["auto", "real", "derived"],
        default="auto",
        help="How to build NYUD edge annotations.",
    )
    parser.add_argument(
        "--edge-root",
        type=Path,
        default=None,
        help="Optional real edge GT root. Used when --edge-source is auto or real.",
    )
    parser.add_argument(
        "--edge-format",
        choices=SUPPORTED_EDGE_FORMATS,
        default="auto",
        help="Format used under --edge-root. auto tries png/npy/npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it exists.",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination {path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_split_indices(splits_mat: Path):
    import scipy.io as sio

    mat = sio.loadmat(splits_mat)
    splits = {}
    for key in mat:
        key_lower = key.lower()
        if key_lower.startswith("train"):
            splits["train"] = mat[key].squeeze()
        elif key_lower.startswith("test"):
            splits["test"] = mat[key].squeeze()
        elif key_lower.startswith("val"):
            splits["val"] = mat[key].squeeze()
    return splits


def resolve_ids_from_indices(indices):
    ids = []
    for idx in indices:
        if isinstance(idx, np.ndarray):
            idx = int(idx.item())
        ids.append(f"{int(idx):05d}")
    return ids


def collect_ids_from_dir(split_dir: Path) -> List[str]:
    return sorted(p.stem for p in split_dir.glob("*.png"))


def write_split_file(path: Path, ids: List[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(ids))


def convert_image_to_jpg(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    img.save(dst, format="JPEG", quality=95)


def convert_depth_to_npy(src: Path, dst: Path) -> None:
    depth = np.array(Image.open(src)).astype(np.float32)
    np.save(dst, depth)


def load_segmentation_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.int32)


def derive_edge_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 2:
        raise ValueError(f"expected a 2D segmentation map, got {segmentation.shape}")

    valid = (segmentation > 0) & (segmentation < 255)
    edge = np.zeros_like(segmentation, dtype=bool)

    horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (segmentation[:, :-1] != segmentation[:, 1:])
    )
    vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (segmentation[:-1, :] != segmentation[1:, :])
    )

    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical

    return thin(edge).astype(np.float32)


def _normalize_edge_stack(edge_stack: np.ndarray) -> np.ndarray:
    edge_stack = np.asarray(edge_stack)
    if edge_stack.ndim == 2:
        edge_stack = edge_stack[np.newaxis, ...]
    if edge_stack.ndim == 3 and edge_stack.shape[-1] == 1:
        edge_stack = np.transpose(edge_stack, (2, 0, 1))
    if edge_stack.ndim != 3:
        raise ValueError(f"expected a 2D or 3D edge stack, got {edge_stack.shape}")
    if edge_stack.max() > 1.0 or edge_stack.min() < 0.0:
        edge_stack = edge_stack / 255.0
    return (edge_stack >= 0.5).astype(np.float32)


def fuse_edge_annotations(edge_stack: np.ndarray) -> np.ndarray:
    edge_stack = _normalize_edge_stack(edge_stack)
    votes = edge_stack.sum(axis=0)
    threshold = edge_stack.shape[0] // 2 + 1
    return (votes >= threshold).astype(np.float32)


def _load_edge_stack(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        if "gts" in data:
            return _normalize_edge_stack(data["gts"])
        first_key = next(iter(data.files))
        return _normalize_edge_stack(data[first_key])
    if suffix == ".npy":
        return _normalize_edge_stack(np.load(path))
    if suffix == ".png":
        return _normalize_edge_stack(np.array(Image.open(path)))
    raise ValueError(f"unsupported edge stack format: {path.suffix}")


def _allowed_suffixes(edge_format: str) -> Tuple[str, ...]:
    if edge_format == "auto":
        return (".png", ".npy", ".npz")
    return (f".{edge_format}",)


def _collect_multi_annotation_paths(base: Path, sample_id: str, allowed_suffixes: Tuple[str, ...]) -> List[Path]:
    matches = []
    for suffix in allowed_suffixes:
        matches.extend(sorted(base.glob(f"{sample_id}_*{suffix}")))
        matches.extend(sorted(base.glob(f"{sample_id}-*{suffix}")))
    deduped = []
    seen = set()
    for path in matches:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def find_real_edge_source(edge_root: Path, split_name: str, sample_id: str, edge_format: str):
    if edge_root is None:
        return None

    allowed_suffixes = _allowed_suffixes(edge_format)
    search_roots = [edge_root / split_name, edge_root]

    for base in search_roots:
        if not base.exists():
            continue

        sample_dir = base / sample_id
        if sample_dir.is_dir():
            files = [
                path
                for path in sorted(sample_dir.iterdir())
                if path.is_file() and path.suffix.lower() in allowed_suffixes
            ]
            if files:
                return files

        for suffix in allowed_suffixes:
            candidate = base / f"{sample_id}{suffix}"
            if candidate.is_file():
                return candidate

        multi = _collect_multi_annotation_paths(base, sample_id, allowed_suffixes)
        if multi:
            return multi

    return None


def load_real_edge_stack(edge_root: Path, split_name: str, sample_id: str, edge_format: str):
    source = find_real_edge_source(edge_root, split_name, sample_id, edge_format)
    if source is None:
        return None

    if isinstance(source, list):
        stacks = []
        for path in source:
            stacks.extend(list(_normalize_edge_stack(_load_edge_stack(path))))
        return _normalize_edge_stack(np.stack(stacks, axis=0))

    return _load_edge_stack(source)


def write_edge_targets(edge_dst_file: Path, edge_eval_dst_file: Path, edge_stack: np.ndarray) -> None:
    edge_stack = _normalize_edge_stack(edge_stack)
    edge_dst_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(edge_dst_file, fuse_edge_annotations(edge_stack).astype(np.float32))

    if edge_stack.shape[0] > 1:
        edge_eval_dst_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(edge_eval_dst_file, gts=edge_stack.astype(np.float32))


def resolve_source_split(src_root: Path, split_name: str, val_from: str) -> str:
    candidate = src_root / "image" / split_name
    if candidate.is_dir():
        return split_name
    if split_name == "val":
        return val_from
    return split_name


def prepare_split(
    split_name: str,
    source_split_name: str,
    ids: List[str],
    src_root: Path,
    dst_root: Path,
    edge_source: str,
    edge_root: Optional[Path],
    edge_format: str,
) -> None:
    image_src_dir = src_root / "image" / source_split_name
    depth_src_dir = src_root / "depth" / source_split_name
    normal_src_dir = src_root / "normal" / source_split_name
    seg_src_dir = src_root / "seg40" / source_split_name

    images_dst = dst_root / "images"
    depth_dst = dst_root / "depth"
    normals_dst = dst_root / "normals"
    seg_dst = dst_root / "segmentation"
    edge_dst = dst_root / "edge"
    edge_eval_dst = dst_root / "edge_eval"

    for folder in [images_dst, depth_dst, normals_dst, seg_dst, edge_dst]:
        folder.mkdir(parents=True, exist_ok=True)

    for sample_id in ids:
        image_src = image_src_dir / f"{sample_id}.png"
        depth_src = depth_src_dir / f"{sample_id}.png"
        normal_src = normal_src_dir / f"{sample_id}.npy"
        seg_src = seg_src_dir / f"{sample_id}.png"

        image_dst = images_dst / f"{sample_id}.jpg"
        depth_dst_file = depth_dst / f"{sample_id}.npy"
        normal_dst = normals_dst / f"{sample_id}.npy"
        seg_dst_file = seg_dst / f"{sample_id}.png"
        edge_dst_file = edge_dst / f"{sample_id}.npy"
        edge_eval_dst_file = edge_eval_dst / f"{sample_id}.npz"

        convert_image_to_jpg(image_src, image_dst)
        convert_depth_to_npy(depth_src, depth_dst_file)

        if not normal_dst.exists():
            shutil.copy2(normal_src, normal_dst)
        if not seg_dst_file.exists():
            shutil.copy2(seg_src, seg_dst_file)

        real_edge_stack = None
        if edge_source in {"auto", "real"}:
            real_edge_stack = load_real_edge_stack(edge_root, source_split_name, sample_id, edge_format)

        if real_edge_stack is not None:
            write_edge_targets(edge_dst_file, edge_eval_dst_file, real_edge_stack)
            continue

        if edge_source == "real":
            raise FileNotFoundError(
                f"real edge GT not found for split={split_name} sample={sample_id} under {edge_root}"
            )

        derived_edge = derive_edge_from_segmentation(load_segmentation_mask(seg_src))
        write_edge_targets(edge_dst_file, edge_eval_dst_file, derived_edge)


def main():
    args = parse_args()
    src_root = args.src
    dst_root = args.dst

    if args.edge_source == "real" and args.edge_root is None:
        raise ValueError("--edge-root is required when --edge-source=real")

    ensure_empty_dir(dst_root, args.overwrite)

    gt_sets_dir = dst_root / "gt_sets"
    gt_sets_dir.mkdir(parents=True, exist_ok=True)

    if args.splits_mat:
        splits = load_split_indices(args.splits_mat)
        split_ids = {k: resolve_ids_from_indices(v) for k, v in splits.items()}
    else:
        split_ids = {
            "train": collect_ids_from_dir(src_root / "image" / "train"),
            "test": collect_ids_from_dir(src_root / "image" / "test"),
        }

    if "val" not in split_ids:
        split_ids["val"] = split_ids[args.val_from]

    for split_name in ["train", "val", "test"]:
        ids = split_ids.get(split_name, [])
        if ids:
            write_split_file(gt_sets_dir / f"{split_name}.txt", ids)

    for split_name in ["train", "val", "test"]:
        ids = split_ids.get(split_name, [])
        if not ids:
            continue
        source_split_name = resolve_source_split(src_root, split_name, args.val_from)
        prepare_split(
            split_name=split_name,
            source_split_name=source_split_name,
            ids=ids,
            src_root=src_root,
            dst_root=dst_root,
            edge_source=args.edge_source,
            edge_root=args.edge_root,
            edge_format=args.edge_format,
        )

    print(f"Prepared NYUDv2 dataset at {dst_root}")


if __name__ == "__main__":
    main()
