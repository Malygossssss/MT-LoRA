#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


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


def collect_ids_from_dir(split_dir: Path) -> list[str]:
    return sorted(p.stem for p in split_dir.glob("*.png"))


def write_split_file(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(ids))


def convert_image_to_jpg(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    img.save(dst, format="JPEG", quality=95)


def convert_depth_to_npy(src: Path, dst: Path) -> None:
    depth = np.array(Image.open(src)).astype(np.float32)
    np.save(dst, depth)


def ensure_edge_placeholder(dst: Path, shape):
    edge = np.zeros(shape[:2], dtype=np.float32)
    np.save(dst, edge)


def prepare_split(
    split_name: str,
    ids: list[str],
    src_root: Path,
    dst_root: Path,
) -> None:
    image_src_dir = src_root / "image" / split_name
    depth_src_dir = src_root / "depth" / split_name
    normal_src_dir = src_root / "normal" / split_name
    seg_src_dir = src_root / "seg40" / split_name

    images_dst = dst_root / "images"
    depth_dst = dst_root / "depth"
    normals_dst = dst_root / "normals"
    seg_dst = dst_root / "segmentation"
    edge_dst = dst_root / "edge"

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

        convert_image_to_jpg(image_src, image_dst)
        convert_depth_to_npy(depth_src, depth_dst_file)

        if not normal_dst.exists():
            shutil.copy2(normal_src, normal_dst)
        if not seg_dst_file.exists():
            shutil.copy2(seg_src, seg_dst_file)

        if not edge_dst_file.exists():
            img = np.array(Image.open(image_src))
            ensure_edge_placeholder(edge_dst_file, img.shape)


def main():
    args = parse_args()
    src_root = args.src
    dst_root = args.dst
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
        if not ids:
            continue
        write_split_file(gt_sets_dir / f"{split_name}.txt", ids)

    for split_name in ["train", "test"]:
        ids = split_ids.get(split_name, [])
        if not ids:
            continue
        prepare_split(split_name, ids, src_root, dst_root)

    print(f"Prepared NYUDv2 dataset at {dst_root}")


if __name__ == "__main__":
    main()