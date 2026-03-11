#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_nyudv2 import derive_edge_from_segmentation, load_segmentation_mask


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination {path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_segmentation_files(segmentation_dir: Path):
    files = sorted(segmentation_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no segmentation PNG files found under {segmentation_dir}")
    return files


def derive_edge_directory(segmentation_dir: Path, output_dir: Path, overwrite: bool = False):
    segmentation_dir = Path(segmentation_dir)
    output_dir = Path(output_dir)
    ensure_empty_dir(output_dir, overwrite)

    total_nonzero = 0
    segmentation_files = collect_segmentation_files(segmentation_dir)
    for seg_path in segmentation_files:
        edge = derive_edge_from_segmentation(load_segmentation_mask(seg_path)).astype(np.float32)
        np.save(output_dir / f"{seg_path.stem}.npy", edge)
        total_nonzero += int(edge.sum())

    return {
        "num_files": len(segmentation_files),
        "total_nonzero": total_nonzero,
        "output_dir": output_dir,
    }


def swap_edge_directory(dataset_root: Path, derived_edge_dir: Path, backup_name: str):
    dataset_root = Path(dataset_root)
    derived_edge_dir = Path(derived_edge_dir)
    edge_dir = dataset_root / "edge"
    backup_dir = dataset_root / backup_name

    if backup_dir.exists():
        raise FileExistsError(
            f"backup directory already exists: {backup_dir}. Move or remove it before replacing edge."
        )

    if edge_dir.exists():
        edge_dir.rename(backup_dir)

    derived_edge_dir.rename(edge_dir)
    return {
        "edge_dir": edge_dir,
        "backup_dir": backup_dir if backup_dir.exists() else None,
    }


def derive_edges_for_prepared_dataset(
    dataset_root: Path,
    output_dir: Path = None,
    replace_edge: bool = False,
    backup_name: str = "edge_zero_backup",
    overwrite: bool = False,
):
    dataset_root = Path(dataset_root)
    segmentation_dir = dataset_root / "segmentation"
    if not segmentation_dir.is_dir():
        raise FileNotFoundError(f"missing segmentation directory: {segmentation_dir}")

    if output_dir is None:
        output_dir = dataset_root / "__edge_from_semseg_tmp__"
    else:
        output_dir = Path(output_dir)

    edge_dir = dataset_root / "edge"
    if replace_edge and output_dir.resolve() == edge_dir.resolve():
        raise ValueError("--output-dir must differ from dataset_root/edge when --replace-edge is enabled")

    results = derive_edge_directory(segmentation_dir, output_dir, overwrite=overwrite)

    if replace_edge:
        results.update(swap_edge_directory(dataset_root, output_dir, backup_name))

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive NYUD edge labels directly from a prepared dataset root's segmentation folder."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Prepared NYUD dataset root containing segmentation/ and optionally edge/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write derived edge .npy files. Defaults to dataset_root/__edge_from_semseg_tmp__.",
    )
    parser.add_argument(
        "--replace-edge",
        action="store_true",
        help="Replace dataset_root/edge by the derived output after creation.",
    )
    parser.add_argument(
        "--backup-name",
        type=str,
        default="edge_zero_backup",
        help="Backup directory name used when --replace-edge is set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = derive_edges_for_prepared_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        replace_edge=args.replace_edge,
        backup_name=args.backup_name,
        overwrite=args.overwrite,
    )
    print(f"Derived {results['num_files']} edge maps with {results['total_nonzero']} positive pixels in total.")
    if "edge_dir" in results:
        print(f"Active edge directory: {results['edge_dir']}")
    else:
        print(f"Output edge directory: {results['output_dir']}")
    if results.get("backup_dir") is not None:
        print(f"Backup edge directory: {results['backup_dir']}")


if __name__ == "__main__":
    main()
