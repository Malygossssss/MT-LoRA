import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config
from data import build_loader

IGNORE_INDEX = 255
CLASSIFICATION_TASKS = {"semseg", "human_parts"}


def parse_args():
    parser = argparse.ArgumentParser("Label statistics for MT-LoRA datasets")
    parser.add_argument("--cfg", type=str, required=True, help="path to config file")
    parser.add_argument("--nyud", type=str, default=None, help="path to NYUD dataset root")
    parser.add_argument("--pascal", type=str, default=None, help="path to PASCAL dataset root")
    parser.add_argument("--tasks", type=str, required=True, help="comma separated task list")
    parser.add_argument("--data-path", type=str, default=None, help="optional override for DATA.DATA_PATH")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size", help="optional batch size override")
    parser.add_argument("--max-batches", type=int, default=20, help="number of batches to scan")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="which loader split to scan")
    parser.add_argument("--opts", nargs="+", default=None, help="extra KEY VALUE pairs to merge into config")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for DistributedDataParallel compatibility")
    parser.add_argument("--local-rank", type=int, default=0, dest="local_rank", help="alias for --local_rank")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[label_stats] Ignoring unrecognized arguments: {unknown}")
    return args


def _flatten_labels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    return tensor.long().reshape(-1)


def main():
    args = parse_args()
    config = get_config(args)

    dataset_train, dataset_val, loader_train, loader_val, _ = build_loader(config)
    loader = loader_train if args.split == "train" else loader_val

    num_outputs = dict(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT)
    task_stats = {}
    for task, n_classes in num_outputs.items():
        if task in CLASSIFICATION_TASKS:
            task_stats[task] = {
                "n_classes": int(n_classes),
                "total": 0,
                "ignore": 0,
                "invalid": 0,
                "min": None,
                "max": None,
                "invalid_values": set(),
            }

    if not task_stats:
        raise ValueError("No classification segmentation tasks found in --tasks.")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        for task, stats in task_stats.items():
            if task not in batch:
                continue

            labels = _flatten_labels(batch[task])
            n_classes = stats["n_classes"]

            ignore_mask = labels == IGNORE_INDEX
            valid_mask = (labels >= 0) & (labels < n_classes)
            invalid_mask = ~(valid_mask | ignore_mask)

            stats["total"] += labels.numel()
            stats["ignore"] += int(ignore_mask.sum().item())
            stats["invalid"] += int(invalid_mask.sum().item())

            valid_non_ignore = labels[valid_mask]
            if valid_non_ignore.numel() > 0:
                cur_min = int(valid_non_ignore.min().item())
                cur_max = int(valid_non_ignore.max().item())
                stats["min"] = cur_min if stats["min"] is None else min(stats["min"], cur_min)
                stats["max"] = cur_max if stats["max"] is None else max(stats["max"], cur_max)

            if invalid_mask.any():
                invalid_vals = torch.unique(labels[invalid_mask]).cpu().tolist()
                stats["invalid_values"].update(int(v) for v in invalid_vals)

    print("=== Label Range Diagnostics ===")
    print(f"split={args.split} max_batches={args.max_batches}")
    for task, stats in task_stats.items():
        total = max(stats["total"], 1)
        ignore_frac = stats["ignore"] / total
        invalid_frac = stats["invalid"] / total
        invalid_vals_sorted = sorted(stats["invalid_values"])
        preview = invalid_vals_sorted[:20]
        suffix = "..." if len(invalid_vals_sorted) > 20 else ""

        print("---")
        print(f"task={task} n_classes={stats['n_classes']}")
        print(f"valid_min={stats['min']} valid_max={stats['max']}")
        print(f"ignore_pixels={stats['ignore']} ({ignore_frac:.4%})")
        print(f"invalid_pixels={stats['invalid']} ({invalid_frac:.4%})")
        print(f"invalid_values_preview={preview}{suffix}")


if __name__ == "__main__":
    main()