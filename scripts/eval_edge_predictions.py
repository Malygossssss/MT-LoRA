#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_edge import eval_edge_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved edge predictions.")
    parser.add_argument(
        "--dataset",
        choices=["nyud"],
        default="nyud",
        help="Dataset used to resolve GT semantics.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Prediction directory. Expected files live in pred-dir/edge/ by default.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        required=True,
        help="Prepared dataset root containing edge/ and optionally edge_eval/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    database = "NYUD" if args.dataset.lower() == "nyud" else args.dataset
    results = eval_edge_predictions(
        database=database,
        save_dir=str(args.pred_dir),
        gt_root=str(args.gt_root),
        write_outputs=True,
    )
    summary = {key: results[key] for key in ("odsF", "oisF", "ap", "best_threshold", "num_images")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
