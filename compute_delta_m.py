#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PAPER_ST_DEFAULTS = {
    "semseg_miou": 67.21,
    "human_miou": 61.93,
    "saliency_miou": 62.35,
    "normals_mean": 17.97,
    "depth_rmse": 0.6436,
}

TASK_SPECS = {
    "semseg": {
        "metric": "semseg_miou",
        "higher_better": True,
        "arg": "semseg_st",
        "display": "SemSeg mIoU",
    },
    "human": {
        "metric": "human_miou",
        "higher_better": True,
        "arg": "human_st",
        "display": "Human Parts mIoU",
    },
    "saliency": {
        "metric": "saliency_miou",
        "higher_better": True,
        "arg": "saliency_st",
        "display": "Saliency mIoU",
    },
    "normals": {
        "metric": "normals_mean",
        "higher_better": False,
        "arg": "normals_st",
        "display": "Normals mean",
    },
    "depth": {
        "metric": "depth_rmse",
        "higher_better": False,
        "arg": "depth_st",
        "display": "Depth rmse",
    },
    "edge": {
        "metric": "edge_odsf",
        "higher_better": True,
        "arg": "edge_st",
        "display": "Edge odsF",
    },
}

RE_EPOCH = re.compile(r"EPOCH\s+(\d+)\s+training takes")
RE_SEMSEG = re.compile(r"Semantic Segmentation mIoU:\s*([0-9.]+)")
RE_HUMAN = re.compile(r"Human Parts mIoU:\s*([0-9.]+)")
RE_SALIENCY_HEADER = re.compile(r"Results for Saliency Estimation")
RE_SALIENCY_MIOU = re.compile(r"\bmIoU:\s*([0-9.]+)")
RE_NORMALS_HEADER = re.compile(r"Results for Surface Normal Estimation")
RE_NORMALS_MEAN = re.compile(r"\bmean:\s*([0-9.]+)")
RE_DEPTH_HEADER = re.compile(r"Results for (Depth Estimation|depth prediction)", re.IGNORECASE)
RE_DEPTH_RMSE = re.compile(r"\brmse\s*:?\s*([0-9.]+)", re.IGNORECASE)
RE_EDGE_HEADER = re.compile(r"Edge Detection Evaluation")
RE_EDGE_ODSF = re.compile(r"\bodsF:\s*([0-9.]+)", re.IGNORECASE)


def parse_tasks(task_text: str) -> List[str]:
    tasks = [task.strip() for task in task_text.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks cannot be empty")
    unknown = [task for task in tasks if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; choose from {list(TASK_SPECS.keys())}")
    deduped = []
    for task in tasks:
        if task not in deduped:
            deduped.append(task)
    return deduped


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Delta m(%) from MTLoRA logs.")
    parser.add_argument("--log-file", required=True, type=Path, help="Training log path.")
    parser.add_argument(
        "--st-json",
        type=Path,
        default=None,
        help="Optional JSON file containing single-task baselines.",
    )
    parser.add_argument("--semseg-st", type=float, default=None, help="SemSeg single-task mIoU.")
    parser.add_argument("--human-st", type=float, default=None, help="Human Parts single-task mIoU.")
    parser.add_argument("--saliency-st", type=float, default=None, help="Saliency single-task mIoU.")
    parser.add_argument("--normals-st", type=float, default=None, help="Normals single-task mean.")
    parser.add_argument("--depth-st", type=float, default=None, help="Depth single-task rmse.")
    parser.add_argument("--edge-st", type=float, default=None, help="Edge single-task odsF.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="semseg,human,saliency,normals,depth",
        help="Comma-separated task list. Supported: semseg,human,saliency,normals,depth,edge",
    )
    parser.add_argument(
        "--use-paper-st",
        action="store_true",
        help="Use paper defaults for any missing baselines when available.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV output for every parsed validation point.",
    )
    args = parser.parse_args()
    args.tasks = parse_tasks(args.tasks)
    return args


def load_st_baseline(args) -> Dict[str, float]:
    baselines = {}
    if args.st_json is not None:
        with args.st_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "normals_rmse" in data and "normals_mean" not in data:
            data["normals_mean"] = data["normals_rmse"]
        baselines.update(data)

    for task_name, spec in TASK_SPECS.items():
        arg_name = spec["arg"]
        value = getattr(args, arg_name)
        if value is not None:
            baselines[spec["metric"]] = value

    required = [TASK_SPECS[task]["metric"] for task in args.tasks]
    missing = [metric for metric in required if metric not in baselines]
    if not missing:
        return baselines

    if not args.use_paper_st:
        raise ValueError(
            "missing single-task baselines for "
            f"{missing}; provide them through --st-json or explicit --*-st flags"
        )

    unavailable = [metric for metric in missing if metric not in PAPER_ST_DEFAULTS]
    if unavailable:
        raise ValueError(f"paper defaults do not include baselines for {unavailable}")

    for metric in missing:
        baselines[metric] = PAPER_ST_DEFAULTS[metric]
    return baselines


def compute_delta_m(current: Dict[str, float], baselines: Dict[str, float], tasks: List[str]) -> float:
    terms = []
    for task in tasks:
        spec = TASK_SPECS[task]
        metric = spec["metric"]
        delta = (current[metric] - baselines[metric]) / baselines[metric]
        terms.append(delta if spec["higher_better"] else -delta)
    return 100.0 * sum(terms) / len(tasks)


def parse_log(log_path: Path, tasks: List[str]) -> List[Dict[str, float]]:
    records = []
    current_epoch: Optional[int] = None
    in_saliency = False
    in_normals = False
    in_depth = False
    in_edge = False
    buffer: Dict[str, float] = {}
    needed_metrics = [TASK_SPECS[task]["metric"] for task in tasks]

    def maybe_flush():
        nonlocal buffer
        if all(metric in buffer for metric in needed_metrics):
            records.append({"epoch": current_epoch, **buffer})
            buffer = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = RE_EPOCH.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            if RE_SALIENCY_HEADER.search(line):
                in_saliency, in_normals, in_depth, in_edge = True, False, False, False
                continue
            if RE_NORMALS_HEADER.search(line):
                in_saliency, in_normals, in_depth, in_edge = False, True, False, False
                continue
            if RE_DEPTH_HEADER.search(line):
                in_saliency, in_normals, in_depth, in_edge = False, False, True, False
                continue
            if RE_EDGE_HEADER.search(line):
                in_saliency, in_normals, in_depth, in_edge = False, False, False, True
                continue

            semseg_match = RE_SEMSEG.search(line)
            if semseg_match:
                buffer["semseg_miou"] = float(semseg_match.group(1))
                maybe_flush()
                continue

            human_match = RE_HUMAN.search(line)
            if human_match:
                buffer["human_miou"] = float(human_match.group(1))
                maybe_flush()
                continue

            if in_saliency:
                saliency_match = RE_SALIENCY_MIOU.search(line)
                if saliency_match:
                    buffer["saliency_miou"] = float(saliency_match.group(1))
                    in_saliency = False
                    maybe_flush()
                    continue

            if in_normals:
                normals_match = RE_NORMALS_MEAN.search(line)
                if normals_match:
                    buffer["normals_mean"] = float(normals_match.group(1))
                    in_normals = False
                    maybe_flush()
                    continue

            if in_depth:
                depth_match = RE_DEPTH_RMSE.search(line)
                if depth_match:
                    buffer["depth_rmse"] = float(depth_match.group(1))
                    in_depth = False
                    maybe_flush()
                    continue

            if in_edge:
                edge_match = RE_EDGE_ODSF.search(line)
                if edge_match:
                    buffer["edge_odsf"] = float(edge_match.group(1))
                    in_edge = False
                    maybe_flush()
                    continue

    if all(metric in buffer for metric in needed_metrics):
        records.append({"epoch": current_epoch, **buffer})
    return records


def main():
    args = parse_args()
    baselines = load_st_baseline(args)
    records = parse_log(args.log_file, args.tasks)
    if not records:
        raise SystemExit(f"no complete validation records were parsed for tasks={args.tasks}")

    rows: List[Tuple[int, float, Dict[str, float]]] = []
    for record in records:
        current = {TASK_SPECS[task]["metric"]: record[TASK_SPECS[task]["metric"]] for task in args.tasks}
        rows.append((record.get("epoch", -1), compute_delta_m(current, baselines, args.tasks), current))

    print("Per-eval Delta m(%):")
    for epoch, delta_m, current in rows:
        epoch_label = f"epoch {epoch}" if epoch is not None and epoch >= 0 else "epoch N/A"
        metrics = ", ".join(
            f"{TASK_SPECS[task]['display']} {current[TASK_SPECS[task]['metric']]:.4f}"
            for task in args.tasks
        )
        print(f"- {epoch_label}: Delta m = {delta_m:.3f}% | {metrics}")

    best_epoch, best_delta_m, best_metrics = max(rows, key=lambda row: row[1])
    print("\n=== Best Delta m(%) ===")
    print(f"Epoch: {best_epoch if best_epoch is not None and best_epoch >= 0 else 'N/A'}")
    print(f"Delta m: {best_delta_m:.3f}%")
    for task in args.tasks:
        metric = TASK_SPECS[task]["metric"]
        print(f"{TASK_SPECS[task]['display']}: {best_metrics[metric]:.4f}")

    if args.csv_out is not None:
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            headers = [TASK_SPECS[task]["display"].replace(" ", "_") for task in args.tasks]
            writer.writerow(["epoch", *headers, "Delta_m_percent"])
            for epoch, delta_m, current in rows:
                writer.writerow([epoch, *[current[TASK_SPECS[task]["metric"]] for task in args.tasks], delta_m])
            writer.writerow([])
            writer.writerow(
                [
                    f"BEST(epoch {best_epoch})",
                    *[best_metrics[TASK_SPECS[task]["metric"]] for task in args.tasks],
                    best_delta_m,
                ]
            )
        print(f"\nCSV written to: {args.csv_out}")


if __name__ == "__main__":
    main()
