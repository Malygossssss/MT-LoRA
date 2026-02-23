#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Δm (%) per validation from MTLoRA training logs and report the maximum.

Δm 定义见 MTLoRA 论文 Eq. (3):
Δm = (1/T) * Σ_i [ (-1)^{l_i} * (M_i - M_st,i) / M_st,i ] * 100%
其中 l_i = 1 表示该任务是“越小越好”（如 Normals mean、Depth rmse）。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 论文 Table 1 的单任务基线（PASCAL MTL，Swin-T 单任务 fully fine-tuned）
# 注意：你的本地单任务基线可能不同，建议用 --semseg-st 等参数覆盖！
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
}

# 正则与状态解析
RE_EPOCH = re.compile(r"EPOCH\s+(\d+)\s+training takes")
RE_SEMSEG = re.compile(r"Semantic Segmentation mIoU:\s*([0-9.]+)")
RE_HUMAN = re.compile(r"Human Parts mIoU:\s*([0-9.]+)")
RE_SALIENCY_HEADER = re.compile(r"Results for Saliency Estimation")
RE_SALIENCY_MIOU = re.compile(r"\bmIoU:\s*([0-9.]+)")
RE_NORMALS_HEADER = re.compile(r"Results for Surface Normal Estimation")
RE_NORMALS_MEAN = re.compile(r"\bmean:\s*([0-9.]+)")
RE_DEPTH_HEADER = re.compile(r"Results for (Depth Estimation|depth prediction)", re.IGNORECASE)
RE_DEPTH_RMSE = re.compile(r"\brmse\s*:?\s*([0-9.]+)", re.IGNORECASE)


def parse_tasks(task_text: str) -> List[str]:
    tasks = [t.strip() for t in task_text.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--tasks 不能为空")
    unknown = [t for t in tasks if t not in TASK_SPECS]
    if unknown:
        raise ValueError(f"未知任务: {unknown}，可选任务: {list(TASK_SPECS.keys())}")
    deduped = []
    for t in tasks:
        if t not in deduped:
            deduped.append(t)
    return deduped

def parse_args():
    ap = argparse.ArgumentParser(description="Compute Δm(%) from MTLoRA log.")
    ap.add_argument("--log-file", required=True, type=Path, help="训练日志文件路径")
    # 单任务基线的几种输入方式：
    ap.add_argument("--st-json", type=Path, default=None,
                    help="包含单任务基线的 JSON 文件，键可含 semseg_miou/human_miou/saliency_miou/normals_mean/depth_rmse")
    ap.add_argument("--semseg-st", type=float, default=None, help="SemSeg 单任务 mIoU")
    ap.add_argument("--human-st", type=float, default=None, help="Human Parts 单任务 mIoU")
    ap.add_argument("--saliency-st", type=float, default=None, help="Saliency 单任务 mIoU")
    ap.add_argument("--normals-st", type=float, default=None, help="Normals 单任务 mean（越低越好）")
    ap.add_argument("--depth-st", type=float, default=None, help="Depth 单任务 rmse（越低越好）")
    ap.add_argument("--tasks", type=str, default="semseg,human,saliency,normals,depth",
                    help="参与计算 Δm 的任务，逗号分隔。可选: semseg,human,saliency,normals,depth")
    ap.add_argument("--use-paper-st", action="store_true",
                    help="若未显式提供基线，则使用论文 Table 1 的默认单任务基线（建议你用自家基线覆盖）")
    ap.add_argument("--csv-out", type=Path, default=None, help="可选：将每次 eval 的指标与Δm保存成 CSV")
    args = ap.parse_args()
    args.tasks = parse_tasks(args.tasks)
    return args

def load_st_baseline(args) -> Dict[str, float]:
    st = {}
    if args.st_json is not None:
        with args.st_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "normals_rmse" in data and "normals_mean" not in data:
            data["normals_mean"] = data["normals_rmse"]
        st.update(data)

    # 覆盖式读取命令行参数
    if args.semseg_st is not None:
        st["semseg_miou"] = args.semseg_st
    if args.human_st is not None:
        st["human_miou"] = args.human_st
    if args.saliency_st is not None:
        st["saliency_miou"] = args.saliency_st
    if args.normals_st is not None:
        st["normals_mean"] = args.normals_st
    if args.depth_st is not None:
        st["depth_rmse"] = args.depth_st

    # 若仍缺失，用论文默认或报错
    required_metrics = [TASK_SPECS[t]["metric"] for t in args.tasks]
    missing = [k for k in required_metrics if k not in st]
    if missing:
        if args.use_paper_st:
            for k in missing:
                st[k] = PAPER_ST_DEFAULTS[k]
            print("[WARN] 未提供完整单任务基线，使用论文默认：", {k: st[k] for k in missing})
        else:
            raise ValueError(
                f"缺少单任务基线：{missing}。请通过 --st-json 或 --semseg-st/--human-st/--saliency-st/--normals-st/--depth-st "
                f"提供，或加 --use-paper-st 使用论文默认基线（不推荐用于正式对比）。"
            )
    return st

def compute_delta_m(cur: Dict[str, float], st: Dict[str, float], tasks: List[str]) -> float:
    terms = []
    for task in tasks:
        spec = TASK_SPECS[task]
        metric = spec["metric"]
        delta = (cur[metric] - st[metric]) / st[metric]
        terms.append(delta if spec["higher_better"] else -delta)
    return 100.0 * sum(terms) / len(tasks)

def parse_log(log_path: Path, tasks: List[str]) -> List[Dict]:
    """返回按 eval 次序的记录：仅包含所选任务对应指标"""
    records = []
    cur_epoch: Optional[int] = None

    # saliency/normals 的“上下文状态”标记
    in_saliency = False
    in_normals = False
    in_depth = False

    # 临时收集一次 eval 的任务项
    buf: Dict[str, float] = {}
    needed_metrics = [TASK_SPECS[t]["metric"] for t in tasks]

    def maybe_flush():
        nonlocal buf
        if all(k in buf for k in needed_metrics):
            records.append({
                "epoch": cur_epoch,
                **buf
            })
            buf = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # epoch 标记
            m = RE_EPOCH.search(line)
            if m:
                try:
                    cur_epoch = int(m.group(1))
                except:
                    cur_epoch = None

            # 进入/退出 saliency 与 normals 区块
            if RE_SALIENCY_HEADER.search(line):
                in_saliency = True
                in_normals = False
                in_depth = False
                continue
            if RE_NORMALS_HEADER.search(line):
                in_normals = True
                in_saliency = False
                in_depth = False
                continue
            if RE_DEPTH_HEADER.search(line):
                in_depth = True
                in_saliency = False
                in_normals = False
                continue

            # 直接匹配的两项
            m = RE_SEMSEG.search(line)
            if m:
                buf["semseg_miou"] = float(m.group(1))
                maybe_flush()
                continue

            m = RE_HUMAN.search(line)
            if m:
                buf["human_miou"] = float(m.group(1))
                maybe_flush()
                continue

            # saliency 的 mIoU 行只在 saliency 区块内采集，避免与别处 "mIoU:" 混淆
            if in_saliency:
                m = RE_SALIENCY_MIOU.search(line)
                if m:
                    buf["saliency_miou"] = float(m.group(1))
                    in_saliency = False  # 取到 mIoU 即视为该区块结束
                    maybe_flush()
                    continue

            # normals 的 mean 行只在 normals 区块内采集
            if in_normals:
                m = RE_NORMALS_MEAN.search(line)
                if m:
                    buf["normals_mean"] = float(m.group(1))
                    in_normals = False
                    maybe_flush()
                    continue

            # depth 的 rmse 行只在 depth 区块内采集
            if in_depth:
                m = RE_DEPTH_RMSE.search(line)
                if m:
                    buf["depth_rmse"] = float(m.group(1))
                    in_depth = False
                    maybe_flush()
                    continue

    # 收尾（防止日志最后一次 eval 没被 flush）
    if all(k in buf for k in needed_metrics):
        records.append({
            "epoch": cur_epoch,
            **buf
        })

    return records

def main():
    args = parse_args()
    st = load_st_baseline(args)
    recs = parse_log(args.log_file, args.tasks)
    if not recs:
        raise SystemExit(f"未在日志中解析到任何完整的 eval 记录（所选任务: {args.tasks}）。请检查格式/正则。")

    # 计算 Δm
    out_rows: List[Tuple[int, float, Dict[str, float]]] = []
    for r in recs:
        cur = {TASK_SPECS[t]["metric"]: r[TASK_SPECS[t]["metric"]] for t in args.tasks}
        d = compute_delta_m(cur, st, args.tasks)
        out_rows.append((r.get("epoch", -1), d, cur))

    # 输出每次 eval
    print("Per-eval Δm(%)：")
    for ep, d, cur in out_rows:
        ep_str = f"epoch {ep}" if ep is not None and ep >= 0 else "(epoch 未捕获)"
        metric_str = ", ".join(
            f"{TASK_SPECS[t]['display']} {cur[TASK_SPECS[t]['metric']]:.4f}" for t in args.tasks
        )
        print(f"- {ep_str}: Δm = {d:.3f}% | {metric_str}")

    # 最大值
    best = max(out_rows, key=lambda x: x[1])
    bep, bd, bcur = best
    bep_str = f"{bep}" if bep is not None and bep >= 0 else "N/A"
    print("\n=== Best Δm(%) ===")
    print(f"Epoch: {bep_str}")
    print(f"Δm: {bd:.3f}%")
    for t in args.tasks:
        metric = TASK_SPECS[t]["metric"]
        print(f"{TASK_SPECS[t]['display']}: {bcur[metric]:.4f}")

    # 可选导出 CSV
    if args.csv_out is not None:
        try:
            import csv
            with args.csv_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                metric_headers = [TASK_SPECS[t]["display"].replace(" ", "_") for t in args.tasks]
                writer.writerow(["epoch", *metric_headers, "Delta_m_percent"])
                for ep, d, cur in out_rows:
                    writer.writerow([ep, *[cur[TASK_SPECS[t]["metric"]] for t in args.tasks], d])
                # ⚠️ 额外写入 best 结果
                writer.writerow([])
                writer.writerow([f"BEST(epoch {bep})", *[bcur[TASK_SPECS[t]["metric"]] for t in args.tasks], bd])

            print(f"\nCSV 已写出：{args.csv_out}")
        except Exception as e:
            print(f"[WARN] 写 CSV 失败：{e}")

if __name__ == "__main__":
    main()
