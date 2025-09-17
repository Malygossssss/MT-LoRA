#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Δm (%) per validation from MTLoRA training logs and report the maximum.

Δm 定义见 MTLoRA 论文 Eq. (3):
Δm = (1/T) * Σ_i [ (-1)^{l_i} * (M_i - M_st,i) / M_st,i ] * 100%
其中 l_i = 1 表示该任务是“越小越好”（本项目里只有 Normals 的 rmse 属于这种）。
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
    "normals_rmse": 17.97,
}

# 正则与状态解析
RE_EPOCH = re.compile(r"EPOCH\s+(\d+)\s+training takes")
RE_SEMSEG = re.compile(r"Semantic Segmentation mIoU:\s*([0-9.]+)")
RE_HUMAN = re.compile(r"Human Parts mIoU:\s*([0-9.]+)")
RE_SALIENCY_HEADER = re.compile(r"Results for Saliency Estimation")
RE_SALIENCY_MIOU = re.compile(r"\bmIoU:\s*([0-9.]+)")
RE_NORMALS_HEADER = re.compile(r"Results for Surface Normal Estimation")
RE_NORMALS_RMSE = re.compile(r"\brmse:\s*([0-9.]+)")

def parse_args():
    ap = argparse.ArgumentParser(description="Compute Δm(%) from MTLoRA log.")
    ap.add_argument("--log-file", required=True, type=Path, help="训练日志文件路径")
    # 单任务基线的几种输入方式：
    ap.add_argument("--st-json", type=Path, default=None,
                    help="包含单任务基线的 JSON 文件，键为 semseg_miou/human_miou/saliency_miou/normals_rmse")
    ap.add_argument("--semseg-st", type=float, default=None, help="SemSeg 单任务 mIoU")
    ap.add_argument("--human-st", type=float, default=None, help="Human Parts 单任务 mIoU")
    ap.add_argument("--saliency-st", type=float, default=None, help="Saliency 单任务 mIoU")
    ap.add_argument("--normals-st", type=float, default=None, help="Normals 单任务 rmse")
    ap.add_argument("--use-paper-st", action="store_true",
                    help="若未显式提供基线，则使用论文 Table 1 的默认单任务基线（建议你用自家基线覆盖）")
    ap.add_argument("--csv-out", type=Path, default=None, help="可选：将每次 eval 的指标与Δm保存成 CSV")
    return ap.parse_args()

def load_st_baseline(args) -> Dict[str, float]:
    st = {}
    if args.st_json is not None:
        with args.st_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        st.update(data)

    # 覆盖式读取命令行参数
    if args.semseg_st is not None:
        st["semseg_miou"] = args.semseg_st
    if args.human_st is not None:
        st["human_miou"] = args.human_st
    if args.saliency_st is not None:
        st["saliency_miou"] = args.saliency_st
    if args.normals_st is not None:
        st["normals_rmse"] = args.normals_st

    # 若仍缺失，用论文默认或报错
    missing = [k for k in ["semseg_miou", "human_miou", "saliency_miou", "normals_rmse"] if k not in st]
    if missing:
        if args.use_paper_st:
            for k in missing:
                st[k] = PAPER_ST_DEFAULTS[k]
            print("[WARN] 未提供完整单任务基线，使用论文默认：", {k: st[k] for k in missing})
        else:
            raise ValueError(
                f"缺少单任务基线：{missing}。请通过 --st-json 或 --semseg-st/--human-st/--saliency-st/--normals-st "
                f"提供，或加 --use-paper-st 使用论文默认基线（不推荐用于正式对比）。"
            )
    return st

def compute_delta_m(cur: Dict[str, float], st: Dict[str, float]) -> float:
    # 三个 mIoU 是“越大越好”；Normals rmse 是“越小越好”
    terms = []
    terms.append( (cur["semseg_miou"]  - st["semseg_miou"])  / st["semseg_miou"] )
    terms.append( (cur["human_miou"]   - st["human_miou"])   / st["human_miou"] )
    terms.append( (cur["saliency_miou"]- st["saliency_miou"])/ st["saliency_miou"] )
    terms.append( -(cur["normals_rmse"]- st["normals_rmse"])/ st["normals_rmse"] )
    return 100.0 * sum(terms) / 4.0

def parse_log(log_path: Path) -> List[Dict]:
    """返回按 eval 次序的记录：{epoch, semseg_miou, human_miou, saliency_miou, normals_rmse}"""
    records = []
    cur_epoch: Optional[int] = None

    # saliency/normals 的“上下文状态”标记
    in_saliency = False
    in_normals = False

    # 临时收集一次 eval 的四项
    buf: Dict[str, float] = {}

    def maybe_flush():
        nonlocal buf
        if all(k in buf for k in ["semseg_miou", "human_miou", "saliency_miou", "normals_rmse"]):
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
                continue
            if RE_NORMALS_HEADER.search(line):
                in_normals = True
                in_saliency = False
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

            # normals 的 rmse 行只在 normals 区块内采集
            if in_normals:
                m = RE_NORMALS_RMSE.search(line)
                if m:
                    buf["normals_rmse"] = float(m.group(1))
                    in_normals = False
                    maybe_flush()
                    continue

    # 收尾（防止日志最后一次 eval 没被 flush）
    if all(k in buf for k in ["semseg_miou", "human_miou", "saliency_miou", "normals_rmse"]):
        records.append({
            "epoch": cur_epoch,
            **buf
        })

    return records

def main():
    args = parse_args()
    st = load_st_baseline(args)
    recs = parse_log(args.log_file)
    if not recs:
        raise SystemExit(f"未在日志中解析到任何完整的 eval 记录（四项指标）。请检查格式/正则。")

    # 计算 Δm
    out_rows: List[Tuple[int, float, Dict[str, float]]] = []
    for r in recs:
        cur = {
            "semseg_miou": r["semseg_miou"],
            "human_miou": r["human_miou"],
            "saliency_miou": r["saliency_miou"],
            "normals_rmse": r["normals_rmse"],
        }
        d = compute_delta_m(cur, st)
        out_rows.append((r.get("epoch", -1), d, cur))

    # 输出每次 eval
    print("Per-eval Δm(%)：")
    for ep, d, cur in out_rows:
        ep_str = f"epoch {ep}" if ep is not None and ep >= 0 else "(epoch 未捕获)"
        print(f"- {ep_str}: Δm = {d:.3f}% | "
              f"SemSeg {cur['semseg_miou']:.3f}, Human {cur['human_miou']:.3f}, "
              f"Sal {cur['saliency_miou']:.3f}, Normals {cur['normals_rmse']:.4f}")

    # 最大值
    best = max(out_rows, key=lambda x: x[1])
    bep, bd, bcur = best
    bep_str = f"{bep}" if bep is not None and bep >= 0 else "N/A"
    print("\n=== Best Δm(%) ===")
    print(f"Epoch: {bep_str}")
    print(f"Δm: {bd:.3f}%")
    print(f"SemSeg mIoU: {bcur['semseg_miou']:.3f}")
    print(f"Human Parts mIoU: {bcur['human_miou']:.3f}")
    print(f"Saliency mIoU: {bcur['saliency_miou']:.3f}")
    print(f"Normals rmse: {bcur['normals_rmse']:.4f}")

    # 可选导出 CSV
    if args.csv_out is not None:
        try:
            import csv
            with args.csv_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "SemSeg_mIoU", "Human_mIoU", "Saliency_mIoU", "Normals_rmse", "Delta_m_percent"])
                for ep, d, cur in out_rows:
                    writer.writerow([
                        ep, cur["semseg_miou"], cur["human_miou"], cur["saliency_miou"], cur["normals_rmse"], d
                    ])
                # ⚠️ 额外写入 best 结果
                writer.writerow([])
                writer.writerow([
                    f"BEST(epoch {bep})",  # 这里带上 epoch
                    bcur["semseg_miou"], bcur["human_miou"],
                    bcur["saliency_miou"], bcur["normals_rmse"], bd
                ])

            print(f"\nCSV 已写出：{args.csv_out}")
        except Exception as e:
            print(f"[WARN] 写 CSV 失败：{e}")

if __name__ == "__main__":
    main()
