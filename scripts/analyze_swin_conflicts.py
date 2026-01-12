#!/usr/bin/env python3
# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
#
# Conflict analysis script inspired by Recon.
# --------------------------------------------------------

import argparse
import json
import os
import sys
from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config
from data import build_loader
from models import build_model, build_mtl_model
from mtl_loss_schemes import get_loss


def parse_options():
    parser = argparse.ArgumentParser(
        description="Analyze per-block task conflicts for Swin-Tiny.")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to config file")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    parser.add_argument("--batch-size", type=int,
                        help="Batch size per GPU")
    parser.add_argument("--data-path", type=str,
                        help="Path to dataset")
    parser.add_argument("--nyud", type=str,
                        help="Path to NYUD dataset")
    parser.add_argument("--pascal", type=str,
                        help="Path to PASCAL dataset")
    parser.add_argument("--tasks", type=str,
                        default="semseg,human_parts,sal,normals",
                        help="Comma-separated task list")
    parser.add_argument(
        "--grad-target",
        type=str,
        default="shared",
        choices=["shared", "lora_shared"],
        help="Compute gradients over shared parameters or shared LoRA only.",
    )
    parser.add_argument("--num-batches", type=int, default=1,
                        help="Number of batches to average over")
    parser.add_argument("--output", type=str, default="output",
                        help="Root output folder")
    parser.add_argument("--tag", type=str, default="conflict_analysis",
                        help="Experiment tag")
    parser.add_argument("--name", type=str,
                        help="Override model name")
    parser.add_argument("--cache-mode", type=str, default="part",
                        choices=['no', 'full', 'part'],
                        help="Cache mode")
    parser.add_argument("--ckpt-freq", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--eval-training-freq", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--zip", action='store_true')
    parser.add_argument("--pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--resume-backbone")
    parser.add_argument("--freeze-backbone", action='store_true')
    parser.add_argument("--skip_initial_validation", action='store_true')
    parser.add_argument("--decoder_map")
    parser.add_argument("--skip_decoder", action='store_true')
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--use-checkpoint", action='store_true')
    parser.add_argument("--amp-opt-level")
    parser.add_argument("--disable_amp", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--throughput", action='store_true')
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--fused_window_process", action='store_true')
    parser.add_argument("--fused_layernorm", action='store_true')
    parser.add_argument("--optim")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mti", action='store_true')
    parser.add_argument("--save_sample", action='store_true')
    return parser.parse_args()


def _sanitize_tasks(tasks):
    normalized = []
    for task in tasks:
        if task == "normal":
            normalized.append("normals")
        else:
            normalized.append(task)
    return normalized


def _collect_swin_blocks(model):
    backbone = model.backbone if hasattr(model, "backbone") else model
    if not hasattr(backbone, "layers"):
        raise ValueError("Backbone has no Swin layers to analyze.")
    blocks = []
    for layer_idx, layer in enumerate(backbone.layers):
        for block_idx, block in enumerate(layer.blocks):
            block_name = f"layer{layer_idx}.block{block_idx}"
            blocks.append((block_name, block))
    return blocks


def _filter_shared_params(block, grad_target):
    params = []
    for name, param in block.named_parameters(recurse=True):
        if not param.requires_grad:
            continue
        if grad_target == "shared":
            if "lora_tasks_" in name or "lora_task_scale" in name:
                continue
            params.append(param)
        else:
            if "lora_tasks_" in name or "lora_task_scale" in name:
                continue
            if "lora_shared_" in name or name.endswith(".lora_A") or name.endswith(".lora_B"):
                params.append(param)
    return params


def _flatten_grads(grads):
    flat = []
    for grad in grads:
        if grad is None:
            continue
        flat.append(grad.contiguous().view(-1))
    if not flat:
        return None
    return torch.cat(flat)


def _pair_cos(vec_a, vec_b, eps=1e-12):
    if vec_a is None or vec_b is None:
        return 0.0, False
    norm_a = torch.norm(vec_a)
    norm_b = torch.norm(vec_b)
    if norm_a.item() <= eps or norm_b.item() <= eps:
        return 0.0, False
    vec_a = vec_a / (norm_a + eps)
    vec_b = vec_b / (norm_b + eps)
    return torch.dot(vec_a, vec_b).item(), True


def _build_loss_functions(config):
    loss_ft = torch.nn.ModuleDict(
        {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS}
    )
    all_loss_weights = {
        'depth': 1.0,
        'semseg': 1.0,
        'human_parts': 2.0,
        'sal': 5.0,
        'edge': 50.0,
        'normals': 10.0,
    }
    loss_weights = {task: all_loss_weights[task] for task in config.TASKS}
    return loss_ft, loss_weights


def _compute_batch_losses(model, loss_ft, loss_weights, samples, targets):
    preds = model(samples)
    losses = {}
    for task, loss_fn in loss_ft.items():
        losses[task] = loss_fn(preds[task], targets[task]) * loss_weights[task]
    return losses


def analyze_conflicts(config, num_batches, grad_target):
    dataset_train, _, data_loader_train, _, _ = build_loader(config)
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)
    model.cuda()
    model.train()

    loss_ft, loss_weights = _build_loss_functions(config)
    loss_ft = loss_ft.cuda()

    blocks = _collect_swin_blocks(model)
    tasks = list(config.TASKS)
    pairs = list(combinations(tasks, 2))

    pair_scores = {
        block_name: {f"{a}|{b}": 0.0 for a, b in pairs}
        for block_name, _ in blocks
    }
    pair_counts = {
        block_name: {f"{a}|{b}": 0 for a, b in pairs}
        for block_name, _ in blocks
    }

    for batch_idx, batch in enumerate(data_loader_train):
        if batch_idx >= num_batches:
            break
        samples = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in config.TASKS}

        losses = _compute_batch_losses(model, loss_ft, loss_weights, samples, targets)

        for block_name, block in blocks:
            params = _filter_shared_params(block, grad_target)
            if not params:
                continue
            task_grads = {}
            for task in tasks:
                grads = torch.autograd.grad(
                    losses[task],
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                task_grads[task] = _flatten_grads(grads)
            for task_a, task_b in pairs:
                key = f"{task_a}|{task_b}"
                cos_val, valid = _pair_cos(task_grads[task_a], task_grads[task_b])
                if valid:
                    pair_scores[block_name][key] += cos_val
                    pair_counts[block_name][key] += 1

    summary = OrderedDict()
    for block_name, _ in blocks:
        block_entry = OrderedDict()
        block_pair_scores = pair_scores[block_name]
        block_pair_counts = pair_counts[block_name]
        mean_scores = []
        for key, total in block_pair_scores.items():
            count = block_pair_counts[key]
            avg = total / count if count > 0 else 0.0
            block_entry[key] = avg
            mean_scores.append(avg)
        block_entry["mean_pair_cos"] = float(sum(mean_scores) / len(mean_scores))
        summary[block_name] = block_entry

    return summary


def _write_outputs(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "swin_block_conflicts.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "swin_block_conflicts.csv")
    with open(csv_path, "w") as f:
        header_keys = []
        if summary:
            sample_block = next(iter(summary.values()))
            header_keys = list(sample_block.keys())
        f.write("block," + ",".join(header_keys) + "\n")
        for block, values in summary.items():
            row = [block] + [str(values[key]) for key in header_keys]
            f.write(",".join(row) + "\n")

    return json_path, csv_path


def main():
    args = parse_options()
    if args.tasks:
        tasks = _sanitize_tasks(args.tasks.split(","))
        args.tasks = ",".join(tasks)
    config = get_config(args)
    if not config.MTL:
        raise ValueError("This analysis requires --tasks to enable MTL.")

    summary = analyze_conflicts(config, args.num_batches, args.grad_target)
    output_dir = config.OUTPUT
    json_path, csv_path = _write_outputs(summary, output_dir)

    print("Per-block conflict summary (mean pairwise cosine):")
    for block_name, values in summary.items():
        print(f"{block_name}: {values['mean_pair_cos']:.4f}")
    print(f"Saved JSON to {json_path}")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
