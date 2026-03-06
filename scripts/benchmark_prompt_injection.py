#!/usr/bin/env python3
import argparse
import time
import os
import sys
from dataclasses import dataclass

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from config import get_config
from models import build_model, build_mtl_model
from mtl_loss_schemes import MultiTaskLoss, get_loss
from optimizer import build_optimizer


@dataclass
class BenchmarkResult:
    config_path: str
    sec_per_iter: float
    iter_per_sec: float
    peak_memory_gb: float


def _build_config(cfg_path, args):
    config_args = argparse.Namespace(
        cfg=cfg_path,
        opts=None,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        disable_amp=args.disable_amp,
        amp_opt_level=None,
        output=None,
        tag=None,
        eval=False,
        throughput=False,
        local_rank=0,
        local_rank_alias=0,
        fused_window_process=False,
        fused_layernorm=False,
        optim=None,
        tasks=args.tasks,
        nyud=args.data_path if args.dataset == "nyud" else None,
        pascal=args.data_path if args.dataset == "pascal" else None,
        data_path=None,
        epochs=None,
        ckpt_freq=None,
        eval_freq=None,
        pretrained=None,
        resume=None,
        use_checkpoint=False,
        name=None,
        decoder_map=None,
        skip_decoder=False,
        freeze_backbone=False,
        skip_initial_validation=False,
        eval_training_freq=None,
        resume_backbone=None,
        disable_wandb=False,
        run_name=None,
        no_eval_50=True,
        enable_amp=False,
    )
    config = get_config(config_args)
    config.defrost()
    if args.enable_amp is not None:
        config.AMP_ENABLE = args.enable_amp
    config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    config.DATA.BATCH_SIZE = args.batch_size
    config.freeze()
    return config


def _build_loss(config):
    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config["TASKS_CONFIG"], task, config) for task in config.TASKS}
        )
        all_loss_weights = {
            "depth": 1.0,
            "semseg": 1.0,
            "human_parts": 2.0,
            "sal": 5.0,
            "edge": 50.0,
            "normals": 10.0,
        }
        loss_weights = {task: all_loss_weights[task] for task in config.TASKS}
        return MultiTaskLoss(config.TASKS, loss_ft, loss_weights)
    return torch.nn.CrossEntropyLoss()


def _build_targets(config, device, batch_size, img_size):
    if not config.MTL:
        return torch.randint(
            0, config.MODEL.NUM_CLASSES, (batch_size,), device=device, dtype=torch.long
        )

    targets = {}
    for task in config.TASKS:
        num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT[task]
        if task in {"semseg", "human_parts"}:
            targets[task] = torch.randint(
                0, num_outputs, (batch_size, 1, img_size, img_size), device=device, dtype=torch.long
            )
        elif task == "normals":
            targets[task] = torch.rand(
                batch_size, num_outputs, img_size, img_size, device=device
            ) * 2.0 - 1.0
        elif task in {"edge", "sal", "depth"}:
            targets[task] = torch.rand(
                batch_size, num_outputs, img_size, img_size, device=device
            )
        else:
            raise ValueError(f"Unsupported task {task} for synthetic targets.")
    return targets


def _run_benchmark(config, args):
    device = torch.device("cuda")
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)
    model.to(device)
    model.train()

    optimizer = build_optimizer(config, model)
    criterion = _build_loss(config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLE)

    batch_size = config.DATA.BATCH_SIZE
    img_size = config.DATA.IMG_SIZE
    inputs = torch.randn(batch_size, 3, img_size, img_size, device=device)
    targets = _build_targets(config, device, batch_size, img_size)

    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(inputs)
            if config.MTL:
                loss, _ = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.timed_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(inputs)
            if config.MTL:
                loss, _ = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append(end - start)

    sec_per_iter = sum(timings) / len(timings)
    iter_per_sec = 1.0 / sec_per_iter if sec_per_iter > 0 else float("inf")
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
    return BenchmarkResult(
        config_path=args.current_config,
        sec_per_iter=sec_per_iter,
        iter_per_sec=iter_per_sec,
        peak_memory_gb=peak_memory_gb,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark prompt injection overhead with synthetic inputs."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config file paths to benchmark.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="semseg,normals,sal,human_parts,edge",
        help="Comma-separated task list (required for MTL configs).",
    )
    parser.add_argument(
        "--dataset",
        choices=["nyud", "pascal"],
        default="pascal",
        help="Dataset name to infer task output shapes.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=".",
        help="Dataset path (only used to set dataset type in config).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--timed-steps", type=int, default=200)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        default=None,
        help="Enable AMP for all configs.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        default=None,
        help="Disable AMP for all configs.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.enable_amp and args.disable_amp:
        raise ValueError("Choose only one of --enable-amp or --disable-amp.")
    if args.disable_amp:
        args.enable_amp = False

    results = []
    for cfg_path in args.configs:
        args.current_config = cfg_path
        config = _build_config(cfg_path, args)
        result = _run_benchmark(config, args)
        results.append(result)
        torch.cuda.empty_cache()

    print("\nBenchmark results:")
    for result in results:
        print(
            f"{result.config_path} | sec/iter {result.sec_per_iter:.6f} | "
            f"it/s {result.iter_per_sec:.3f} | peak mem {result.peak_memory_gb:.3f} GB"
        )


if __name__ == "__main__":
    main()
