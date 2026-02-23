#!/usr/bin/env python3
"""Export NYUD depth predictions to files for offline metric audits.

Example:
python tools/export_depth_predictions.py \
  --cfg configs/swin/tiny_448/nyud/swin_tiny_patch4_window7_448_nyud_alltask.yaml \
  --nyud NYUv2_MT \
  --tasks semseg,normals,depth \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth \
  --output-dir debug_preds/depth \
  --save-ext npy
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict

import numpy as np
import torch

from config import get_config
from data.mtl_ds import get_transformations, get_mtl_val_dataset, get_mtl_val_dataloader
from models import build_model, build_mtl_model

try:
    import scipy.io as sio
except Exception:
    sio = None


LOGGER = logging.getLogger("export_depth")


def _make_config_args(args: argparse.Namespace) -> SimpleNamespace:
    keys: Dict[str, object] = {
        "cfg": args.cfg,
        "opts": args.opts,
        "batch_size": args.batch_size,
        "ckpt_freq": None,
        "eval_freq": None,
        "skip_initial_validation": False,
        "eval_training_freq": None,
        "epochs": None,
        "mti": None,
        "decoder_map": None,
        "skip_decoder": False,
        "data_path": None,
        "nyud": args.nyud,
        "pascal": None,
        "tasks": args.tasks,
        "zip": False,
        "cache_mode": None,
        "pretrained": None,
        "resume": None,
        "resume_backbone": None,
        "freeze_backbone": False,
        "save_sample": False,
        "accumulation_steps": None,
        "use_checkpoint": False,
        "amp_opt_level": None,
        "disable_amp": args.disable_amp,
        "output": args.output_root,
        "tag": args.tag,
        "eval": False,
        "throughput": False,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "debug_repro_steps": None,
        "enable_amp": args.enable_amp,
        "fused_window_process": False,
        "fused_layernorm": False,
        "optim": None,
        "name": None,
        "local_rank": 0,
    }
    return SimpleNamespace(**keys)


def _filter_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    module_state = module.state_dict()
    out = {}
    for k, v in state_dict.items():
        if k in module_state and module_state[k].shape == v.shape:
            out[k] = v
    return out


def _try_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_weights(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if not args.resume and not args.resume_backbone:
        LOGGER.warning("No --resume or --resume-backbone provided, exporting with random/uninitialized decoder weights.")
        return

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state = _filter_state_dict(model, state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        LOGGER.info("Loaded --resume=%s (matched keys=%d, missing=%d, unexpected=%d)", args.resume, len(state), len(missing), len(unexpected))

    if args.resume_backbone:
        ckpt = torch.load(args.resume_backbone, map_location="cpu")
        state = ckpt.get("model", ckpt)

        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # Try direct backbone keys first.
        cand = _filter_state_dict(model.backbone, state)
        if not cand:
            # Try checkpoint saved from full MTL model.
            stripped = _try_strip_prefix(state, "backbone.")
            cand = _filter_state_dict(model.backbone, stripped)

        missing, unexpected = model.backbone.load_state_dict(cand, strict=False)
        LOGGER.info("Loaded --resume-backbone=%s (matched keys=%d, missing=%d, unexpected=%d)", args.resume_backbone, len(cand), len(missing), len(unexpected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--nyud", required=True, type=str, help="Path to NYUD root containing image/, depth/, gt_sets/")
    parser.add_argument("--tasks", default="semseg,normals,depth", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--resume", default="", type=str, help="Full model checkpoint")
    parser.add_argument("--resume-backbone", default="", type=str, help="Backbone checkpoint")
    parser.add_argument("--output-dir", required=True, type=str, help="Folder to save per-image depth predictions")
    parser.add_argument("--save-ext", choices=["npy", "mat"], default="npy")
    parser.add_argument("--mat-key", default="depth", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-batches", default=0, type=int, help="0 means all")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--enable-amp", default=None, type=str)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-root", default="output", type=str)
    parser.add_argument("--tag", default="export_depth", type=str)
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    if args.save_ext == "mat" and sio is None:
        raise RuntimeError("scipy is required when --save-ext mat")

    cfg_args = _make_config_args(args)
    config = get_config(cfg_args)

    if "depth" not in config.TASKS:
        raise ValueError(f"Current tasks {config.TASKS} do not include 'depth'.")

    config.defrost()
    config.DATA.NUM_WORKERS = args.num_workers
    config.DATA.BATCH_SIZE = args.batch_size
    config.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)
    model.to(device)
    model.eval()

    _load_weights(model, args)

    _, val_transforms = get_transformations("NYUD", config.TASKS_CONFIG)
    val_dataset = get_mtl_val_dataset("NYUD", config, val_transforms)
    val_loader = get_mtl_val_dataloader(config, val_dataset)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.disable_amp)):
                output = model(images)

            depth_pred = output["depth"].detach().float().cpu().squeeze(1).numpy()
            image_ids = batch["meta"]["image"]

            for idx, image_id in enumerate(image_ids):
                arr = depth_pred[idx]
                save_path = out_dir / f"{image_id}.{args.save_ext}"
                if args.save_ext == "npy":
                    np.save(save_path, arr.astype(np.float32))
                else:
                    sio.savemat(save_path, {args.mat_key: arr.astype(np.float32)})
                saved += 1

            if (bi + 1) % 20 == 0:
                LOGGER.info("Processed %d batches, saved %d predictions", bi + 1, saved)

    LOGGER.info("Done. Saved %d depth predictions to %s", saved, out_dir)


if __name__ == "__main__":
    main()
