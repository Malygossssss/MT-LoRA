# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper

from mtl_loss_schemes import MultiTaskLoss, get_loss
from evaluation.evaluate_utils import PerformanceMeter, get_output
from ptflops import get_model_complexity_info
from models.lora import mark_only_lora_as_trainable
from models.dora_mtlora import prune_lora_scaler as dora_prune_lora_scaler
from models.dora_mtlora import regularization_loss as dora_regularization_loss

try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb library not found. Logging is disabled.")
    wandb_available = False


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=5,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='List of tasks to run in MTL setup.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--decoder_map', type=str,
                        help='Path to JSON file containing the type of decoder heads')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Skip loading decoder head weights')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--run_name', type=str,
                        help='wandb run name')
    parser.add_argument('--no_eval_50', action='store_false',
                        help='Disable the iniital eval at 50 epochs.')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    teacher = None
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {n_parameters / 1e6} M")

    model.cuda()
    macs, params = get_model_complexity_info(model, (3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)

    logger.info(f"ptflops GMACS = {macs} and params = {params}")

    model_without_ddp = model

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(
            data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
        all_loss_weights = {
            'depth': 1.0,
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'edge': 50.0,
            'normals': 10.0,
        }
        loss_weights = {}
        for t in config.TASKS:
            loss_weights[t] = all_loss_weights[t]

        criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)

        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model, 0)
        if config.EVAL_MODE:
            return

    if config.MODEL.RESUME_BACKBONE:
        max_accuracy = load_checkpoint(
            config, model_without_ddp.backbone, optimizer, lr_scheduler, loss_scaler, logger, True)
        if config.EVAL_MODE:
            validate(config, data_loader_val, model, 0)
            return

    if config.EVAL_MODE:
        validate(config, data_loader_val, model, 0)
        return

    if teacher is not None:
        print("loading teacher.......")
        load_checkpoint(config, teacher, optimizer, lr_scheduler,
                        loss_scaler, logger, quiet=True)

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if not config.SKIP_INITIAL_EVAL:
            acc1, _, _ = validate(config, data_loader_val, model, 0)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    if config.MODEL.MTLORA.ENABLED:
        if config.MODEL.MTLORA.FREEZE_PRETRAINED:
            print("\nMarking LoRA params only as trainable:")
            mark_only_lora_as_trainable(model.backbone,
                                        bias=config.MODEL.MTLORA.BIAS,
                                        freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
                                        freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
                                        free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
                                        freeze_downsample_reduction=True if config.MODEL.MTLORA.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION)
            if config.MODEL.PROMPT.ENABLED:
                for name, param in model.backbone.named_parameters():
                    if (
                        "prompt_embeddings" in name
                        or "deep_prompt_embeddings" in name
                        or "deep_prompt_pools" in name
                        or "deep_prompt_gates" in name
                    ):
                        param.requires_grad = True
        else:
            print("Marking all layers as trainable")
    if config.MODEL.FREEZE_BACKBONE:
        assert (not config.MODEL.MTLORA.ENABLED)
        print("Freezing backbone.........")
        model.freeze_backbone()
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters()
                      if p.requires_grad and 'lora' in name)
    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_params_without_lora = total_model_params - lora_params
    decoder_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'backbone' not in name)

    logger.info(
        f"\nNumber of trainable params: {trainable_params:,}\n"
        f"Decoder params:             {decoder_params:,}\n"
        f"LoRA params:                {lora_params:,}\n"
        f"Extra params:                {(trainable_params - (lora_params + decoder_params)):,}\n"
        f"Total params:               {total_model_params:,} (trainable ratio: {trainable_params/total_model_params * 100:2.2f}%)\n"
        f"Total params without LoRA:  {total_model_params_without_lora:,} (trainable ratio: {trainable_params/total_model_params_without_lora * 100:2.2f}%)"
    )
    logger.info("Start training")
    start_time = time.perf_counter()

    record_cr_values = config.TRAIN.ENABLE_CONFLICT_RATIO and dist.get_rank() == 0
    all_batch_cr_records = [] if record_cr_values else None
    total_cr_sum = 0.0
    total_cr_count = 0.0

    epoch = 0
    for epoch in range(config.TRAIN.EPOCHS):
        if not config.MTL:
            data_loader_train.sampler.set_epoch(epoch)

        epoch_cr_records, epoch_cr_sum, epoch_cr_count = train_one_epoch(config, model, criterion, data_loader_train,
                                                                         optimizer, epoch, mixup_fn, lr_scheduler,
                                                                         loss_scaler, teacher=teacher)
        if config.TRAIN.ENABLE_CONFLICT_RATIO:
            total_cr_sum += epoch_cr_sum
            total_cr_count += epoch_cr_count
            if record_cr_values and epoch_cr_records:
                all_batch_cr_records.extend(epoch_cr_records)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        if epoch % config.EVAL_FREQ == 0 or (not args.no_eval_50 and epoch == 50):
            if config.MTL:
                validate(config, data_loader_val, model, epoch)
            else:
                acc1, _, _ = validate(config, data_loader_val, model, epoch)
                max_accuracy = max(max_accuracy, acc1)

    if config.TRAIN.ENABLE_CONFLICT_RATIO:
        cr_device = next(model.parameters()).device
        cr_tensor = torch.tensor([total_cr_sum, total_cr_count], dtype=torch.float64, device=cr_device)
        if dist.is_initialized():
            dist.all_reduce(cr_tensor, op=dist.ReduceOp.SUM)
        total_count_value = cr_tensor[1].item()
        global_avg_cr = (cr_tensor[0] / total_count_value).item() if total_count_value > 0 else 0.0
        if dist.get_rank() == 0:
            os.makedirs(config.OUTPUT, exist_ok=True)
            cr_output_path = os.path.join(config.OUTPUT, "conflict_ratio.txt")
            with open(cr_output_path, "w") as f:
                if all_batch_cr_records:
                    for record in all_batch_cr_records:
                        f.write(
                            f"epoch={record['epoch']}, batch={record['batch']}, global_step={record['global_step']}, cr={record['cr']:.6f}\n"
                        )
                else:
                    f.write("# No conflict ratio values were recorded.\n")
                f.write(f"global_average_cr: {global_avg_cr:.6f}\n")
            logger.info(f"Conflict gradient ratio metrics saved to {cr_output_path}")

    # final eval
    validate(config, data_loader_val, model, epoch)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def _is_per_task_parameter(name, tasks):
    for task in tasks:
        if f".{task}." in name or name.endswith(f".{task}") or f".{task}_" in name or f"_{task}." in name:
            return True
    return False


def _get_shared_parameters_for_cr(model, config):
    tasks = set(config.TASKS) if hasattr(config, "TASKS") else set()
    shared_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("decoders."):
            continue
        if name.startswith("downsampler."):
            parts = name.split(".")
            if len(parts) > 1 and parts[1] in tasks:
                continue
        if tasks and _is_per_task_parameter(name, tasks):
            continue
        shared_params.append(param)
    return shared_params


def _compute_conflict_ratio(loss_dict, shared_params, task_order, task_weights):
    if loss_dict is None or not shared_params:
        return 0.0

    active_tasks = [task for task in task_order if task in loss_dict]
    if not active_tasks:
        return 0.0

    task_losses = {
        task: loss_dict[task] * float(task_weights.get(task, 1.0))
        for task in active_tasks
    }

    if not task_losses:
        return 0.0

    # Accumulate total gradient across tasks and per-task norms
    sum_total = [torch.zeros_like(param, dtype=torch.float32) for param in shared_params]
    task_norm_sq = {task: 0.0 for task in task_losses}

    for task, loss_value in task_losses.items():
        grads = torch.autograd.grad(loss_value, shared_params, retain_graph=True, allow_unused=True)
        norm_sq = 0.0
        for idx, (grad, accum) in enumerate(zip(grads, sum_total)):
            if grad is None:
                continue
            grad_detached = grad.detach()
            if grad_detached.dtype != torch.float32:
                grad_detached = grad_detached.float()
            accum.add_(grad_detached)
            norm_sq += torch.sum(grad_detached * grad_detached).item()
        task_norm_sq[task] = norm_sq

    sum_total_norm_sq = 0.0
    for accum in sum_total:
        sum_total_norm_sq += torch.sum(accum * accum).item()

    task_dot_with_sum = {}
    for task, loss_value in task_losses.items():
        grads = torch.autograd.grad(loss_value, shared_params, retain_graph=True, allow_unused=True)
        dot_val = 0.0
        for grad, accum in zip(grads, sum_total):
            if grad is None:
                continue
            grad_detached = grad.detach()
            if grad_detached.dtype != torch.float32:
                grad_detached = grad_detached.float()
            dot_val += torch.sum(grad_detached * accum).item()
        task_dot_with_sum[task] = dot_val

    eps = 1e-12
    ratios = []
    for task in task_losses:
        grad_norm_sq = task_norm_sq.get(task, 0.0)
        if grad_norm_sq <= eps:
            ratios.append(0.0)
            continue
        dot_val = task_dot_with_sum.get(task, 0.0) - grad_norm_sq
        other_norm_sq = sum_total_norm_sq - grad_norm_sq - 2.0 * dot_val
        if other_norm_sq <= eps:
            ratios.append(0.0)
            continue
        proj_norm_sq = (dot_val ** 2) / (other_norm_sq + eps)
        conflict_norm_sq = grad_norm_sq - proj_norm_sq
        if conflict_norm_sq < 0.0:
            conflict_norm_sq = 0.0
        ratio = math.sqrt(conflict_norm_sq / (grad_norm_sq + eps))
        ratios.append(ratio)

    if not ratios:
        return 0.0

    return float(sum(ratios) / len(ratios))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, task=None, teacher=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    compute_cr = bool(config.MTL and config.TRAIN.ENABLE_CONFLICT_RATIO and hasattr(criterion, 'loss_weights'))
    shared_params_for_cr = None
    task_order_for_cr = list(config.TASKS) if compute_cr else []
    task_weights_for_cr = {}
    cr_records = [] if compute_cr and dist.get_rank() == 0 else None
    cr_period = 1
    if compute_cr:
        period_value = getattr(config.TRAIN, 'CONFLICT_RATIO_PERIOD', 1)
        try:
            cr_period = int(period_value)
        except (TypeError, ValueError):
            cr_period = 1
        if cr_period < 1:
            cr_period = 1
    cr_sum = 0.0
    cr_count = 0.0

    if compute_cr:
        shared_params_for_cr = _get_shared_parameters_for_cr(model, config)
        if not shared_params_for_cr:
            compute_cr = False
            if dist.get_rank() == 0:
                logger.warning("Conflict ratio computation disabled because no shared parameters were found.")
        else:
            task_weights_for_cr = {
                task: float(criterion.loss_weights.get(task, 1.0))
                for task in task_order_for_cr if task in criterion.loss_weights
            }
            if not task_weights_for_cr:
                compute_cr = False
                if dist.get_rank() == 0:
                    logger.warning("Conflict ratio computation disabled because no task weights were available.")
            elif cr_records is not None:
                cr_records = []

    for idx, batch in enumerate(data_loader):
        global_step = epoch * num_steps + idx + 1
        max_step = config.TRAIN.EPOCHS * num_steps
        if not config.MTL:
            samples, targets = batch
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            samples = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(
                non_blocking=True) for task in config.TASKS}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss, loss_dict = criterion(outputs, targets)
            if config.MODEL.MTLORA.ADAPTIVE:
                reg = dora_regularization_loss(model, global_step, max_step, config.MODEL.MTLORA)
                loss = loss + config.MODEL.MTLORA.REGULARIZATION_LOSS_ALPHA * reg

        should_measure_cr = False
        if compute_cr:
            should_measure_cr = ((idx + 1) % cr_period == 0) or (idx == num_steps - 1)

        if should_measure_cr:
            batch_cr = _compute_conflict_ratio(loss_dict, shared_params_for_cr, task_order_for_cr, task_weights_for_cr)
            cr_sum += batch_cr
            cr_count += 1
            if cr_records is not None:
                cr_records.append({
                    'epoch': epoch,
                    'batch': idx,
                    'global_step': global_step,
                    'cr': batch_cr,
                })

        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if config.MODEL.MTLORA.ADAPTIVE:
                dora_prune_lora_scaler(model, global_step, max_step, config.MODEL.MTLORA)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # torch.cuda.synchronize()

        if not config.MTL:
            loss_meter.update(loss.item(), targets.size(0))
        else:
            loss_meter.update(loss.item())

        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        if wandb_available:
            metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": idx,
                "train/train_loss": loss_meter.val,
                "train/train_loss_avg": loss_meter.avg,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/weight_decay": optimizer.param_groups[0]['weight_decay'],
                "train/time": batch_time.val,
                "train/time_avg": batch_time.avg,
                "train/grad_norm": norm_meter.val,
                "train/grad_norm_avg": norm_meter.avg,
                "train/loss_scale": scaler_meter.val,
                "train/loss_scale_avg": scaler_meter.avg,
                "train/memory": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"train/tasks/{task}/loss"] = task_loss.item()
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        performance_meter.update(
            {t: get_output(outputs[t], t) for t in config.TASKS}, targets)

        scores = performance_meter.get_score(verbose=True)
        if wandb_available:
            scores_logs = {
                "train/epoch": epoch,
            }
            if 'semseg' in scores:
                scores_logs["train/tasks/semseg/mIoU"] = scores['semseg']['mIoU']
            if 'normals' in loss_dict:
                scores_logs["train/tasks/normals/mean"] = scores['normals']['mean']
                scores_logs["train/tasks/normals/rmse"] = scores['normals']['rmse']
                scores_logs["train/tasks/normals/mean_v2"] = scores['normals']['mean_v2']
                scores_logs["train/tasks/normals/rmse_v2"] = scores['normals']['rmse_v2']
            if 'human_parts' in loss_dict:
                scores_logs["train/tasks/human_parts/mIoU"] = scores['human_parts']['mIoU']
            if 'sal' in loss_dict:
                scores_logs["train/tasks/sal/maxF"] = scores['sal']['maxF']
                scores_logs["train/tasks/sal/Beta maxF"] = scores['sal']['Beta maxF']
                scores_logs["train/tasks/sal/mIoU"] = scores['sal']['mIoU']
            if 'edge' in loss_dict:
                scores_logs["train/tasks/sal/loss"] = scores['edge']['loss']
            if 'depth' in loss_dict:
                scores_logs["train/tasks/depth/rmse"] = scores['depth']['rmse']
                scores_logs["train/tasks/depth/log_rmse"] = scores['depth']['log_rmse']

            wandb.log(scores_logs)

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return (cr_records if compute_cr else None, cr_sum if compute_cr else 0.0, cr_count if compute_cr else 0.0)


@torch.no_grad()
def validate(config, data_loader, model, epoch):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = config.TASKS
    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)
    loss_meter = AverageMeter()

    loss_ft = torch.nn.ModuleDict(
        {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
    all_loss_weights = {
        'depth': 1.0,
        'semseg': 1.0,
        'human_parts': 2.0,
        'sal': 5.0,
        'edge': 50.0,
        'normals': 10.0,
    }
    loss_weights = {}
    for t in config.TASKS:
        loss_weights[t] = all_loss_weights[t]
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    model.eval()
    num_val_points = 0
    logger.info("Start eval")
    start = time.perf_counter()
    loss_dict = None
    for i, batch in enumerate(data_loader):
        # Forward pass
        logger.debug(f"Image ID = {batch['meta']['image']}")
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(
            non_blocking=True) for task in tasks}

        output = model(images)

        num_val_points += 1

        # Measure performance
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            loss, loss_dict = criterion(
                output, targets)
            loss_meter.update(loss.item())
        processed_output = {t: get_output(
            output[t], t) for t in tasks}
        performance_meter.update(processed_output, targets)
        if wandb_available:
            metrics = {
                "val/epoch_ndx": epoch,
                "val/batch_ndx": i,
                "val/val_loss": loss_meter.val,
                "val/val_loss_avg": loss_meter.avg,
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"val/tasks/{task}/loss"] = task_loss.item()
            wandb.log(metrics)

    logger.info(f"val loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t")

    eval_results = performance_meter.get_score(verbose=True)
    epoch_time = time.perf_counter() - start
    logger.info(
        f"eval takes {datetime.timedelta(seconds=int(epoch_time))}")
    if wandb_available:
        scores_logs = {
            "val/epoch": epoch,
        }
        if 'semseg' in eval_results:
            scores_logs["val/tasks/semseg/mIoU"] = eval_results['semseg']['mIoU']
        if 'normals' in eval_results:
            scores_logs["val/tasks/normals/mean"] = eval_results['normals']['mean']
            scores_logs["val/tasks/normals/rmse"] = eval_results['normals']['rmse']
            scores_logs["val/tasks/normals/mean_v2"] = eval_results['normals']['mean_v2']
            scores_logs["val/tasks/normals/rmse_v2"] = eval_results['normals']['rmse_v2']
        if 'human_parts' in eval_results:
            scores_logs["val/tasks/human_parts/mIoU"] = eval_results['human_parts']['mIoU']
        if 'sal' in eval_results:
            scores_logs["val/tasks/sal/maxF"] = eval_results['sal']['maxF']
            scores_logs["val/tasks/sal/Beta maxF"] = eval_results['sal']['Beta maxF']
            scores_logs["val/tasks/sal/mIoU"] = eval_results['sal']['mIoU']
        if 'edge' in eval_results:
            scores_logs["val/tasks/sal/loss"] = eval_results['edge']['loss']
        if 'depth' in eval_results:
            scores_logs["val/tasks/depth/rmse"] = eval_results['depth']['rmse']
            scores_logs["val/tasks/depth/log_rmse"] = eval_results['depth']['log_rmse']

        wandb.log(scores_logs)

    return eval_results


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        # torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.perf_counter()
        for i in range(30):
            model(images)
        # torch.cuda.synchronize()
        tic2 = time.perf_counter()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    eval_logger = create_logger(output_dir=config.OUTPUT,
                                dist_rank=dist.get_rank(), name="eval")
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.disable_wandb:
        wandb_available = False
        logger.info("Wandb logging disabled.")
    elif wandb_available:
        try:
            if not os.getenv("WANDB_API_KEY"):
                wandb.login()
            else:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
            config_name = f"{os.path.basename(os.path.dirname(args.cfg))}/{os.path.basename(args.cfg)}"
            wandb.init(project='MTLoRA', config=config,
                       name=config_name if not args.run_name else args.run_name)
            wandb.config.update({'args': vars(args)})
        except wandb.exc.LaunchError:
            logger.warnning("Could not initialize wandb. Logging is disabled.")
            wandb_available = False

    main(config)
