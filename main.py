# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn import Module
from torch.nn.utils.stateless import functional_call
from collections import OrderedDict

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
# from models.lora import mark_only_lora_as_trainable
from models.lora import (
    mark_only_lora_as_trainable,
    consolidate_task_lora_to_shared,
    print_lora_layer_summary,
    freeze_task_specific_lora,
)

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

    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    teacher = None
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    if getattr(config.MODEL.MTLORA, 'METASGD_MODE', False):
        init_meta_sgd_lrs(model, config.MODEL.MTLORA.METASGD_INIT,
                          freeze_ts=config.MODEL.MTLORA.FREEZE_TS_LORA)

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
            if config.MODEL.MTLORA.FREEZE_TS_LORA:
                freeze_task_specific_lora(model.backbone)
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

    print(f"""
Number of trainable params: {trainable_params:,}
Decoder params:             {decoder_params:,}
LoRA params:                {lora_params:,}
Extra params:                {(trainable_params - (lora_params + decoder_params)):,}
Total params:               {total_model_params:,} (trainable ratio: {trainable_params/total_model_params * 100:2.2f}%)
Total params without LoRA:  {total_model_params_without_lora:,} (trainable ratio: {trainable_params/total_model_params_without_lora * 100:2.2f}%)
""")
    # print_lora_layer_summary(model)
    logger.info("Start training")
    # for name, param in model.named_parameters():
    #     if 'lora_shared_' in name:
    #         print(name)
    start_time = time.perf_counter()

    epoch = 0
    for epoch in range(config.TRAIN.EPOCHS):
        if not config.MTL:
            data_loader_train.sampler.set_epoch(epoch)

        #⭐Meta-learning logic
        if getattr(config.MODEL.MTLORA, 'REPTILE_MODE', False):
            print("--------------------REPTILE---------------------")
            reptile_train_one_epoch(
                config, model, criterion, data_loader_train, optimizer, epoch,
                mixup_fn, lr_scheduler, loss_scaler, teacher=teacher
            )
        elif getattr(config.MODEL.MTLORA, 'METASGD_MODE', False):
            print("--------------------Meta-SGD---------------------")
            meta_sgd_train_one_epoch(
                config, model, criterion, data_loader_train, optimizer, epoch,
                mixup_fn, lr_scheduler, loss_scaler, teacher=teacher
            )
        elif config.MODEL.MTLORA.MAML_MODE:
            print("--------------------MAML---------------------")
            maml_train_one_epoch(
                config, model, criterion, data_loader_train, optimizer, epoch,
                mixup_fn, lr_scheduler, loss_scaler, teacher=teacher
            )
        else:
            print("--------------------MT-LORA---------------------")
            train_one_epoch(
                config, model, criterion, data_loader_train, optimizer, epoch,
                mixup_fn, lr_scheduler, loss_scaler, teacher=teacher)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        if epoch % config.EVAL_FREQ == 0 or (not args.no_eval_50 and epoch == 50):
            if config.MTL:
                validate(config, data_loader_val, model, epoch)
            else:
                acc1, _, _ = validate(config, data_loader_val, model, epoch)
                max_accuracy = max(max_accuracy, acc1)

    # final eval
    validate(config, data_loader_val, model, epoch)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

class _TaskWrapper(Module):
    """Simple wrapper to expose ``forward_task`` as ``forward`` for functional_call."""

    def __init__(self, model: Module, task: str):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        return self.model.forward_task(x, self.task)[self.task]


def init_meta_sgd_lrs(model: Module, init_lr: float, freeze_ts: bool = False) -> None:
    """Attach per-parameter learnable inner-loop lrs to ``model`` for Meta-SGD."""

    lora_named_params = [
        (n, p)
        for n, p in model.named_parameters()
        if 'lora_shared_' in n or (not freeze_ts and 'lora_tasks_' in n)
    ]
    model.meta_sgd_lrs = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.full_like(p, init_lr)) for _, p in lora_named_params]
    )
    model.meta_sgd_lr_map = {f"model.{n}": i for i, (n, _) in enumerate(lora_named_params)}

def maml_train_one_epoch(
        config,
        model,
        criterion,
        data_loader,
        optimizer,
        epoch,
        mixup_fn,
        lr_scheduler,
        loss_scaler,
        task=None,
        teacher=None):
    """Train one epoch using a per-task MAML update.

    After computing the meta loss over all tasks, gradients are applied to
    **all** LoRA weights (both shared and task specific).
    """

    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    for idx, batch in enumerate(data_loader):
        samples = batch['image'].cuda(non_blocking=True)
        targets = {
            t: batch[t].cuda(non_blocking=True) for t in config.TASKS
        }

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        batch_size = samples.size(0)
        support_size = batch_size // 2
        support_samples = samples[:support_size]
        query_samples = samples[support_size:]

        for p in model.parameters():
            p.requires_grad = False

        lora_params = [
            p for n, p in model.named_parameters()
            if 'lora_shared_' in n or (
                    not config.MODEL.MTLORA.FREEZE_TS_LORA and 'lora_tasks_' in n
            )
        ]
        for p in lora_params:
            p.requires_grad = True

        query_losses = []
        loss_dict = {}

        for task_name in config.TASKS:
            task_params = [
                p for n, p in model.named_parameters()
                if 'lora_shared_' in n or (
                        not config.MODEL.MTLORA.FREEZE_TS_LORA and 'lora_tasks_' in n and f'.{task_name}' in n
                )
            ]
            backup = [p.detach().clone() for p in task_params]

            starget = targets[task_name][:support_size]
            qtarget = targets[task_name][support_size:]

            inner_steps = max(1, config.MODEL.MTLORA.MAML_INNER_STEPS)
            for _ in range(inner_steps):
                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    sout = model.forward_task(support_samples, task_name)[task_name]
                    s_loss = criterion.loss_ft[task_name](sout, starget)
                    s_loss = criterion.loss_weights[task_name] * s_loss

                grads = torch.autograd.grad(
                    s_loss,
                    task_params,
                    allow_unused=True,
                    create_graph=False,
                )
                # replace missing grads with zeros to avoid runtime errors
                grads = [torch.zeros_like(p) if g is None else g
                         for p, g in zip(task_params, grads)]
                inner_lr = config.MODEL.MTLORA.MAML_INNER_LR
                for p, g in zip(task_params, grads):
                    p.data = p.data - inner_lr * g

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                qout = model.forward_task(query_samples, task_name)[task_name]
                q_loss = criterion.loss_ft[task_name](qout, qtarget)
                q_loss = criterion.loss_weights[task_name] * q_loss

            query_losses.append(q_loss)
            loss_dict[task_name] = q_loss.detach()

            for p, b in zip(task_params, backup):
                p.data.copy_(b)

        meta_loss = torch.stack(query_losses).mean()

        grad_norm = loss_scaler(
            meta_loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=lora_params,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss_meter.update(meta_loss.item())
        if grad_norm is not None:
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
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'MAML Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

def meta_sgd_train_one_epoch(
        config,
        model,
        criterion,
        data_loader,
        optimizer,
        epoch,
        mixup_fn,
        lr_scheduler,
        loss_scaler,
        task=None,
        teacher=None):
    """Train one epoch using Meta-SGD with learnable per-parameter inner learning rates."""

    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    lora_named = [
        (n, p)
        for n, p in model.named_parameters()
        if 'lora_shared_' in n or (
                not config.MODEL.MTLORA.FREEZE_TS_LORA and 'lora_tasks_' in n)
    ]
    lora_names = [f"model.{n}" for n, _ in lora_named]
    lora_params = [p for _, p in lora_named]

    assert hasattr(model, 'meta_sgd_lrs'), 'call init_meta_sgd_lrs before training'
    meta_lrs = model.meta_sgd_lrs
    lr_map = model.meta_sgd_lr_map

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    for idx, batch in enumerate(data_loader):
        samples = batch['image'].cuda(non_blocking=True)
        targets = {t: batch[t].cuda(non_blocking=True) for t in config.TASKS}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        batch_size = samples.size(0)
        support_size = batch_size // 2
        support_samples = samples[:support_size]
        query_samples = samples[support_size:]

        for p in model.parameters():
            p.requires_grad = False
        for p in lora_params:
            p.requires_grad = True
        for a in meta_lrs:
            a.requires_grad = True

        base_params = OrderedDict(
            {
                **{f"model.{k}": v for k, v in model.named_parameters()},
                **{f"model.{k}": v for k, v in model.named_buffers()},
            }
        )

        query_losses = []
        loss_dict = {}

        for task_name in config.TASKS:
            idxs = [i for i, n in enumerate(lora_names) if ('lora_shared_' in n) or (
                    not config.MODEL.MTLORA.FREEZE_TS_LORA and f'.{task_name}' in n)]
            task_params = [lora_params[i] for i in idxs]
            task_alphas = [meta_lrs[i] for i in idxs]
            task_names = [lora_names[i] for i in idxs]

            starget = targets[task_name][:support_size]
            qtarget = targets[task_name][support_size:]

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                sout = model.forward_task(support_samples, task_name)[task_name]
                s_loss = criterion.loss_ft[task_name](sout, starget)
                s_loss = criterion.loss_weights[task_name] * s_loss

            grads = torch.autograd.grad(
                s_loss,
                task_params,
                allow_unused=True,
                create_graph=True,
            )
            grads = [torch.zeros_like(p) if g is None else g for p, g in zip(task_params, grads)]

            adapted = base_params.copy()
            for name, p, g, a_lr in zip(task_names, task_params, grads, task_alphas):
                adapted[name] = p - a_lr * g

            task_model = _TaskWrapper(model, task_name)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                qout = functional_call(task_model, adapted, (query_samples,))
                q_loss = criterion.loss_ft[task_name](qout, qtarget)
                q_loss = criterion.loss_weights[task_name] * q_loss

            query_losses.append(q_loss)
            loss_dict[task_name] = q_loss.detach()

        meta_loss = torch.stack(query_losses).mean()

        grad_norm = loss_scaler(
            meta_loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=list(meta_lrs) + lora_params,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss_meter.update(meta_loss.item())
        if grad_norm is not None:
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
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Meta-SGD Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

def reptile_train_one_epoch(
        config,
        model,
        criterion,
        data_loader,
        optimizer,
        epoch,
        mixup_fn,
        lr_scheduler,
        loss_scaler,
        task=None,
        teacher=None):
    """Train one epoch using Reptile style updates.

    Each task performs a small inner-loop adaptation on a support set.
    The parameter differences after adaptation are averaged and applied
    directly to the model weights.
    """

    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    for idx, batch in enumerate(data_loader):
        samples = batch['image'].cuda(non_blocking=True)
        targets = {t: batch[t].cuda(non_blocking=True) for t in config.TASKS}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        for p in model.parameters():
            p.requires_grad = False

        lora_shared_params = [
            p for n, p in model.named_parameters() if 'lora_shared_' in n
        ]
        lora_task_params = {
            t: [
                p for n, p in model.named_parameters()
                if 'lora_tasks_' in n and f'.{t}' in n
            ] for t in config.TASKS
        }
        lora_params = lora_shared_params + [p for params in lora_task_params.values() for p in params]
        for p in lora_params:
            p.requires_grad = True

        shared_init_state = [p.detach().clone() for p in lora_shared_params]
        shared_diff_list = []

        task_losses = []
        loss_dict = {}

        for task_name in config.TASKS:
            # reset shared parameters to their initial state for this task
            for p, init in zip(lora_shared_params, shared_init_state):
                p.data.copy_(init)

            task_specific_params = lora_task_params[task_name]
            task_init_state = [p.detach().clone() for p in task_specific_params]

            task_params = lora_shared_params + task_specific_params

            target = targets[task_name]

            inner_steps = max(1, config.MODEL.MTLORA.MAML_INNER_STEPS)
            for _ in range(inner_steps):
                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    out = model.forward_task(samples, task_name)[task_name]
                    loss = criterion.loss_ft[task_name](out, target)
                    loss = criterion.loss_weights[task_name] * loss

                grads = torch.autograd.grad(
                    loss,
                    task_params,
                    allow_unused=True,
                    create_graph=False,
                )
                grads = [torch.zeros_like(p) if g is None else g
                         for p, g in zip(task_params, grads)]
                inner_lr = config.MODEL.MTLORA.MAML_INNER_LR
                for p, g in zip(task_params, grads):
                    p.data = p.data - inner_lr * g

            # compute loss after adaptation (for logging only)
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                out = model.forward_task(samples, task_name)[task_name]
                t_loss = criterion.loss_ft[task_name](out, target)
                t_loss = criterion.loss_weights[task_name] * t_loss

            task_losses.append(t_loss)
            loss_dict[task_name] = t_loss.detach()

            shared_diff = [p.data - init for p, init in zip(lora_shared_params, shared_init_state)]
            shared_diff_list.append(shared_diff)

            meta_lr = optimizer.param_groups[0]['lr']
            for p, init in zip(task_specific_params, task_init_state):
                diff = p.data - init
                p.data = init + meta_lr * diff

        meta_lr = optimizer.param_groups[0]['lr']
        if shared_diff_list:
            final_diff = [
                torch.stack([d[i] for d in shared_diff_list]).mean(0)
                for i in range(len(lora_shared_params))
            ]
            for p, init, d in zip(lora_shared_params, shared_init_state, final_diff):
                p.data = init + meta_lr * d

        lr_scheduler.step_update(epoch * num_steps + idx)

        meta_loss = torch.stack(task_losses).mean()

        loss_meter.update(meta_loss.item())
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if wandb_available:
            metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": idx,
                "train/train_loss": loss_meter.val,
                "train/train_loss_avg": loss_meter.avg,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/time": batch_time.val,
                "train/time_avg": batch_time.avg,
                "train/memory": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            }
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Reptile Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

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

    for idx, batch in enumerate(data_loader):
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

        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
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
        print("--------------------------------performance_meter--------------------------")
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
    # args：argparse.Namespace 类型，记录所有命令行参数；
    #
    # config：CfgNode 或自定义 get_config() 返回的配置对象（一般是 YAML 的结构化表示，支持嵌套）。
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
        backend='gloo', init_method='env://', world_size=world_size, rank=rank)
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