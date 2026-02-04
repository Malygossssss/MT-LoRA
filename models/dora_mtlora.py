# --------------------------------------------------------
# MTLoRA DoRA extension
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon DoRA (https://github.com/hiyouga/DoRA)
#
# This file adds Adaptive LoRA (DoRA) variants of the multi-task
# linear layers used in MTLoRA. It reuses the Adaptive_Lora_Linear
# implementation from the original DoRA repository and extends it to
# support task-shared (TS) and task-adaptive (TA) branches.
# --------------------------------------------------------

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Union

import torch
import torch.nn as nn


class AdaptiveLoRALinear(nn.Module):
    """Implementation of DoRA's Adaptive LoRA linear layer.

    This class is copied from the external DoRA project with minimal
    changes to keep the logic intact. It freezes the wrapped linear
    layer and introduces trainable low-rank matrices together with a
    learnable ``lora_scaler`` vector that enables dynamic rank pruning.
    """

    def __init__(self, in_features: int, out_features: int, r: int,
                 lora_alpha: int = 1, lora_dropout: float = 0.0,
                 **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_scaler = nn.Parameter(torch.zeros(r, dtype=torch.float32))
        self.lora_b = nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_b.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(x)
        lora = self.lora_dropout(x)
        lora = self.lora_a(lora)
        lora = lora * self.lora_scaler
        lora = self.lora_b(lora)
        lora = lora * self.scaling
        return hidden + lora


class MTDoRALinear(nn.Module):
    """Multi-task extension of DoRA linear layers for MTLoRA.

    This layer mirrors :class:`models.lora.MTLoRALinear` but replaces the
    LoRA branches with adaptive DoRA branches. Both the task-shared
    (TS-LoRA) and task-adaptive (TA-LoRA) components have learnable
    ``lora_scaler`` vectors allowing dynamic rank adjustment.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: Union[int, Mapping[str, int]] = 0,
            lora_shared_scale: float = 1.0,
            lora_task_scale: float = 1.0,
            lora_dropout: float = 0.0,
            tasks: Optional[list[str]] = None,
            trainable_scale_shared: bool = False,
            trainable_scale_per_task: bool = False,
            shared_mode: str = "matrix",
            **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(r, int):
            r = {"shared": r}
        assert shared_mode in ["matrix", "matrixv2", "add", "addition", "lora_only"]
        if shared_mode == "add":
            shared_mode = "addition"
        if shared_mode == "lora_only":
            tasks = None
        self.shared_mode = shared_mode
        self.tasks = tasks

        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_dropout = nn.Dropout(lora_dropout)

        self.r_shared = r["shared"]
        if self.r_shared > 0:
            self.lora_shared_a = nn.Linear(in_features, self.r_shared, bias=False)
            self.lora_shared_scaler = nn.Parameter(torch.zeros(self.r_shared, dtype=torch.float32))
            self.lora_shared_b = nn.Linear(self.r_shared, out_features, bias=False)
            if trainable_scale_shared:
                self.lora_shared_alpha = nn.Parameter(torch.tensor([lora_shared_scale], dtype=torch.float32))
            else:
                self.lora_shared_alpha = lora_shared_scale

        if tasks is not None:
            self.lora_tasks_a = nn.ModuleDict()
            self.lora_tasks_scaler = nn.ParameterDict()
            self.lora_tasks_b = nn.ModuleDict()
            self.lora_task_alpha: Dict[str, Union[float, torch.Tensor]] = {}
            for task in tasks:
                r_task = r.get(task, 0)
                self.lora_tasks_a[task] = nn.Linear(in_features, r_task, bias=False)
                self.lora_tasks_scaler[task] = nn.Parameter(torch.zeros(r_task, dtype=torch.float32))
                self.lora_tasks_b[task] = nn.Linear(r_task, out_features, bias=False)
                scale_value = lora_task_scale[task] if isinstance(lora_task_scale, Mapping) else lora_task_scale
                if trainable_scale_per_task:
                    self.lora_task_alpha[task] = nn.Parameter(torch.tensor([scale_value], dtype=torch.float32))
                else:
                    self.lora_task_alpha[task] = scale_value

        if self.shared_mode == "addition" and tasks is not None:
            self.lora_norm = nn.LayerNorm(out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.r_shared > 0:
            nn.init.kaiming_uniform_(self.lora_shared_a.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_shared_b.weight, a=math.sqrt(5))
        if self.tasks is not None:
            for task in self.tasks:
                nn.init.kaiming_uniform_(self.lora_tasks_a[task].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_tasks_b[task].weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, x_tasks: Optional[Dict[str, torch.Tensor]] = None):
        pretrained = self.linear(x)
        if self.r_shared == 0:
            return pretrained, None

        shared_inp = self.lora_dropout(x)
        shared = self.lora_shared_a(shared_inp)
        shared = shared * self.lora_shared_scaler
        shared = self.lora_shared_b(shared)
        alpha = self.lora_shared_alpha
        if isinstance(alpha, torch.Tensor):
            shared = shared * (alpha / self.r_shared)
        else:
            shared = shared * (alpha / self.r_shared)

        if self.tasks is None:
            return pretrained + shared, None

        lora_tasks = {}
        for task in self.tasks:
            task_inp = self.lora_dropout(x if x_tasks is None else x_tasks[task])
            task_lora = self.lora_tasks_a[task](task_inp)
            task_lora = task_lora * self.lora_tasks_scaler[task]
            task_lora = self.lora_tasks_b[task](task_lora)
            alpha = self.lora_task_alpha[task]
            r_task = self.lora_tasks_a[task].out_features
            if isinstance(alpha, torch.Tensor):
                task_lora = task_lora * (alpha / r_task)
            else:
                task_lora = task_lora * (alpha / r_task)
            if self.shared_mode == "matrix":
                lora_tasks[task] = pretrained + task_lora
            elif self.shared_mode == "matrixv2":
                lora_tasks[task] = pretrained + shared + task_lora
            elif self.shared_mode == "addition":
                lora_tasks[task] = pretrained + task_lora
            else:
                raise NotImplementedError

        if self.shared_mode == "addition":
            shared = self.lora_norm(torch.sum(torch.stack(list(lora_tasks.values()), dim=0), dim=0))

        return pretrained + shared, lora_tasks


class MTDoRAQKV(nn.Module):
    """QKV projection composed of three :class:`MTDoRALinear` layers."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: Union[int, Mapping[str, int]] = 0,
            lora_shared_scale: float = 1.0,
            lora_task_scale: float = 1.0,
            lora_dropout: float = 0.0,
            tasks: Optional[list[str]] = None,
            trainable_scale_shared: bool = False,
            trainable_scale_per_task: bool = False,
            shared_mode: str = "matrix",
            **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(r, int):
            r = {"shared": r}
        self.tasks = tasks
        self.q = MTDoRALinear(in_features, out_features, r=r, lora_shared_scale=lora_shared_scale,
                              lora_task_scale=lora_task_scale, lora_dropout=lora_dropout, tasks=tasks,
                              trainable_scale_shared=trainable_scale_shared,
                              trainable_scale_per_task=trainable_scale_per_task,
                              shared_mode=shared_mode, **kwargs)
        self.k = MTDoRALinear(in_features, out_features, r=r, lora_shared_scale=lora_shared_scale,
                              lora_task_scale=lora_task_scale, lora_dropout=lora_dropout, tasks=tasks,
                              trainable_scale_shared=trainable_scale_shared,
                              trainable_scale_per_task=trainable_scale_per_task,
                              shared_mode=shared_mode, **kwargs)
        self.v = MTDoRALinear(in_features, out_features, r=r, lora_shared_scale=lora_shared_scale,
                              lora_task_scale=lora_task_scale, lora_dropout=lora_dropout, tasks=tasks,
                              trainable_scale_shared=trainable_scale_shared,
                              trainable_scale_per_task=trainable_scale_per_task,
                              shared_mode=shared_mode, **kwargs)

    def forward(self, x: torch.Tensor, x_tasks: Optional[Dict[str, torch.Tensor]] = None):
        q, qt = self.q(x, x_tasks)
        k, kt = self.k(x, x_tasks)
        v, vt = self.v(x, x_tasks)
        if self.tasks is None:
            return torch.cat([q, k, v], dim=-1), None
        tasks_out = {}
        for task in self.tasks:
            tasks_out[task] = torch.cat([qt[task], kt[task], vt[task]], dim=-1)
        return torch.cat([q, k, v], dim=-1), tasks_out

    #
# ---- Adaptive DoRA utilities ----

from types import SimpleNamespace


def _compute_score(a: torch.Tensor, b: torch.Tensor, scaler: torch.Tensor, eps: float) -> torch.Tensor:
    """Return per-rank importance score for a single LoRA branch."""
    delta_w = b @ (a * scaler.unsqueeze(1))
    norm_w = delta_w.norm(p="fro") + eps
    prod = torch.einsum('ik,kj->kij', b, a * scaler.unsqueeze(1))
    norm_i = prod.norm(p="fro", dim=(1, 2))
    return norm_i / norm_w


def update_importance_score(model: nn.Module, cfg: SimpleNamespace) -> None:
    """EMA update of Frobenius-norm-based importance scores for all DoRA layers."""
    beta, eps = cfg.ADAPTIVE_SENSITIVITY_BETA, cfg.ADAPTIVE_EPS
    if not hasattr(model, "dora_sensitivity_score"):
        model.dora_sensitivity_score = {}
        model.dora_final_mask = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, MTDoRALinear):
                if module.r_shared > 0:
                    score = _compute_score(module.lora_shared_a.weight, module.lora_shared_b.weight,
                                           module.lora_shared_scaler, eps)
                    key = f"{name}.shared"
                    if key not in model.dora_sensitivity_score:
                        model.dora_sensitivity_score[key] = torch.zeros_like(score)
                        model.dora_final_mask[key] = torch.zeros_like(score, dtype=torch.bool)
                    model.dora_sensitivity_score[key] = (
                            beta * model.dora_sensitivity_score[key] + (1 - beta) * score
                    )
                if module.tasks is not None:
                    for task in module.tasks:
                        score = _compute_score(module.lora_tasks_a[task].weight,
                                               module.lora_tasks_b[task].weight,
                                               module.lora_tasks_scaler[task], eps)
                        key = f"{name}.{task}"
                        if key not in model.dora_sensitivity_score:
                            model.dora_sensitivity_score[key] = torch.zeros_like(score)
                            model.dora_final_mask[key] = torch.zeros_like(score, dtype=torch.bool)
                        model.dora_sensitivity_score[key] = (
                                beta * model.dora_sensitivity_score[key] + (1 - beta) * score
                        )


def get_prune_step(step: int, max_step: int, cfg: SimpleNamespace) -> tuple[int, int]:
    cur = int(step - max_step * cfg.ADAPTIVE_START_PRUNE_STEP_RATIO)
    max_prune = int(max_step * (1 - cfg.ADAPTIVE_START_PRUNE_STEP_RATIO - cfg.ADAPTIVE_END_PRUNE_STEP_RATIO))
    return cur, max_prune


def prune_lora_scaler(model: nn.Module, step: int, max_step: int, cfg: SimpleNamespace) -> bool:
    """Prune low-importance ranks following cubic decay schedule."""
    current_prune_step, max_prune_step = get_prune_step(step, max_step, cfg)
    if current_prune_step > max_prune_step:
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, MTDoRALinear):
                    if module.r_shared > 0:
                        key = f"{name}.shared"
                        mask = model.dora_final_mask.get(key)
                        if mask is not None:
                            module.lora_shared_scaler[mask] = 0
                    if module.tasks is not None:
                        for task in module.tasks:
                            key = f"{name}.{task}"
                            mask = model.dora_final_mask.get(key)
                            if mask is not None:
                                module.lora_tasks_scaler[task][mask] = 0
        return False

    update_importance_score(model, cfg)
    if current_prune_step < 0:
        return False
    if not (current_prune_step % cfg.ADAPTIVE_PRUNE_INTERVAL_STEP == 0 or current_prune_step == max_prune_step):
        return False
    with torch.no_grad():
        all_scores = torch.cat(list(model.dora_sensitivity_score.values()))
        start_rank = cfg.ADAPTIVE_START_RANK
        end_rank = cfg.ADAPTIVE_END_AVG_RANK
        prune_rate = ((start_rank - end_rank) / start_rank) * ((current_prune_step / max_prune_step) ** 3)
        prune_num = int(all_scores.numel() * prune_rate)
        threshold = torch.kthvalue(all_scores, max(prune_num, 1)).values
        for name, module in model.named_modules():
            if isinstance(module, MTDoRALinear):
                if module.r_shared > 0:
                    key = f"{name}.shared"
                    mask = model.dora_sensitivity_score[key] <= threshold
                    module.lora_shared_scaler[mask] = 0
                    if current_prune_step == max_prune_step:
                        model.dora_final_mask[key] = mask
                if module.tasks is not None:
                    for task in module.tasks:
                        key = f"{name}.{task}"
                        mask = model.dora_sensitivity_score[key] <= threshold
                        module.lora_tasks_scaler[task][mask] = 0
                        if current_prune_step == max_prune_step:
                            model.dora_final_mask[key] = mask
    return current_prune_step == max_prune_step


def regularization_loss(model: nn.Module, step: int, max_step: int, cfg: SimpleNamespace) -> torch.Tensor:
    """Compute DEM variance regularization used in DoRA."""
    current_prune_step, max_prune_step = get_prune_step(step, max_step, cfg)
    if current_prune_step > max_prune_step:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    sum_var = torch.tensor(0.0, device=next(model.parameters()).device)
    branch_cnt = 0
    for module in model.modules():
        if isinstance(module, MTDoRALinear):
            if module.r_shared > 0:
                sum_var += module.lora_shared_a.weight.var(dim=1).sum()
                sum_var += module.lora_shared_b.weight.var(dim=0).sum()
                branch_cnt += 1
            if module.tasks is not None:
                for task in module.tasks:
                    sum_var += module.lora_tasks_a[task].weight.var(dim=1).sum()
                    sum_var += module.lora_tasks_b[task].weight.var(dim=0).sum()
                    branch_cnt += 1
    mean_var = sum_var / (branch_cnt * cfg.ADAPTIVE_START_RANK * 2)
    return mean_var