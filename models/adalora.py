from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

class AdaLoRALinear(nn.Module):
    """SVD-based low-rank adapter used by AdaLoRA.

    This module mirrors the interface of :class:`MTLoRALinear` but replaces
    the original A/B factorization with ``P @ diag(lambda) @ Q``. ``P`` and
    ``Q`` are initialized to be orthogonal and ``lambda`` contains the
    trainable singular values. Setting entries of ``lambda`` to zero prunes
    the corresponding rank components.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: Mapping[str, int] | int = 0,
                 lora_shared_scale: float = 1.0,
                 lora_task_scale: float = 1.0,
                 lora_dropout: float = 0.0,
                 tasks: Optional[list[str]] = None,
                 trainable_scale_shared: bool = False,
                 trainable_scale_per_task: bool = False,
                 shared_mode: str = 'matrix',
                 use_adapter: bool = False,
                 **kwargs) -> None:
        super().__init__()
        assert shared_mode == 'matrix', 'AdaLoRA only supports shared_mode="matrix"'
        assert not use_adapter, 'AdaLoRA does not support adapter mode'
        if isinstance(r, int):
            r = {'shared': r}
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.r = r['shared']
        self.tasks = tasks
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else (lambda x: x)

        if self.r > 0:
            # shared parameters
            self.lora_shared_P = nn.Parameter(torch.empty(out_features, r['shared']))
            self.lora_shared_Q = nn.Parameter(torch.empty(r['shared'], in_features))
            self.lora_shared_L = nn.Parameter(torch.ones(r['shared']))
            self.register_buffer('lora_shared_mask', torch.ones(r['shared']))
            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(torch.tensor(lora_shared_scale))
            else:
                self.lora_shared_scale = lora_shared_scale

            # task specific parameters
            if tasks is not None:
                self.lora_tasks_P = nn.ParameterDict()
                self.lora_tasks_Q = nn.ParameterDict()
                self.lora_tasks_L = nn.ParameterDict()
                self.lora_tasks_mask = {}
                self.lora_task_scale = {}
                for t in tasks:
                    r_t = r[t]
                    self.lora_tasks_P[t] = nn.Parameter(torch.empty(out_features, r_t))
                    self.lora_tasks_Q[t] = nn.Parameter(torch.empty(r_t, in_features))
                    self.lora_tasks_L[t] = nn.Parameter(torch.ones(r_t))
                    mask = torch.ones(r_t)
                    self.register_buffer(f'lora_tasks_mask_{t}', mask)
                    self.lora_tasks_mask[t] = mask
                    if trainable_scale_per_task:
                        self.lora_task_scale[t] = nn.Parameter(torch.tensor(lora_task_scale[t]))
                    else:
                        self.lora_task_scale[t] = lora_task_scale[t]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, 'lora_shared_P'):
            nn.init.orthogonal_(self.lora_shared_P)
        if hasattr(self, 'lora_shared_Q'):
            nn.init.orthogonal_(self.lora_shared_Q)
        if hasattr(self, 'lora_tasks_P'):
            for t in self.lora_tasks_P:
                nn.init.orthogonal_(self.lora_tasks_P[t])
                nn.init.orthogonal_(self.lora_tasks_Q[t])

    # interfaces for scheduler -------------------------------------------------
    def get_lambda(self, task: Optional[str] = None) -> torch.Tensor:
        if task is None:
            return self.lora_shared_L
        return self.lora_tasks_L[task]

    def get_mask(self, task: Optional[str] = None) -> torch.Tensor:
        if task is None:
            return self.lora_shared_mask
        return self.lora_tasks_mask[task]

    def zero_lambda(self, indices: torch.Tensor, task: Optional[str] = None) -> None:
        """Set selected ``lambda`` values to zero and mask them out."""
        lam = self.get_lambda(task)
        mask = self.get_mask(task)
        lam.data[indices] = 0
        mask[indices] = 0

    def active_rank(self, task: Optional[str] = None) -> int:
        return int(self.get_mask(task).sum().item())

    def orthogonal_regularization(self) -> torch.Tensor:
        if self.r == 0:
            return torch.tensor(0., device=self.linear.weight.device)
        reg = ((self.lora_shared_P.t() @ self.lora_shared_P - torch.eye(self.lora_shared_P.size(1), device=self.lora_shared_P.device))**2).sum()
        reg = reg + ((self.lora_shared_Q @ self.lora_shared_Q.t() - torch.eye(self.lora_shared_Q.size(0), device=self.lora_shared_Q.device))**2).sum()
        if self.tasks is not None:
            for t in self.tasks:
                P = self.lora_tasks_P[t]
                Q = self.lora_tasks_Q[t]
                reg = reg + ((P.t() @ P - torch.eye(P.size(1), device=P.device))**2).sum()
                reg = reg + ((Q @ Q.t() - torch.eye(Q.size(0), device=Q.device))**2).sum()
        return reg

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, x_tasks: Optional[Dict[str, torch.Tensor]] = None):
        pretrained = self.linear(x)
        if self.r == 0:
            return pretrained, None
        x_drop = self.lora_dropout(x)
        shared = (x_drop @ self.lora_shared_Q.t()) * (self.lora_shared_L * self.lora_shared_mask)
        shared = shared @ self.lora_shared_P.t()
        if isinstance(self.lora_shared_scale, torch.Tensor):
            shared = shared * self.lora_shared_scale
        else:
            shared = shared * self.lora_shared_scale
        lora = shared
        lora_tasks = None
        if self.tasks is not None:
            lora_tasks = {}
            for t in self.tasks:
                xt = x_drop if x_tasks is None else self.lora_dropout(x_tasks[t])
                lt = (xt @ self.lora_tasks_Q[t].t()) * (self.lora_tasks_L[t] * self.lora_tasks_mask[t])
                lt = lt @ self.lora_tasks_P[t].t()
                scale = self.lora_task_scale[t]
                if isinstance(scale, torch.Tensor):
                    scale = scale
                lt = lt * scale
                lora_tasks[t] = pretrained + lt
        return pretrained + lora, lora_tasks

class AdaLoRAScheduler:
    """Prunes singular values of all AdaLoRA modules according to importance."""

    def __init__(self, model: nn.Module, cfg) -> None:
        self.modules = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
        self.step_interval = cfg.SCHEDULE_STEP
        self.target_rank = cfg.TARGET_RANK
        self.min_rank = cfg.MIN_RANK
        self.mode = cfg.MODE
        self.ortho_reg_coeff = cfg.ORTHO_REG
        self.importance = cfg.IMPORTANCE
        self.current_step = 0

        self.total_init = sum(m.get_lambda().numel() for m in self.modules)
        self.target_total = cfg.TARGET_RANK

    def orthogonal_regularization(self) -> torch.Tensor:
        if not self.modules:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        regs = [m.orthogonal_regularization() for m in self.modules]
        return torch.stack(regs).sum()

    def step(self) -> None:
        self.current_step += 1
        if self.current_step % self.step_interval != 0:
            return
        params = []
        for m in self.modules:
            lam = m.get_lambda()
            if lam.grad is None:
                continue
            score = (lam * lam.grad).abs()
            params.append((m, None, score))
            if m.tasks is not None and self.mode.lower() == 'ts':
                for t in m.tasks:
                    lam_t = m.get_lambda(t)
                    if lam_t.grad is None:
                        continue
                    score_t = (lam_t * lam_t.grad).abs()
                    params.append((m, t, score_t))
        if not params:
            return
        flat = []
        for m, t, s in params:
            for i, v in enumerate(s.detach().cpu()):
                flat.append((v.item(), m, t, i))
        flat.sort(key=lambda x: x[0])
        for _, m, t, idx in flat:
            total = sum(p.get_lambda().numel() for p in self.modules)
            current = sum(p.active_rank() for p in self.modules)
            if current <= self.target_total:
                break
            if m.active_rank(t) <= self.min_rank:
                continue
            m.zero_lambda(torch.tensor([idx], dtype=torch.long, device=m.get_lambda().device), t)