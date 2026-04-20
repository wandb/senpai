# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import copy

import torch


class Lion(torch.optim.Optimizer):
    """Minimal Lion optimizer used by the paper sprint."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if wd != 0:
                    param.mul_(1 - lr * wd)
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                param.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer, alpha: float = 0.5, k: int = 5):
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = self.optimizer.state
        self._slow_weights = [
            [param.detach().clone() for param in group["params"]]
            for group in self.param_groups
        ]

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            for slow_group, fast_group in zip(self._slow_weights, self.param_groups, strict=True):
                for slow, fast in zip(slow_group, fast_group["params"], strict=True):
                    slow.lerp_(fast.data, self.alpha)
                    fast.data.copy_(slow)
        return loss


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, start_step: int = 0):
        self.decay = decay
        self.start_step = start_step
        self.step_counter = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step_counter += 1
        if self.step_counter < self.start_step:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = None


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)
