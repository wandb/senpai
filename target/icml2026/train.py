# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.architectures import ABUPTReference, ANPSurfaceDecoder, ReferenceTransolver, SenpaiTransolver
from core.contracts import ABUPTBatch, DatasetBundle, GroupedBatch, TargetTransformStats
from core.datasets import ABUPTCollate, build_dataset_bundle, collate_grouped
from core.features import (
    append_batched_fourier_features,
    compute_cp_panel,
    compute_te_features,
    compute_vortex_panel_velocity,
    compute_wake_deficit_features,
)
from core.optim import Lion, Lookahead


class EMAWithWarmup:
    """EMA with timm-style adaptive decay warmup: min(target, (1+step)/(10+step))."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step_counter = 0
        self.shadow = {
            self._clean(name): param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _clean(name: str) -> str:
        return name.replace("_orig_mod.", "")

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step_counter += 1
        actual_decay = min(self.decay, (1 + self.step_counter) / (10 + self.step_counter))
        for name, param in model.named_parameters():
            key = self._clean(name)
            if not param.requires_grad or key not in self.shadow:
                continue
            self.shadow[key].mul_(actual_decay).add_(param.detach(), alpha=1 - actual_decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        self.backup = {
            self._clean(name): param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and self._clean(name) in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            key = self._clean(name)
            if param.requires_grad and key in self.shadow:
                param.data.copy_(self.shadow[key])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        for name, param in model.named_parameters():
            key = self._clean(name)
            if param.requires_grad and key in self.backup:
                param.data.copy_(self.backup[key])
        self.backup = None


LEGACY_VAL_ALIAS = {
    "val_in_dist": "p_in",
    "val_ood_cond": "p_oodc",
    "val_tandem_transfer": "p_tan",
    "val_ood_re": "p_re",
}
AIRFRANS_FIELD_NAMES = ("Ux", "Uy", "p", "nut")
TANDEM_FIELD_NAMES = ("Ux", "Uy", "p")


@dataclass
class TrainConfig:
    dataset: str = "tandemfoil"
    model: str = "senpai_transolver"
    model_layers: int = 3
    model_hidden_dim: int = 192
    model_heads: int = 3
    model_mlp_ratio: int = 4
    model_slices: int = 96
    model_dropout: float = 0.0
    attn_dropout: float = 0.0
    drivaerml_train_surface_points: int = 0
    drivaerml_eval_surface_points: int = 0
    drivaerml_train_volume_points: int = 0
    drivaerml_eval_volume_points: int = 0
    drivaerml_manifest: str = ""
    drivaerml_root: str = ""
    agent: str = ""
    wandb_name: str = ""
    wandb_group: str = ""
    output_dir: str = "outputs/icml2026"
    epochs: int = 2
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "lion"
    use_lookahead: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    num_workers: int = -1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    amp_mode: str = "bf16"
    compile_model: bool = True
    debug: bool = False
    max_train_batches: int = 0
    max_eval_batches: int = 0
    surface_only_drivaerml: bool = True
    airfrans_task: str = "full"
    tandem_manifest: str = ""
    tandem_stats: str = ""
    tandemfoil_paper_task: str = "cruise_random_uniform"
    tandemfoil_paper_manifest: str = ""
    tandemfoil_paper_stats: str = ""
    enable_fourier: bool = False
    enable_te_coord_frame: bool = False
    enable_cp_panel: bool = False
    enable_cp_panel_tandem_only: bool = False
    cp_panel_scale: float = 1.0
    enable_wake_deficit: bool = False
    enable_wake_angle: bool = False
    enable_vortex_panel_velocity: bool = False
    vortex_panel_scale: float = 0.1
    vortex_panel_n: int = 64
    enable_pressure_prior_addition: bool = False
    surface_refine: bool = True
    surface_refine_hidden: int = 128
    surface_refine_layers: int = 2
    anp_srf: bool = False
    asinh_pressure: bool = False
    asinh_scale: float = 0.75
    residual_prediction: bool = False
    re_stratified_sampling: bool = False
    cosine_t_max: int = 150
    geometry_points: int = 25_000
    geometry_supernodes: int = 4_096
    surface_anchor_points: int = 8_000
    volume_anchor_points: int = 8_000
    grad_clip: float = 0.0
    grad_accum_steps: int = 1
    save_checkpoint: bool = False
    seed: int = 0


@dataclass
class TandemPreparedBatch:
    surface_x: torch.Tensor
    volume_x: torch.Tensor | None
    surface_target: torch.Tensor
    volume_target: torch.Tensor | None
    surface_raw_y: torch.Tensor
    volume_raw_y: torch.Tensor | None
    surface_mask: torch.Tensor
    volume_mask: torch.Tensor | None
    umag: torch.Tensor
    q: torch.Tensor
    freestream: torch.Tensor | None
    surface_raw_xy: torch.Tensor
    surface_saf_vec: torch.Tensor
    surface_cp_panel: torch.Tensor | None
    is_tandem: torch.Tensor
    fore_surface_mask: torch.Tensor
    aft_surface_mask: torch.Tensor


class TargetTransform:
    def __init__(
        self,
        *,
        pressure_index: int | None,
        stats_mean: torch.Tensor | None,
        stats_std: torch.Tensor | None,
        asinh_pressure: bool = False,
        asinh_scale: float = 1.0,
    ):
        self.pressure_index = pressure_index
        self.stats_mean = stats_mean
        self.stats_std = stats_std
        self.asinh_pressure = asinh_pressure
        self.asinh_scale = asinh_scale

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        # FIX (Bug 3): clamp non-finite input values before normalization so that
        # ±inf from float16 overflow in raw TandemFoil Paper pickles do not
        # propagate as NaN through the normalization arithmetic.
        if not out.is_floating_point():
            out = out.float()
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        if self.asinh_pressure and self.pressure_index is not None:
            out[..., self.pressure_index] = torch.asinh(out[..., self.pressure_index] * self.asinh_scale)
        if self.stats_mean is not None and self.stats_std is not None and self.stats_mean.numel() == out.shape[-1]:
            out = (out - self.stats_mean.to(out.device)) / self.stats_std.to(out.device).clamp(min=1e-6)
        out = torch.where(out.isfinite(), out, torch.zeros_like(out))
        return out

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        if self.stats_mean is not None and self.stats_std is not None and self.stats_mean.numel() == out.shape[-1]:
            out = out * self.stats_std.to(out.device) + self.stats_mean.to(out.device)
        if self.asinh_pressure and self.pressure_index is not None:
            out[..., self.pressure_index] = torch.sinh(out[..., self.pressure_index]) / max(self.asinh_scale, 1e-6)
        return out


def comparison_metric_tensors(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    train_transform: TargetTransform,
    metric_transform: TargetTransform | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric_transform is None:
        return preds, train_transform.apply(target)
    pred_raw = train_transform.invert(preds)
    return metric_transform.apply(pred_raw), metric_transform.apply(target)


class TandemTargetTransform:
    def __init__(
        self,
        *,
        stats: TargetTransformStats,
        phys_stats: TargetTransformStats,
        config: TrainConfig,
    ):
        if stats.x_mean is None or stats.x_std is None:
            raise ValueError("Tandem parity path requires x_mean/x_std in split stats")
        if phys_stats.y_mean is None or phys_stats.y_std is None:
            raise ValueError("Tandem parity path requires physics-normalized y stats")
        self.x_mean = stats.x_mean
        self.x_std = stats.x_std
        self.phys_mean = phys_stats.y_mean
        self.phys_std = phys_stats.y_std
        self.config = config

    @staticmethod
    def _umag_q(y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        ux_mean = (y[..., 0] * mask.float()).sum(dim=1, keepdim=True) / n_nodes
        uy_mean = (y[..., 1] * mask.float()).sum(dim=1, keepdim=True) / n_nodes
        umag = (ux_mean.square() + uy_mean.square()).sqrt().clamp(min=1.0).unsqueeze(-1)
        q = 0.5 * umag.square()
        return umag, q

    @staticmethod
    def _phys_norm(y: torch.Tensor, umag: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        out[..., 0:1] = y[..., 0:1] / umag
        out[..., 1:2] = y[..., 1:2] / umag
        out[..., 2:3] = y[..., 2:3] / q
        return out

    @staticmethod
    def _phys_denorm(y_phys: torch.Tensor, umag: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        out = y_phys.clone()
        out[..., 0:1] = y_phys[..., 0:1].clamp(-10, 10) * umag
        out[..., 1:2] = y_phys[..., 1:2].clamp(-10, 10) * umag
        out[..., 2:3] = y_phys[..., 2:3].clamp(-20, 20) * q
        return out

    def _split_full(
        self,
        full_tensor: torch.Tensor,
        surface_tokens: int,
        volume_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        surface = full_tensor[:, :surface_tokens]
        volume = full_tensor[:, surface_tokens : surface_tokens + volume_tokens] if volume_tokens > 0 else None
        return surface, volume

    def prepare_batch(self, batch: GroupedBatch) -> TandemPreparedBatch:
        surface_tokens = batch.surface_x.shape[1]
        volume_tokens = 0 if batch.volume_x is None else batch.volume_x.shape[1]
        volume_x_raw = batch.volume_x if batch.volume_x is not None else batch.surface_x.new_zeros(batch.surface_x.shape[0], 0, batch.surface_x.shape[-1])
        volume_y_raw = batch.volume_y if batch.volume_y is not None else batch.surface_y.new_zeros(batch.surface_y.shape[0], 0, batch.surface_y.shape[-1])
        volume_mask = batch.volume_mask if batch.volume_mask is not None else batch.surface_mask.new_zeros(batch.surface_mask.shape[0], 0)

        full_x_raw = torch.cat([batch.surface_x, volume_x_raw], dim=1)
        full_y_raw = torch.cat([batch.surface_y, volume_y_raw], dim=1)
        full_mask = torch.cat([batch.surface_mask, volume_mask], dim=1)
        full_is_surface = torch.cat([batch.surface_mask, volume_mask.new_zeros(volume_mask.shape)], dim=1)

        raw_xy = full_x_raw[..., :2]
        raw_saf_vec = full_x_raw[..., 2:4]
        raw_saf_norm = raw_saf_vec.norm(dim=-1)
        raw_aoa = full_x_raw[:, 0, 14:15]
        raw_gap = full_x_raw[:, :, 22].mean(dim=1)
        is_tandem = full_x_raw[:, 0, 22].abs() > 0.01

        x = (full_x_raw - self.x_mean.to(full_x_raw.device)) / self.x_std.to(full_x_raw.device).clamp(min=1e-6)
        raw_dsdf = full_x_raw[..., 2:10]
        dist_feat = torch.log1p(raw_dsdf.abs().min(dim=-1, keepdim=True).values * 10.0)
        curv = x[..., 2:6].norm(dim=-1, keepdim=True) * full_is_surface.float().unsqueeze(-1)
        x = torch.cat([x, curv, dist_feat], dim=-1)

        fore_te_x = None
        fore_te_y = None
        if self.config.enable_te_coord_frame or self.config.enable_wake_deficit:
            te_feats, fore_te_x, fore_te_y = compute_te_features(raw_xy, full_is_surface, raw_saf_norm)
            if self.config.enable_te_coord_frame:
                x = torch.cat([x, te_feats], dim=-1)
        if self.config.enable_wake_deficit:
            wake_feats = compute_wake_deficit_features(
                raw_xy,
                full_is_surface,
                raw_saf_norm,
                raw_gap,
                fore_te_x=fore_te_x,
                fore_te_y=fore_te_y,
                include_angle=self.config.enable_wake_angle,
            )
            x = torch.cat([x, wake_feats], dim=-1)
        if self.config.enable_fourier:
            x = append_batched_fourier_features(x)

        cp_panel_unscaled = None
        if self.config.enable_cp_panel:
            cp_panel = compute_cp_panel(raw_xy, raw_aoa, full_is_surface, raw_saf_norm)
            cp_panel_unscaled = cp_panel.clone()
            if self.config.enable_cp_panel_tandem_only:
                cp_panel = cp_panel * is_tandem[:, None, None]
            if self.config.cp_panel_scale != 1.0:
                cp_panel = cp_panel * self.config.cp_panel_scale
            x = torch.cat([x, cp_panel], dim=-1)
        if self.config.enable_vortex_panel_velocity:
            vortex = compute_vortex_panel_velocity(
                raw_xy,
                raw_aoa,
                full_is_surface,
                raw_saf_norm,
                n_panels=self.config.vortex_panel_n,
            )
            if self.config.vortex_panel_scale != 1.0:
                vortex = vortex * self.config.vortex_panel_scale
            x = torch.cat([x, vortex], dim=-1)

        umag, q = self._umag_q(full_y_raw, full_mask)
        y_phys = self._phys_norm(full_y_raw, umag, q)
        if self.config.asinh_pressure:
            y_phys = y_phys.clone()
            y_phys[..., 2:3] = torch.asinh(y_phys[..., 2:3] * self.config.asinh_scale)
        y_norm = (y_phys - self.phys_mean.to(full_y_raw.device)) / self.phys_std.to(full_y_raw.device).clamp(min=1e-6)

        freestream = None
        if self.config.residual_prediction:
            freestream_phys = torch.zeros(full_y_raw.shape[0], 1, 3, device=full_y_raw.device, dtype=full_y_raw.dtype)
            freestream_phys[:, 0, 0] = torch.cos(raw_aoa.squeeze(-1))
            freestream_phys[:, 0, 1] = torch.sin(raw_aoa.squeeze(-1))
            freestream = (freestream_phys - self.phys_mean.to(full_y_raw.device)) / self.phys_std.to(full_y_raw.device).clamp(min=1e-6)
            y_norm = y_norm - freestream

        x = x * full_mask.unsqueeze(-1)
        surface_x, volume_x = self._split_full(x, surface_tokens, volume_tokens)
        surface_target, volume_target = self._split_full(y_norm, surface_tokens, volume_tokens)
        surface_raw_y, volume_raw_y = self._split_full(full_y_raw, surface_tokens, volume_tokens)
        surface_cp_panel, _ = self._split_full(
            cp_panel_unscaled if cp_panel_unscaled is not None else x.new_zeros(x.shape[0], x.shape[1], 1),
            surface_tokens,
            volume_tokens,
        )
        surface_raw_xy = batch.surface_x[..., :2]
        surface_saf_vec = batch.surface_x[..., 2:4]
        surface_saf_norm = surface_saf_vec.norm(dim=-1)
        fore_surface_mask = batch.surface_mask & (surface_saf_norm <= 0.005)
        aft_surface_mask = batch.surface_mask & (surface_saf_norm > 0.005) & is_tandem[:, None]

        return TandemPreparedBatch(
            surface_x=surface_x * batch.surface_mask.unsqueeze(-1),
            volume_x=None if volume_x is None else volume_x * volume_mask.unsqueeze(-1),
            surface_target=surface_target * batch.surface_mask.unsqueeze(-1),
            volume_target=None if volume_target is None else volume_target * volume_mask.unsqueeze(-1),
            surface_raw_y=surface_raw_y,
            volume_raw_y=volume_raw_y,
            surface_mask=batch.surface_mask,
            volume_mask=volume_mask if volume_tokens > 0 else None,
            umag=umag,
            q=q,
            freestream=freestream,
            surface_raw_xy=surface_raw_xy,
            surface_saf_vec=surface_saf_vec,
            surface_cp_panel=surface_cp_panel if self.config.enable_cp_panel else None,
            is_tandem=is_tandem,
            fore_surface_mask=fore_surface_mask,
            aft_surface_mask=aft_surface_mask,
        )

    def invert_predictions(
        self,
        prepared: TandemPreparedBatch,
        *,
        surface_preds: torch.Tensor,
        volume_preds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        volume_tokens = 0 if volume_preds is None else volume_preds.shape[1]
        full_pred = torch.cat(
            [
                surface_preds,
                volume_preds
                if volume_preds is not None
                else surface_preds.new_zeros(surface_preds.shape[0], 0, surface_preds.shape[-1]),
            ],
            dim=1,
        )
        if self.config.residual_prediction and prepared.freestream is not None:
            full_pred = full_pred + prepared.freestream
        pred_phys = full_pred * self.phys_std.to(full_pred.device) + self.phys_mean.to(full_pred.device)
        if self.config.asinh_pressure:
            pred_phys = pred_phys.clone()
            pred_phys[..., 2:3] = torch.sinh(pred_phys[..., 2:3]) / max(self.config.asinh_scale, 1e-6)
        pred_orig = self._phys_denorm(pred_phys, prepared.umag, prepared.q)
        return self._split_full(pred_orig, surface_preds.shape[1], volume_tokens)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Unified ICML 2026 CFD trainer")
    for field_name, field_value in TrainConfig().__dict__.items():
        arg_name = f"--{field_name.replace('_', '-')}"
        if isinstance(field_value, bool):
            parser.add_argument(arg_name, action="store_true", default=field_value)
            parser.add_argument(f"--no-{field_name.replace('_', '-')}", action="store_false", dest=field_name)
        else:
            parser.add_argument(arg_name, type=type(field_value), default=field_value)
    namespace = parser.parse_args()
    return TrainConfig(**vars(namespace))


def build_bundle(config: TrainConfig) -> DatasetBundle:
    bundle_kwargs = {
        "dataset_name": config.dataset,
        "debug": config.debug,
        "tandemfoil_paper_task": config.tandemfoil_paper_task,
        "airfrans_task": config.airfrans_task,
        "drivaerml_surface_only": config.surface_only_drivaerml,
        "drivaerml_train_surface_points": config.drivaerml_train_surface_points,
        "drivaerml_eval_surface_points": config.drivaerml_eval_surface_points,
        "drivaerml_train_volume_points": config.drivaerml_train_volume_points,
        "drivaerml_eval_volume_points": config.drivaerml_eval_volume_points,
        "enable_fourier": config.enable_fourier,
        "enable_te_coord_frame": config.enable_te_coord_frame,
        "enable_cp_panel": config.enable_cp_panel,
        "enable_wake_deficit": config.enable_wake_deficit,
        "enable_wake_angle": config.enable_wake_angle,
        "enable_vortex_panel_velocity": config.enable_vortex_panel_velocity,
    }
    if config.tandem_manifest:
        bundle_kwargs["tandem_manifest"] = config.tandem_manifest
    if config.tandem_stats:
        bundle_kwargs["tandem_stats"] = config.tandem_stats
    if config.drivaerml_manifest:
        bundle_kwargs["drivaerml_manifest"] = config.drivaerml_manifest
    if config.drivaerml_root:
        bundle_kwargs["drivaerml_root"] = config.drivaerml_root
    if config.tandemfoil_paper_manifest:
        bundle_kwargs["tandemfoil_paper_manifest"] = config.tandemfoil_paper_manifest
    if config.tandemfoil_paper_stats:
        bundle_kwargs["tandemfoil_paper_stats"] = config.tandemfoil_paper_stats
    return build_dataset_bundle(**bundle_kwargs)


def cp_panel_prior_index(config: TrainConfig, bundle: DatasetBundle) -> int | None:
    if not config.enable_cp_panel or not config.enable_pressure_prior_addition:
        return None
    base_dim = bundle.spec.surface_input_dim
    if config.enable_vortex_panel_velocity:
        base_dim -= 4
    if config.enable_cp_panel:
        base_dim -= 1
        return base_dim
    return None


def build_model(config: TrainConfig, bundle: DatasetBundle) -> torch.nn.Module:
    transolver_kwargs = {
        "n_layers": config.model_layers,
        "n_hidden": config.model_hidden_dim,
        "dropout": config.attn_dropout if config.attn_dropout > 0 else config.model_dropout,
        "n_head": config.model_heads,
        "mlp_ratio": config.model_mlp_ratio,
        "slice_num": config.model_slices,
    }
    if config.model == "reference_transolver":
        return ReferenceTransolver(
            space_dim=bundle.spec.space_dim,
            surface_input_dim=bundle.spec.surface_input_dim,
            surface_output_dim=bundle.spec.surface_output_dim,
            volume_input_dim=bundle.spec.volume_input_dim,
            volume_output_dim=bundle.spec.volume_output_dim,
            **transolver_kwargs,
        )
    if config.model == "senpai_transolver":
        prior_idx = cp_panel_prior_index(config, bundle)
        return SenpaiTransolver(
            space_dim=bundle.spec.space_dim,
            surface_input_dim=bundle.spec.surface_input_dim,
            surface_output_dim=bundle.spec.surface_output_dim,
            volume_input_dim=bundle.spec.volume_input_dim,
            volume_output_dim=bundle.spec.volume_output_dim,
            pressure_output_index=bundle.spec.pressure_output_index,
            surface_refine=config.surface_refine,
            surface_refine_hidden_dim=config.surface_refine_hidden,
            surface_refine_layers=config.surface_refine_layers,
            surface_pressure_prior_idx=prior_idx,
            volume_pressure_prior_idx=prior_idx,
            **transolver_kwargs,
        )
    if config.model == "reference_abupt":
        return ABUPTReference(
            space_dim=bundle.spec.space_dim,
            surface_output_dim=bundle.spec.surface_output_dim,
            volume_output_dim=bundle.spec.volume_output_dim,
        )
    raise ValueError(f"Unknown model: {config.model}")


def build_optimizer(params, config: TrainConfig):
    if config.optimizer == "lion":
        optimizer = Lion(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    if config.use_lookahead:
        optimizer = Lookahead(optimizer)
    return optimizer


def resolve_num_workers(config: TrainConfig, dataset_name: str) -> int:
    if config.num_workers >= 0:
        return config.num_workers
    if not torch.cuda.is_available():
        return 0
    cpu_count = os.cpu_count() or 8
    target = 8 if dataset_name == "drivaerml" else 4
    return min(target, cpu_count)


def autocast_context(device: torch.device, amp_mode: str):
    if amp_mode != "bf16" or device.type != "cuda":
        return nullcontext()
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: True)
    if not supports_bf16():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def float_outputs(outputs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {
        name: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for name, value in outputs.items()
    }


def build_loaders(
    config: TrainConfig,
    bundle: DatasetBundle,
    *,
    num_workers: int | None = None,
) -> tuple[DataLoader, dict[str, DataLoader], dict[str, DataLoader]]:
    if config.model == "reference_abupt":
        train_collate = ABUPTCollate(
            geometry_points=config.geometry_points,
            geometry_supernodes=config.geometry_supernodes,
            surface_anchor_points=config.surface_anchor_points,
            volume_anchor_points=None if bundle.spec.volume_output_dim == 0 else config.volume_anchor_points,
            fixed_per_case=True,
            seed=0,
        )
        eval_collate = ABUPTCollate(
            geometry_points=config.geometry_points,
            geometry_supernodes=config.geometry_supernodes,
            surface_anchor_points=None,
            volume_anchor_points=None,
            fixed_per_case=True,
            seed=0,
        )
    else:
        train_collate = collate_grouped
        eval_collate = collate_grouped

    train_sampler = None
    shuffle = True
    if config.re_stratified_sampling and bundle.sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=bundle.sample_weights,
            num_samples=len(bundle.sample_weights),
            replacement=True,
        )
        shuffle = False

    resolved_num_workers = resolve_num_workers(config, bundle.spec.name) if num_workers is None else num_workers
    loader_kwargs = {
        "num_workers": resolved_num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        collate_fn=train_collate,
        **loader_kwargs,
    )
    val_loaders = {
        name: DataLoader(
            dataset,
            batch_size=1 if config.model == "reference_abupt" else config.batch_size,
            shuffle=False,
            collate_fn=eval_collate,
            **loader_kwargs,
        )
        for name, dataset in bundle.val_datasets.items()
    }
    test_loaders = {
        name: DataLoader(
            dataset,
            batch_size=1 if config.model == "reference_abupt" else config.batch_size,
            shuffle=False,
            collate_fn=eval_collate,
            **loader_kwargs,
        )
        for name, dataset in bundle.test_datasets.items()
    }
    return train_loader, val_loaders, test_loaders


def add_primary_metric_aliases(
    bundle: DatasetBundle,
    metrics: dict[str, float],
    *,
    phase: str,
) -> dict[str, float]:
    default_metric = bundle.spec.default_metric
    if bundle.spec.name == "tandemfoilset":
        eq4_key = f"{phase}_eq4/surface_pressure_mae"
        if eq4_key in metrics:
            metrics[f"{phase}_primary/surface_pressure_mae"] = metrics[eq4_key]
            return metrics

        legacy_keys = [f"{split}/mae_surf_p" for split in LEGACY_VAL_ALIAS]
        legacy_values = [metrics[key] for key in legacy_keys if key in metrics]
        if legacy_values:
            metrics[f"{phase}_primary/mae_surf_p"] = sum(legacy_values) / len(legacy_values)
        return metrics

    suffix = f"/{default_metric}"
    matching = [value for key, value in metrics.items() if key.endswith(suffix)]
    if len(matching) == 1:
        metrics[f"{phase}_primary/{default_metric}"] = matching[0]
    return metrics


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1).float() * pred.shape[-1]
    return diff.sum() / denom


def _case_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if not valid.any():
        return float("nan")
    return float(values[valid].mean().detach().cpu().item())


def _case_masked_channel_means(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    valid = mask.bool()
    if not valid.any():
        return None
    return values[valid].mean(dim=0)


def _case_masked_rel_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if not valid.any():
        return float("nan")
    pred_valid = pred[valid]
    target_valid = target[valid]
    denom = target_valid.square().sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return float("nan")
    rel = torch.sqrt((pred_valid - target_valid).square().sum() / denom)
    return float(rel.detach().cpu().item())


def _accumulate_case_rel_l2_sums(
    case_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    case_ids: list[str],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    mask_float = mask.float()
    error_sq_sum = ((pred - target).square().sum(dim=-1) * mask_float).sum(dim=1)
    target_sq_sum = (target.square().sum(dim=-1) * mask_float).sum(dim=1)
    valid_cases = mask.any(dim=1)
    for case_idx, case_id in enumerate(case_ids):
        if not bool(valid_cases[case_idx]):
            continue
        prev = case_sums.get(case_id)
        if prev is None:
            case_sums[case_id] = (error_sq_sum[case_idx], target_sq_sum[case_idx])
        else:
            case_sums[case_id] = (prev[0] + error_sq_sum[case_idx], prev[1] + target_sq_sum[case_idx])


def _finalize_case_rel_l2(case_sums: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> float | None:
    if not case_sums:
        return None
    error_sq = torch.stack([value[0] for value in case_sums.values()])
    target_sq = torch.stack([value[1] for value in case_sums.values()])
    valid = target_sq > 0
    if not bool(valid.any()):
        return None
    rel_l2 = torch.sqrt(error_sq[valid] / target_sq[valid].clamp(min=1e-12))
    return float(rel_l2.mean().detach().cpu().item())


def _named_channel_metrics(prefix: str, values: torch.Tensor, names: tuple[str, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for idx, value in enumerate(values):
        label = names[idx] if idx < len(names) else f"var{idx}"
        metrics[f"{prefix}_{label}"] = float(value.detach().cpu().item())
    return metrics


def loss_grouped(
    batch: GroupedBatch,
    outputs: dict[str, torch.Tensor | None],
    transform: TargetTransform,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=batch.surface_x.device)
    metrics: dict[str, float] = {}
    if batch.surface_y is not None and outputs["surface_preds"] is not None:
        target = transform.apply(batch.surface_y)
        surf_loss = F.mse_loss(outputs["surface_preds"][batch.surface_mask], target[batch.surface_mask])
        total = total + surf_loss
        metrics["surface_loss"] = float(surf_loss.detach().cpu().item())
    if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
        target = transform.apply(batch.volume_y)
        vol_loss = F.mse_loss(outputs["volume_preds"][batch.volume_mask], target[batch.volume_mask])
        total = total + vol_loss
        metrics["volume_loss"] = float(vol_loss.detach().cpu().item())
    metrics["loss"] = float(total.detach().cpu().item())
    return total, metrics


def loss_grouped_tandem(
    prepared: TandemPreparedBatch,
    outputs: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=prepared.surface_x.device)
    metrics: dict[str, float] = {}
    if outputs["surface_preds"] is not None:
        surf_loss = F.mse_loss(outputs["surface_preds"][prepared.surface_mask], prepared.surface_target[prepared.surface_mask])
        total = total + surf_loss
        metrics["surface_loss"] = float(surf_loss.detach().cpu().item())
    if prepared.volume_target is not None and outputs["volume_preds"] is not None and prepared.volume_mask is not None:
        vol_loss = F.mse_loss(outputs["volume_preds"][prepared.volume_mask], prepared.volume_target[prepared.volume_mask])
        total = total + vol_loss
        metrics["volume_loss"] = float(vol_loss.detach().cpu().item())
    metrics["loss"] = float(total.detach().cpu().item())
    return total, metrics


def loss_abupt(
    batch: ABUPTBatch,
    outputs: dict[str, torch.Tensor | None],
    transform: TargetTransform,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=batch.geometry_position.device)
    metrics: dict[str, float] = {}
    if batch.surface_anchor_target is not None and outputs["surface_preds"] is not None:
        surf_target = transform.apply(batch.surface_anchor_target)
        surf_loss = F.mse_loss(outputs["surface_preds"], surf_target)
        total = total + surf_loss
        metrics["surface_loss"] = float(surf_loss.detach().cpu().item())
    if batch.volume_anchor_target is not None and outputs["volume_preds"] is not None:
        vol_target = transform.apply(batch.volume_anchor_target)
        vol_loss = F.mse_loss(outputs["volume_preds"], vol_target)
        total = total + vol_loss
        metrics["volume_loss"] = float(vol_loss.detach().cpu().item())
    metrics["loss"] = float(total.detach().cpu().item())
    return total, metrics


def maybe_apply_anp(
    outputs: dict[str, torch.Tensor | None],
    prepared: TandemPreparedBatch,
    anp_head: ANPSurfaceDecoder | None,
) -> dict[str, torch.Tensor | None]:
    if anp_head is None or outputs["surface_preds"] is None:
        return outputs
    correction = anp_head(
        outputs["surface_hidden"],
        prepared.surface_cp_panel,
        prepared.surface_raw_xy,
        prepared.surface_saf_vec,
        prepared.surface_mask,
        prepared.is_tandem,
        prepared.fore_surface_mask,
        prepared.aft_surface_mask,
    ).float()
    outputs = dict(outputs)
    outputs["surface_preds"] = (outputs["surface_preds"] + correction) * prepared.surface_mask.unsqueeze(-1)
    return outputs


@torch.no_grad()
def evaluate_grouped(
    model: torch.nn.Module,
    anp_head: ANPSurfaceDecoder | None,
    loader: DataLoader,
    transform: TargetTransform | TandemTargetTransform,
    metric_transform: TargetTransform | None,
    device: torch.device,
    *,
    amp_mode: str = "none",
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    if anp_head is not None:
        anp_head.eval()

    dataset_name: str | None = None
    total_surface = 0.0
    total_volume = 0.0
    count_surface = 0
    count_volume = 0
    surf_pressure_abs_sum = 0.0
    surf_pressure_count = 0
    mae_surf = torch.zeros(3, device=device)
    mae_vol = torch.zeros(3, device=device)
    n_surf = torch.zeros(3, device=device)
    n_vol = torch.zeros(3, device=device)
    surf_channel_sum = None
    vol_channel_sum = None
    paper_surface_sq_sum = None
    paper_volume_sq_sum = None
    paper_surface_nodes = 0
    paper_volume_nodes = 0
    drivaer_surface_case_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    drivaer_volume_case_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    batches = loader if max_batches <= 0 else itertools.islice(loader, max_batches)
    for batch in batches:
        batch = batch.to(device)
        dataset_name = batch.dataset_name

        if isinstance(transform, TandemTargetTransform):
            prepared = transform.prepare_batch(batch)
            with autocast_context(device, amp_mode):
                outputs = model(
                    surface_x=prepared.surface_x,
                    surface_mask=prepared.surface_mask,
                    volume_x=prepared.volume_x,
                    volume_mask=prepared.volume_mask,
                )
                outputs = maybe_apply_anp(outputs, prepared, anp_head)
            outputs = float_outputs(outputs)
            surface_pred_orig, volume_pred_orig = transform.invert_predictions(
                prepared,
                surface_preds=outputs["surface_preds"],
                volume_preds=outputs["volume_preds"],
            )
            surface_err = (surface_pred_orig - prepared.surface_raw_y).abs()
            finite = surface_err.isfinite()
            mae_surf += (surface_err * prepared.surface_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += (prepared.surface_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
            surf_pressure_abs_sum += float((surface_err[..., 2] * prepared.surface_mask).sum().detach().cpu().item())
            surf_pressure_count += int(prepared.surface_mask.sum().detach().cpu().item())
            total_surface += float(_masked_mae(surface_pred_orig, prepared.surface_raw_y, prepared.surface_mask).detach().cpu().item())
            count_surface += 1

            if volume_pred_orig is not None and prepared.volume_raw_y is not None and prepared.volume_mask is not None:
                volume_err = (volume_pred_orig - prepared.volume_raw_y).abs()
                finite = volume_err.isfinite()
                mae_vol += (volume_err * prepared.volume_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_vol += (prepared.volume_mask.unsqueeze(-1) * finite).sum(dim=(0, 1)).float()
                total_volume += float(_masked_mae(volume_pred_orig, prepared.volume_raw_y, prepared.volume_mask).detach().cpu().item())
                count_volume += 1
            continue

        with autocast_context(device, amp_mode):
            outputs = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
        outputs = float_outputs(outputs)
        if dataset_name == "airfrans":
            if not isinstance(transform, TargetTransform):
                raise TypeError("AirfRANS evaluation requires TargetTransform")
            if batch.surface_y is not None and outputs["surface_preds"] is not None:
                surface_pred, surface_target = comparison_metric_tensors(
                    outputs["surface_preds"],
                    batch.surface_y,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                case_values = (surface_pred - surface_target).square()
                for case_idx in range(case_values.shape[0]):
                    channel_mean = _case_masked_channel_means(case_values[case_idx], batch.surface_mask[case_idx])
                    if channel_mean is not None:
                        total_surface += float(channel_mean.mean().detach().cpu().item())
                        count_surface += 1
                        if surf_channel_sum is None:
                            surf_channel_sum = torch.zeros(channel_mean.shape[0], device=device)
                        surf_channel_sum += channel_mean.to(device)
            if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
                volume_pred, volume_target = comparison_metric_tensors(
                    outputs["volume_preds"],
                    batch.volume_y,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                case_values = (volume_pred - volume_target).square()
                for case_idx in range(case_values.shape[0]):
                    channel_mean = _case_masked_channel_means(case_values[case_idx], batch.volume_mask[case_idx])
                    if channel_mean is not None:
                        total_volume += float(channel_mean.mean().detach().cpu().item())
                        count_volume += 1
                        if vol_channel_sum is None:
                            vol_channel_sum = torch.zeros(channel_mean.shape[0], device=device)
                        vol_channel_sum += channel_mean.to(device)
            continue

        if dataset_name == "tandemfoilset_paper":
            if not isinstance(transform, TargetTransform):
                raise TypeError("TandemFoil paper evaluation requires TargetTransform")
            if batch.surface_y is not None and outputs["surface_preds"] is not None:
                surface_pred, surface_target = comparison_metric_tensors(
                    outputs["surface_preds"],
                    batch.surface_y,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                case_values = (surface_pred - surface_target).square()
                valid = batch.surface_mask.unsqueeze(-1)
                if paper_surface_sq_sum is None:
                    paper_surface_sq_sum = torch.zeros(case_values.shape[-1], device=device)
                paper_surface_sq_sum += (case_values * valid).sum(dim=(0, 1)).to(device)
                paper_surface_nodes += int(batch.surface_mask.sum().detach().cpu().item())
            if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
                volume_pred, volume_target = comparison_metric_tensors(
                    outputs["volume_preds"],
                    batch.volume_y,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                case_values = (volume_pred - volume_target).square()
                valid = batch.volume_mask.unsqueeze(-1)
                if paper_volume_sq_sum is None:
                    paper_volume_sq_sum = torch.zeros(case_values.shape[-1], device=device)
                paper_volume_sq_sum += (case_values * valid).sum(dim=(0, 1)).to(device)
                paper_volume_nodes += int(batch.volume_mask.sum().detach().cpu().item())
            continue

        pred_surface = None
        pred_volume = None
        if batch.surface_y is not None and outputs["surface_preds"] is not None:
            pred_surface = transform.invert(outputs["surface_preds"])
        if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
            pred_volume = transform.invert(outputs["volume_preds"])

        if dataset_name == "drivaerml":
            if pred_surface is not None and batch.surface_y is not None:
                _accumulate_case_rel_l2_sums(
                    drivaer_surface_case_sums,
                    case_ids=batch.case_ids,
                    pred=pred_surface,
                    target=batch.surface_y,
                    mask=batch.surface_mask,
                )
            if pred_volume is not None and batch.volume_mask is not None and batch.volume_y is not None:
                _accumulate_case_rel_l2_sums(
                    drivaer_volume_case_sums,
                    case_ids=batch.case_ids,
                    pred=pred_volume,
                    target=batch.volume_y,
                    mask=batch.volume_mask,
                )
            continue

        if pred_surface is not None:
            total_surface += float(_masked_mae(pred_surface, batch.surface_y, batch.surface_mask).cpu().item())
            count_surface += 1
            if transform.pressure_index is not None:
                pressure_idx = transform.pressure_index
                surf_pressure_abs_sum += float(
                    (
                        (pred_surface[..., pressure_idx] - batch.surface_y[..., pressure_idx]).abs()
                        * batch.surface_mask
                    ).sum().detach().cpu().item()
                )
                surf_pressure_count += int(batch.surface_mask.sum().detach().cpu().item())
        if pred_volume is not None and batch.volume_mask is not None and batch.volume_y is not None:
            total_volume += float(_masked_mae(pred_volume, batch.volume_y, batch.volume_mask).cpu().item())
            count_volume += 1

    if dataset_name == "airfrans":
        metrics = {
            "surface_mse": total_surface / max(count_surface, 1),
            "volume_mse": total_volume / max(count_volume, 1),
        }
        if surf_channel_sum is not None and count_surface > 0:
            metrics.update(_named_channel_metrics("surface_mse", surf_channel_sum / count_surface, AIRFRANS_FIELD_NAMES))
        if vol_channel_sum is not None and count_volume > 0:
            metrics.update(_named_channel_metrics("volume_mse", vol_channel_sum / count_volume, AIRFRANS_FIELD_NAMES))
        return metrics
    if dataset_name == "drivaerml":
        surface_rel_l2 = _finalize_case_rel_l2(drivaer_surface_case_sums) or 0.0
        metrics = {
            "surface_rel_l2": surface_rel_l2,
            "surface_rel_l2_pct": surface_rel_l2 * 100.0,
        }
        volume_rel_l2 = _finalize_case_rel_l2(drivaer_volume_case_sums)
        if volume_rel_l2 is not None:
            metrics["volume_rel_l2"] = volume_rel_l2
            metrics["volume_rel_l2_pct"] = volume_rel_l2 * 100.0
        return metrics
    if dataset_name == "tandemfoilset_paper":
        metrics: dict[str, float] = {}
        total_sq_sum = 0.0
        total_nodes = 0
        if paper_surface_sq_sum is not None and paper_surface_nodes > 0:
            surface_channel = paper_surface_sq_sum / paper_surface_nodes
            metrics["surface_mse"] = float(surface_channel.mean().detach().cpu().item())
            metrics.update(_named_channel_metrics("surface_mse", surface_channel, TANDEM_FIELD_NAMES))
            total_sq_sum += float(paper_surface_sq_sum.sum().detach().cpu().item())
            total_nodes += paper_surface_nodes
        if paper_volume_sq_sum is not None and paper_volume_nodes > 0:
            volume_channel = paper_volume_sq_sum / paper_volume_nodes
            metrics["volume_mse"] = float(volume_channel.mean().detach().cpu().item())
            metrics.update(_named_channel_metrics("volume_mse", volume_channel, TANDEM_FIELD_NAMES))
            total_sq_sum += float(paper_volume_sq_sum.sum().detach().cpu().item())
            total_nodes += paper_volume_nodes
        if total_nodes > 0:
            metrics["field_mse"] = total_sq_sum / (total_nodes * len(TANDEM_FIELD_NAMES))
        return metrics
    metrics = {
        "surface_mae": total_surface / max(count_surface, 1),
        "volume_mae": total_volume / max(count_volume, 1),
    }
    if surf_pressure_count > 0:
        metrics["surface_pressure_mae"] = surf_pressure_abs_sum / surf_pressure_count
    if n_surf.sum().item() > 0:
        mae_surf = mae_surf / n_surf.clamp(min=1)
        metrics["mae_surf_Ux"] = float(mae_surf[0].detach().cpu().item())
        metrics["mae_surf_Uy"] = float(mae_surf[1].detach().cpu().item())
        metrics["mae_surf_p"] = float(mae_surf[2].detach().cpu().item())
    if n_vol.sum().item() > 0:
        mae_vol = mae_vol / n_vol.clamp(min=1)
        metrics["mae_vol_Ux"] = float(mae_vol[0].detach().cpu().item())
        metrics["mae_vol_Uy"] = float(mae_vol[1].detach().cpu().item())
        metrics["mae_vol_p"] = float(mae_vol[2].detach().cpu().item())
    return metrics


@torch.no_grad()
def evaluate_abupt(
    model: torch.nn.Module,
    loader: DataLoader,
    transform: TargetTransform,
    metric_transform: TargetTransform | None,
    device: torch.device,
    *,
    amp_mode: str = "none",
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    dataset_name: str | None = None
    total_surface = 0.0
    total_volume = 0.0
    count_surface = 0
    count_volume = 0
    surf_channel_sum = None
    vol_channel_sum = None
    paper_surface_sq_sum = None
    paper_volume_sq_sum = None
    paper_surface_nodes = 0
    paper_volume_nodes = 0
    batches = loader if max_batches <= 0 else itertools.islice(loader, max_batches)
    for batch in batches:
        batch = batch.to(device)
        dataset_name = batch.dataset_name
        with autocast_context(device, amp_mode):
            outputs = model(
                geometry_position=batch.geometry_position,
                geometry_supernode_idx=batch.geometry_supernode_idx,
                geometry_batch_idx=batch.geometry_batch_idx,
                surface_anchor_position=batch.surface_anchor_position,
                volume_anchor_position=batch.volume_anchor_position,
            )
        outputs = float_outputs(outputs)
        if dataset_name == "airfrans":
            if batch.surface_anchor_target is not None and outputs["surface_preds"] is not None:
                surface_pred, surface_target = comparison_metric_tensors(
                    outputs["surface_preds"],
                    batch.surface_anchor_target,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                channel_mean = (surface_pred - surface_target).square().mean(dim=(0, 1))
                total_surface += float(channel_mean.mean().detach().cpu().item())
                count_surface += 1
                if surf_channel_sum is None:
                    surf_channel_sum = torch.zeros(channel_mean.shape[0], device=device)
                surf_channel_sum += channel_mean.to(device)
            if batch.volume_anchor_target is not None and outputs["volume_preds"] is not None:
                volume_pred, volume_target = comparison_metric_tensors(
                    outputs["volume_preds"],
                    batch.volume_anchor_target,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                channel_mean = (volume_pred - volume_target).square().mean(dim=(0, 1))
                total_volume += float(channel_mean.mean().detach().cpu().item())
                count_volume += 1
                if vol_channel_sum is None:
                    vol_channel_sum = torch.zeros(channel_mean.shape[0], device=device)
                vol_channel_sum += channel_mean.to(device)
            continue

        if dataset_name == "tandemfoilset_paper":
            if batch.surface_anchor_target is not None and outputs["surface_preds"] is not None:
                surface_pred, surface_target = comparison_metric_tensors(
                    outputs["surface_preds"],
                    batch.surface_anchor_target,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                if paper_surface_sq_sum is None:
                    paper_surface_sq_sum = torch.zeros(surface_target.shape[-1], device=device)
                paper_surface_sq_sum += (surface_pred - surface_target).square().sum(dim=(0, 1)).to(device)
                paper_surface_nodes += batch.surface_anchor_target.shape[0] * batch.surface_anchor_target.shape[1]
            if batch.volume_anchor_target is not None and outputs["volume_preds"] is not None:
                volume_pred, volume_target = comparison_metric_tensors(
                    outputs["volume_preds"],
                    batch.volume_anchor_target,
                    train_transform=transform,
                    metric_transform=metric_transform,
                )
                if paper_volume_sq_sum is None:
                    paper_volume_sq_sum = torch.zeros(volume_target.shape[-1], device=device)
                paper_volume_sq_sum += (volume_pred - volume_target).square().sum(dim=(0, 1)).to(device)
                paper_volume_nodes += batch.volume_anchor_target.shape[0] * batch.volume_anchor_target.shape[1]
            continue

        if batch.surface_anchor_target is not None and outputs["surface_preds"] is not None:
            pred = transform.invert(outputs["surface_preds"])
            if dataset_name == "drivaerml":
                value = _case_masked_rel_l2(
                    pred[0],
                    batch.surface_anchor_target[0],
                    torch.ones(pred.shape[1], dtype=torch.bool, device=pred.device),
                )
                if value == value:
                    total_surface += value
                    count_surface += 1
            else:
                total_surface += float((pred - batch.surface_anchor_target).abs().mean().cpu().item())
                count_surface += 1
        if batch.volume_anchor_target is not None and outputs["volume_preds"] is not None:
            pred = transform.invert(outputs["volume_preds"])
            if dataset_name == "drivaerml":
                value = _case_masked_rel_l2(
                    pred[0],
                    batch.volume_anchor_target[0],
                    torch.ones(pred.shape[1], dtype=torch.bool, device=pred.device),
                )
                if value == value:
                    total_volume += value
                    count_volume += 1
            else:
                total_volume += float((pred - batch.volume_anchor_target).abs().mean().cpu().item())
                count_volume += 1

    if dataset_name == "airfrans":
        metrics = {"surface_mse": total_surface / max(count_surface, 1), "volume_mse": total_volume / max(count_volume, 1)}
        if surf_channel_sum is not None and count_surface > 0:
            metrics.update(_named_channel_metrics("surface_mse", surf_channel_sum / count_surface, AIRFRANS_FIELD_NAMES))
        if vol_channel_sum is not None and count_volume > 0:
            metrics.update(_named_channel_metrics("volume_mse", vol_channel_sum / count_volume, AIRFRANS_FIELD_NAMES))
        return metrics
    if dataset_name == "drivaerml":
        surface_rel_l2 = total_surface / max(count_surface, 1)
        metrics = {
            "surface_rel_l2": surface_rel_l2,
            "surface_rel_l2_pct": surface_rel_l2 * 100.0,
        }
        if count_volume > 0:
            volume_rel_l2 = total_volume / count_volume
            metrics["volume_rel_l2"] = volume_rel_l2
            metrics["volume_rel_l2_pct"] = volume_rel_l2 * 100.0
        return metrics
    if dataset_name == "tandemfoilset_paper":
        metrics: dict[str, float] = {}
        total_sq_sum = 0.0
        total_nodes = 0
        if paper_surface_sq_sum is not None and paper_surface_nodes > 0:
            surface_channel = paper_surface_sq_sum / paper_surface_nodes
            metrics["surface_mse"] = float(surface_channel.mean().detach().cpu().item())
            metrics.update(_named_channel_metrics("surface_mse", surface_channel, TANDEM_FIELD_NAMES))
            total_sq_sum += float(paper_surface_sq_sum.sum().detach().cpu().item())
            total_nodes += paper_surface_nodes
        if paper_volume_sq_sum is not None and paper_volume_nodes > 0:
            volume_channel = paper_volume_sq_sum / paper_volume_nodes
            metrics["volume_mse"] = float(volume_channel.mean().detach().cpu().item())
            metrics.update(_named_channel_metrics("volume_mse", volume_channel, TANDEM_FIELD_NAMES))
            total_sq_sum += float(paper_volume_sq_sum.sum().detach().cpu().item())
            total_nodes += paper_volume_nodes
        if total_nodes > 0:
            metrics["field_mse"] = total_sq_sum / (total_nodes * len(TANDEM_FIELD_NAMES))
        return metrics
    return {"surface_mae": total_surface / max(count_surface, 1), "volume_mae": total_volume / max(count_volume, 1)}


def train_one_epoch(
    model: torch.nn.Module,
    anp_head: ANPSurfaceDecoder | None,
    loader: DataLoader,
    optimizer,
    scheduler,
    ema: EMAWithWarmup | None,
    anp_ema: EMAWithWarmup | None,
    transform: TargetTransform | TandemTargetTransform,
    device: torch.device,
    model_name: str,
    *,
    amp_mode: str = "none",
    max_batches: int = 0,
    grad_clip: float = 0.0,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    model.train()
    if anp_head is not None:
        anp_head.train()
    running = {"loss": 0.0}
    steps = 0
    micro_batches_total = 0
    batches = loader if max_batches <= 0 else itertools.islice(loader, max_batches)
    batch_iter = iter(batches)
    exhausted = False

    while not exhausted:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_count = 0

        for _ in range(grad_accum_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                exhausted = True
                break

            if model_name == "reference_abupt":
                batch = batch.to(device)
                with autocast_context(device, amp_mode):
                    outputs = model(
                        geometry_position=batch.geometry_position,
                        geometry_supernode_idx=batch.geometry_supernode_idx,
                        geometry_batch_idx=batch.geometry_batch_idx,
                        surface_anchor_position=batch.surface_anchor_position,
                        volume_anchor_position=batch.volume_anchor_position,
                    )
                    loss, _ = loss_abupt(batch, outputs, transform)
            else:
                batch = batch.to(device)
                if isinstance(transform, TandemTargetTransform):
                    prepared = transform.prepare_batch(batch)
                    with autocast_context(device, amp_mode):
                        outputs = model(
                            surface_x=prepared.surface_x,
                            surface_mask=prepared.surface_mask,
                            volume_x=prepared.volume_x,
                            volume_mask=prepared.volume_mask,
                        )
                        outputs = maybe_apply_anp(outputs, prepared, anp_head)
                        loss, _ = loss_grouped_tandem(prepared, outputs)
                else:
                    with autocast_context(device, amp_mode):
                        outputs = model(
                            surface_x=batch.surface_x,
                            surface_mask=batch.surface_mask,
                            volume_x=batch.volume_x,
                            volume_mask=batch.volume_mask,
                        )
                        loss, _ = loss_grouped(batch, outputs, transform)

            accum_loss += float(loss.detach().cpu().item())
            (loss / grad_accum_steps).backward()
            micro_count += 1

        if micro_count == 0:
            break

        all_params = list(model.parameters())
        if anp_head is not None:
            all_params += list(anp_head.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip if grad_clip > 0 else float("inf"))
        running.setdefault("grad_norm_mean", 0.0)
        running["grad_norm_mean"] += float(grad_norm)
        if grad_clip > 0 and float(grad_norm) > grad_clip:
            running.setdefault("grad_clip_events", 0.0)
            running["grad_clip_events"] += 1.0
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
            running["ema_decay_actual"] = min(ema.decay, (1 + ema.step_counter) / (10 + ema.step_counter))
        if anp_head is not None and anp_ema is not None:
            anp_ema.update(anp_head)
        running["loss"] += accum_loss / micro_count
        micro_batches_total += micro_count
        steps += 1

    running["loss"] /= max(steps, 1)
    if "grad_norm_mean" in running:
        running["grad_norm_mean"] /= max(steps, 1)
    running["train_steps"] = float(steps)
    running["micro_batches"] = float(micro_batches_total)
    if scheduler is not None:
        running["lr"] = float(optimizer.param_groups[0]["lr"])
    return running


def write_run_summary(
    path: Path,
    config: TrainConfig,
    bundle: DatasetBundle,
    history: list[dict[str, float]],
    best_epoch: int | None = None,
    best_val_primary_metric_name: str | None = None,
    best_val_primary_metric: float | None = None,
    best_val_metrics: dict[str, float] | None = None,
    best_test_metrics: dict[str, float] | None = None,
    final_test_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "dataset_spec": {
            "name": bundle.spec.name,
            "space_dim": bundle.spec.space_dim,
            "surface_input_dim": bundle.spec.surface_input_dim,
            "surface_output_dim": bundle.spec.surface_output_dim,
            "volume_input_dim": bundle.spec.volume_input_dim,
            "volume_output_dim": bundle.spec.volume_output_dim,
            "pressure_output_index": bundle.spec.pressure_output_index,
            "default_metric": bundle.spec.default_metric,
            "notes": bundle.spec.notes,
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_val_primary_metric_name": best_val_primary_metric_name,
        "best_val_primary_metric": best_val_primary_metric,
        "best_val_metrics": best_val_metrics or {},
        "best_test_metrics": best_test_metrics or {},
        "final_test_metrics": final_test_metrics or {},
    }
    drivaerml_split = drivaerml_split_summary(bundle)
    if drivaerml_split is not None:
        payload["drivaerml_split"] = drivaerml_split
    path.write_text(json.dumps(payload, indent=2))


def drivaerml_split_summary(bundle: DatasetBundle) -> dict[str, object] | None:
    if bundle.spec.name != "drivaerml":
        return None
    store = getattr(bundle.train_dataset, "store", None)
    manifest = getattr(store, "manifest", None)
    if not isinstance(manifest, dict):
        return None
    return {
        "manifest_path": str(store.manifest_path),
        "case_root": str(store.root),
        "surface_split_counts": manifest.get("surface_split_counts", {}),
        "surface_splits": manifest.get("surface_splits", {}),
        "volume_split_counts": manifest.get("volume_split_counts", {}),
        "volume_splits": manifest.get("volume_splits", {}),
        "excluded_case_count": manifest.get("excluded_case_count"),
        "excluded_case_ids": manifest.get("excluded_case_ids", []),
    }


def primary_metric_key(bundle: DatasetBundle, *, phase: str) -> str:
    if bundle.spec.name == "tandemfoilset":
        return f"{phase}_primary/surface_pressure_mae"
    return f"{phase}_primary/{bundle.spec.default_metric}"


def snapshot_module_state(module: torch.nn.Module | None) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def restore_module_state(module: torch.nn.Module | None, state: dict[str, torch.Tensor] | None) -> None:
    if module is None or state is None:
        return
    module.load_state_dict(state)


def best_checkpoint_metric_aliases(metrics: dict[str, float]) -> dict[str, float]:
    return {f"best_{key}": value for key, value in metrics.items()}


def evaluate_phase_metrics(
    *,
    bundle: DatasetBundle,
    config: TrainConfig,
    forward_model: torch.nn.Module,
    anp_head: ANPSurfaceDecoder | None,
    loaders: dict[str, DataLoader],
    transform: TargetTransform | TandemTargetTransform,
    metric_transform: TargetTransform | None,
    device: torch.device,
    phase: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name, loader in loaders.items():
        if config.model == "reference_abupt":
            split_metrics = evaluate_abupt(
                forward_model,
                loader,
                transform,
                metric_transform,
                device,
                amp_mode=config.amp_mode,
                max_batches=config.max_eval_batches,
            )
        else:
            split_metrics = evaluate_grouped(
                forward_model,
                anp_head,
                loader,
                transform,
                metric_transform,
                device,
                amp_mode=config.amp_mode,
                max_batches=config.max_eval_batches,
            )
        metrics.update({f"{split_name}/{name}": value for name, value in split_metrics.items()})
        if phase == "val" and split_name in LEGACY_VAL_ALIAS and "mae_surf_p" in split_metrics:
            metrics[f"legacy_noam/{LEGACY_VAL_ALIAS[split_name]}"] = split_metrics["mae_surf_p"]

    if bundle.spec.name == "tandemfoilset":
        suffix = {
            "val": "val_eq4/surface_pressure_mae",
            "test": "test_eq4/surface_pressure_mae",
        }[phase]
        eq4_keys = {
            "val": [
                "val_single_in_dist/surface_pressure_mae",
                "val_geom_camber_rc/surface_pressure_mae",
                "val_geom_camber_cruise/surface_pressure_mae",
                "val_re_rand/surface_pressure_mae",
            ],
            "test": [
                "test_single_in_dist/surface_pressure_mae",
                "test_geom_camber_rc/surface_pressure_mae",
                "test_geom_camber_cruise/surface_pressure_mae",
                "test_re_rand/surface_pressure_mae",
            ],
        }[phase]
        eq4_values = [metrics[key] for key in eq4_keys if key in metrics]
        if len(eq4_values) == 4:
            metrics[suffix] = sum(eq4_values) / 4.0

    return add_primary_metric_aliases(bundle, metrics, phase=phase)


def compute_tandem_phys_stats(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    asinh_pressure: bool,
    asinh_scale: float,
) -> TargetTransformStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_grouped)
    phys_sum = torch.zeros(3, device=device)
    phys_sq_sum = torch.zeros(3, device=device)
    phys_n = 0.0
    for batch in loader:
        batch = batch.to(device)
        volume_y = batch.volume_y if batch.volume_y is not None else batch.surface_y.new_zeros(batch.surface_y.shape[0], 0, batch.surface_y.shape[-1])
        volume_mask = batch.volume_mask if batch.volume_mask is not None else batch.surface_mask.new_zeros(batch.surface_mask.shape[0], 0)
        full_y = torch.cat([batch.surface_y, volume_y], dim=1)
        full_mask = torch.cat([batch.surface_mask, volume_mask], dim=1)
        umag, q = TandemTargetTransform._umag_q(full_y, full_mask)
        y_phys = TandemTargetTransform._phys_norm(full_y, umag, q)
        if asinh_pressure:
            y_phys = y_phys.clone()
            y_phys[..., 2:3] = torch.asinh(y_phys[..., 2:3] * asinh_scale)
        mask = full_mask.float().unsqueeze(-1)
        phys_sum += (y_phys * mask).sum(dim=(0, 1))
        phys_sq_sum += (y_phys.square() * mask).sum(dim=(0, 1))
        phys_n += full_mask.float().sum().item()
    mean = (phys_sum / phys_n).float()
    std = ((phys_sq_sum / phys_n - mean.square()).clamp(min=0.0).sqrt()).clamp(min=1e-6).float()
    return TargetTransformStats(y_mean=mean, y_std=std)


def main() -> None:
    config = parse_args()
    if config.seed != 0:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs_env = os.getenv("SENPAI_MAX_EPOCHS")
    timeout_env = os.getenv("SENPAI_TIMEOUT_MINUTES")
    if max_epochs_env:
        config.epochs = min(config.epochs, int(max_epochs_env))

    import core.datasets as _ds
    _ds.RUNTIME_CACHE_CASES = 3000

    if config.dataset in ("tandemfoil", "tandemfoilset"):
        from tandemfoil.data import prepare_multi
        _orig_init = _ds.TandemFoilCaseDataset.__init__
        _shared_base = [None]

        def _patched_init(self, split_indices, manifest_path=_ds.DEFAULT_TANDEM_MANIFEST, *, debug=False):
            if _shared_base[0] is None:
                _orig_init(self, split_indices, manifest_path, debug=debug)
                _shared_base[0] = self.base
            else:
                self.base = _shared_base[0]
                self.indices = list(split_indices)

        _ds.TandemFoilCaseDataset.__init__ = _patched_init

    bundle = build_bundle(config)
    resolved_num_workers = resolve_num_workers(config, bundle.spec.name)
    if bundle.spec.name == "tandemfoilset":
        phys_stats = compute_tandem_phys_stats(
            bundle.train_dataset,
            batch_size=config.batch_size,
            num_workers=resolved_num_workers,
            device=device,
            asinh_pressure=config.asinh_pressure,
            asinh_scale=config.asinh_scale,
        )
        transform: TargetTransform | TandemTargetTransform = TandemTargetTransform(
            stats=bundle.target_stats,
            phys_stats=phys_stats,
            config=config,
        )
        metric_transform = None
    else:
        transform = TargetTransform(
            pressure_index=bundle.spec.pressure_output_index,
            stats_mean=bundle.target_stats.y_mean,
            stats_std=bundle.target_stats.y_std,
            asinh_pressure=config.asinh_pressure,
            asinh_scale=config.asinh_scale,
        )
        metric_transform = None
        if bundle.spec.name in {"airfrans", "tandemfoilset_paper"}:
            # Keep literature-facing metrics in the dataset's raw z-score space
            # even when training applies an auxiliary target transform.
            metric_transform = TargetTransform(
                pressure_index=bundle.spec.pressure_output_index,
                stats_mean=bundle.target_stats.y_mean,
                stats_std=bundle.target_stats.y_std,
                asinh_pressure=False,
                asinh_scale=config.asinh_scale,
            )

    train_loader, val_loaders, test_loaders = build_loaders(config, bundle, num_workers=resolved_num_workers)
    model = build_model(config, bundle).to(device)
    forward_model = torch.compile(model) if config.compile_model and device.type == "cuda" else model
    anp_head = None
    if bundle.spec.name == "tandemfoilset" and config.anp_srf:
        anp_head = ANPSurfaceDecoder(
            hidden_dim=getattr(model, "n_hidden", 192),
            output_dim=bundle.spec.surface_output_dim,
        ).to(device)

    params = list(model.parameters())
    if anp_head is not None:
        params.extend(list(anp_head.parameters()))
    optimizer = build_optimizer(params, config)
    scheduler = None
    if config.cosine_t_max > 0:
        base_optimizer = optimizer.optimizer if isinstance(optimizer, Lookahead) else optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=config.cosine_t_max)

    ema = EMAWithWarmup(model, decay=config.ema_decay) if config.use_ema else None
    anp_ema = EMAWithWarmup(anp_head, decay=config.ema_decay) if config.use_ema and anp_head is not None else None
    history: list[dict[str, float]] = []
    best_epoch: int | None = None
    best_val_primary_metric_name = primary_metric_key(bundle, phase="val")
    best_val_primary_metric: float | None = None
    best_val_metrics: dict[str, float] = {}
    best_model_state: dict[str, torch.Tensor] | None = None
    best_anp_state: dict[str, torch.Tensor] | None = None

    run = None
    if config.wandb_name:
        run_config = asdict(config)
        run_config["effective_batch_size"] = config.batch_size * config.grad_accum_steps
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "senpai-v1"),
            entity=os.getenv("WANDB_ENTITY"),
            name=config.wandb_name,
            group=config.wandb_group or None,
            config=run_config,
            tags=[config.dataset, config.model],
        )

    start_time = time.monotonic()
    timeout_seconds = None if not timeout_env else float(timeout_env) * 60.0

    for epoch in range(1, config.epochs + 1):
        if timeout_seconds is not None and epoch > 1 and (time.monotonic() - start_time) >= timeout_seconds:
            break
        train_metrics = train_one_epoch(
            model=forward_model,
            anp_head=anp_head,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            anp_ema=anp_ema,
            transform=transform,
            device=device,
            model_name=config.model,
            amp_mode=config.amp_mode,
            max_batches=config.max_train_batches,
            grad_clip=config.grad_clip,
            grad_accum_steps=config.grad_accum_steps,
        )

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        if anp_head is not None and anp_ema is not None:
            anp_ema.store(anp_head)
            anp_ema.copy_to(anp_head)

        eval_metrics = evaluate_phase_metrics(
            bundle=bundle,
            config=config,
            forward_model=forward_model,
            anp_head=anp_head,
            loaders=val_loaders,
            transform=transform,
            metric_transform=metric_transform,
            device=device,
            phase="val",
        )

        current_primary_metric = eval_metrics.get(best_val_primary_metric_name)
        if current_primary_metric is not None and (
            best_val_primary_metric is None or current_primary_metric < best_val_primary_metric
        ):
            best_epoch = epoch
            best_val_primary_metric = current_primary_metric
            best_val_metrics = dict(eval_metrics)
            best_model_state = snapshot_module_state(model)
            best_anp_state = snapshot_module_state(anp_head)

        if ema is not None:
            ema.restore(model)
        if anp_head is not None and anp_ema is not None:
            anp_ema.restore(anp_head)

        epoch_metrics = {"epoch": float(epoch), **train_metrics, **eval_metrics}
        history.append(epoch_metrics)
        if run is not None:
            wandb.log(epoch_metrics, step=epoch)
            run.summary["epoch"] = epoch
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

    if best_epoch is not None:
        print(
            json.dumps(
                {
                    "best_checkpoint": {
                        "epoch": float(best_epoch),
                        "val_primary_metric_name": best_val_primary_metric_name,
                        "val_primary_metric": best_val_primary_metric,
                    }
                },
                sort_keys=True,
            )
        )

    final_test_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    terminal_model_state: dict[str, torch.Tensor] | None = None
    terminal_anp_state: dict[str, torch.Tensor] | None = None
    if test_loaders:
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        if anp_head is not None and anp_ema is not None:
            anp_ema.store(anp_head)
            anp_ema.copy_to(anp_head)

        final_test_metrics = evaluate_phase_metrics(
            bundle=bundle,
            config=config,
            forward_model=forward_model,
            anp_head=anp_head,
            loaders=test_loaders,
            transform=transform,
            metric_transform=metric_transform,
            device=device,
            phase="test",
        )

        if ema is not None:
            ema.restore(model)
        if anp_head is not None and anp_ema is not None:
            anp_ema.restore(anp_head)
        terminal_model_state = snapshot_module_state(model)
        terminal_anp_state = snapshot_module_state(anp_head)
        if run is not None:
            wandb.log(final_test_metrics, step=int(history[-1]["epoch"]) if history else 0)
            run.summary.update(final_test_metrics)
        print(json.dumps({"final_test_metrics": final_test_metrics}, sort_keys=True), flush=True)

        if best_model_state is not None:
            restore_module_state(model, best_model_state)
            restore_module_state(anp_head, best_anp_state)
            best_test_metrics = evaluate_phase_metrics(
                bundle=bundle,
                config=config,
                forward_model=forward_model,
                anp_head=anp_head,
                loaders=test_loaders,
                transform=transform,
                metric_transform=metric_transform,
                device=device,
                phase="test",
            )
            if run is not None:
                best_checkpoint_metrics: dict[str, float | int | str] = {
                    "best_epoch": int(best_epoch) if best_epoch is not None else -1,
                    "best_val_primary_metric_name": best_val_primary_metric_name or "",
                }
                if best_val_primary_metric is not None:
                    best_checkpoint_metrics["best_val_primary_metric"] = best_val_primary_metric
                best_checkpoint_metrics.update({f"best_val/{key}": value for key, value in best_val_metrics.items()})
                best_checkpoint_metrics.update({f"best_test/{key}": value for key, value in best_test_metrics.items()})
                best_checkpoint_metrics.update(best_checkpoint_metric_aliases(best_val_metrics))
                best_checkpoint_metrics.update(best_checkpoint_metric_aliases(best_test_metrics))
                wandb.log(
                    best_checkpoint_metrics,
                    step=int(best_epoch) if best_epoch is not None else (int(history[-1]["epoch"]) if history else 0),
                )
                run.summary.update(best_checkpoint_metrics)
            print(
                json.dumps(
                    {
                        "best_test_metrics": {
                            "epoch": float(best_epoch) if best_epoch is not None else None,
                            **best_test_metrics,
                        }
                    },
                    sort_keys=True,
                )
            )
            restore_module_state(model, terminal_model_state)
            restore_module_state(anp_head, terminal_anp_state)

    output_dir = Path(config.output_dir)
    write_run_summary(
        output_dir / f"{config.dataset}_{config.model}_summary.json",
        config,
        bundle,
        history,
        best_epoch=best_epoch,
        best_val_primary_metric_name=best_val_primary_metric_name,
        best_val_primary_metric=best_val_primary_metric,
        best_val_metrics=best_val_metrics,
        best_test_metrics=best_test_metrics,
        final_test_metrics=final_test_metrics,
    )
    if config.save_checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        if terminal_model_state is not None:
            torch.save(terminal_model_state, output_dir / f"{config.dataset}_{config.model}.pt")
        else:
            torch.save(model.state_dict(), output_dir / f"{config.dataset}_{config.model}.pt")
        if best_model_state is not None:
            torch.save(best_model_state, output_dir / f"{config.dataset}_{config.model}_best.pt")
        if anp_head is not None:
            if terminal_anp_state is not None:
                torch.save(terminal_anp_state, output_dir / f"{config.dataset}_{config.model}_anp.pt")
            else:
                torch.save(anp_head.state_dict(), output_dir / f"{config.dataset}_{config.model}_anp.pt")
            if best_anp_state is not None:
                torch.save(best_anp_state, output_dir / f"{config.dataset}_{config.model}_anp_best.pt")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
