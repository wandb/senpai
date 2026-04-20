# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.architectures import ABUPTReference, ReferenceTransolver, SenpaiTransolver
from core.contracts import ABUPTBatch, DatasetBundle, GroupedBatch
from core.datasets import ABUPTCollate, build_dataset_bundle, collate_grouped
from core.optim import EMA, Lion, Lookahead


@dataclass
class TrainConfig:
    dataset: str = "tandemfoil"
    model: str = "senpai_transolver"
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
    ema_decay: float = 0.999
    ema_start_step: int = 50
    num_workers: int = 0
    debug: bool = False
    surface_only_drivaerml: bool = True
    airfrans_task: str = "full"
    enable_fourier: bool = False
    enable_cp_panel: bool = False
    enable_wake_deficit: bool = False
    enable_wake_angle: bool = False
    asinh_pressure: bool = False
    asinh_scale: float = 0.75
    re_stratified_sampling: bool = False
    geometry_points: int = 25_000
    geometry_supernodes: int = 4_096
    surface_anchor_points: int = 8_000
    volume_anchor_points: int = 8_000
    save_checkpoint: bool = False


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
        if self.asinh_pressure and self.pressure_index is not None:
            out[..., self.pressure_index] = torch.asinh(out[..., self.pressure_index] * self.asinh_scale)
        if self.stats_mean is not None and self.stats_std is not None and self.stats_mean.numel() == out.shape[-1]:
            out = (out - self.stats_mean.to(out.device)) / self.stats_std.to(out.device).clamp(min=1e-6)
        return out

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        out = y.clone()
        if self.stats_mean is not None and self.stats_std is not None and self.stats_mean.numel() == out.shape[-1]:
            out = out * self.stats_std.to(out.device) + self.stats_mean.to(out.device)
        if self.asinh_pressure and self.pressure_index is not None:
            out[..., self.pressure_index] = torch.sinh(out[..., self.pressure_index]) / max(self.asinh_scale, 1e-6)
        return out


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
    return build_dataset_bundle(
        config.dataset,
        debug=config.debug,
        airfrans_task=config.airfrans_task,
        drivaerml_surface_only=config.surface_only_drivaerml,
        enable_fourier=config.enable_fourier,
        enable_cp_panel=config.enable_cp_panel,
        enable_wake_deficit=config.enable_wake_deficit,
        enable_wake_angle=config.enable_wake_angle,
    )


def cp_panel_prior_index(config: TrainConfig, bundle: DatasetBundle) -> int | None:
    if not config.enable_cp_panel:
        return None
    base_dim = bundle.spec.surface_input_dim
    if config.enable_wake_deficit:
        base_dim -= 2 + (1 if config.enable_wake_angle else 0)
    return base_dim - 1


def build_model(config: TrainConfig, bundle: DatasetBundle) -> torch.nn.Module:
    if config.model == "reference_transolver":
        return ReferenceTransolver(
            space_dim=bundle.spec.space_dim,
            surface_input_dim=bundle.spec.surface_input_dim,
            surface_output_dim=bundle.spec.surface_output_dim,
            volume_input_dim=bundle.spec.volume_input_dim,
            volume_output_dim=bundle.spec.volume_output_dim,
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
            surface_refine=True,
            surface_pressure_prior_idx=prior_idx,
            volume_pressure_prior_idx=prior_idx,
        )
    if config.model == "reference_abupt":
        return ABUPTReference(
            space_dim=bundle.spec.space_dim,
            surface_output_dim=bundle.spec.surface_output_dim,
            volume_output_dim=bundle.spec.volume_output_dim,
        )
    raise ValueError(f"Unknown model: {config.model}")


def build_optimizer(model: torch.nn.Module, config: TrainConfig):
    if config.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    if config.use_lookahead:
        optimizer = Lookahead(optimizer)
    return optimizer


def build_loaders(config: TrainConfig, bundle: DatasetBundle) -> tuple[DataLoader, dict[str, DataLoader], dict[str, DataLoader]]:
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

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=train_collate,
    )
    val_loaders = {
        name: DataLoader(
            ds,
            batch_size=1 if config.model == "reference_abupt" else config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=eval_collate,
        )
        for name, ds in bundle.val_datasets.items()
    }
    test_loaders = {
        name: DataLoader(
            ds,
            batch_size=1 if config.model == "reference_abupt" else config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=eval_collate,
        )
        for name, ds in bundle.test_datasets.items()
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

    suffix = f"/{default_metric}"
    matching = [value for key, value in metrics.items() if key.endswith(suffix)]
    if len(matching) == 1:
        metrics[f"{phase}_primary/{default_metric}"] = matching[0]
    return metrics


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1).float() * pred.shape[-1]
    return diff.sum() / denom


def _case_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    valid = mask.bool()
    if not valid.any():
        return float("nan")
    return float(values[valid].mean().detach().cpu().item())


def _case_masked_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
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


def loss_grouped(
    batch: GroupedBatch,
    outputs: dict[str, torch.Tensor | None],
    transform: TargetTransform,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.tensor(0.0, device=batch.surface_x.device)
    metrics: dict[str, float] = {}
    if batch.surface_y is not None and outputs["surface_preds"] is not None:
        target = transform.apply(batch.surface_y)
        pred = outputs["surface_preds"]
        surf_loss = F.mse_loss(pred[batch.surface_mask], target[batch.surface_mask])
        total = total + surf_loss
        metrics["surface_loss"] = float(surf_loss.detach().cpu().item())
    if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
        target = transform.apply(batch.volume_y)
        pred = outputs["volume_preds"]
        volume_mask = batch.volume_mask
        vol_loss = F.mse_loss(pred[volume_mask], target[volume_mask])
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


@torch.no_grad()
def evaluate_grouped(
    model: torch.nn.Module,
    loader: DataLoader,
    transform: TargetTransform,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dataset_name: str | None = None
    total_surface = 0.0
    total_volume = 0.0
    count_surface = 0
    count_volume = 0
    surf_pressure_abs_sum = 0.0
    surf_pressure_count = 0
    for batch in loader:
        batch = batch.to(device)
        dataset_name = batch.dataset_name
        outputs = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        if dataset_name == "airfrans":
            if batch.surface_y is not None and outputs["surface_preds"] is not None:
                target = transform.apply(batch.surface_y)
                pred = outputs["surface_preds"]
                case_values = (pred - target).square().mean(dim=-1)
                for case_idx in range(case_values.shape[0]):
                    value = _case_masked_mean(case_values[case_idx], batch.surface_mask[case_idx])
                    if value == value:
                        total_surface += value
                        count_surface += 1
            if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
                target = transform.apply(batch.volume_y)
                pred = outputs["volume_preds"]
                case_values = (pred - target).square().mean(dim=-1)
                for case_idx in range(case_values.shape[0]):
                    value = _case_masked_mean(case_values[case_idx], batch.volume_mask[case_idx])
                    if value == value:
                        total_volume += value
                        count_volume += 1
            continue

        pred_surface = None
        pred_volume = None
        if batch.surface_y is not None and outputs["surface_preds"] is not None:
            pred_surface = transform.invert(outputs["surface_preds"])
        if batch.volume_y is not None and outputs["volume_preds"] is not None and batch.volume_mask is not None:
            pred_volume = transform.invert(outputs["volume_preds"])

        if dataset_name == "drivaerml":
            if pred_surface is not None:
                for case_idx in range(pred_surface.shape[0]):
                    value = _case_masked_rel_l2(
                        pred_surface[case_idx],
                        batch.surface_y[case_idx],
                        batch.surface_mask[case_idx],
                    )
                    if value == value:
                        total_surface += value * 100.0
                        count_surface += 1
            if pred_volume is not None and batch.volume_mask is not None and batch.volume_y is not None:
                for case_idx in range(pred_volume.shape[0]):
                    value = _case_masked_rel_l2(
                        pred_volume[case_idx],
                        batch.volume_y[case_idx],
                        batch.volume_mask[case_idx],
                    )
                    if value == value:
                        total_volume += value * 100.0
                        count_volume += 1
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
        return {
            "surface_mse": total_surface / max(count_surface, 1),
            "volume_mse": total_volume / max(count_volume, 1),
        }
    if dataset_name == "drivaerml":
        metrics = {
            "surface_rel_l2_pct": total_surface / max(count_surface, 1),
        }
        if count_volume > 0:
            metrics["volume_rel_l2_pct"] = total_volume / max(count_volume, 1)
        return metrics

    metrics = {
        "surface_mae": total_surface / max(count_surface, 1),
        "volume_mae": total_volume / max(count_volume, 1),
    }
    if surf_pressure_count > 0:
        metrics["surface_pressure_mae"] = surf_pressure_abs_sum / surf_pressure_count
    return metrics


@torch.no_grad()
def evaluate_abupt(
    model: torch.nn.Module,
    loader: DataLoader,
    transform: TargetTransform,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dataset_name: str | None = None
    total_surface = 0.0
    total_volume = 0.0
    count_surface = 0
    count_volume = 0
    for batch in loader:
        batch = batch.to(device)
        dataset_name = batch.dataset_name
        outputs = model(
            geometry_position=batch.geometry_position,
            geometry_supernode_idx=batch.geometry_supernode_idx,
            geometry_batch_idx=batch.geometry_batch_idx,
            surface_anchor_position=batch.surface_anchor_position,
            volume_anchor_position=batch.volume_anchor_position,
        )
        if dataset_name == "airfrans":
            if batch.surface_anchor_target is not None and outputs["surface_preds"] is not None:
                target = transform.apply(batch.surface_anchor_target)
                case_values = (outputs["surface_preds"] - target).square().mean(dim=-1)
                total_surface += float(case_values.mean().detach().cpu().item())
                count_surface += 1
            if batch.volume_anchor_target is not None and outputs["volume_preds"] is not None:
                target = transform.apply(batch.volume_anchor_target)
                case_values = (outputs["volume_preds"] - target).square().mean(dim=-1)
                total_volume += float(case_values.mean().detach().cpu().item())
                count_volume += 1
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
                    total_surface += value * 100.0
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
                    total_volume += value * 100.0
                    count_volume += 1
            else:
                total_volume += float((pred - batch.volume_anchor_target).abs().mean().cpu().item())
                count_volume += 1

    if dataset_name == "airfrans":
        return {
            "surface_mse": total_surface / max(count_surface, 1),
            "volume_mse": total_volume / max(count_volume, 1),
        }
    if dataset_name == "drivaerml":
        metrics = {
            "surface_rel_l2_pct": total_surface / max(count_surface, 1),
        }
        if count_volume > 0:
            metrics["volume_rel_l2_pct"] = total_volume / max(count_volume, 1)
        return metrics
    return {
        "surface_mae": total_surface / max(count_surface, 1),
        "volume_mae": total_volume / max(count_volume, 1),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    ema: EMA | None,
    transform: TargetTransform,
    device: torch.device,
    model_name: str,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0}
    steps = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        if model_name == "reference_abupt":
            batch = batch.to(device)
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
            outputs = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
            loss, _ = loss_grouped(batch, outputs, transform)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        running["loss"] += float(loss.detach().cpu().item())
        steps += 1
    running["loss"] /= max(steps, 1)
    return running


def write_run_summary(
    path: Path,
    config: TrainConfig,
    bundle: DatasetBundle,
    history: list[dict[str, float]],
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
        "final_test_metrics": final_test_metrics or {},
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs_env = os.getenv("SENPAI_MAX_EPOCHS")
    timeout_env = os.getenv("SENPAI_TIMEOUT_MINUTES")
    if max_epochs_env:
        config.epochs = min(config.epochs, int(max_epochs_env))
    bundle = build_bundle(config)
    transform = TargetTransform(
        pressure_index=bundle.spec.pressure_output_index,
        stats_mean=bundle.target_stats.y_mean,
        stats_std=bundle.target_stats.y_std,
        asinh_pressure=config.asinh_pressure,
        asinh_scale=config.asinh_scale,
    )
    train_loader, val_loaders, test_loaders = build_loaders(config, bundle)
    model = build_model(config, bundle).to(device)
    optimizer = build_optimizer(model, config)
    ema = EMA(model, decay=config.ema_decay, start_step=config.ema_start_step) if config.use_ema else None
    history: list[dict[str, float]] = []
    run = None
    if config.wandb_name:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "senpai-v1"),
            entity=os.getenv("WANDB_ENTITY"),
            name=config.wandb_name,
            group=config.wandb_group or None,
            config=asdict(config),
            tags=[config.dataset, config.model],
        )
    start_time = time.monotonic()
    timeout_seconds = None if not timeout_env else float(timeout_env) * 60.0

    for epoch in range(1, config.epochs + 1):
        if timeout_seconds is not None and epoch > 1 and (time.monotonic() - start_time) >= timeout_seconds:
            break
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ema=ema,
            transform=transform,
            device=device,
            model_name=config.model,
        )
        eval_model = model
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        eval_metrics = {}
        for split_name, loader in val_loaders.items():
            if config.model == "reference_abupt":
                split_metrics = evaluate_abupt(model, loader, transform, device)
            else:
                split_metrics = evaluate_grouped(model, loader, transform, device)
            eval_metrics.update({f"{split_name}/{name}": value for name, value in split_metrics.items()})
        if bundle.spec.name == "tandemfoilset":
            eq4_keys = [
                "val_single_in_dist/surface_pressure_mae",
                "val_geom_camber_rc/surface_pressure_mae",
                "val_geom_camber_cruise/surface_pressure_mae",
                "val_re_rand/surface_pressure_mae",
            ]
            eq4_values = [eval_metrics[k] for k in eq4_keys if k in eval_metrics]
            if len(eq4_values) == 4:
                eval_metrics["val_eq4/surface_pressure_mae"] = sum(eq4_values) / 4.0
        eval_metrics = add_primary_metric_aliases(bundle, eval_metrics, phase="val")
        if ema is not None:
            ema.restore(model)
        epoch_metrics = {"epoch": float(epoch), **train_metrics, **eval_metrics}
        history.append(epoch_metrics)
        if run is not None:
            wandb.log(epoch_metrics, step=epoch)
        print(json.dumps(epoch_metrics, sort_keys=True))

    output_dir = Path(config.output_dir)
    final_test_metrics: dict[str, float] = {}
    if test_loaders:
        eval_model = model
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        for split_name, loader in test_loaders.items():
            if config.model == "reference_abupt":
                split_metrics = evaluate_abupt(model, loader, transform, device)
            else:
                split_metrics = evaluate_grouped(model, loader, transform, device)
            final_test_metrics.update({f"{split_name}/{name}": value for name, value in split_metrics.items()})
        if bundle.spec.name == "tandemfoilset":
            eq4_keys = [
                "test_single_in_dist/surface_pressure_mae",
                "test_geom_camber_rc/surface_pressure_mae",
                "test_geom_camber_cruise/surface_pressure_mae",
                "test_re_rand/surface_pressure_mae",
            ]
            eq4_values = [final_test_metrics[k] for k in eq4_keys if k in final_test_metrics]
            if len(eq4_values) == 4:
                final_test_metrics["test_eq4/surface_pressure_mae"] = sum(eq4_values) / 4.0
        final_test_metrics = add_primary_metric_aliases(bundle, final_test_metrics, phase="test")
        if ema is not None:
            ema.restore(model)
        if run is not None:
            wandb.log(final_test_metrics, step=int(history[-1]["epoch"]) if history else 0)
        print(json.dumps({"final_test_metrics": final_test_metrics}, sort_keys=True))

    write_run_summary(
        output_dir / f"{config.dataset}_{config.model}_summary.json",
        config,
        bundle,
        history,
        final_test_metrics=final_test_metrics,
    )
    if config.save_checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / f"{config.dataset}_{config.model}.pt")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
