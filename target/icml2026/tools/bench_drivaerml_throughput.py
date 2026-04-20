from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import TrainConfig, TargetTransform, build_bundle, build_loaders, build_model, build_optimizer, loss_grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microbenchmark DrivAerML train/eval throughput")
    parser.add_argument("--model", default="senpai_transolver", choices=["senpai_transolver", "reference_transolver"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp-mode", default="none", choices=["none", "bf16"])
    parser.add_argument("--surface-refine", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-surface-points", type=int, default=1_048_576)
    parser.add_argument("--eval-surface-points", type=int, default=1_048_576)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def forward_loss(
    model: torch.nn.Module,
    batch,
    transform: TargetTransform,
    amp_mode: str,
):
    if amp_mode == "bf16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
            loss, _ = loss_grouped(batch, outputs, transform)
        return outputs, loss
    outputs = model(
        surface_x=batch.surface_x,
        surface_mask=batch.surface_mask,
        volume_x=batch.volume_x,
        volume_mask=batch.volume_mask,
    )
    loss, _ = loss_grouped(batch, outputs, transform)
    return outputs, loss


def benchmark_eval_step(model, batch, transform: TargetTransform, amp_mode: str) -> None:
    if amp_mode == "bf16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                surface_x=batch.surface_x,
                surface_mask=batch.surface_mask,
                volume_x=batch.volume_x,
                volume_mask=batch.volume_mask,
            )
            pred_surface = transform.invert(outputs["surface_preds"])
    else:
        outputs = model(
            surface_x=batch.surface_x,
            surface_mask=batch.surface_mask,
            volume_x=batch.volume_x,
            volume_mask=batch.volume_mask,
        )
        pred_surface = transform.invert(outputs["surface_preds"])

    valid = batch.surface_mask[0].bool()
    if valid.any():
        pred_valid = pred_surface[0][valid]
        target_valid = batch.surface_y[0][valid]
        target_sq = float(target_valid.square().sum().detach().cpu().item())
        if target_sq > 0.0:
            _ = float((pred_valid - target_valid).square().sum().detach().cpu().item())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(
        dataset="drivaerml",
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=1,
        surface_refine=args.surface_refine,
        drivaerml_train_surface_points=args.train_surface_points,
        drivaerml_eval_surface_points=args.eval_surface_points,
    )
    bundle = build_bundle(config)
    transform = TargetTransform(
        pressure_index=bundle.spec.pressure_output_index,
        stats_mean=bundle.target_stats.y_mean,
        stats_std=bundle.target_stats.y_std,
        asinh_pressure=config.asinh_pressure,
        asinh_scale=config.asinh_scale,
    )
    train_loader, val_loaders, test_loaders = build_loaders(config, bundle)
    val_loader = next(iter(val_loaders.values()))
    test_loader = next(iter(test_loaders.values()))

    model = build_model(config, bundle).to(device)
    optimizer = build_optimizer(model.parameters(), config)
    base_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=config.cosine_t_max)

    summary = {
        "model": args.model,
        "num_workers": args.num_workers,
        "amp_mode": args.amp_mode,
        "surface_refine": args.surface_refine,
        "compile": args.compile,
        "train_views": len(bundle.train_dataset),
        "val_views": len(next(iter(bundle.val_datasets.values()))),
        "test_views": len(next(iter(bundle.test_datasets.values()))),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader),
    }
    if args.compile:
        model = torch.compile(model)
    print(json.dumps({"config": summary}, sort_keys=True))

    torch.cuda.reset_peak_memory_stats(device)

    train_iter = iter(train_loader)
    train_times = {"fetch": [], "to_device": [], "forward_loss": [], "backward_step": [], "total": []}
    finite_losses = 0
    for step in range(args.steps):
        t0 = time.perf_counter()
        batch = next(train_iter)
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        sync()
        t2 = time.perf_counter()
        _, loss = forward_loss(model, batch, transform, args.amp_mode)
        sync()
        t3 = time.perf_counter()
        loss.backward()
        optimizer.step()
        scheduler.step()
        sync()
        t4 = time.perf_counter()
        if torch.isfinite(loss):
            finite_losses += 1
        if step >= args.warmup:
            train_times["fetch"].append(t1 - t0)
            train_times["to_device"].append(t2 - t1)
            train_times["forward_loss"].append(t3 - t2)
            train_times["backward_step"].append(t4 - t3)
            train_times["total"].append(t4 - t0)

    model.eval()
    eval_iter = iter(val_loader)
    eval_times = {"fetch": [], "to_device": [], "forward_metric": [], "total": []}
    with torch.no_grad():
        for step in range(args.eval_steps):
            t0 = time.perf_counter()
            batch = next(eval_iter)
            t1 = time.perf_counter()
            batch = batch.to(device)
            sync()
            t2 = time.perf_counter()
            benchmark_eval_step(model, batch, transform, args.amp_mode)
            sync()
            t3 = time.perf_counter()
            if step >= args.warmup:
                eval_times["fetch"].append(t1 - t0)
                eval_times["to_device"].append(t2 - t1)
                eval_times["forward_metric"].append(t3 - t2)
                eval_times["total"].append(t3 - t0)

    train_total = mean(train_times["total"])
    eval_total = mean(eval_times["total"])
    epoch_estimate_s = (
        len(train_loader) * train_total
        + (len(val_loader) + len(test_loader)) * eval_total
    )
    result = {
        **summary,
        "finite_losses": finite_losses,
        "train_fetch_s": round(mean(train_times["fetch"]), 4),
        "train_to_device_s": round(mean(train_times["to_device"]), 4),
        "train_forward_loss_s": round(mean(train_times["forward_loss"]), 4),
        "train_backward_step_s": round(mean(train_times["backward_step"]), 4),
        "train_total_s": round(train_total, 4),
        "eval_fetch_s": round(mean(eval_times["fetch"]), 4),
        "eval_to_device_s": round(mean(eval_times["to_device"]), 4),
        "eval_forward_metric_s": round(mean(eval_times["forward_metric"]), 4),
        "eval_total_s": round(eval_total, 4),
        "epoch_estimate_min": round(epoch_estimate_s / 60.0, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
    }
    print("BENCH_SUMMARY " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
