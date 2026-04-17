# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Chart rendering for LLM vision consumption.

These functions produce PNGs that Claude reads back with the Read tool to
form gestalt impressions of training runs — the visual complement to the
numeric features in `training_diagnostics.py`.

Default output directory: `/tmp/wandb_plots/<run_id>/`. Ephemeral per
design — repo should not fill up with run artefacts.

All public functions return the path to the written PNG.

Usage:
    from curve_plots import (
        plot_single_run_overview,
        plot_run_comparison,
        plot_grad_histogram_heatmap,
        plot_grad_norm_by_layer,
    )

    png = plot_single_run_overview(run, step_key="_step")
    # Then Read the png path with the Read tool so Claude sees the image.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # safe for headless pods
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from training_diagnostics import (  # noqa: E402
    curve_features,
    grad_histogram_features,
)
from wandb_helpers import fast_scan_history  # noqa: E402


DEFAULT_ROOT = Path("/tmp/wandb_plots")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_out_dir(run_id: str, out_dir: str | os.PathLike | None) -> Path:
    root = Path(out_dir) if out_dir else DEFAULT_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_history(run: Any, keys: list[str], samples: int = 2000) -> pd.DataFrame:
    """Fast downsampled history for plotting. Uses server-side bucketing."""
    df = run.history(samples=samples, keys=keys, pandas=True)
    if df is None or len(df) == 0:
        # Fall back to full scan (prefers beta_scan_history parquet).
        df = pd.DataFrame(list(fast_scan_history(run, keys=keys)))
    return df


def _mark_spikes(ax: plt.Axes, values: np.ndarray, steps: np.ndarray, features: dict) -> None:
    spikes = features.get("spikes", [])
    if not spikes:
        return
    spike_steps = [s["step"] for s in spikes]
    spike_vals = [s["value"] for s in spikes]
    ax.scatter(spike_steps, spike_vals, c="red", s=16, zorder=5, label=f"{len(spikes)} spike(s)")


def _panel(ax: plt.Axes, steps: np.ndarray, values: np.ndarray, title: str,
           direction: str = "decreasing", log_y: bool = False) -> None:
    ax.plot(steps, values, linewidth=1.2, color="#1f77b4")
    feats = curve_features(values, steps, direction=direction)
    _mark_spikes(ax, values, steps, feats)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_y and np.nanmin(values) > 0:
        ax.set_yscale("log")
    # Annotate: final, min, smoothness.
    txt = (f"final={feats.get('last', float('nan')):.3g}  "
           f"min={feats.get('min', float('nan')):.3g}  "
           f"smooth={feats.get('smoothness', float('nan')):.2g}  "
           f"spikes={feats.get('spike_count', 0)}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", family="monospace",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7})


# ---------------------------------------------------------------------------
# Single-run overview
# ---------------------------------------------------------------------------

DEFAULT_OVERVIEW_METRICS: dict[str, dict[str, Any]] = {
    "train_loss": {"keys": ["loss", "train/loss", "train_loss"], "direction": "decreasing", "log_y": True},
    "val_loss": {"keys": ["val_loss", "val/loss", "validation/loss"], "direction": "decreasing", "log_y": True},
    "lr": {"keys": ["lr", "learning_rate", "train/lr"], "direction": "auto", "log_y": False},
    "grad_norm": {"keys": ["grad_norm", "train/grad_norm", "gradients/global_norm"], "direction": "decreasing", "log_y": True},
    "accuracy": {"keys": ["accuracy", "val/accuracy", "val_acc"], "direction": "increasing", "log_y": False},
    "throughput": {"keys": ["throughput", "samples_per_sec", "train/throughput"], "direction": "auto", "log_y": False},
}


def _resolve_metric(df_columns: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df_columns:
            return c
    return None


def plot_single_run_overview(
    run: Any,
    step_key: str,
    metrics: list[tuple[str, list[str], str, bool]] | None = None,
    out_dir: str | os.PathLike | None = None,
    samples: int = 2000,
) -> Path:
    """Render a 2x3 small-multiples composite PNG for one run.

    Args:
        run: A W&B Run object.
        step_key: Confirmed x-axis step key.
        metrics: Optional override list of `(title, candidate_keys, direction,
            log_y)` tuples. If omitted, uses the default 6-panel layout
            (train loss, val loss, lr, grad norm, accuracy, throughput),
            auto-resolving each candidate against the run's logged keys.
        out_dir: Override output directory. Default
            `/tmp/wandb_plots/<run_id>/`.
        samples: Server-side sample count passed to `run.history`.

    Returns:
        Path to the written PNG.
    """
    out = _ensure_out_dir(run.id, out_dir)

    # Decide the 6 panels.
    if metrics is None:
        panels: list[tuple[str, list[str], str, bool]] = [
            (title, cfg["keys"], cfg["direction"], cfg["log_y"])
            for title, cfg in DEFAULT_OVERVIEW_METRICS.items()
        ]
    else:
        panels = metrics

    # Pull all needed keys in one history call.
    all_keys: list[str] = [step_key]
    for _, candidates, _, _ in panels:
        all_keys.extend(candidates)
    df = _load_history(run, keys=list(dict.fromkeys(all_keys)), samples=samples)
    cols = set(df.columns)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    steps = df[step_key].to_numpy() if step_key in df.columns else np.arange(len(df))

    for ax, (title, candidates, direction, log_y) in zip(axes.flat, panels):
        resolved = _resolve_metric(cols, candidates)
        if resolved is None:
            ax.set_title(f"{title} — not logged", fontsize=10, color="gray")
            ax.text(0.5, 0.5, "metric not found", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
            ax.grid(True, alpha=0.3)
            continue
        series = df[[step_key, resolved]].dropna()
        if len(series) == 0:
            ax.set_title(f"{title} ({resolved}) — empty", fontsize=10, color="gray")
            continue
        s = series[step_key].to_numpy()
        v = series[resolved].to_numpy()
        _panel(ax, s, v, f"{title} ({resolved})", direction=direction, log_y=log_y)

    for ax in axes[-1]:
        ax.set_xlabel(step_key)
    fig.suptitle(f"{run.name} [{run.id}]   x-axis: {step_key}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = out / "overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------

def plot_run_comparison(
    runs: list[Any],
    metric: str,
    step_key: str,
    out_dir: str | os.PathLike | None = None,
    max_runs: int = 6,
    highlight: str | None = None,
    samples: int = 2000,
    log_y: bool = True,
) -> Path:
    """Overlay a single metric across multiple runs.

    Args:
        runs: List of W&B Run objects. Must be <= `max_runs` unless caller
            explicitly raises the cap.
        metric: Metric to compare.
        step_key: Confirmed x-axis.
        out_dir: Override output directory. Default
            `/tmp/wandb_plots/compare_<N>runs_<metric>/`.
        max_runs: Hard cap on overlay density. Default 6.
        highlight: Optional run id or name to render bold.
        samples: history() sample count.
        log_y: Log-scale the y-axis (sensible for losses).

    Returns:
        Path to the written PNG.
    """
    if len(runs) > max_runs:
        raise ValueError(
            f"plot_run_comparison got {len(runs)} runs but max_runs={max_runs}. "
            "Pass a smaller list (e.g. top-k by final metric) or explicitly "
            "raise max_runs if you know the chart will still be readable."
        )
    if not runs:
        raise ValueError("plot_run_comparison got an empty run list.")

    slug = metric.replace("/", "_")
    root = Path(out_dir) if out_dir else DEFAULT_ROOT / f"compare_{len(runs)}runs_{slug}"
    root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors  # up to 10 distinct colors
    aligned: list[pd.Series] = []

    for i, run in enumerate(runs):
        df = _load_history(run, keys=[metric, step_key], samples=samples)
        if metric not in df.columns or step_key not in df.columns:
            continue
        df = df[[step_key, metric]].dropna()
        if len(df) == 0:
            continue
        is_hl = highlight is not None and (highlight == run.id or highlight == run.name)
        lw = 2.4 if is_hl else 1.2
        alpha = 1.0 if is_hl else 0.85
        ax.plot(df[step_key].to_numpy(), df[metric].to_numpy(),
                label=run.name, linewidth=lw, alpha=alpha,
                color=colors[i % len(colors)])
        aligned.append(df.set_index(step_key)[metric])

    # Faint min-max band across runs.
    if len(aligned) >= 2:
        concat = pd.concat(aligned, axis=1).sort_index()
        lo = concat.min(axis=1)
        hi = concat.max(axis=1)
        ax.fill_between(concat.index, lo, hi, alpha=0.08, color="gray",
                        label="min-max band")

    if log_y and all((s.dropna() > 0).all() for s in aligned if len(s)):
        ax.set_yscale("log")

    ax.set_title(f"{metric}  —  {len(runs)} runs  (x: {step_key})", fontsize=12)
    ax.set_xlabel(step_key)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    path = root / f"comparison_{slug}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Gradient histogram heatmap
# ---------------------------------------------------------------------------

def plot_grad_histogram_heatmap(
    run: Any,
    layer_prefix: str = "gradients/",
    metric: str = "mean_abs",
    step_key: str = "_step",
    out_dir: str | os.PathLike | None = None,
    n_step_buckets: int = 60,
) -> Path:
    """Per-layer gradient-histogram stat as a heatmap (layer x step).

    Reads histogram-logged gradients (typically from `wandb.watch`),
    computes per-bin summary stats, and renders a heatmap where:
      - rows = layers (ordered by name)
      - columns = step buckets (binned for readability)
      - color = chosen stat (`mean_abs`, `kurtosis`, `variance`, `max_abs`)

    Args:
        run: A W&B Run object.
        layer_prefix: Prefix for histogram keys.
        metric: Which per-histogram stat to visualize.
        step_key: Confirmed step key.
        out_dir: Override output directory.
        n_step_buckets: Column count in the heatmap.

    Returns:
        Path to the written PNG.
    """
    df = grad_histogram_features(run, layer_prefix=layer_prefix, step_key=step_key)
    if len(df) == 0:
        raise ValueError(f"No gradient histogram data for run {run.id}.")
    if metric not in df.columns:
        raise ValueError(f"Unknown histogram metric `{metric}`. "
                         f"Available: {list(df.columns)}")

    # Bucket steps into N evenly-spaced bins.
    step_min, step_max = df["step"].min(), df["step"].max()
    edges = np.linspace(step_min, step_max, n_step_buckets + 1)
    df["step_bucket"] = np.clip(
        np.searchsorted(edges, df["step"], side="right") - 1,
        0, n_step_buckets - 1,
    )

    pivot = df.pivot_table(
        index="layer", columns="step_bucket", values=metric, aggfunc="mean",
    )
    pivot = pivot.sort_index()
    bucket_centers = 0.5 * (edges[:-1] + edges[1:])

    out = _ensure_out_dir(run.id, out_dir)
    fig, ax = plt.subplots(figsize=(max(10, n_step_buckets * 0.15),
                                    max(4, len(pivot) * 0.25)))

    data = pivot.to_numpy()
    norm = matplotlib.colors.LogNorm(vmin=max(np.nanmin(data[data > 0]), 1e-12),
                                     vmax=np.nanmax(data)) if (data > 0).any() else None
    im = ax.imshow(data, aspect="auto", cmap="viridis", norm=norm,
                   interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    step_tick_idx = np.linspace(0, n_step_buckets - 1, min(10, n_step_buckets)).astype(int)
    ax.set_xticks(step_tick_idx)
    ax.set_xticklabels([f"{bucket_centers[i]:.0f}" for i in step_tick_idx], fontsize=8)
    ax.set_xlabel(step_key)
    ax.set_ylabel("layer")
    ax.set_title(f"{run.name} — grad histograms, {metric} (log color)")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()

    path = out / f"grad_histograms_{metric}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Per-layer grad norm small-multiples
# ---------------------------------------------------------------------------

def plot_grad_norm_by_layer(
    run: Any,
    step_key: str,
    layer_prefix: str = "parameters/",
    max_layers: int = 16,
    out_dir: str | os.PathLike | None = None,
    samples: int = 2000,
) -> Path:
    """Small-multiples of per-layer scalar grad norms.

    Looks for scalar (non-histogram) keys matching `layer_prefix*` — e.g.
    `parameters/layer1.weight_grad_norm`. Renders up to `max_layers`
    subplots on a shared x-axis.
    """
    # Probe keys.
    sample = []
    for i, row in enumerate(fast_scan_history(run)):
        sample.append(row)
        if i + 1 >= 20:
            break
    layer_keys = sorted({
        k for row in sample for k, v in row.items()
        if k.startswith(layer_prefix)
        and isinstance(v, (int, float)) and not isinstance(v, bool)
    })
    if not layer_keys:
        raise ValueError(
            f"No scalar per-layer keys found with prefix `{layer_prefix}`. "
            "If you want per-layer grad norms, log them as scalars "
            f"(e.g. `wandb.log({{'{layer_prefix}layer1_grad_norm': ...}})`)."
        )
    overflow = max(0, len(layer_keys) - max_layers)
    layer_keys = layer_keys[:max_layers]

    df = _load_history(run, keys=[step_key, *layer_keys], samples=samples)
    out = _ensure_out_dir(run.id, out_dir)

    n = len(layer_keys)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows),
                             sharex=True, squeeze=False)

    for ax, key in zip(axes.flat, layer_keys):
        series = df[[step_key, key]].dropna()
        if len(series) == 0:
            ax.set_title(key, fontsize=8, color="gray")
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=8)
            continue
        ax.plot(series[step_key].to_numpy(), series[key].to_numpy(),
                linewidth=1.0, color="#2ca02c")
        ax.set_title(key[len(layer_prefix):], fontsize=8)
        ax.grid(True, alpha=0.3)
        if (series[key] > 0).all():
            ax.set_yscale("log")

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel(step_key)
    title = f"{run.name} — per-layer grad norm"
    if overflow:
        title += f"   (showing {n}/{n + overflow} layers)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = out / "grad_norm_by_layer.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
