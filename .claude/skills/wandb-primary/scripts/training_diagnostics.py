# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Curve-shape diagnostics for W&B training runs.

This is the numerical counterpart to `curve_plots.py`. It turns a noisy loss
curve into a compact dict of features that mirror how an experienced ML
researcher eyeballs a chart: spikes, smoothness, checkpoint slopes,
plateaus, divergence. Pair these features with a PNG that Claude reads via
vision — the two cross-check each other.

No plotting, no filesystem side-effects. Only wandb scan_history access;
caller is expected to pass a confirmed step key.

Usage:
    from training_diagnostics import (
        curve_features,
        compare_runs_curves,
        lr_schedule_features,
        grad_norm_features,
        grad_histogram_features,
    )

    feats = curve_features(loss_series, step_series, direction="decreasing")
    df = compare_runs_curves([run_a, run_b], metric="loss", step_key="_step")
"""

from __future__ import annotations

from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from wandb_helpers import discover_history_keys, fast_scan_history


Direction = Literal["decreasing", "increasing", "auto"]


# ---------------------------------------------------------------------------
# Core single-curve features
# ---------------------------------------------------------------------------

def curve_features(
    values: Iterable[float] | pd.Series | np.ndarray,
    steps: Iterable[float] | pd.Series | np.ndarray | None = None,
    n_checkpoints: int = 20,
    direction: Direction = "auto",
    spike_z: float = 3.0,
) -> dict[str, Any]:
    """Extract researcher-intuition features from a single metric curve.

    Args:
        values: Metric values, one per step.
        steps: Step values matching `values`. If None, uses positional index.
        n_checkpoints: Number of evenly-spaced segments to compute a slope
                       over (default 20, i.e. every 5% of the run).
        direction: "decreasing" (loss-like), "increasing" (accuracy-like),
                   or "auto" — pick based on whether first half mean > second
                   half mean.
        spike_z: Z-score threshold for spike detection.

    Returns:
        Dict with numeric features. NaN/Inf values are dropped before stats
        but counted separately.
    """
    v = np.asarray(list(values), dtype=float)
    s = np.arange(len(v), dtype=float) if steps is None else np.asarray(list(steps), dtype=float)

    if len(v) == 0:
        return {"n_points": 0}

    nan_mask = ~np.isfinite(v)
    nan_count = int(nan_mask.sum())
    first_nan_step = float(s[nan_mask][0]) if nan_count else None

    # Drop NaN/Inf for all subsequent stats.
    good = np.isfinite(v)
    v_clean = v[good]
    s_clean = s[good]
    n = len(v_clean)
    if n == 0:
        return {
            "n_points": 0,
            "nan_inf_count": nan_count,
            "first_nan_step": first_nan_step,
        }

    # Direction.
    if direction == "auto":
        half = n // 2 or 1
        direction = "decreasing" if v_clean[:half].mean() > v_clean[-half:].mean() else "increasing"

    # Basic stats.
    argmin_i = int(np.argmin(v_clean))
    argmax_i = int(np.argmax(v_clean))
    final_tail = max(1, n // 10)
    final_slice = v_clean[-final_tail:]

    # Smoothness: rolling std / rolling mean, averaged over the curve.
    window = max(5, n // 50)  # ~2% of steps
    series = pd.Series(v_clean)
    rolling_mean = series.rolling(window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window, center=True, min_periods=1).std()
    # Scale rolling std by the curve's range so smoothness is comparable
    # across metrics of different magnitudes.
    curve_range = float(v_clean.max() - v_clean.min()) or 1.0
    smoothness = float((rolling_std / curve_range).dropna().mean())

    # Spikes: points > spike_z std above local mean (for loss-like) or below
    # (for accuracy-like). We flag in whichever direction is "bad".
    residual = series - rolling_mean
    if direction == "decreasing":
        spike_mask = residual > spike_z * rolling_std
    else:
        spike_mask = residual < -spike_z * rolling_std
    spike_idx = np.where(spike_mask.fillna(False))[0]
    spikes = [
        {
            "step": float(s_clean[i]),
            "value": float(v_clean[i]),
            "z_score": float(residual.iloc[i] / rolling_std.iloc[i])
            if rolling_std.iloc[i] and np.isfinite(rolling_std.iloc[i])
            else float("inf"),
            "magnitude": float(abs(residual.iloc[i])),
        }
        for i in spike_idx
    ]

    # Monotonicity: % of consecutive deltas in the expected direction.
    deltas = np.diff(v_clean)
    if direction == "decreasing":
        mono_pct = float((deltas <= 0).mean() * 100)
    else:
        mono_pct = float((deltas >= 0).mean() * 100)

    # Checkpoint slopes: linear fit over n_checkpoints segments.
    checkpoint_slopes: list[dict[str, float]] = []
    if n >= n_checkpoints * 2:
        edges = np.linspace(0, n, n_checkpoints + 1, dtype=int)
        for i in range(n_checkpoints):
            a, b = edges[i], edges[i + 1]
            if b - a < 2:
                continue
            xs = s_clean[a:b]
            ys = v_clean[a:b]
            # Guard: if steps are constant, skip slope.
            if xs[-1] == xs[0]:
                continue
            slope = float(np.polyfit(xs, ys, 1)[0])
            checkpoint_slopes.append({
                "pct_start": round(100 * a / n, 1),
                "pct_end": round(100 * b / n, 1),
                "step_start": float(xs[0]),
                "step_end": float(xs[-1]),
                "slope": slope,
            })

    # Plateau regions: stretches where rolling |change| is tiny relative
    # to the curve range.
    change = series.diff().abs().rolling(window, min_periods=1).mean()
    plateau_tol = curve_range * 1e-4
    plateau_mask = (change < plateau_tol).fillna(False).to_numpy()
    plateau_regions: list[dict[str, float]] = []
    if plateau_mask.any():
        # Find contiguous runs of True longer than `window`.
        idx = 0
        while idx < len(plateau_mask):
            if plateau_mask[idx]:
                start = idx
                while idx < len(plateau_mask) and plateau_mask[idx]:
                    idx += 1
                end = idx
                if end - start >= window:
                    plateau_regions.append({
                        "start_step": float(s_clean[start]),
                        "end_step": float(s_clean[end - 1]),
                        "value": float(v_clean[start:end].mean()),
                    })
            else:
                idx += 1

    # Divergence: in the last 20%, is the rolling mean of deltas
    # consistently in the wrong direction?
    tail_start = max(0, n - max(window, n // 5))
    tail_delta = deltas[max(0, tail_start - 1):]
    if direction == "decreasing":
        divergent = bool(len(tail_delta) > window and tail_delta.mean() > 0)
    else:
        divergent = bool(len(tail_delta) > window and tail_delta.mean() < 0)
    divergence_start = None
    if divergent:
        # Earliest step in the tail where the running mean crossed zero in
        # the wrong direction.
        rolling_delta = pd.Series(deltas).rolling(window, min_periods=1).mean()
        if direction == "decreasing":
            bad = rolling_delta > 0
        else:
            bad = rolling_delta < 0
        bad_idx = np.where(bad.fillna(False))[0]
        if len(bad_idx):
            divergence_start = float(s_clean[bad_idx[0]])

    return {
        "n_points": n,
        "direction": direction,
        "first": float(v_clean[0]),
        "last": float(v_clean[-1]),
        "min": float(v_clean.min()),
        "max": float(v_clean.max()),
        "argmin_step": float(s_clean[argmin_i]),
        "argmax_step": float(s_clean[argmax_i]),
        "final_10pct_mean": float(final_slice.mean()),
        "final_10pct_std": float(final_slice.std()),
        "smoothness": smoothness,
        "monotonicity_pct": mono_pct,
        "spike_count": len(spikes),
        "spikes": spikes,
        "checkpoint_slopes": checkpoint_slopes,
        "plateau_regions": plateau_regions,
        "divergence": {"detected": divergent, "start_step": divergence_start},
        "nan_inf_count": nan_count,
        "first_nan_step": first_nan_step,
    }


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------

SCALAR_FEATURE_KEYS: tuple[str, ...] = (
    "n_points",
    "first",
    "last",
    "min",
    "max",
    "argmin_step",
    "argmax_step",
    "final_10pct_mean",
    "final_10pct_std",
    "smoothness",
    "monotonicity_pct",
    "spike_count",
    "nan_inf_count",
)


def compare_runs_curves(
    runs: list[Any],
    metric: str,
    step_key: str,
    names: list[str] | None = None,
    direction: Direction = "auto",
) -> pd.DataFrame:
    """Build a DataFrame of curve features across multiple runs.

    One row per run, columns are the scalar features from `curve_features`.
    Non-scalar fields (spike list, slopes, plateaus) are omitted to keep
    the frame compact; grab them per-run via `curve_features` if needed.

    Args:
        runs: List of W&B Run objects.
        metric: Metric name to pull from each run's history.
        step_key: Confirmed x-axis step key. MUST be known — never guessed.
        names: Optional display names; defaults to run.name.
        direction: Passed through to `curve_features`.

    Returns:
        DataFrame indexed by run name. If `len(runs) > 6`, attaches a
        warning in `df.attrs["warning"]`; caller decides whether to proceed.
    """
    rows: list[dict[str, Any]] = []
    warning = None
    if len(runs) > 6:
        warning = (
            f"compare_runs_curves got {len(runs)} runs. Overlay plots become "
            "unreadable past 6; consider selecting top-k or grouping before "
            "plotting. Feature table is still computed."
        )
    for i, run in enumerate(runs):
        display = names[i] if names else run.name
        history = list(fast_scan_history(run, keys=[metric, step_key]))
        values = [r.get(metric) for r in history if r.get(metric) is not None]
        steps = [r.get(step_key) for r in history if r.get(metric) is not None]
        feats = curve_features(values, steps, direction=direction)
        row: dict[str, Any] = {"run": display, "run_id": run.id}
        for k in SCALAR_FEATURE_KEYS:
            row[k] = feats.get(k)
        row["divergence_detected"] = feats.get("divergence", {}).get("detected")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("run")
    if warning:
        df.attrs["warning"] = warning
    return df


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_schedule_features(
    values: Iterable[float] | pd.Series | np.ndarray,
    steps: Iterable[float] | pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize a learning rate schedule.

    Detects warmup length, peak LR + step, and the shape of the decay
    segment (linear / cosine / exponential / constant / unknown). Flags
    any unexpected restarts (LR increases after peak).
    """
    v = np.asarray(list(values), dtype=float)
    s = np.arange(len(v), dtype=float) if steps is None else np.asarray(list(steps), dtype=float)
    good = np.isfinite(v)
    v = v[good]
    s = s[good]
    n = len(v)
    if n == 0:
        return {"n_points": 0}

    peak_i = int(np.argmax(v))
    peak_lr = float(v[peak_i])
    peak_step = float(s[peak_i])

    # Warmup: from start up to peak (if peak isn't right at step 0).
    warmup_len = int(peak_i)
    warmup_slope = None
    if warmup_len >= 3:
        warmup_slope = float(np.polyfit(s[:peak_i + 1], v[:peak_i + 1], 1)[0])

    # Decay segment.
    decay_v = v[peak_i:]
    decay_s = s[peak_i:]
    decay_shape = "unknown"
    if len(decay_v) >= 5:
        # Constant?
        if float(decay_v.std()) / (abs(peak_lr) or 1.0) < 1e-3:
            decay_shape = "constant"
        else:
            # Fit linear vs exponential (log-linear) and compare residuals.
            xs = decay_s - decay_s[0]
            lin_coef = np.polyfit(xs, decay_v, 1)
            lin_resid = float(np.mean((decay_v - np.polyval(lin_coef, xs)) ** 2))
            positive = decay_v[decay_v > 0]
            positive_xs = xs[: len(positive)]
            exp_resid = float("inf")
            if len(positive) >= 5 and len(positive) == len(positive_xs):
                exp_coef = np.polyfit(positive_xs, np.log(positive), 1)
                exp_pred = np.exp(np.polyval(exp_coef, positive_xs))
                exp_resid = float(np.mean((positive - exp_pred) ** 2))
            # Cosine: 0.5 * (peak) * (1 + cos(pi * t / T)).
            t_norm = xs / (xs[-1] or 1.0)
            cosine_pred = 0.5 * peak_lr * (1.0 + np.cos(np.pi * t_norm))
            cos_resid = float(np.mean((decay_v - cosine_pred) ** 2))
            scores = {"linear": lin_resid, "exponential": exp_resid, "cosine": cos_resid}
            decay_shape = min(scores, key=scores.get)

    # Restarts: after a real decay segment, LR rises meaningfully from a local
    # trough. This catches cosine restarts that stay below the initial peak.
    restart_steps: list[float] = []
    local_trough = peak_lr
    i = peak_i + 1
    while i < n:
        local_trough = min(local_trough, float(v[i - 1]))
        if v[i] <= v[i - 1]:
            local_trough = min(local_trough, float(v[i]))
            i += 1
            continue
        if local_trough < peak_lr * 0.99 and v[i] > local_trough * 1.01:
            restart_steps.append(float(s[i]))
            while i + 1 < n and v[i + 1] >= v[i]:
                i += 1
            local_trough = float(v[i])
        i += 1

    return {
        "n_points": n,
        "peak_lr": peak_lr,
        "peak_step": peak_step,
        "warmup_steps": warmup_len,
        "warmup_slope": warmup_slope,
        "final_lr": float(v[-1]),
        "decay_shape": decay_shape,
        "restart_steps": restart_steps,
    }


# ---------------------------------------------------------------------------
# Grad norm
# ---------------------------------------------------------------------------

def grad_norm_features(
    values: Iterable[float] | pd.Series | np.ndarray,
    steps: Iterable[float] | pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Grad-norm specific features on top of generic curve_features.

    Adds tail heaviness (kurtosis — excess kurtosis; heavy-tailed updates
    are concerning) and a `dead_flag` when grad-norm collapses near zero
    and stays there, which usually means a layer has stopped learning.
    """
    feats = curve_features(values, steps, direction="decreasing")
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        feats["kurtosis"] = None
        feats["dead_flag"] = False
        return feats

    mean = v.mean()
    std = v.std() or 1.0
    kurt = float(((v - mean) ** 4).mean() / (std ** 4) - 3.0)
    feats["kurtosis"] = kurt

    # Dead flag: final 10% mean is <1% of the run-wide median and doesn't
    # move.
    tail = v[-max(1, len(v) // 10):]
    median = float(np.median(v)) or 1.0
    feats["dead_flag"] = bool(tail.mean() < median * 0.01 and tail.std() < median * 0.001)
    return feats


# ---------------------------------------------------------------------------
# Grad histograms (per-layer over time)
# ---------------------------------------------------------------------------

def _histogram_stats(hist: Any) -> dict[str, float] | None:
    """Compute summary stats from a W&B-logged histogram payload.

    W&B stores histograms as a dict-like: `{"bins": [...], "values": [...]}`
    or as a wandb.Histogram repr. `values` are counts per bin (len = len(bins) - 1).
    """
    if hist is None:
        return None
    if isinstance(hist, dict):
        bins = hist.get("bins")
        counts = hist.get("values")
    else:
        bins = getattr(hist, "bins", None)
        counts = getattr(hist, "histogram", None) or getattr(hist, "values", None)
    if bins is None or counts is None:
        return None
    bins = np.asarray(bins, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(bins) < 2 or len(counts) < 1:
        return None

    centers = 0.5 * (bins[:-1] + bins[1:])
    total = counts.sum()
    if total <= 0:
        return None
    probs = counts / total

    mean = float((probs * centers).sum())
    variance = float((probs * (centers - mean) ** 2).sum())
    std = variance ** 0.5 or 1e-12
    kurtosis = float((probs * ((centers - mean) / std) ** 4).sum() - 3.0)
    mean_abs = float((probs * np.abs(centers)).sum())
    max_abs = float(np.abs(centers[counts > 0]).max()) if (counts > 0).any() else 0.0
    max_bin_idx = int(np.argmax(counts))
    pct_at_max_bin = float(counts[max_bin_idx] / total * 100)

    return {
        "mean": mean,
        "mean_abs": mean_abs,
        "variance": variance,
        "kurtosis": kurtosis,
        "max_abs": max_abs,
        "pct_at_max_bin": pct_at_max_bin,
    }


def grad_histogram_features(
    run: Any,
    layer_prefix: str = "gradients/",
    step_key: str = "_step",
    probe_rows: int = 500,
) -> pd.DataFrame:
    """Parse W&B-logged gradient histograms into a (layer, step) frame.

    Args:
        run: A W&B Run object.
        layer_prefix: Prefix for gradient histogram keys. Default
            `gradients/` matches wandb.watch's default layout.
        step_key: Confirmed step-axis key.
        probe_rows: Max history rows to scan while discovering histogram keys.

    Returns:
        DataFrame with one row per (layer, step), columns:
        mean, mean_abs, variance, kurtosis, max_abs, pct_at_max_bin.
        Raises ValueError with a helpful message if no histogram keys found.
    """
    hist_keys = discover_history_keys(
        run,
        lambda key, _value: key.startswith(layer_prefix),
        max_rows=probe_rows,
    )
    if not hist_keys:
        raise ValueError(
            f"No histogram keys found with prefix `{layer_prefix}` in run "
            f"{run.id}. If you want per-layer grad histograms, call "
            "`wandb.watch(model, log='gradients', log_freq=N)` during training."
        )

    rows: list[dict[str, Any]] = []
    for row in fast_scan_history(run, keys=[*hist_keys, step_key]):
        step = row.get(step_key)
        if step is None:
            continue
        for key in hist_keys:
            stats = _histogram_stats(row.get(key))
            if stats is None:
                continue
            layer = key[len(layer_prefix):]
            rows.append({"layer": layer, "step": step, **stats})
    return pd.DataFrame(rows)
