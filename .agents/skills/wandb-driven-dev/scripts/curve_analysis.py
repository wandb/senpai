#!/usr/bin/env python3
"""Pandas curve-analysis primitives for comparing W&B training curves.

Humans rarely need every point in a plot. For experiment review we usually need
bounded, explainable features:

- value at selected steps
- smoothed local slope over a recent window
- relative movement over that window
- residual noise around the local slope
- early-training launch health
- mid/late-training spike and slope-shift checks
- trend classification
- best run by value and by slope

This module exposes those primitives and a JSON CLI. It intentionally analyzes
selected runs and selected metrics only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

_HERE = Path(__file__).resolve().parent
_WBAGENT_SCRIPTS = _HERE.parent.parent / "wbagent" / "scripts"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_WBAGENT_SCRIPTS))
from wandb_helpers import get_api, scan_history, scan_history_until_step  # noqa: E402
from wdd_helpers import CONFIG_PATH, curve_step_keys, read_config  # noqa: E402


DEFAULT_WINDOW_STEPS = 1000
DEFAULT_SMOOTHING_POINTS = 5
DEFAULT_SMOOTHING_METHOD = "median"
DEFAULT_STAGE = "auto"
DEFAULT_EARLY_STEP_THRESHOLD = 5000
DEFAULT_EARLY_MIN_POINTS = 4

LOWER_IS_BETTER_HINTS = (
    "loss",
    "error",
    "rel_l2",
    "mae",
    "mse",
    "rmse",
    "wer",
    "cer",
    "perplexity",
)
HIGHER_IS_BETTER_HINTS = (
    "acc",
    "accuracy",
    "auc",
    "f1",
    "precision",
    "recall",
    "samples_per_sec",
    "throughput",
    "tokens_per_sec",
)
def infer_lower_is_better(metric: str) -> bool:
    lowered = metric.lower()
    if any(hint in lowered for hint in HIGHER_IS_BETTER_HINTS):
        return False
    if any(hint in lowered for hint in LOWER_IS_BETTER_HINTS):
        return True
    return True


def metric_group(metric: str) -> str:
    return metric.split("/", 1)[0] if "/" in metric else metric


def _unique_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)
    return out


def rows_to_curve_frame(
    rows: list[dict[str, Any]],
    metrics: list[str],
    step_key: str = "_step",
) -> pd.DataFrame:
    """Convert W&B history rows to a sorted numeric frame."""
    columns = _unique_columns(["_step", step_key, *metrics])
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[step_key]).sort_values([step_key, "_step"])
    return frame.reset_index(drop=True)


def metric_frame(frame: pd.DataFrame, metric: str, step_key: str = "_step") -> pd.DataFrame:
    """Return rows where a metric exists, with duplicate steps collapsed to latest."""
    if metric not in frame.columns:
        return pd.DataFrame(columns=_unique_columns(["_step", step_key, metric]))
    columns = _unique_columns(["_step", step_key, metric])
    out = frame[columns].dropna(subset=[step_key, metric]).copy()
    if out.empty:
        return out
    out = out.sort_values([step_key, "_step"]).drop_duplicates(subset=[step_key], keep="last")
    return out.reset_index(drop=True)


def smooth_metric_frame(
    frame: pd.DataFrame,
    metric: str,
    smoothing_points: int = 5,
    smoothing_method: str = "median",
) -> pd.DataFrame:
    """Add a trailing smoothed metric column for noisy curves.

    Smoothing is trailing rather than centered so analysis at step N never uses
    points logged after N. Median is the default because isolated spikes are
    common in training curves.
    """
    out = frame.copy()
    smoothed_col = f"{metric}__smoothed"
    smoothing_points = max(1, int(smoothing_points))
    values = pd.to_numeric(out[metric], errors="coerce")
    if smoothing_points <= 1:
        out[smoothed_col] = values
        return out
    rolling = values.rolling(window=smoothing_points, min_periods=1)
    if smoothing_method == "mean":
        out[smoothed_col] = rolling.mean()
    elif smoothing_method == "median":
        out[smoothed_col] = rolling.median()
    else:
        raise ValueError(f"Unsupported smoothing method: {smoothing_method!r}")
    return out


def _finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _linear_slope(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[float | None, float | None]:
    """Return least-squares slope and residual std for a small curve window."""
    data = frame[[x_col, y_col]].dropna()
    if len(data) < 2:
        return None, None
    x = data[x_col].astype(float)
    y = data[y_col].astype(float)
    x_centered = x - x.mean()
    denom = float((x_centered * x_centered).sum())
    if denom == 0:
        return None, None
    slope = float((x_centered * (y - y.mean())).sum() / denom)
    intercept = float(y.mean() - slope * x.mean())
    residuals = y - (slope * x + intercept)
    residual_std = float(residuals.std(ddof=0)) if len(data) >= 3 else 0.0
    return slope, residual_std


def _trend_from_slope(
    slope_per_1k: float,
    noise_per_1k: float | None,
    lower_is_better: bool,
    epsilon: float,
) -> tuple[str, str, float | None]:
    signal = abs(slope_per_1k)
    noise = noise_per_1k or 0.0
    threshold = max(epsilon, noise)
    noise_to_signal = None if signal == 0 else noise / signal
    if signal <= threshold:
        confidence = "low" if noise > epsilon else "high"
        return "flat", confidence, noise_to_signal
    trend = (
        "improving"
        if (slope_per_1k < 0 and lower_is_better)
        or (slope_per_1k > 0 and not lower_is_better)
        else "worsening"
    )
    if noise_to_signal is None or noise_to_signal <= 0.25:
        confidence = "high"
    elif noise_to_signal <= 1.0:
        confidence = "medium"
    else:
        confidence = "low"
    return trend, confidence, noise_to_signal


def _improvement_delta(current: float, reference: float, lower_is_better: bool) -> float:
    return reference - current if lower_is_better else current - reference


def _improvement_fraction(current: float, reference: float, lower_is_better: bool) -> float | None:
    if reference == 0:
        return None
    return _improvement_delta(current, reference, lower_is_better) / abs(reference)


def _directional_fraction(
    frame: pd.DataFrame,
    value_col: str,
    lower_is_better: bool,
    epsilon: float = 1e-12,
) -> float | None:
    deltas = frame[value_col].diff().dropna()
    if deltas.empty:
        return None
    if lower_is_better:
        improving = deltas < -epsilon
    else:
        improving = deltas > epsilon
    return float(improving.sum() / len(deltas))


def detect_spikes(
    frame: pd.DataFrame,
    metric: str,
    target_step: int | float | None = None,
    window_steps: int | float | None = None,
    step_key: str = "_step",
    lower_is_better: bool = True,
    smoothing_points: int = 5,
    smoothing_method: str = "median",
    spike_z: float = 4.0,
    spike_relative: float = 0.2,
    max_events: int = 5,
) -> dict[str, Any]:
    """Detect sudden bad-direction spikes against the smoothed local curve."""
    mf = metric_frame(frame, metric, step_key=step_key)
    if mf.empty:
        return {"points": 0, "bad_spike_count": 0, "events": [], "error": "no metric rows"}
    mf = smooth_metric_frame(
        mf,
        metric,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
    )
    if target_step is not None:
        mf = mf[mf[step_key] <= target_step]
    if window_steps is not None and not mf.empty:
        end_step = float(mf.iloc[-1][step_key])
        mf = mf[mf[step_key] >= end_step - float(window_steps)]
    if mf.empty:
        return {"points": 0, "bad_spike_count": 0, "events": [], "error": "no rows in window"}

    smoothed_col = f"{metric}__smoothed"
    residual = pd.to_numeric(mf[metric] - mf[smoothed_col], errors="coerce")
    centered = residual - residual.median()
    mad = float(centered.abs().median()) if not centered.dropna().empty else 0.0
    robust_scale = 1.4826 * mad
    if robust_scale == 0.0 and len(residual.dropna()) >= 2:
        robust_scale = float(residual.std(ddof=0))
    relative_base_value = _finite_float(pd.to_numeric(mf[smoothed_col].abs(), errors="coerce").median())
    relative_base = relative_base_value or 0.0
    threshold = max(spike_z * robust_scale, spike_relative * relative_base, 0.0)

    badness = residual if lower_is_better else -residual
    events: list[dict[str, Any]] = []
    if threshold > 0:
        spike_rows = mf[badness > threshold].copy()
        spike_rows["_badness"] = badness.loc[spike_rows.index]
        spike_rows["_residual"] = residual.loc[spike_rows.index]
        spike_rows = spike_rows.sort_values("_badness", ascending=False)
        for _, row in spike_rows.head(max_events).iterrows():
            events.append(
                {
                    "step": _finite_float(row[step_key]),
                    "value": _finite_float(row[metric]),
                    "smoothed_value": _finite_float(row[smoothed_col]),
                    "residual": _finite_float(row["_residual"]),
                    "badness": _finite_float(row["_badness"]),
                }
            )

    return {
        "points": int(len(mf)),
        "bad_spike_count": int((badness > threshold).sum()) if threshold > 0 else 0,
        "spike_threshold": threshold,
        "robust_noise_scale": robust_scale,
        "events": events,
    }


def curve_summary(
    frame: pd.DataFrame,
    metric: str,
    step_key: str = "_step",
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Summarize a single metric curve across the supplied frame."""
    mf = metric_frame(frame, metric, step_key=step_key)
    if mf.empty:
        return {"metric": metric, "points": 0, "error": "no metric rows"}

    start = mf.iloc[0]
    end = mf.iloc[-1]
    best_idx = mf[metric].idxmin() if lower_is_better else mf[metric].idxmax()
    best = mf.loc[best_idx]
    delta = float(end[metric] - start[metric])
    pct_delta = None if float(start[metric]) == 0 else delta / abs(float(start[metric]))
    return {
        "metric": metric,
        "points": int(len(mf)),
        "start_step": _finite_float(start[step_key]),
        "start_value": _finite_float(start[metric]),
        "final_step": _finite_float(end[step_key]),
        "final_value": _finite_float(end[metric]),
        "best_step": _finite_float(best[step_key]),
        "best_value": _finite_float(best[metric]),
        "delta": delta,
        "pct_delta": pct_delta,
    }


def early_training_health(
    frame: pd.DataFrame,
    metric: str,
    target_step: int | float,
    step_key: str = "_step",
    lower_is_better: bool = True,
    smoothing_points: int = 3,
    smoothing_method: str = "median",
    min_points: int = 4,
    min_directional_fraction: float = 0.5,
    max_noise_to_signal: float = 2.0,
    max_bad_spikes: int = 1,
    min_relative_improvement: float = 0.0,
    slope_epsilon_per_1k: float = 1e-12,
) -> dict[str, Any]:
    """Check whether an early run has launched, stabilized, and started moving correctly."""
    mf = metric_frame(frame, metric, step_key=step_key)
    mf = mf[mf[step_key] <= target_step]
    if mf.empty:
        return {
            "stage": "early",
            "metric": metric,
            "target_step": target_step,
            "status": "insufficient",
            "reasons": ["no metric rows at or before target_step"],
        }

    smoothed_col = f"{metric}__smoothed"
    mf = smooth_metric_frame(
        mf,
        metric,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
    )
    first = mf.iloc[0]
    latest = mf.iloc[-1]
    window_steps = max(float(latest[step_key] - first[step_key]), 1.0)
    snapshot = curve_snapshot(
        frame,
        metric,
        target_step=target_step,
        window_steps=window_steps,
        step_key=step_key,
        lower_is_better=lower_is_better,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
        slope_epsilon_per_1k=slope_epsilon_per_1k,
    )
    directional_fraction = _directional_fraction(
        mf,
        smoothed_col,
        lower_is_better=lower_is_better,
        epsilon=slope_epsilon_per_1k / 1000.0,
    )
    relative_improvement = _improvement_fraction(
        float(latest[smoothed_col]),
        float(first[smoothed_col]),
        lower_is_better,
    )
    spikes = detect_spikes(
        frame,
        metric,
        target_step=target_step,
        window_steps=window_steps,
        step_key=step_key,
        lower_is_better=lower_is_better,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
    )

    reasons: list[str] = []
    points = int(len(mf))
    trend = snapshot.get("trend")
    confidence = snapshot.get("trend_confidence")
    noise_to_signal = snapshot.get("noise_to_signal")
    bad_spike_count = int(spikes.get("bad_spike_count", 0))

    if points < min_points:
        reasons.append(f"only {points} metric points")
    if trend == "worsening" and confidence != "low":
        reasons.append("smoothed early slope is worsening")
    if trend == "flat":
        reasons.append("smoothed early slope is flat")
    if directional_fraction is not None and directional_fraction < min_directional_fraction:
        reasons.append("too few smoothed intervals move in the right direction")
    if noise_to_signal is not None and noise_to_signal > max_noise_to_signal:
        reasons.append("noise is high relative to slope")
    if bad_spike_count > max_bad_spikes:
        reasons.append("bad-direction spikes exceed early tolerance")
    if relative_improvement is not None and relative_improvement < min_relative_improvement:
        reasons.append("smoothed value has not improved enough from launch")

    if points < min_points:
        status = "insufficient"
    elif trend == "worsening" and confidence in {"high", "medium"}:
        status = "fail"
    elif bad_spike_count > max_bad_spikes and trend != "improving":
        status = "fail"
    elif reasons:
        status = "watch"
    else:
        status = "healthy"

    return {
        "stage": "early",
        "metric": metric,
        "target_step": target_step,
        "status": status,
        "reasons": reasons,
        "points": points,
        "start_step": _finite_float(first[step_key]),
        "start_value": _finite_float(first[metric]),
        "start_smoothed_value": _finite_float(first[smoothed_col]),
        "latest_step": _finite_float(latest[step_key]),
        "latest_value": _finite_float(latest[metric]),
        "latest_smoothed_value": _finite_float(latest[smoothed_col]),
        "relative_improvement": relative_improvement,
        "directional_fraction": directional_fraction,
        "snapshot": snapshot,
        "spikes": spikes,
    }


def curve_snapshot(
    frame: pd.DataFrame,
    metric: str,
    target_step: int | float,
    window_steps: int | float,
    step_key: str = "_step",
    lower_is_better: bool = True,
    smoothing_points: int = 5,
    smoothing_method: str = "median",
    slope_epsilon_per_1k: float = 1e-12,
) -> dict[str, Any]:
    """Compute value and local slope at the latest logged step <= target."""
    mf = metric_frame(frame, metric, step_key=step_key)
    smoothed_col = f"{metric}__smoothed"
    mf = smooth_metric_frame(
        mf,
        metric,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
    )
    candidates = mf[mf[step_key] <= target_step]
    if candidates.empty:
        return {
            "metric": metric,
            "target_step": target_step,
            "error": "no point at or before target_step",
        }

    point = candidates.iloc[-1]
    point_step = float(point[step_key])
    point_value = float(point[metric])
    point_smoothed_value = float(point[smoothed_col])
    window_start = point_step - float(window_steps)
    window = mf[(mf[step_key] >= window_start) & (mf[step_key] <= point_step)]
    if len(window) < 2:
        window = candidates.tail(min(2, len(candidates)))

    slope_per_step = None
    slope_per_1k = None
    window_delta = None
    raw_window_delta = None
    pct_delta = None
    raw_pct_delta = None
    noise_std = None
    noise_per_1k = None
    noise_to_signal = None
    trend = "unknown"
    trend_confidence = "unknown"
    window_start_step = None
    window_start_value = None
    window_start_smoothed_value = None
    if len(window) >= 2:
        first = window.iloc[0]
        last = window.iloc[-1]
        span = float(last[step_key] - first[step_key])
        if span > 0:
            window_start_step = float(first[step_key])
            window_start_value = float(first[metric])
            window_start_smoothed_value = float(first[smoothed_col])
            window_delta = float(last[smoothed_col] - first[smoothed_col])
            raw_window_delta = float(last[metric] - first[metric])
            pct_delta = (
                None
                if float(first[smoothed_col]) == 0
                else window_delta / abs(float(first[smoothed_col]))
            )
            raw_pct_delta = None if float(first[metric]) == 0 else raw_window_delta / abs(float(first[metric]))
            slope_per_step, noise_std = _linear_slope(window, step_key, smoothed_col)
            if slope_per_step is None:
                slope_per_step = window_delta / span
            slope_per_1k = slope_per_step * 1000.0
            if noise_std is not None:
                noise_per_1k = noise_std / span * 1000.0
            trend, trend_confidence, noise_to_signal = _trend_from_slope(
                slope_per_1k,
                noise_per_1k,
                lower_is_better,
                slope_epsilon_per_1k,
            )

    return {
        "metric": metric,
        "target_step": target_step,
        "step": point_step,
        "value": point_value,
        "smoothed_value": point_smoothed_value,
        "window_steps": window_steps,
        "smoothing_points": smoothing_points,
        "smoothing_method": smoothing_method,
        "window_start_step": window_start_step,
        "window_start_value": window_start_value,
        "window_start_smoothed_value": window_start_smoothed_value,
        "window_delta": window_delta,
        "raw_window_delta": raw_window_delta,
        "pct_delta": pct_delta,
        "raw_pct_delta": raw_pct_delta,
        "slope_per_step": slope_per_step,
        "slope_per_1k": slope_per_1k,
        "noise_std": noise_std,
        "noise_per_1k": noise_per_1k,
        "noise_to_signal": noise_to_signal,
        "trend": trend,
        "trend_confidence": trend_confidence,
    }


def progress_curve_insight(
    frame: pd.DataFrame,
    metric: str,
    target_step: int | float,
    window_steps: int | float,
    step_key: str = "_step",
    lower_is_better: bool = True,
    smoothing_points: int = 5,
    smoothing_method: str = "median",
    slope_epsilon_per_1k: float = 1e-12,
) -> dict[str, Any]:
    """Analyze an in-progress or mature curve with slope, spike, and slope-shift features."""
    current = curve_snapshot(
        frame,
        metric,
        target_step=target_step,
        window_steps=window_steps,
        step_key=step_key,
        lower_is_better=lower_is_better,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
        slope_epsilon_per_1k=slope_epsilon_per_1k,
    )
    previous_target = current.get("window_start_step")
    previous = None
    slope_shift_per_1k = None
    effective_slope_shift_per_1k = None
    if previous_target is not None:
        previous = curve_snapshot(
            frame,
            metric,
            target_step=previous_target,
            window_steps=window_steps,
            step_key=step_key,
            lower_is_better=lower_is_better,
            smoothing_points=smoothing_points,
            smoothing_method=smoothing_method,
            slope_epsilon_per_1k=slope_epsilon_per_1k,
        )
        if current.get("slope_per_1k") is not None and previous.get("slope_per_1k") is not None:
            slope_shift_per_1k = current["slope_per_1k"] - previous["slope_per_1k"]
            multiplier = -1.0 if lower_is_better else 1.0
            effective_slope_shift_per_1k = multiplier * slope_shift_per_1k

    spikes = detect_spikes(
        frame,
        metric,
        target_step=target_step,
        window_steps=window_steps,
        step_key=step_key,
        lower_is_better=lower_is_better,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
    )

    reasons: list[str] = []
    trend = current.get("trend")
    confidence = current.get("trend_confidence")
    bad_spike_count = int(spikes.get("bad_spike_count", 0))
    if trend == "worsening" and confidence in {"high", "medium"}:
        reasons.append("current smoothed slope is worsening")
    if bad_spike_count:
        reasons.append("recent bad-direction spikes detected")
    if effective_slope_shift_per_1k is not None and effective_slope_shift_per_1k < 0:
        reasons.append("slope is decelerating versus previous window")

    if trend == "improving" and confidence in {"high", "medium"} and bad_spike_count == 0:
        status = "improving"
    elif trend == "worsening" and confidence in {"high", "medium"}:
        status = "regressing"
    elif reasons:
        status = "watch"
    elif trend == "flat":
        status = "plateau"
    else:
        status = "uncertain"

    return {
        "stage": "progress",
        "metric": metric,
        "target_step": target_step,
        "status": status,
        "reasons": reasons,
        "current": current,
        "previous": previous,
        "slope_shift_per_1k": slope_shift_per_1k,
        "effective_slope_shift_per_1k": effective_slope_shift_per_1k,
        "spikes": spikes,
    }


def compare_snapshots(
    snapshots: dict[str, dict[str, Any]],
    lower_is_better: bool,
) -> dict[str, Any]:
    """Rank run snapshots by value and slope."""
    valid_values = {
        run_id: snap
        for run_id, snap in snapshots.items()
        if snap.get("value") is not None
    }
    if not valid_values:
        return {"best_run": None, "best_slope_run": None, "runs": snapshots}

    best_run = (
        min(valid_values, key=lambda run_id: valid_values[run_id]["value"])
        if lower_is_better
        else max(valid_values, key=lambda run_id: valid_values[run_id]["value"])
    )
    best_value = valid_values[best_run]["value"]

    valid_slopes = {
        run_id: snap
        for run_id, snap in valid_values.items()
        if snap.get("slope_per_1k") is not None
    }
    best_slope_run = None
    if valid_slopes:
        best_slope_run = (
            min(valid_slopes, key=lambda run_id: valid_slopes[run_id]["slope_per_1k"])
            if lower_is_better
            else max(valid_slopes, key=lambda run_id: valid_slopes[run_id]["slope_per_1k"])
        )

    enriched = {}
    for run_id, snap in snapshots.items():
        value = snap.get("value")
        row = dict(snap)
        if value is not None and best_value not in (None, 0):
            raw_gap = value - best_value if lower_is_better else best_value - value
            row["gap_to_best"] = raw_gap
            row["pct_gap_to_best"] = raw_gap / abs(best_value)
        enriched[run_id] = row

    return {
        "best_run": best_run,
        "best_slope_run": best_slope_run,
        "runs": enriched,
    }


def _resolve_stage(stage: str, step: int | float, early_step_threshold: int | float) -> str:
    if stage != "auto":
        return stage
    return "early" if float(step) <= float(early_step_threshold) else "progress"


def analyze_curve_frames(
    frames: dict[str, pd.DataFrame],
    metrics: list[str],
    steps: list[int],
    window_steps: int,
    lower_is_better: dict[str, bool] | None = None,
    step_key: str = "_step",
    smoothing_points: int = 5,
    smoothing_method: str = "median",
    stage: str = "progress",
    early_step_threshold: int | float = 5000,
    early_min_points: int = 4,
) -> dict[str, Any]:
    """Analyze already-fetched curve frames."""
    lower_is_better = lower_is_better or {}
    result: dict[str, Any] = {
        "step_key": step_key,
        "window_steps": window_steps,
        "smoothing_points": smoothing_points,
        "smoothing_method": smoothing_method,
        "stage": stage,
        "early_step_threshold": early_step_threshold,
        "metrics": {},
    }

    for metric in metrics:
        minimize = lower_is_better.get(metric, infer_lower_is_better(metric))
        metric_result = {
            "lower_is_better": minimize,
            "summaries": {
                run_id: curve_summary(frame, metric, step_key=step_key, lower_is_better=minimize)
                for run_id, frame in frames.items()
            },
            "points": [],
        }
        for step in steps:
            snapshots = {
                run_id: curve_snapshot(
                    frame,
                    metric,
                    target_step=step,
                    window_steps=window_steps,
                    step_key=step_key,
                    lower_is_better=minimize,
                    smoothing_points=smoothing_points,
                    smoothing_method=smoothing_method,
                )
                for run_id, frame in frames.items()
            }
            point = compare_snapshots(snapshots, lower_is_better=minimize)
            point["target_step"] = step
            resolved_stage = _resolve_stage(stage, step, early_step_threshold)
            point["resolved_stage"] = resolved_stage
            if resolved_stage == "early":
                point["stage_analysis"] = {
                    run_id: early_training_health(
                        frame,
                        metric,
                        target_step=step,
                        step_key=step_key,
                        lower_is_better=minimize,
                        smoothing_points=min(smoothing_points, 3),
                        smoothing_method=smoothing_method,
                        min_points=early_min_points,
                    )
                    for run_id, frame in frames.items()
                }
            else:
                point["stage_analysis"] = {
                    run_id: progress_curve_insight(
                        frame,
                        metric,
                        target_step=step,
                        window_steps=window_steps,
                        step_key=step_key,
                        lower_is_better=minimize,
                        smoothing_points=smoothing_points,
                        smoothing_method=smoothing_method,
                    )
                    for run_id, frame in frames.items()
                }
            metric_result["points"].append(point)
        result["metrics"][metric] = metric_result
    return result


def fetch_run_curve_frame(
    api: Any,
    project: str,
    run_id: str,
    metrics: list[str],
    step_key: str = "_step",
    max_step: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Fetch selected history rows for one run and return a curve frame."""
    run = api.run(f"{project}/{run_id}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for metric in metrics:
        grouped[metric_group(metric)].append(metric)

    rows: list[dict[str, Any]] = []
    for group_metrics in grouped.values():
        keys = _unique_columns(["_step", step_key, *group_metrics])
        if max_step is None:
            rows.extend(scan_history(run, keys=keys, max_rows=max_rows))
        else:
            rows.extend(
                scan_history_until_step(
                    run,
                    keys=keys,
                    step_key=step_key,
                    target_step=max_step,
                    max_rows=max_rows,
                )
            )
    return rows_to_curve_frame(rows, metrics, step_key=step_key)


def analyze_wandb_runs(
    api: Any,
    project: str,
    run_ids: list[str],
    metrics: list[str],
    steps: list[int],
    window_steps: int = 1000,
    step_key: str = "_step",
    lower_is_better: dict[str, bool] | None = None,
    max_rows: int | None = None,
    workers: int | None = None,
    api_factory: Any | None = get_api,
    smoothing_points: int = 5,
    smoothing_method: str = "median",
    stage: str = "progress",
    early_step_threshold: int | float = 5000,
    early_min_points: int = 4,
) -> dict[str, Any]:
    max_step = max(steps) if steps else None
    workers = workers or min(12, max(1, len(run_ids)))

    def fetch_one(run_id: str) -> tuple[str, pd.DataFrame]:
        worker_api = api_factory() if api_factory is not None else api
        return run_id, fetch_run_curve_frame(
            worker_api,
            project,
            run_id,
            metrics,
            step_key=step_key,
            max_step=max_step,
            max_rows=max_rows,
        )

    if workers <= 1 or len(run_ids) <= 1:
        frames = dict(fetch_one(run_id) for run_id in run_ids)
    else:
        frames_unsorted: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_one, run_id): run_id for run_id in run_ids}
            for future in as_completed(futures):
                run_id, frame = future.result()
                frames_unsorted[run_id] = frame
        frames = {run_id: frames_unsorted[run_id] for run_id in run_ids}

    result = analyze_curve_frames(
        frames,
        metrics,
        steps,
        window_steps=window_steps,
        lower_is_better=lower_is_better,
        step_key=step_key,
        smoothing_points=smoothing_points,
        smoothing_method=smoothing_method,
        stage=stage,
        early_step_threshold=early_step_threshold,
        early_min_points=early_min_points,
    )
    result["project"] = project
    result["runs"] = run_ids
    result["fetched_through_step"] = max_step
    result["workers"] = workers
    return result


def compare_curve_frames(
    frames: dict[str, pd.DataFrame],
    metrics: list[str],
    steps: list[int],
    step_key: str = "_step",
) -> dict[str, Any]:
    """Opinionated in-memory curve comparison for the common review path."""
    return analyze_curve_frames(
        frames,
        metrics,
        steps,
        window_steps=DEFAULT_WINDOW_STEPS,
        step_key=step_key,
        smoothing_points=DEFAULT_SMOOTHING_POINTS,
        smoothing_method=DEFAULT_SMOOTHING_METHOD,
        stage=DEFAULT_STAGE,
        early_step_threshold=DEFAULT_EARLY_STEP_THRESHOLD,
        early_min_points=DEFAULT_EARLY_MIN_POINTS,
    )


def compare_wandb_curves(
    api: Any,
    project: str,
    run_ids: list[str],
    metrics: list[str],
    steps: list[int],
    step_key: str = "_step",
) -> dict[str, Any]:
    """Opinionated W&B curve comparison with only the required inputs."""
    return analyze_wandb_runs(
        api,
        project,
        run_ids=run_ids,
        metrics=metrics,
        steps=steps,
        window_steps=DEFAULT_WINDOW_STEPS,
        step_key=step_key,
        smoothing_points=DEFAULT_SMOOTHING_POINTS,
        smoothing_method=DEFAULT_SMOOTHING_METHOD,
        stage=DEFAULT_STAGE,
        early_step_threshold=DEFAULT_EARLY_STEP_THRESHOLD,
        early_min_points=DEFAULT_EARLY_MIN_POINTS,
    )


def compare_wandb_curves_from_config(
    api: Any,
    run_ids: list[str],
    metrics: list[str],
    steps: list[int],
    config_path: Path | str = CONFIG_PATH,
    project: str | None = None,
    step_key: str | None = None,
) -> dict[str, Any]:
    """Compare curves using project and metric step keys from local config.

    Setup should have populated `.claude/wandb-driven-dev.local.md`. If a metric
    maps to a different semantic step key, metrics are grouped and fetched in
    separate bounded scans.
    """
    cfg = read_config(config_path)
    resolved_project = project or (cfg or {}).get("wandb_project")
    if not resolved_project:
        raise RuntimeError(
            f"wandb_project is required; pass project or create {config_path}"
        )

    if step_key is not None:
        metric_step_keys = {metric: step_key for metric in metrics}
    else:
        metric_step_keys = curve_step_keys(cfg, metrics)

    grouped: dict[str, list[str]] = defaultdict(list)
    for metric, key in metric_step_keys.items():
        grouped[key].append(metric)

    result: dict[str, Any] = {
        "project": resolved_project,
        "runs": run_ids,
        "steps": steps,
        "metric_step_keys": metric_step_keys,
        "metrics": {},
        "groups": {},
    }
    max_workers = 0
    for key, group_metrics in grouped.items():
        group_result = compare_wandb_curves(
            api,
            resolved_project,
            run_ids=run_ids,
            metrics=group_metrics,
            steps=steps,
            step_key=key,
        )
        result["metrics"].update(group_result["metrics"])
        result["groups"][key] = {
            "metrics": group_metrics,
            "fetched_through_step": group_result.get("fetched_through_step"),
            "workers": group_result.get("workers"),
        }
        max_workers = max(max_workers, int(group_result.get("workers") or 0))

    result["fetched_through_step"] = max(steps) if steps else None
    result["workers"] = max_workers
    return result


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _direction_overrides(lower: str | None, higher: str | None) -> dict[str, bool]:
    out = {metric: True for metric in _split_csv(lower)}
    out.update({metric: False for metric in _split_csv(higher)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze selected W&B curves and print JSON features.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_p = subparsers.add_parser("compare", help="Analyze selected run curves")
    compare_p.add_argument("project", nargs="?", help="Optional entity/project. Defaults to local config.")
    compare_p.add_argument("--runs", required=True, help="Run IDs, comma-separated")
    compare_p.add_argument("--metrics", required=True, help="Metric keys, comma-separated")
    compare_p.add_argument("--steps", required=True, help="Target steps, comma-separated")
    compare_p.add_argument("--config", default=str(CONFIG_PATH), help=argparse.SUPPRESS)
    compare_p.add_argument("--window", type=int, default=DEFAULT_WINDOW_STEPS, help=argparse.SUPPRESS)
    compare_p.add_argument("--step-key", help=argparse.SUPPRESS)
    compare_p.add_argument("--lower-is-better", default="", help=argparse.SUPPRESS)
    compare_p.add_argument("--higher-is-better", default="", help=argparse.SUPPRESS)
    compare_p.add_argument("--max-rows", type=int, help=argparse.SUPPRESS)
    compare_p.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    compare_p.add_argument("--smooth-points", type=int, default=DEFAULT_SMOOTHING_POINTS, help=argparse.SUPPRESS)
    compare_p.add_argument("--smooth-method", choices=["median", "mean"], default=DEFAULT_SMOOTHING_METHOD, help=argparse.SUPPRESS)
    compare_p.add_argument(
        "--stage",
        choices=["early", "progress", "auto"],
        default=DEFAULT_STAGE,
        help=argparse.SUPPRESS,
    )
    compare_p.add_argument(
        "--early-step-threshold",
        type=int,
        default=DEFAULT_EARLY_STEP_THRESHOLD,
        help=argparse.SUPPRESS,
    )
    compare_p.add_argument("--early-min-points", type=int, default=DEFAULT_EARLY_MIN_POINTS, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    start = time.perf_counter()
    api = get_api()

    if args.command == "compare":
        run_ids = _split_csv(args.runs)
        metrics = _split_csv(args.metrics)
        steps = [int(step) for step in _split_csv(args.steps)]
        simple_defaults = (
            args.window == DEFAULT_WINDOW_STEPS
            and args.max_rows is None
            and args.workers is None
            and args.smooth_points == DEFAULT_SMOOTHING_POINTS
            and args.smooth_method == DEFAULT_SMOOTHING_METHOD
            and args.stage == DEFAULT_STAGE
            and args.early_step_threshold == DEFAULT_EARLY_STEP_THRESHOLD
            and args.early_min_points == DEFAULT_EARLY_MIN_POINTS
            and not args.lower_is_better
            and not args.higher_is_better
        )
        if simple_defaults:
            result = compare_wandb_curves_from_config(
                api,
                run_ids=run_ids,
                metrics=metrics,
                steps=steps,
                config_path=args.config,
                project=args.project,
                step_key=args.step_key,
            )
        else:
            cfg = read_config(args.config)
            project = args.project or (cfg or {}).get("wandb_project")
            if not project:
                raise RuntimeError(f"wandb_project is required; pass project or create {args.config}")
            result = analyze_wandb_runs(
                api,
                project,
                run_ids=run_ids,
                metrics=metrics,
                steps=steps,
                window_steps=args.window,
                step_key=args.step_key or "_step",
                lower_is_better=_direction_overrides(args.lower_is_better, args.higher_is_better),
                max_rows=args.max_rows,
                workers=args.workers,
                api_factory=get_api,
                smoothing_points=args.smooth_points,
                smoothing_method=args.smooth_method,
                stage=args.stage,
                early_step_threshold=args.early_step_threshold,
                early_min_points=args.early_min_points,
            )
    else:
        raise AssertionError(f"unhandled command {args.command}")

    print(json.dumps({"latency_s": round(time.perf_counter() - start, 4), "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
