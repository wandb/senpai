# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for working with W&B (Weights & Biases) training data.

The wandb SDK is for the W&B Models product — training runs, metrics history,
hyperparameter sweeps, artifacts, and system metrics. These helpers convert
run data into pandas-friendly structures for analysis.

Usage (in sandbox):
    import sys
    sys.path.insert(0, "skills/wandb-primary/scripts")
    from wandb_helpers import (
        runs_to_dataframe,   # Convert runs to a clean pandas DataFrame
        diagnose_run,        # Quick diagnostic summary of a training run
        compare_configs,     # Side-by-side config diff between two runs
    )
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import pandas as pd


def fast_scan_history(
    run: Any,
    keys: list[str] | None = None,
    page_size: int = 1000,
    min_step: int = 0,
    max_step: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate a run's history via `beta_scan_history` (local parquet, no API round-trips)."""
    yield from run.beta_scan_history(
        keys=keys,
        page_size=page_size,
        min_step=min_step,
        max_step=max_step,
    )


def discover_history_keys(
    run: Any,
    predicate: Callable[[str, Any], bool],
    max_rows: int = 500,
) -> list[str]:
    """Find history keys matching `predicate` — useful for sparsely-logged keys."""
    found: set[str] = set()
    for i, row in enumerate(fast_scan_history(run)):
        for key, value in row.items():
            if predicate(key, value):
                found.add(key)
        if found or i + 1 >= max_rows:
            break
    return sorted(found)


# ---------------------------------------------------------------------------
# Runs -> DataFrame
# ---------------------------------------------------------------------------

def runs_to_dataframe(
    runs: Any,
    limit: int = 200,
    metric_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten up to `limit` runs into pd.DataFrame-ready dicts (metadata + config + summary metrics)."""
    if metric_keys is None:
        metric_keys = ["loss", "val_loss", "accuracy"]

    rows = []
    for run in runs[:limit]:
        row = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
        }
        # Config (skip internal keys)
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[f"config.{k}"] = v
        # Summary metrics
        for key in metric_keys:
            row[key] = run.summary_metrics.get(key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------------------------

def diagnose_run(run: Any) -> dict[str, Any]:
    """Quick diagnostic summary: convergence, overfit gap, NaNs, tail stats."""
    df = pd.DataFrame(list(fast_scan_history(run, keys=["loss", "val_loss"])))
    loss = df["loss"].dropna()

    diagnostics: dict[str, Any] = {
        "total_steps": len(loss),
        "final_loss": loss.iloc[-1] if len(loss) else None,
        "min_loss": loss.min() if len(loss) else None,
        "min_loss_step": int(loss.idxmin()) if len(loss) else None,
        "has_nan": bool(loss.isna().any()),
        "final_10pct_mean": float(loss.tail(max(1, len(loss) // 10)).mean())
        if len(loss)
        else None,
    }

    # Overfitting check (val_loss diverging from train loss)
    if "val_loss" in df.columns:
        val = df["val_loss"].dropna()
        if len(val) > 10:
            tail_size = max(1, len(val) // 5)
            train_tail = float(loss.tail(tail_size).mean())
            val_tail = float(val.tail(tail_size).mean())
            diagnostics["train_val_gap"] = round(val_tail - train_tail, 6)
            diagnostics["likely_overfit"] = val_tail > train_tail * 1.2

    # Convergence check
    if len(loss) > 100:
        last_pct = loss.tail(max(1, len(loss) // 10))
        diagnostics["converged"] = bool(last_pct.std() < last_pct.mean() * 0.01)

    return diagnostics


# ---------------------------------------------------------------------------
# Config comparison
# ---------------------------------------------------------------------------

def compare_configs(run_a: Any, run_b: Any) -> list[dict[str, Any]]:
    """Return config keys that differ between two runs, with both values."""
    config_a = {k: v for k, v in run_a.config.items() if not k.startswith("_")}
    config_b = {k: v for k, v in run_b.config.items() if not k.startswith("_")}

    all_keys = sorted(set(config_a) | set(config_b))
    diffs = []
    for k in all_keys:
        val_a = config_a.get(k)
        val_b = config_b.get(k)
        if val_a != val_b:
            diffs.append({
                "key": k,
                run_a.name: val_a,
                run_b.name: val_b,
            })
    return diffs
