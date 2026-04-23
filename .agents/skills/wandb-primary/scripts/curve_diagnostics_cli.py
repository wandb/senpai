# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""One-shot CLI for the training-curve diagnostics workflow.

This wrapper is intentionally opinionated:

- it confirms or refuses `step_key` rather than silently guessing
- it prints helper-derived tables instead of raw history rows
- it writes the standard PNGs so the caller can inspect them visually

Examples:
    python .agents/skills/wandb-primary/scripts/curve_diagnostics_cli.py \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --run abc123

    python .agents/skills/wandb-primary/scripts/curve_diagnostics_cli.py \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --run abc123 \
        --run def456 \
        --metric val_loss
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

pd = None
wandb = None
DEFAULT_OVERVIEW_METRICS = None
plot_run_comparison = None
plot_single_run_overview = None
format_step_candidates = None
guess_step_key_from_workspace = None
list_candidate_step_keys = None
compare_runs_curves = None
curve_features = None
grad_norm_features = None
lr_schedule_features = None
fast_scan_history = None


def _import_runtime_deps() -> None:
    global pd
    global wandb
    global DEFAULT_OVERVIEW_METRICS
    global plot_run_comparison
    global plot_single_run_overview
    global format_step_candidates
    global guess_step_key_from_workspace
    global list_candidate_step_keys
    global compare_runs_curves
    global curve_features
    global grad_norm_features
    global lr_schedule_features
    global fast_scan_history

    required = ["pandas", "wandb", "numpy", "matplotlib"]
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing dependency `{exc.name}`. Install it in the detected project "
                "environment before using this CLI."
            ) from exc

    import pandas as pandas_module
    import wandb as wandb_module
    from curve_plots import DEFAULT_OVERVIEW_METRICS as overview_metrics
    from curve_plots import plot_run_comparison as plot_comparison
    from curve_plots import plot_single_run_overview as plot_overview
    from step_axis import format_step_candidates as render_step_candidates
    from step_axis import guess_step_key_from_workspace as workspace_step_guess
    from step_axis import list_candidate_step_keys as candidate_step_keys
    from training_diagnostics import compare_runs_curves as compare_curves
    from training_diagnostics import curve_features as one_curve_features
    from training_diagnostics import grad_norm_features as one_grad_norm_features
    from training_diagnostics import lr_schedule_features as one_lr_schedule_features
    from wandb_helpers import fast_scan_history as fast_history

    pd = pandas_module
    wandb = wandb_module
    DEFAULT_OVERVIEW_METRICS = overview_metrics
    plot_run_comparison = plot_comparison
    plot_single_run_overview = plot_overview
    format_step_candidates = render_step_candidates
    guess_step_key_from_workspace = workspace_step_guess
    list_candidate_step_keys = candidate_step_keys
    compare_runs_curves = compare_curves
    curve_features = one_curve_features
    grad_norm_features = one_grad_norm_features
    lr_schedule_features = one_lr_schedule_features
    fast_scan_history = fast_history


def _load_history_frame(run: Any, keys: list[str], step_key: str, samples: int = 2000) -> pd.DataFrame:
    try:
        df = run.history(samples=samples, keys=keys, x_axis=step_key, pandas=True)
    except Exception:
        df = None
    if df is None or len(df) == 0 or step_key not in df.columns:
        df = pd.DataFrame(list(fast_scan_history(run, keys=keys)))
    return df


def _resolve_metric(df_columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df_columns:
            return candidate
    return None


def _choose_step_key(
    runs: list[Any],
    entity: str,
    project: str,
    explicit_step_key: str | None,
) -> tuple[str, list[str], str | None]:
    candidate_lists = [list_candidate_step_keys(run) for run in runs]
    common_candidates = set(candidate_lists[0]) if candidate_lists else set()
    for candidates in candidate_lists[1:]:
        common_candidates &= set(candidates)

    ordered_common = [c for c in candidate_lists[0] if c in common_candidates] if candidate_lists else []
    workspace_guess = guess_step_key_from_workspace(entity, project)

    if explicit_step_key:
        missing = [run.id for run, candidates in zip(runs, candidate_lists) if explicit_step_key not in candidates]
        if missing:
            raise ValueError(
                f"`{explicit_step_key}` is not a detected candidate for run(s): {', '.join(missing)}"
            )
        return explicit_step_key, ordered_common, workspace_guess

    if len(ordered_common) == 1:
        return ordered_common[0], ordered_common, workspace_guess

    if workspace_guess and workspace_guess in ordered_common and len(ordered_common) == 1:
        return workspace_guess, ordered_common, workspace_guess

    rendered = format_step_candidates(ordered_common or candidate_lists[0], workspace_guess)
    raise ValueError(
        "Ambiguous step_key. Re-run with `--step-key` after checking these candidates:\n"
        f"{rendered}"
    )


def _single_run_tables(run: Any, step_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_panels = [
        ("train_loss", DEFAULT_OVERVIEW_METRICS["train_loss"]["keys"], "decreasing"),
        ("val_loss", DEFAULT_OVERVIEW_METRICS["val_loss"]["keys"], "decreasing"),
        ("grad_norm", DEFAULT_OVERVIEW_METRICS["grad_norm"]["keys"], "decreasing"),
    ]
    lr_candidates = DEFAULT_OVERVIEW_METRICS["lr"]["keys"]

    history_keys = [step_key]
    for _, candidates, _ in core_panels:
        history_keys.extend(candidates)
    history_keys.extend(lr_candidates)

    df = _load_history_frame(run, keys=list(dict.fromkeys(history_keys)), step_key=step_key)
    cols = set(df.columns)

    curve_rows: list[dict[str, Any]] = []
    for label, candidates, direction in core_panels:
        resolved = _resolve_metric(cols, candidates)
        if resolved is None:
            continue
        series = df[[step_key, resolved]].dropna()
        if len(series) < 5:
            continue
        values = series[resolved].to_numpy()
        steps = series[step_key].to_numpy()
        if label == "grad_norm":
            feats = grad_norm_features(values, steps)
        else:
            feats = curve_features(values, steps, direction=direction)
        curve_rows.append(
            {
                "panel": label,
                "metric_key": resolved,
                "n_points": feats.get("n_points"),
                "final_10pct_mean": feats.get("final_10pct_mean"),
                "final_10pct_std": feats.get("final_10pct_std"),
                "smoothness": feats.get("smoothness"),
                "monotonicity_pct": feats.get("monotonicity_pct"),
                "spike_count": feats.get("spike_count"),
                "argmin_step": feats.get("argmin_step"),
                "divergence_detected": feats.get("divergence", {}).get("detected"),
                "kurtosis": feats.get("kurtosis"),
                "dead_flag": feats.get("dead_flag"),
            }
        )

    lr_rows: list[dict[str, Any]] = []
    lr_key = _resolve_metric(cols, lr_candidates)
    if lr_key is not None:
        series = df[[step_key, lr_key]].dropna()
        if len(series) >= 5:
            feats = lr_schedule_features(
                series[lr_key].to_numpy(),
                series[step_key].to_numpy(),
            )
            lr_rows.append(
                {
                    "metric_key": lr_key,
                    "peak_lr": feats.get("peak_lr"),
                    "peak_step": feats.get("peak_step"),
                    "warmup_steps": feats.get("warmup_steps"),
                    "final_lr": feats.get("final_lr"),
                    "decay_shape": feats.get("decay_shape"),
                    "restart_steps": feats.get("restart_steps"),
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(lr_rows)


def _print_frame(title: str, df: pd.DataFrame) -> None:
    print(f"\n## {title}")
    if df.empty:
        print("(no matching data found)")
        return
    print(df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True, help="W&B entity")
    parser.add_argument("--project", required=True, help="W&B project")
    parser.add_argument("--run", dest="runs", action="append", required=True, help="Run id (repeatable)")
    parser.add_argument("--metric", help="Metric to compare across runs when passing multiple --run values")
    parser.add_argument("--step-key", help="Confirmed step-axis key. Required if the script reports ambiguity.")
    parser.add_argument(
        "--direction",
        choices=["auto", "decreasing", "increasing"],
        default="auto",
        help="Curve direction for multi-run comparisons.",
    )
    args = parser.parse_args()

    try:
        _import_runtime_deps()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    runs = [api.run(f"{path}/{run_id}") for run_id in args.runs]

    try:
        step_key, common_candidates, workspace_guess = _choose_step_key(
            runs,
            entity=args.entity,
            project=args.project,
            explicit_step_key=args.step_key,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Chosen step_key: {step_key}")
    if common_candidates:
        print(f"Common candidates: {common_candidates}")
    if workspace_guess:
        print(f"Workspace guess: {workspace_guess}")

    if len(runs) == 1:
        run = runs[0]
        curve_df, lr_df = _single_run_tables(run, step_key=step_key)
        overview_png = plot_single_run_overview(run, step_key=step_key)

        print(f"Run: {run.name} [{run.id}]")
        _print_frame("Curve Features", curve_df)
        _print_frame("LR Schedule Features", lr_df)
        print(f"\nOverview PNG: {overview_png}")
        return 0

    if not args.metric:
        print("`--metric` is required when comparing multiple runs.", file=sys.stderr)
        return 2

    compare_df = compare_runs_curves(
        runs,
        metric=args.metric,
        step_key=step_key,
        direction=args.direction,
    )
    sort_ascending = args.direction != "increasing"
    compare_df = compare_df.sort_values("final_10pct_mean", ascending=sort_ascending)

    plot_runs = runs
    if len(runs) > 6:
        top_ids = compare_df["run_id"].tolist()[:6]
        by_id = {run.id: run for run in runs}
        plot_runs = [by_id[run_id] for run_id in top_ids]
        print("More than 6 runs supplied; plotting the top 6 by final_10pct_mean.")

    comparison_png = plot_run_comparison(plot_runs, metric=args.metric, step_key=step_key)

    _print_frame(f"Run Comparison ({args.metric})", compare_df.reset_index())
    if compare_df.attrs.get("warning"):
        print(f"\nWarning: {compare_df.attrs['warning']}")
    print(f"\nComparison PNG: {comparison_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
