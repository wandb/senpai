#!/usr/bin/env python3
"""Setup-time W&B metadata discovery for wandb-driven-dev projects.

This script is intentionally more opinionated than the curve analysis CLI:

- read the project-local wandb-driven-dev config
- use configured decision/health metrics
- sample a few recent finished runs when run IDs are not supplied
- infer semantic step keys from selected summaries
- write only project metadata and step-key decisions back to the config

It does not persist run summaries or metric values. W&B remains the source of
truth for live data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_HERE = Path(__file__).resolve().parent
_WBAGENT_SCRIPTS = _HERE.parent.parent / "wbagent" / "scripts"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_WBAGENT_SCRIPTS))
from wandb_helpers import (  # noqa: E402
    fetch_run_summaries,
    fetch_runs,
    get_api,
    scan_history,
)
from wdd_helpers import CONFIG_PATH, read_config, update_preflight_config  # noqa: E402


COMMON_STEP_KEY_CANDIDATES = (
    "_step",
    "global_step",
    "train/global_step",
    "trainer/global_step",
    "step",
    "epoch",
)


def _unique_columns(columns: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)
    return out


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def configured_metrics(cfg: dict[str, Any]) -> list[str]:
    metrics = cfg.get("metrics") or {}
    return _unique_columns([*(metrics.get("decision") or []), *(metrics.get("health") or [])])


def configured_candidate_step_keys(cfg: dict[str, Any]) -> list[str]:
    curves = cfg.get("curves") or {}
    metric_step_keys = curves.get("metric_step_keys") or {}
    return _unique_columns(
        [
            "_step",
            *(curves.get("candidate_step_keys") or []),
            *(value for value in metric_step_keys.values() if value),
        ]
    )


def step_key_candidates(metrics: list[str], extra: list[str] | None = None) -> list[str]:
    """Return generic and metric-namespace-specific step-key candidates."""
    candidates = list(COMMON_STEP_KEY_CANDIDATES)
    for metric in metrics:
        if "/" not in metric:
            continue
        namespace = metric.split("/", 1)[0]
        candidates.extend(
            [
                f"{namespace}/global_step",
                f"{namespace}/step",
                f"{namespace}/epoch",
            ]
        )
    if extra:
        candidates.extend(extra)
    return _unique_columns(candidates)


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except TypeError:
        return True
    if isinstance(missing, bool):
        return not missing
    return True


def _summary_span(values: list[Any]) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max() - numeric.min())


def step_key_score(frame: pd.DataFrame, metric: str, step_key: str) -> dict[str, Any]:
    """Score how usable a step key is for one metric in one fetched frame."""
    if metric not in frame.columns or step_key not in frame.columns:
        return {
            "step_key": step_key,
            "points": 0,
            "unique_steps": 0,
            "span": None,
            "monotonic": False,
            "usable": False,
        }
    data = frame[[step_key, metric]].dropna()
    if data.empty:
        return {
            "step_key": step_key,
            "points": 0,
            "unique_steps": 0,
            "span": None,
            "monotonic": False,
            "usable": False,
        }
    data = data.sort_values(step_key).drop_duplicates(subset=[step_key], keep="last")
    steps = pd.to_numeric(data[step_key], errors="coerce").dropna()
    if steps.empty:
        return {
            "step_key": step_key,
            "points": 0,
            "unique_steps": 0,
            "span": None,
            "monotonic": False,
            "usable": False,
        }
    span = float(steps.max() - steps.min())
    return {
        "step_key": step_key,
        "points": int(len(steps)),
        "unique_steps": int(steps.nunique()),
        "span": span,
        "monotonic": bool(steps.is_monotonic_increasing),
        "usable": bool(len(steps) >= 2 and steps.nunique() >= 2 and span > 0),
    }


def recommend_metric_step_key(
    frames: dict[str, pd.DataFrame],
    metric: str,
    candidates: list[str] | None = None,
    min_point_ratio: float = 0.8,
) -> dict[str, Any]:
    """Recommend a step key for a metric across fetched runs."""
    candidates = _unique_columns(candidates or step_key_candidates([metric]))
    per_candidate: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        scores = {
            run_id: step_key_score(frame, metric, candidate)
            for run_id, frame in frames.items()
        }
        points = sum(score["points"] for score in scores.values())
        unique_steps = sum(score["unique_steps"] for score in scores.values())
        usable_runs = sum(1 for score in scores.values() if score["usable"])
        spans = [score["span"] for score in scores.values() if score["span"] is not None]
        per_candidate[candidate] = {
            "points": points,
            "unique_steps": unique_steps,
            "usable_runs": usable_runs,
            "max_span": max(spans) if spans else None,
            "runs": scores,
        }

    best_points = max((score["points"] for score in per_candidate.values()), default=0)
    threshold = best_points * min_point_ratio
    eligible = [
        (key, score)
        for key, score in per_candidate.items()
        if score["points"] >= threshold and score["usable_runs"] > 0
    ]
    if eligible:
        recommended, _ = max(
            eligible,
            key=lambda item: (
                item[0] != "_step",
                item[1]["usable_runs"],
                item[1]["points"],
                item[1]["max_span"] or 0,
            ),
        )
        reason = "custom step key has comparable coverage" if recommended != "_step" else "using W&B default _step"
    else:
        recommended = "_step"
        reason = "no candidate had enough usable points; falling back to _step"

    return {
        "metric": metric,
        "recommended_step_key": recommended,
        "reason": reason,
        "candidates": per_candidate,
    }


def recommend_metric_step_keys(
    frames: dict[str, pd.DataFrame],
    metrics: list[str],
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Recommend step keys for several metrics from already-fetched frames."""
    candidate_keys = _unique_columns(candidates or step_key_candidates(metrics))
    return {
        "default_step_key": "_step",
        "candidate_step_keys": candidate_keys,
        "metric_step_keys": {
            metric: recommend_metric_step_key(frames, metric, candidate_keys)
            for metric in metrics
        },
    }


def recommend_metric_step_key_from_summaries(
    rows: list[dict[str, Any]],
    metric: str,
    candidates: list[str],
    min_run_coverage_ratio: float = 0.8,
) -> dict[str, Any]:
    """Recommend a step key from selected run summaries."""
    metric_runs = [row for row in rows if _present(row.get(metric))]
    metric_run_count = len(metric_runs)
    per_candidate: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        if candidate == "_step":
            paired_rows = metric_runs
            candidate_values: list[Any] = []
        else:
            paired_rows = [
                row
                for row in metric_runs
                if _present(row.get(candidate))
            ]
            candidate_values = [row.get(candidate) for row in paired_rows]

        coverage = 0.0 if metric_run_count == 0 else len(paired_rows) / metric_run_count
        per_candidate[candidate] = {
            "mode": "summary",
            "metric_summary_runs": metric_run_count,
            "paired_summary_runs": len(paired_rows),
            "coverage": coverage,
            "unique_steps": 0 if candidate == "_step" else int(pd.Series(candidate_values).nunique(dropna=True)),
            "span": None if candidate == "_step" else _summary_span(candidate_values),
            "usable": bool(candidate == "_step" and metric_run_count > 0)
            or bool(candidate != "_step" and len(paired_rows) > 0),
        }

    eligible_custom = [
        candidate
        for candidate in candidates
        if candidate != "_step"
        and per_candidate[candidate]["paired_summary_runs"] > 0
        and per_candidate[candidate]["coverage"] >= min_run_coverage_ratio
    ]
    if eligible_custom:
        recommended = eligible_custom[0]
        reason = "custom step key is present in summaries for comparable metric coverage"
    else:
        recommended = "_step"
        reason = "no custom step key had enough summary coverage; using W&B default _step"

    return {
        "metric": metric,
        "recommended_step_key": recommended,
        "reason": reason,
        "candidates": per_candidate,
    }


def recommend_metric_step_keys_from_summaries(
    rows: list[dict[str, Any]],
    metrics: list[str],
    candidates: list[str],
) -> dict[str, Any]:
    return {
        "default_step_key": "_step",
        "candidate_step_keys": candidates,
        "metric_step_keys": {
            metric: recommend_metric_step_key_from_summaries(rows, metric, candidates)
            for metric in metrics
        },
    }


def build_preflight_metadata(
    project: str,
    summary_rows: list[dict[str, Any]],
    checked_summary_keys: list[str],
    metric_step_keys: dict[str, str],
    candidate_step_keys: list[str],
) -> dict[str, Any]:
    """Build project-level W&B metadata stored in the local config.

    Persist keys and decisions only. Do not persist metric values or per-run
    summaries; W&B remains the source of truth for live data.
    """
    observed_summary_keys: list[str] = []
    for row in summary_rows:
        summary = row.get("summary") or {}
        observed_summary_keys.extend(key for key, value in summary.items() if _present(value))

    observed_summary_keys = _unique_columns(observed_summary_keys)
    checked_set = set(checked_summary_keys)
    observed_metric_keys = [key for key in observed_summary_keys if not key.startswith("_")]
    observed_system_keys = [key for key in observed_summary_keys if key.startswith("_")]
    return {
        "preflight": {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "setup_project.py",
            "project": project,
            "sampled_run_count": len(summary_rows),
            "checked_summary_keys": checked_summary_keys,
            "observed_summary_keys": observed_summary_keys,
            "observed_metric_keys": observed_metric_keys,
            "observed_system_keys": observed_system_keys,
            "missing_checked_summary_keys": [
                key for key in checked_summary_keys if key not in observed_summary_keys
            ],
            "observed_step_keys": [
                key for key in candidate_step_keys if key != "_step" and key in observed_summary_keys
            ],
            "metric_step_keys": metric_step_keys,
            "candidate_step_keys": candidate_step_keys,
            "unchecked_observed_summary_keys": [
                key for key in observed_summary_keys if key not in checked_set
            ],
        }
    }


def compact_preflight_result(
    recommendations: dict[str, Any],
    project: str,
    run_ids: list[str],
    mode: str,
    missing_runs: list[str] | None = None,
    checked_summary_keys: list[str] | None = None,
    summary_rows: list[dict[str, Any]] | None = None,
    max_rows: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Return the stable setup preflight contract agents should persist."""
    detailed_metrics = recommendations["metric_step_keys"]
    metric_step_keys = {
        metric: detail["recommended_step_key"]
        for metric, detail in detailed_metrics.items()
    }
    configured_candidates = _unique_columns(
        ["_step", *(step_key for step_key in metric_step_keys.values() if step_key != "_step")]
    )
    result: dict[str, Any] = {
        "mode": mode,
        "project": project,
        "runs": run_ids,
        "missing_runs": missing_runs or [],
        "default_step_key": recommendations.get("default_step_key", "_step"),
        "metric_step_keys": metric_step_keys,
        "candidate_step_keys": configured_candidates,
        "config_patch": {
            "curves": {
                "default_step_key": recommendations.get("default_step_key", "_step"),
                "metric_step_keys": metric_step_keys,
                "candidate_step_keys": configured_candidates,
            }
        },
    }
    if checked_summary_keys is not None:
        result["checked_summary_keys"] = checked_summary_keys
    if summary_rows is not None:
        result["config_patch"]["wandb_metadata"] = build_preflight_metadata(
            project,
            summary_rows,
            checked_summary_keys or [],
            metric_step_keys,
            configured_candidates,
        )
    if max_rows is not None:
        result["max_rows"] = max_rows
    if workers is not None:
        result["workers"] = workers
    return result


def _history_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame | None:
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[columns].dropna(subset=["_step"])
    if frame.empty:
        return None
    return frame.drop_duplicates(subset=["_step"], keep="last")


def _merge_history_frame(merged: pd.DataFrame | None, part: pd.DataFrame | None) -> pd.DataFrame | None:
    if part is None:
        return merged
    if merged is None:
        return part
    out = merged.merge(part, on="_step", how="outer", suffixes=("", "__new"))
    for column in part.columns:
        if column == "_step":
            continue
        new_column = f"{column}__new"
        if new_column not in out.columns:
            continue
        if column in out.columns:
            out[column] = out[column].combine_first(out[new_column])
        else:
            out[column] = out[new_column]
        out = out.drop(columns=[new_column])
    return out


def _rows_to_preflight_frame(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["_step"]).sort_values("_step")
    return frame.reset_index(drop=True)


def fetch_step_key_preflight_frame(
    api: Any,
    project: str,
    run_id: str,
    metrics: list[str],
    candidate_step_keys: list[str],
    max_rows: int | None = 5000,
    min_pair_coverage_ratio: float = 0.8,
) -> pd.DataFrame:
    """Fetch bounded metric and candidate-step samples for recommendation."""
    run = api.run(f"{project}/{run_id}")
    summary_keys = set(getattr(run, "summary_metrics", {}).keys())
    candidates = _unique_columns(
        [
            candidate
            for candidate in candidate_step_keys
            if candidate == "_step" or candidate in summary_keys
        ]
    )

    merged: pd.DataFrame | None = None
    frame_columns = _unique_columns(["_step", *metrics, *(key for key in candidates if key != "_step")])
    metric_point_counts: dict[str, int] = {}

    for metric in metrics:
        metric_rows = scan_history(run, keys=["_step", metric], max_rows=max_rows)
        metric_part = _history_frame(metric_rows, ["_step", metric])
        metric_point_counts[metric] = 0 if metric_part is None else int(metric_part[metric].notna().sum())
        merged = _merge_history_frame(merged, metric_part)

    for candidate in candidates:
        if candidate == "_step":
            continue
        step_rows = scan_history(run, keys=["_step", candidate], max_rows=max_rows)
        step_part = _history_frame(step_rows, ["_step", candidate])
        merged = _merge_history_frame(merged, step_part)

    if merged is not None:
        for metric in metrics:
            metric_points = metric_point_counts.get(metric, 0)
            if metric_points < 2 or metric not in merged.columns:
                continue
            min_pair_points = metric_points * min_pair_coverage_ratio
            for candidate in candidates:
                if candidate == "_step":
                    continue
                pair_points = (
                    int(merged[[metric, candidate]].dropna().shape[0])
                    if candidate in merged.columns
                    else 0
                )
                if pair_points >= min_pair_points:
                    continue
                pair_rows = scan_history(run, keys=["_step", candidate, metric], max_rows=max_rows)
                pair_part = _history_frame(pair_rows, ["_step", candidate, metric])
                merged = _merge_history_frame(merged, pair_part)

    rows = [] if merged is None else merged.to_dict("records")
    return _rows_to_preflight_frame(rows, frame_columns)


def select_representative_runs(
    api: Any,
    project: str,
    sample_runs: int = 3,
) -> list[str]:
    """Select a small recent-finished run sample without materializing a project."""
    rows = fetch_runs(
        api,
        project,
        metric_keys=[],
        filters={"state": "finished"},
        config_keys=None,
        order="-created_at",
        limit=sample_runs,
        per_page=sample_runs,
    )
    return [row["name"] for row in rows]


def preflight_wandb_step_keys(
    api: Any,
    project: str,
    run_ids: list[str],
    metrics: list[str],
    candidate_step_keys: list[str],
    max_rows: int | None = 5000,
    workers: int | None = None,
    api_factory: Any | None = get_api,
    validate_history: bool = False,
) -> dict[str, Any]:
    """Recommend step keys for configured curve metrics."""
    candidates = step_key_candidates(metrics, candidate_step_keys)
    if not validate_history:
        summary_keys = _unique_columns([*metrics, *(key for key in candidates if key != "_step")])
        summary_rows = fetch_run_summaries(
            api,
            project,
            run_ids=run_ids,
            summary_keys=summary_keys,
            order="-created_at",
            per_page=min(50, max(1, len(run_ids))),
        )
        rows = [
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "display_name": row.get("display_name"),
                "state": row.get("state"),
                "created_at": row.get("created_at"),
                **(row.get("summary") or {}),
            }
            for row in summary_rows
        ]
        by_name = {row["name"]: row for row in rows}
        ordered_rows = [by_name[run_id] for run_id in run_ids if run_id in by_name]
        recommendations = recommend_metric_step_keys_from_summaries(ordered_rows, metrics, candidates)
        return compact_preflight_result(
            recommendations,
            project=project,
            run_ids=run_ids,
            mode="summary",
            missing_runs=[run_id for run_id in run_ids if run_id not in by_name],
            checked_summary_keys=summary_keys,
            summary_rows=summary_rows,
        )

    workers = workers or min(12, max(1, len(run_ids)))

    def fetch_one(run_id: str) -> tuple[str, pd.DataFrame]:
        worker_api = api_factory() if api_factory is not None else api
        return run_id, fetch_step_key_preflight_frame(
            worker_api,
            project,
            run_id,
            metrics,
            candidates,
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

    recommendations = recommend_metric_step_keys(frames, metrics, candidates)
    return compact_preflight_result(
        recommendations,
        project=project,
        run_ids=run_ids,
        mode="history",
        max_rows=max_rows,
        workers=workers,
    )


def run_setup_preflight(
    api: Any,
    config_path: Path | str = CONFIG_PATH,
    project: str | None = None,
    run_ids: list[str] | None = None,
    metrics: list[str] | None = None,
    extra_step_keys: list[str] | None = None,
    sample_runs: int = 3,
    validate_history: bool = False,
    max_rows: int | None = 5000,
    workers: int | None = None,
    write_config: bool = True,
) -> dict[str, Any]:
    """Run the setup preflight using config defaults and return JSON-ready data."""
    cfg = read_config(config_path)
    if cfg is None:
        raise RuntimeError(f"{config_path} is missing; run setup and write the project config first")

    resolved_project = project or cfg.get("wandb_project")
    if not resolved_project:
        raise RuntimeError("wandb_project is required in config or via --project")

    selected_metrics = _unique_columns(metrics or configured_metrics(cfg))
    if not selected_metrics:
        raise RuntimeError("No metrics configured; add metrics.decision/health or pass --metrics")

    candidate_keys = _unique_columns([*configured_candidate_step_keys(cfg), *(extra_step_keys or [])])
    selected_runs = _unique_columns(run_ids or select_representative_runs(api, resolved_project, sample_runs=sample_runs))
    if not selected_runs:
        raise RuntimeError(f"No representative finished runs found in {resolved_project}")

    result = preflight_wandb_step_keys(
        api,
        resolved_project,
        run_ids=selected_runs,
        metrics=selected_metrics,
        candidate_step_keys=candidate_keys,
        max_rows=max_rows,
        workers=workers,
        api_factory=get_api,
        validate_history=validate_history,
    )
    result["config_path"] = str(config_path)
    result["selected_metrics"] = selected_metrics
    result["sample_runs"] = sample_runs
    result["write_config"] = write_config

    if write_config:
        update_preflight_config(
            result["config_patch"],
            project=resolved_project,
            path=config_path,
        )
        result["config_written"] = str(config_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover project curve metadata and update the wandb-driven-dev local config."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Project config path")
    parser.add_argument("--project", help=argparse.SUPPRESS)
    parser.add_argument("--runs", help="Optional run IDs, comma-separated. Defaults to recent finished runs.")
    parser.add_argument("--metrics", help=argparse.SUPPRESS)
    parser.add_argument("--extra-step-keys", default="", help="Extra step-key candidates, comma-separated")
    parser.add_argument("--sample-runs", type=int, default=3, help="Recent finished runs to sample when --runs is omitted")
    parser.add_argument(
        "--validate-history",
        action="store_true",
        help="Run bounded history probes when summary coverage is ambiguous. Slower.",
    )
    parser.add_argument("--max-rows", type=int, default=5000, help="Rows per scan for --validate-history")
    parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print the config patch without writing it")

    args = parser.parse_args(argv)
    start = time.perf_counter()
    api = get_api()
    result = run_setup_preflight(
        api,
        config_path=args.config,
        project=args.project,
        run_ids=_split_csv(args.runs),
        metrics=_split_csv(args.metrics),
        extra_step_keys=_split_csv(args.extra_step_keys),
        sample_runs=args.sample_runs,
        validate_history=args.validate_history,
        max_rows=args.max_rows,
        workers=args.workers,
        write_config=not args.dry_run,
    )
    print(json.dumps({"latency_s": round(time.perf_counter() - start, 4), "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
