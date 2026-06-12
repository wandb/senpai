# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""Helpers for working with W&B (Weights & Biases) training data.

Optimized for projects of any size, including large projects (10K+ runs,
1K+ metrics per run).

Key features:
- fetch_runs: Direct GraphQL with summaryMetrics field selection (15-25x
  faster than SDK iteration on large projects)
- get_api: Uses timeout=120 to prevent timeouts on large projects
- probe_project: Discovers project scale and available metrics
- runs_to_dataframe: Selective config/metric access
- diagnose_run: Configurable metric keys, uses beta_scan_history (parquet)
  for large histories
- scan_history: Auto-selects beta_scan_history for runs with 10K+ steps
- All history methods require explicit keys to avoid 502s on runs with
  thousands of metrics

    Usage (in sandbox):
    import sys
    sys.path.insert(0, ".claude/skills/wbagent/scripts")
    from wandb_helpers import (
        get_api,             # Create API with large-project-safe timeout
        probe_project,       # Discover project scale, metrics, config keys, artifacts
        fetch_runs,          # Fast selected-summary GraphQL run fetch
        fetch_run_summaries, # Fast selected/full summary fetch by run id
        count_runs,          # Exact lazy server-side run count
        runs_to_dataframe,   # Convert runs to a clean pandas DataFrame
        diagnose_run,        # Quick diagnostic summary of a training run
        compare_configs,     # Side-by-side config diff between two runs
        scan_history,        # Smart history scan (beta_scan_history for large runs)
        scan_history_until_step,  # Bounded scan that stops after a target step
    )
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# API factory
# ---------------------------------------------------------------------------


def get_api(timeout: int = 120) -> Any:
    """Create a wandb.Api with a safe timeout for large projects.

    The default wandb timeout (19s) causes frequent timeouts on projects
    with 10K+ runs or runs with 1K+ metrics.
    """
    import wandb

    return wandb.Api(timeout=timeout)


# ---------------------------------------------------------------------------
# Project probe
# ---------------------------------------------------------------------------


def probe_project(api: Any, path: str, sample_size: int = 3) -> dict[str, Any]:
    """Discover project characteristics before running queries.

    Call this FIRST on an unfamiliar project. It returns the project scale,
    available metric keys, config shape, whether runs have step history, and
    what artifacts have been logged. This lets you choose the right query
    strategy upfront instead of hitting timeouts or 502s by accident.

    Args:
        api: wandb.Api instance (use get_api()).
        path: "entity/project" string.
        sample_size: Number of runs to sample for metric/config inspection.

    Returns:
        Dict with: run_count_estimate, sample_metrics, sample_config_keys,
        has_step_history, recommended_per_page, artifact_names, warnings.
        artifact_names is a dict mapping artifact base name -> artifact type
        for all artifacts logged by the sampled runs.
    """
    result: dict[str, Any] = {"path": path, "warnings": []}

    # Fetch a small sample of runs — avoid triggering len() or large pages
    runs = api.runs(
        path, filters={"state": "finished"}, order="-created_at", per_page=sample_size
    )
    sample = runs[:sample_size]

    if not sample:
        result["run_count_estimate"] = 0
        result["warnings"].append("No finished runs found")
        return result

    # Inspect sampled runs
    all_metric_keys: set[str] = set()
    all_config_keys: set[str] = set()
    has_history = False
    artifact_names: dict[str, str] = {}  # base_name -> type

    for run in sample:
        metric_keys = {k for k in run.summary_metrics.keys() if not k.startswith("_")}
        config_keys = {k for k in run.config.keys() if not k.startswith("_")}
        all_metric_keys |= metric_keys
        all_config_keys |= config_keys
        if getattr(run, "lastHistoryStep", -1) >= 0:
            has_history = True
        try:
            for art in run.logged_artifacts():
                base = art.name.split(":")[0]
                if base not in artifact_names:
                    artifact_names[base] = art.type
        except Exception:
            pass

    n_metrics = len(all_metric_keys)
    result["sample_metric_count"] = n_metrics
    result["sample_metric_keys"] = sorted(all_metric_keys)[:50]
    result["sample_config_keys"] = sorted(all_config_keys)[:50]
    result["has_step_history"] = has_history
    result["artifact_names"] = artifact_names

    # Scale warnings
    if n_metrics > 500:
        result["warnings"].append(
            f"Runs have {n_metrics} metrics — ALWAYS pass keys= to history/scan_history"
        )
    if n_metrics > 5000:
        result["warnings"].append(
            f"Runs have {n_metrics} metrics — history() without keys WILL 502"
        )

    # Recommend per_page based on metric density
    if n_metrics > 1000:
        result["recommended_per_page"] = 10
    elif n_metrics > 100:
        result["recommended_per_page"] = 50
    else:
        result["recommended_per_page"] = 100

    return result


# ---------------------------------------------------------------------------
# Smart history scan
# ---------------------------------------------------------------------------


def scan_history(
    run: Any,
    keys: list[str],
    max_rows: int | None = None,
    use_beta: bool | None = None,
) -> list[dict[str, Any]]:
    """Read history rows from a run, choosing the fastest available method.

    Uses beta_scan_history (parquet-backed) for runs with large step counts
    (10K+ steps) since it avoids GraphQL pagination. Falls back to
    scan_history for smaller runs where parquet download overhead isn't worth it.

    IMPORTANT: keys is required. Never call without explicit keys on large
    projects — runs with 1K+ metrics will 502 or timeout without key filtering.

    Args:
        run: A W&B Run object.
        keys: Metric keys to fetch. REQUIRED.
        max_rows: Stop after this many rows. None = all rows.
        use_beta: Force beta_scan_history (True), force regular (False),
                  or auto-detect (None, default).

    Returns:
        List of dicts with the requested keys + _step.
    """
    if not keys:
        raise ValueError(
            "keys is required — never scan without explicit keys on large projects"
        )

    # Auto-detect: use beta for runs with 10K+ steps
    if use_beta is None:
        total_steps = getattr(run, "lastHistoryStep", -1)
        use_beta = total_steps >= 10_000

    rows = []
    if use_beta and hasattr(run, "beta_scan_history"):
        scanner = run.beta_scan_history(
            keys=keys, page_size=min(max_rows or 10_000, 10_000)
        )
    else:
        scanner = run.scan_history(keys=keys)

    for row in scanner:
        rows.append(dict(row))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# Fast run fetcher (direct GraphQL with field selection)
# ---------------------------------------------------------------------------

def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "none", "None"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        return json.loads(value)
    return value


def parse_filter_expr(expr: str) -> tuple[str, Any]:
    """Parse `key=value`, `key<value`, `key<=value`, `key>value`, or `key>=value`."""
    match = re.match(r"^([^<>=!]+?)\s*(<=|>=|=|<|>)\s*(.+)$", expr)
    if not match:
        raise ValueError(f"Invalid filter expression: {expr!r}")
    key, op, raw_value = match.groups()
    key = key.strip()
    value = _parse_scalar(raw_value)
    if op == "=":
        return key, value
    op_map = {"<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}
    return key, {op_map[op]: value}


def build_filters(
    filters: list[str] | None = None,
    filters_json: str | None = None,
    default_state: str | None = "finished",
) -> dict[str, Any]:
    """Build a W&B filter dict from CLI-friendly expressions."""
    out: dict[str, Any] = {}
    if default_state:
        out["state"] = default_state
    if filters_json:
        out.update(json.loads(filters_json))
    for expr in filters or []:
        key, value = parse_filter_expr(expr)
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _unwrap_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict) and "value" in value:
            out[key] = value["value"]
        else:
            out[key] = value
    return out


def nested_get(data: dict[str, Any], key: str) -> Any:
    """Read a dict key, supporting both flat keys and dotted nested paths."""
    if key in data:
        return data[key]
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


_nested_get = nested_get


def _post_graphql(api: Any, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    import requests

    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", api.api_key),
        headers={"Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=getattr(api, "_timeout", 120),
    )
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    project = payload.get("data", {}).get("project")
    if project is None:
        raise RuntimeError("GraphQL returned project=null; check project path and credentials")
    return payload


_RUNS_QUERY = """\
query Runs($project: String!, $entity: String!, $cursor: String,
           $perPage: Int!, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    state
                    createdAt
                    displayName
                    summaryMetrics(keys: %KEYS%)
                    config
                }
                cursor
            }
            pageInfo {
                endCursor
                hasNextPage
            }
        }
    }
}
"""


_RUN_SUMMARIES_QUERY = """\
query Runs($project: String!, $entity: String!, $cursor: String,
           $perPage: Int!, $order: String, $filters: JSONString) {
    project(name: $project, entityName: $entity) {
        runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
            edges {
                node {
                    id
                    name
                    state
                    createdAt
                    displayName
                    %SUMMARY_METRICS%
                }
                cursor
            }
            pageInfo {
                endCursor
                hasNextPage
            }
        }
    }
}
"""


def fetch_runs(
    api: Any,
    path: str,
    metric_keys: list[str] | None = None,
    limit: int = 200,
    filters: dict[str, Any] | None = None,
    order: str = "-created_at",
    config_keys: list[str] | None = None,
    per_page: int = 50,
) -> list[dict[str, Any]]:
    """Fetch runs using direct GraphQL with summaryMetrics field selection.

    This is DRAMATICALLY faster than iterating run objects on large projects.
    The standard SDK fetches ALL summary metrics per run (771KB+ per run on
    projects with 20K+ metrics). This function uses the GraphQL
    summaryMetrics(keys: [...]) parameter to fetch ONLY the requested metrics,
    reducing payload from 771KB to ~50 bytes per run.

    Benchmarks (wandb/large_runs_demo, 72K runs, 44K metrics/run):
        Standard SDK:  ~600ms/run (12s for 20 runs)
        This function: ~34ms/run  (0.67s for 20 runs) — 17x faster

    Args:
        api: wandb.Api instance (use get_api()).
        path: "entity/project" string.
        metric_keys: Summary metric keys to fetch. None or empty skips summaryMetrics.
        limit: Max runs to return.
        filters: W&B filter dict (e.g., {"state": "finished"}).
        order: Sort order (e.g., "-created_at", "+summary_metrics.loss").
        config_keys: Specific config keys to extract. None = skip config.
        per_page: Runs per GraphQL page (default 50).

    Returns:
        List of flat dicts with run metadata + selected metrics + selected config.
    """
    entity, project = path.split("/", 1)
    metric_keys = metric_keys or []

    # Build the query with specific metric keys
    if metric_keys:
        query = _RUNS_QUERY.replace("%KEYS%", json.dumps(metric_keys))
    else:
        query = _RUNS_QUERY.replace("                    summaryMetrics(keys: %KEYS%)\n", "")

    # If we don't need config, remove it from the query to save bandwidth
    if config_keys is None:
        query = query.replace("                    config\n", "")

    filter_str = json.dumps(filters or {})

    rows: list[dict[str, Any]] = []
    cursor = None
    remaining = limit

    while remaining > 0:
        page_size = min(per_page, remaining)
        variables: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "perPage": page_size,
            "order": order,
            "filters": filter_str,
        }
        if cursor:
            variables["cursor"] = cursor

        data = _post_graphql(api, query, variables)
        runs_data = data.get("data", {}).get("project", {}).get("runs", {})
        edges = runs_data.get("edges", [])
        page_info = runs_data.get("pageInfo", {})

        for edge in edges:
            node = edge["node"]
            summary = json.loads(node.get("summaryMetrics") or "{}")

            row: dict[str, Any] = {
                "id": node["id"],
                "name": node["name"],
                "display_name": node.get("displayName"),
                "state": node["state"],
                "created_at": node["createdAt"],
            }

            # Config — selective
            if config_keys is not None:
                config = _unwrap_config(json.loads(node.get("config") or "{}"))
                for k in config_keys:
                    row[f"config.{k}"] = _nested_get(config, k)

            # Summary metrics — already filtered server-side
            for key in metric_keys:
                row[key] = summary.get(key)

            rows.append(row)

        remaining -= len(edges)
        if not page_info.get("hasNextPage") or not edges:
            break
        cursor = page_info.get("endCursor")

    return rows[:limit]


def fetch_run_summaries(
    api: Any,
    path: str,
    run_ids: list[str],
    summary_keys: list[str] | None = None,
    order: str = "-created_at",
    per_page: int = 50,
) -> list[dict[str, Any]]:
    """Fetch selected runs with summary metrics via GraphQL.

    `summary_keys=None` requests the full run summary. Use this for small sets
    of representative runs when discovering project-level metadata; pass
    explicit keys for very wide projects.
    """
    if not run_ids:
        return []

    entity, project = path.split("/", 1)
    summary_clause = (
        "summaryMetrics"
        if summary_keys is None
        else f"summaryMetrics(keys: {json.dumps(summary_keys)})"
    )
    query = _RUN_SUMMARIES_QUERY.replace("%SUMMARY_METRICS%", summary_clause)
    limit = len(run_ids)
    rows: list[dict[str, Any]] = []
    cursor = None
    remaining = limit
    while remaining > 0:
        page_size = min(per_page, remaining)
        variables: dict[str, Any] = {
            "project": project,
            "entity": entity,
            "perPage": page_size,
            "order": order,
            "filters": json.dumps({"name": {"$in": run_ids}}),
        }
        if cursor:
            variables["cursor"] = cursor

        payload = _post_graphql(api, query, variables)
        runs_data = payload["data"]["project"]["runs"]
        edges = runs_data.get("edges", [])
        for edge in edges:
            node = edge["node"]
            rows.append(
                {
                    "id": node["id"],
                    "name": node["name"],
                    "display_name": node.get("displayName"),
                    "state": node["state"],
                    "created_at": node["createdAt"],
                    "summary": json.loads(node.get("summaryMetrics") or "{}"),
                }
            )

        remaining -= len(edges)
        if not edges or not runs_data.get("pageInfo", {}).get("hasNextPage"):
            break
        cursor = runs_data.get("pageInfo", {}).get("endCursor")

    by_name = {row["name"]: row for row in rows}
    return [by_name[run_id] for run_id in run_ids if run_id in by_name]


def count_runs(
    api: Any,
    path: str,
    filters: dict[str, Any] | None = None,
    include_sweeps: bool = False,
) -> int:
    """Exact server-side run count. Does not materialize runs."""
    return len(
        api.runs(
            path,
            filters=filters or {},
            per_page=1,
            include_sweeps=include_sweeps,
            lazy=True,
        )
    )


def scan_history_until_step(
    run: Any,
    keys: list[str],
    step_key: str,
    target_step: int | float,
    max_rows: int | None = None,
    use_beta: bool | None = None,
) -> list[dict[str, Any]]:
    """Scan selected history keys until `step_key` passes `target_step`.

    W&B history rows are ordered by `_step`; most training step counters are
    monotonic. Stopping after the first row beyond the target keeps at-step
    comparisons bounded while preserving the large-history beta-scan behavior
    used by `scan_history()`.
    """
    if not keys:
        raise ValueError(
            "keys is required — never scan without explicit keys on large projects"
        )

    if use_beta is None:
        total_steps = getattr(run, "lastHistoryStep", -1)
        use_beta = total_steps >= 10_000

    if use_beta and hasattr(run, "beta_scan_history"):
        scanner = run.beta_scan_history(
            keys=keys,
            page_size=min(max_rows or 10_000, 10_000),
        )
    else:
        scanner = run.scan_history(keys=keys)

    rows: list[dict[str, Any]] = []
    for raw_row in scanner:
        row = dict(raw_row)
        step_value = row.get(step_key)
        if isinstance(step_value, (int, float)) and step_value > target_step:
            break
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _metric_group(metric: str) -> str:
    return metric.split("/", 1)[0] if "/" in metric else metric


def latest_at_or_before(
    rows: list[dict[str, Any]],
    step_key: str,
    metric: str,
    step: int | float,
) -> dict[str, Any] | None:
    """Return the latest row with a metric value at or before a target step."""
    candidates = [
        row
        for row in rows
        if row.get(step_key) is not None
        and row[step_key] <= step
        and row.get(metric) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[step_key])


def compare_runs_at_step(
    api: Any,
    path: str,
    run_ids: list[str],
    step: int | float,
    metrics: list[str],
    step_key: str = "_step",
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Compare selected history metrics at the latest logged step <= target step.

    W&B history scans require all requested keys to be present in a row. Training
    and validation metrics often log on different cadences, so metrics are
    grouped by namespace and retried one-by-one only when a grouped scan misses.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for metric in metrics:
        grouped[_metric_group(metric)].append(metric)

    comparison: dict[str, Any] = {}
    for run_id in run_ids:
        run = api.run(f"{path}/{run_id}")
        run_result: dict[str, Any] = {
            "display_name": getattr(run, "display_name", None) or getattr(run, "name", None),
            "target_step": step,
            "metrics": {},
        }
        for group_metrics in grouped.values():
            keys = list(dict.fromkeys(["_step", step_key, *group_metrics]))
            rows = scan_history_until_step(run, keys, step_key, step, max_rows=max_rows)
            for metric in group_metrics:
                row = latest_at_or_before(rows, step_key, metric, step)
                if row is None and len(group_metrics) > 1:
                    rows_single = scan_history_until_step(
                        run,
                        keys=list(dict.fromkeys(["_step", step_key, metric])),
                        step_key=step_key,
                        target_step=step,
                        max_rows=max_rows,
                    )
                    row = latest_at_or_before(rows_single, step_key, metric, step)
                run_result["metrics"][metric] = (
                    None
                    if row is None
                    else {
                        "value": row[metric],
                        step_key: row[step_key],
                        "_step": row.get("_step"),
                    }
                )
        comparison[run_id] = run_result
    return comparison



# ---------------------------------------------------------------------------
# Runs -> DataFrame (legacy wrapper, uses fetch_runs when possible)
# ---------------------------------------------------------------------------


def runs_to_dataframe(
    runs: Any,
    limit: int = 200,
    metric_keys: list[str] | None = None,
    config_keys: list[str] | None = None,
    include_all_config: bool = False,
) -> list[dict[str, Any]]:
    """Convert W&B runs to a list of flat dicts (ready for pd.DataFrame).

    For best performance on large projects, use fetch_runs() directly instead.
    This function exists for backward compatibility with code that already has
    a runs object.

    Args:
        runs: W&B Runs object from api.runs().
        limit: Max runs to process (default 200).
        metric_keys: Summary metric keys to include. If None, includes
                     "loss", "val_loss", "accuracy".
        config_keys: Specific config keys to include. None = skip config.
        include_all_config: If True, include all non-internal config keys.
                           Ignored if config_keys is set. Can be slow on
                           runs with large configs.

    Returns:
        List of dicts with run metadata + selected config + selected metrics.
    """
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
        # Config — selective by default
        if config_keys is not None:
            for k in config_keys:
                row[f"config.{k}"] = run.config.get(k)
        elif include_all_config:
            for k, v in run.config.items():
                if not k.startswith("_"):
                    row[f"config.{k}"] = v
        # Summary metrics — only requested keys
        for key in metric_keys:
            row[key] = run.summary_metrics.get(key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------------------------


def diagnose_run(
    run: Any,
    train_key: str = "loss",
    val_key: str | None = "val_loss",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Quick diagnostic summary of a training run.

    Checks for convergence, overfitting, NaN values, and other common
    training issues. Uses beta_scan_history for runs with large step counts.

    Args:
        run: A W&B Run object from api.run().
        train_key: Primary training metric key (default "loss").
        val_key: Validation metric key (default "val_loss"). None to skip.
        max_steps: Limit rows read. None = all.

    Returns:
        Dict with diagnostic keys. Returns {"error": ...} if the
        requested keys don't exist.
    """
    import pandas as pd

    # Verify keys exist in summary before scanning history
    available_keys = set(run.summary_metrics.keys())
    if train_key not in available_keys:
        return {
            "error": f"Key '{train_key}' not in run summary. Available: {sorted(k for k in available_keys if not k.startswith('_'))[:20]}"
        }

    keys = [train_key]
    if val_key and val_key in available_keys:
        keys.append(val_key)
    elif val_key and val_key not in available_keys:
        val_key = None  # skip val check

    rows = scan_history(run, keys=keys, max_rows=max_steps)
    if not rows:
        return {
            "error": "No history rows found",
            "summary_value": run.summary_metrics.get(train_key),
        }

    df = pd.DataFrame(rows)
    if train_key not in df.columns:
        return {
            "error": f"Key '{train_key}' not in history columns: {list(df.columns)}"
        }

    loss = df[train_key].dropna()

    diagnostics: dict[str, Any] = {
        "total_steps": len(loss),
        "final_value": float(loss.iloc[-1]) if len(loss) else None,
        "min_value": float(loss.min()) if len(loss) else None,
        "min_value_step": int(loss.idxmin()) if len(loss) else None,
        "has_nan": bool(df[train_key].isna().any()),
        "final_10pct_mean": float(loss.tail(max(1, len(loss) // 10)).mean())
        if len(loss)
        else None,
    }

    # Overfitting check
    if val_key and val_key in df.columns:
        val = df[val_key].dropna()
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


def compare_configs(
    run_a: Any,
    run_b: Any,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Side-by-side config comparison between two W&B runs.

    Args:
        run_a: First W&B Run object.
        run_b: Second W&B Run object.
        keys: Specific config keys to compare. None = all non-internal keys.

    Returns:
        List of dicts with differing keys and their values per run.
    """
    if keys is not None:
        config_a = {k: run_a.config.get(k) for k in keys}
        config_b = {k: run_b.config.get(k) for k in keys}
    else:
        config_a = {k: v for k, v in run_a.config.items() if not k.startswith("_")}
        config_b = {k: v for k, v in run_b.config.items() if not k.startswith("_")}

    all_keys = sorted(set(config_a) | set(config_b))
    diffs = []
    for k in all_keys:
        val_a = config_a.get(k)
        val_b = config_b.get(k)
        if val_a != val_b:
            diffs.append(
                {
                    "key": k,
                    run_a.name: val_a,
                    run_b.name: val_b,
                }
            )
    return diffs
