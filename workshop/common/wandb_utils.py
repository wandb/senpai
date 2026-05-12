"""W&B helpers for the runnable workshop."""

from __future__ import annotations

import os
from typing import Any

from .config import WorkshopConfig


def wandb_api(config: WorkshopConfig):
    os.environ["WANDB_API_KEY"] = config.wandb_api_key
    import wandb

    return wandb.Api()


def validate_project(config: WorkshopConfig) -> dict[str, str]:
    api = wandb_api(config)
    project = api.project(config.wandb_project, entity=config.wandb_entity)
    return {
        "entity": config.wandb_entity,
        "project": config.wandb_project,
        "name": project.name,
    }


def recent_runs(config: WorkshopConfig, *, limit: int = 10, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    api = wandb_api(config)
    runs = api.runs(config.wandb_path, filters=filters or {}, order="-created_at")
    rows = []
    for idx, run in enumerate(runs):
        if idx >= limit:
            break
        rows.append(run_summary(run))
    return rows


def run_summary(run: Any) -> dict[str, Any]:
    summary = dict(run.summary or {})
    config = {
        key: value
        for key, value in dict(run.config or {}).items()
        if not key.startswith("_")
    }
    metric_keys = [
        key for key in summary
        if any(part in key.lower() for part in ("val", "test", "loss", "metric", "mae", "mse", "rel_l2"))
    ]
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(getattr(run, "created_at", "")),
        "url": getattr(run, "url", ""),
        "config": {key: config[key] for key in sorted(config)[:25]},
        "metrics": {key: summary[key] for key in sorted(metric_keys)[:30]},
    }


def compact_history(run: Any, keys: list[str], *, samples: int = 200) -> dict[str, Any]:
    rows = list(run.scan_history(keys=keys, page_size=min(samples, 1000)))
    if not rows:
        return {"rows": 0, "keys": keys, "stats": {}}
    stats = {}
    for key in keys:
        vals = [row.get(key) for row in rows if isinstance(row.get(key), int | float)]
        if vals:
            stats[key] = {
                "count": len(vals),
                "first": vals[0],
                "last": vals[-1],
                "min": min(vals),
                "max": max(vals),
            }
    return {"rows": len(rows), "keys": keys, "stats": stats}


def local_trace_shape() -> dict[str, Any]:
    return {
        "trace_id": "local-teaching-trace",
        "root_call": "advisor_review_pr",
        "calls": [
            {"type": "tool", "name": "pr_all_comments", "summary": "Found terminal SENPAI-RESULT."},
            {"type": "tool", "name": "wandb.Api().run", "summary": "Loaded final validation and test metrics."},
            {"type": "llm", "name": "advisor_decision", "summary": "Rejected validation-only improvement."},
        ],
    }
