#!/usr/bin/env python3
"""Create and record the Phase 4 W&B Report for an experiment.

This is the hardcoded Phase 4 report workflow:

1. Read `.claude/wandb-driven-dev.local.md`.
2. Read `experiments/<slug>/plan.md`.
3. Resolve pinned run URLs from plan.md plus launched W&B runs named
   `exp-<slug>-baseline`, `exp-<slug>-variant`, or `exp-<slug>-variant-*`.
4. Verify decision + health metrics exist on every resolved run.
5. Optionally build a focused table from selected config/summary columns.
6. Create the experiment report and splice run/report URLs into `## Runs`.

The script prints a JSON summary to stdout and exits nonzero on gate failures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from wdd_helpers import (
    _run_id,
    create_experiment_report,
    read_config,
    verify_required_metrics,
)

_HERE = Path(__file__).resolve().parent
_WBAGENT_SCRIPTS = _HERE.parent.parent / "wbagent" / "scripts"
sys.path.insert(0, str(_WBAGENT_SCRIPTS))
from wandb_helpers import get_api  # noqa: E402


_WANDb_RUN_URL_RE = re.compile(r"https://wandb\.ai/[^\s)>\]]+/runs/[A-Za-z0-9_.-]+")


def _section(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^##\s+{re.escape(name)}[^\n]*\n(?P<body>.*?)(?=^##\s+|\Z)",
        text,
    )
    return match.group("body").strip() if match else ""


def _first_field(text: str, label: str) -> str | None:
    match = re.search(rf"(?mi)^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    if not value or value.startswith("TODO"):
        return None
    return value


def _parse_metric_list(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    lowered = value.lower()
    if (
        "config default" in lowered
        or "override list" in lowered
        or value.strip() in {"...", "TODO", "TBD"}
    ):
        return fallback
    value = value.strip().strip("[]")
    parts = [
        p.strip().strip("`'\"")
        for p in re.split(r",|\s+", value)
        if p.strip().strip("`'\"")
    ]
    return parts or fallback


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [
        p.strip().strip("`'\"")
        for p in re.split(r",|\s+", value.strip().strip("[]"))
        if p.strip().strip("`'\"")
    ]
    return parts


def _plan_metrics(plan_text: str, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    metrics = _section(plan_text, "Metrics")
    decision_match = re.search(r"(?mi)^-\s+\*\*Decision:\*\*\s*(.+?)\s*$", metrics)
    health_match = re.search(r"(?mi)^-\s+\*\*Health:\*\*\s*(.+?)\s*$", metrics)
    cfg_metrics = cfg.get("metrics", {})
    decision = _parse_metric_list(
        decision_match.group(1) if decision_match else None,
        list(cfg_metrics.get("decision") or []),
    )
    health = _parse_metric_list(
        health_match.group(1) if health_match else None,
        list(cfg_metrics.get("health") or []),
    )
    return decision, health


def _plan_report_columns(plan_text: str) -> list[str]:
    body = _section(plan_text, "Report Columns")
    if not body:
        return []
    columns: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        columns.extend(_parse_list(stripped[1:].strip()))
    return list(dict.fromkeys(columns))


def _role_from_line(line: str) -> str | None:
    match = re.search(r"`?((?:baseline)|(?:variant(?:-[A-Za-z0-9_.-]+)?))`?", line)
    return match.group(1) if match else None


def _runs_from_plan(plan_text: str) -> dict[str, str]:
    runs: dict[str, str] = {}

    runs_section = _section(plan_text, "Runs")
    for line in runs_section.splitlines():
        url_match = _WANDb_RUN_URL_RE.search(line)
        if not url_match:
            continue
        role = _role_from_line(line)
        if role:
            runs[role] = url_match.group(0)

    baseline = _section(plan_text, "Baseline")
    baseline_url = _WANDb_RUN_URL_RE.search(baseline)
    if baseline_url and "baseline" not in runs:
        runs["baseline"] = baseline_url.group(0)

    return runs


def _display_name(run: Any) -> str:
    return getattr(run, "name", None) or getattr(run, "display_name", None) or ""


def _run_url(project: str, run: Any) -> str:
    entity, project_name = project.split("/", 1)
    run_id = getattr(run, "id", None) or getattr(run, "name", None)
    return getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project_name}/runs/{run_id}"


def _discover_runs(api: Any, project: str, slug: str) -> dict[str, str]:
    pattern = rf"^exp-{re.escape(slug)}-(baseline|variant(?:-[A-Za-z0-9_.-]+)?)$"
    runs = api.runs(
        project,
        filters={"display_name": {"$regex": pattern}},
        order="+created_at",
        per_page=50,
        include_sweeps=False,
    )
    out: dict[str, str] = {}
    for run in runs:
        name = _display_name(run)
        match = re.match(pattern, name)
        if not match:
            continue
        out[match.group(1)] = _run_url(project, run)
    return out


def _validate_roles(runs: dict[str, str], roles: list[str] | None) -> None:
    if roles:
        missing = [role for role in roles if role not in runs]
    else:
        missing = []
        if "baseline" not in runs:
            missing.append("baseline")
        if not any(role == "variant" or role.startswith("variant-") for role in runs):
            missing.append("variant")
    if missing:
        raise SystemExit(
            "FAIL: missing required run role(s): "
            + ", ".join(missing)
            + ". Record pinned URLs in plan.md or wait for W&B runs to appear."
        )


def _run_paths(project: str, runs: dict[str, str]) -> dict[str, str]:
    return {role: f"{project}/{_run_id(url_or_id)}" for role, url_or_id in runs.items()}


def _nested_get(data: Any, key: str) -> Any:
    if not isinstance(data, dict):
        return None
    if key in data:
        return data[key]
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _column_value(run: Any, column: str) -> Any:
    config = getattr(run, "config", {}) or {}
    summary = getattr(run, "summary_metrics", {}) or {}

    if column.startswith("config."):
        return _nested_get(config, column.removeprefix("config."))
    if column.startswith("summary."):
        return _nested_get(summary, column.removeprefix("summary."))
    if column.startswith("metric."):
        return _nested_get(summary, column.removeprefix("metric."))

    summary_value = _nested_get(summary, column)
    if summary_value is not None:
        return summary_value
    return _nested_get(config, column)


def _format_column_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (str, int, bool)):
        return str(value)
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _report_column_values(
    api: Any,
    project: str,
    runs: dict[str, str],
    columns: list[str],
) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for role, run_url_or_id in runs.items():
        run = api.run(f"{project}/{_run_id(run_url_or_id)}")
        values[role] = {
            column: _format_column_value(_column_value(run, column))
            for column in columns
        }
    return values


def _replace_runs_section(plan_text: str, runs: dict[str, str], report_url: str) -> str:
    lines = ["", *[f"- **{role}:** {url}" for role, url in sorted(runs.items())]]
    lines.append(f"- **Report:** {report_url}")
    body = "\n".join(lines).rstrip() + "\n"

    match = re.search(r"(?m)^##\s+Runs[^\n]*$", plan_text)
    if not match:
        return plan_text.rstrip() + "\n\n## Runs\n" + body

    next_heading = re.search(r"(?m)^##\s+", plan_text[match.end() + 1 :])
    end = len(plan_text) if not next_heading else match.end() + 1 + next_heading.start()
    return plan_text[: match.end()] + body + plan_text[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="Experiment slug, e.g. 20260507-my-ablation")
    parser.add_argument(
        "--plan",
        default=None,
        help="Path to plan.md (default: experiments/<slug>/plan.md)",
    )
    parser.add_argument(
        "--config",
        default=".claude/wandb-driven-dev.local.md",
        help="Path to wandb-driven-dev local config",
    )
    parser.add_argument(
        "--roles",
        default=None,
        help="Comma-separated required roles. Default requires baseline and one variant.",
    )
    parser.add_argument(
        "--x-axis",
        "--x_axis",
        dest="x_axis",
        default=None,
        help="Report x-axis. Defaults to cfg.curves.default_step_key (else _step).",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help=(
            "Comma/space-separated config or summary keys for the focused report table. "
            "Examples: config.lr,config.model.depth,val/loss"
        ),
    )
    parser.add_argument("--publish", action="store_true", help="Save a non-draft report")
    parser.add_argument(
        "--skip-metric-check",
        "--skip_metric_check",
        dest="skip_metric_check",
        action="store_true",
    )
    parser.add_argument(
        "--no-write",
        "--no_write",
        dest="no_write",
        action="store_true",
        help="Do not modify plan.md",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan) if args.plan else Path(f"experiments/{args.slug}/plan.md")
    if not plan_path.exists():
        raise SystemExit(f"FAIL: missing plan file: {plan_path}")

    cfg = read_config(args.config)
    if cfg is None:
        raise SystemExit(f"FAIL: missing config file: {args.config}")
    project = cfg.get("wandb_project")
    if not project:
        raise SystemExit(f"FAIL: config {args.config} has no wandb_project")

    plan_text = plan_path.read_text()
    decision_metrics, health_metrics = _plan_metrics(plan_text, cfg)
    if not decision_metrics:
        raise SystemExit("FAIL: no decision metrics found in plan.md or config")
    report_columns = _parse_list(args.columns) or _plan_report_columns(plan_text)

    roles = [role.strip() for role in args.roles.split(",") if role.strip()] if args.roles else None

    api = get_api()
    runs = _discover_runs(api, project, args.slug)
    runs.update(_runs_from_plan(plan_text))
    _validate_roles(runs, roles)
    if roles:
        runs = {role: runs[role] for role in roles}

    metric_check: dict[str, list[str]] = {}
    required_metrics = decision_metrics + health_metrics
    if not args.skip_metric_check:
        paths_by_role = _run_paths(project, runs)
        metric_check = verify_required_metrics(api, list(paths_by_role.values()), required_metrics)
        missing = {path: missing for path, missing in metric_check.items() if missing}
        if missing:
            raise SystemExit("FAIL: required metrics missing: " + json.dumps(missing, indent=2))

    report_column_values = {}
    if report_columns:
        report_column_values = _report_column_values(api, project, runs, report_columns)

    question = _first_field(plan_text, "Question")
    falsifier = _section(plan_text, "Falsifier").replace("\n", " ").strip() or None

    x_axis = args.x_axis or (cfg.get("curves") or {}).get("default_step_key") or "_step"

    report_url = create_experiment_report(
        project=project,
        slug=args.slug,
        decision_metrics=decision_metrics,
        health_metrics=health_metrics,
        question=question,
        falsifier=falsifier,
        report_columns=report_columns,
        report_column_values=report_column_values,
        runs=runs,
        x_axis=x_axis,
        draft=not args.publish,
    )

    if not args.no_write:
        plan_path.write_text(_replace_runs_section(plan_text, runs, report_url))

    print(json.dumps({
        "slug": args.slug,
        "project": project,
        "plan": str(plan_path),
        "report_url": report_url,
        "draft": not args.publish,
        "runs": runs,
        "decision_metrics": decision_metrics,
        "health_metrics": health_metrics,
        "report_columns": report_columns,
        "report_column_values": report_column_values,
        "metric_check": metric_check,
        "plan_updated": not args.no_write,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
