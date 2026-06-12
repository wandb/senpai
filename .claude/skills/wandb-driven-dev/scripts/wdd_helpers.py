"""wandb-driven-dev helpers: project config IO and experiment Reports.

Generic wandb querying lives in `../../wbagent/scripts/wandb_helpers.py`.
This module only contains things tied to the wandb-driven-dev experiment workflow:

- Project config schema + read/write at `.claude/wandb-driven-dev.local.md`.
  The file is YAML frontmatter (structured fields) followed by a markdown body
  (free-form project notes the agents read verbatim).
- Project-local experiment launch configuration. `launcher.command` is the
  repo-specific command used to start/submit training; `training.script` is the
  underlying training entrypoint used for flag validation. This is distinct from
  W&B Launch.
- Experiment gates for config-based lookup, required metric checks, run
  resolution, runtime estimates, and training flag validation.
- Report builders for experiment dashboards and run comparisons.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(".claude/wandb-driven-dev.local.md")
_FRONTMATTER_DELIM = "---"


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------

def default_config() -> dict[str, Any]:
    """Schema for the structured frontmatter of `.claude/wandb-driven-dev.local.md`."""
    return {
        "wandb_project": "",
        "launcher": {
            "command": "",
            "reproduction": "working_tree",  # working_tree | clone | shared_fs | image
        },
        "training": {
            "script": "",  # underlying training entrypoint, even if launcher is a wrapper
            "config_dir": "",
        },
        "gpus": {
            "smoke": 1,
            "full": 1,
        },
        "metrics": {
            "decision": [],
            "health": [],
        },
        "curves": {
            # W&B always has `_step`; projects can override with custom step
            # metrics such as `train/global_step` or `stage_4e/epoch`.
            "default_step_key": "_step",
            "metric_step_keys": {},
            "candidate_step_keys": ["_step"],
        },
        "wandb_metadata": {},
    }


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split YAML frontmatter from the markdown body.

    Returns (frontmatter_text, body_text). Raises ValueError if no frontmatter.
    """
    if not text.startswith(_FRONTMATTER_DELIM):
        raise ValueError(
            f"Config file must start with YAML frontmatter delimited by '{_FRONTMATTER_DELIM}'"
        )
    lines = text.splitlines(keepends=True)
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("Config file frontmatter is not closed (no trailing '---')")
    fm = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1:])
    return fm, body


def read_config(path: Path | str = CONFIG_PATH) -> dict[str, Any] | None:
    """Read the project config. Returns None if the file doesn't exist.

    The returned dict contains the parsed frontmatter plus a `_notes` key
    holding the markdown body (stripped of leading/trailing whitespace).
    """
    import yaml

    p = Path(path)
    if not p.exists():
        return None
    fm, body = _split_frontmatter(p.read_text())
    cfg = yaml.safe_load(fm) or {}
    cfg["_notes"] = body.strip()
    return cfg


def write_config(
    cfg: dict[str, Any],
    notes: str = "",
    path: Path | str = CONFIG_PATH,
) -> None:
    """Write the project config preserving key order from `default_config`.

    `notes` becomes the markdown body (free-form context for the agents).
    Any `_notes` key in `cfg` is preferred over the `notes` argument.
    """
    import yaml

    cfg = dict(cfg)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    body = cfg.pop("_notes", notes).strip()
    fm = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).rstrip()
    contents = f"{_FRONTMATTER_DELIM}\n{fm}\n{_FRONTMATTER_DELIM}\n"
    if body:
        contents += "\n" + body + "\n"
    p.write_text(contents)


def _unique_list(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def merge_curve_config(cfg: dict[str, Any], curves: dict[str, Any]) -> dict[str, Any]:
    """Merge discovered curve settings into an existing project config."""
    default_curves = default_config()["curves"]
    existing = cfg.get("curves") or {}
    merged = {
        "default_step_key": existing.get("default_step_key") or default_curves["default_step_key"],
        "metric_step_keys": dict(existing.get("metric_step_keys") or {}),
        "candidate_step_keys": list(existing.get("candidate_step_keys") or default_curves["candidate_step_keys"]),
    }

    if curves.get("default_step_key"):
        merged["default_step_key"] = curves["default_step_key"]
    if curves.get("metric_step_keys"):
        merged["metric_step_keys"].update(curves["metric_step_keys"])
    if curves.get("candidate_step_keys"):
        merged["candidate_step_keys"] = _unique_list(
            [*merged["candidate_step_keys"], *curves["candidate_step_keys"]]
        )

    cfg["curves"] = merged
    return cfg


def merge_wandb_metadata(cfg: dict[str, Any], wandb_metadata: dict[str, Any]) -> dict[str, Any]:
    """Merge W&B discovery metadata sections into an existing project config."""
    existing = dict(cfg.get("wandb_metadata") or {})
    for key, value in (wandb_metadata or {}).items():
        existing[key] = value
    cfg["wandb_metadata"] = existing
    return cfg


def update_preflight_config(
    config_patch: dict[str, Any],
    project: str | None = None,
    path: Path | str = CONFIG_PATH,
    create_if_missing: bool = False,
) -> dict[str, Any]:
    """Persist preflight settings/metadata into `.claude/wandb-driven-dev.local.md`.

    By default the config must already exist; setup discovery should enrich the
    project file created by the setup interview, not silently create an
    incomplete default. If a config exists for a different project, raise
    instead of silently writing project-specific discoveries into the wrong
    file.
    """
    cfg = read_config(path)
    if cfg is None:
        if not create_if_missing:
            raise FileNotFoundError(
                f"{path} is missing; run setup and write the project config before preflight"
            )
        cfg = default_config()
        if project:
            cfg["wandb_project"] = project

    existing_project = cfg.get("wandb_project")
    if project and existing_project and existing_project != project:
        raise ValueError(
            f"Config project mismatch: {path} has {existing_project!r}, got {project!r}"
        )

    notes = cfg.get("_notes", "")
    if config_patch.get("curves"):
        cfg = merge_curve_config(cfg, config_patch["curves"])
    if config_patch.get("wandb_metadata"):
        cfg = merge_wandb_metadata(cfg, config_patch["wandb_metadata"])
    write_config(cfg, notes=notes, path=path)
    loaded = read_config(path)
    if loaded is None:
        raise RuntimeError(f"Failed to write config to {path}")
    return loaded


def update_curve_config(
    curves: dict[str, Any],
    project: str | None = None,
    path: Path | str = CONFIG_PATH,
    wandb_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist discovered curve settings into `.claude/wandb-driven-dev.local.md`."""
    patch: dict[str, Any] = {"curves": curves}
    if wandb_metadata:
        patch["wandb_metadata"] = wandb_metadata
    return update_preflight_config(patch, project=project, path=path)


def curve_step_key(cfg: dict[str, Any] | None, metric: str) -> str:
    """Return the configured step key for a metric, defaulting to W&B `_step`.

    `curves.metric_step_keys` supports exact metric keys and namespace wildcards
    like `train/*` or `stage_4e/*`.
    """
    curves = (cfg or {}).get("curves") or {}
    metric_step_keys = curves.get("metric_step_keys") or {}
    if metric in metric_step_keys:
        return metric_step_keys[metric]
    if "/" in metric:
        namespace = metric.split("/", 1)[0]
        wildcard = f"{namespace}/*"
        if wildcard in metric_step_keys:
            return metric_step_keys[wildcard]
    return curves.get("default_step_key") or "_step"


def curve_step_keys(cfg: dict[str, Any] | None, metrics: list[str]) -> dict[str, str]:
    """Return configured step keys for a list of curve metrics."""
    return {metric: curve_step_key(cfg, metric) for metric in metrics}


# ---------------------------------------------------------------------------
# Experiment gate helpers
# ---------------------------------------------------------------------------

def _load_fetch_runs() -> Any:
    """Import wbagent's bounded GraphQL fetcher without caller setup."""
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "wbagent" / "scripts"
    scripts_dir_s = str(scripts_dir)
    if scripts_dir_s not in sys.path:
        sys.path.insert(0, scripts_dir_s)
    from wandb_helpers import fetch_runs

    return fetch_runs


def _load_scan_history() -> Any:
    """Import wbagent's `scan_history` without requiring caller sys.path setup."""
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "wbagent" / "scripts"
    scripts_dir_s = str(scripts_dir)
    if scripts_dir_s not in sys.path:
        sys.path.insert(0, scripts_dir_s)
    from wandb_helpers import scan_history

    return scan_history


def find_runs_by_config(
    api: Any,
    project: str,
    config_filters: dict[str, Any],
    metric_keys: list[str] | None = None,
    extra_config_keys: list[str] | None = None,
    state: str | None = "finished",
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Find runs matching exact or operator-based config values.

    Wrapper around WDD's bounded GraphQL fetcher. It prefixes config filters,
    requests only selected summary/config fields, and supports dotted nested
    config output such as `config.model.depth`.
    """
    fetch_runs = _load_fetch_runs()

    filters: dict[str, Any] = {}
    for k, v in config_filters.items():
        filters[k if k.startswith("config.") else f"config.{k}"] = v
    if state is not None:
        filters["state"] = state

    config_keys = list({k.removeprefix("config.") for k in config_filters})
    if extra_config_keys:
        config_keys.extend(k for k in extra_config_keys if k not in config_keys)

    return fetch_runs(
        api,
        project,
        metric_keys=metric_keys or [],
        config_keys=config_keys,
        filters=filters,
        limit=limit,
    )


def verify_required_metrics(
    api: Any,
    run_paths: list[str],
    required: list[str],
) -> dict[str, list[str]]:
    """Confirm every required metric appears on each run summary or history."""
    if not required:
        return {run_path: [] for run_path in run_paths}

    results: dict[str, list[str]] = {}
    scan_history = _load_scan_history()
    for run_path in run_paths:
        run = api.run(run_path)
        missing = [
            metric
            for metric in required
            if metric not in run.summary_metrics or run.summary_metrics[metric] is None
        ]
        if missing:
            rows = scan_history(run, keys=missing, max_rows=10_000)
            seen = {
                key
                for row in rows
                for key in missing
                if row.get(key) is not None
            }
            missing = [metric for metric in missing if metric not in seen]
        results[run_path] = missing
    return results


def find_run_by_name(
    api: Any,
    project: str,
    name: str,
    timeout_s: int = 120,
    poll_interval_s: int = 10,
) -> Any | None:
    """Find a W&B run by exact display name, polling until it appears."""
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        runs = api.runs(
            project,
            filters={"display_name": name},
            per_page=1,
            include_sweeps=False,
        )
        try:
            return next(iter(runs))
        except StopIteration:
            pass
        time.sleep(poll_interval_s)
    return None


def runtime_estimate(
    api: Any,
    project: str,
    name_pattern: str,
    target_steps: int,
    sample: int = 5,
    min_steps: int | None = None,
) -> dict[str, Any] | None:
    """Estimate wall-clock for target steps from prior finished runs."""
    if min_steps is None:
        min_steps = max(target_steps // 10, 1000)

    runs = api.runs(
        project,
        filters={"state": "finished", "display_name": {"$regex": name_pattern}},
        order="-created_at",
        per_page=sample,
        include_sweeps=False,
    )
    samples: list[tuple[float, int]] = []
    excluded = 0
    for run in runs[:sample]:
        runtime = run.summary_metrics.get("_runtime")
        steps = run.summary_metrics.get("train/global_step")
        if not (runtime and steps and steps > 0):
            continue
        if steps < min_steps:
            excluded += 1
            continue
        samples.append((runtime / 3600.0, int(steps)))

    if not samples:
        return None

    hours_per_step = [hours / steps for hours, steps in samples]
    target_low = target_steps * min(hours_per_step)
    target_high = target_steps * max(hours_per_step)

    return {
        "runs_used": len(samples),
        "excluded_short_runs": excluded,
        "min_steps": min_steps,
        "samples": samples,
        "target_hours_low": round(target_low, 2),
        "target_hours_high": round(target_high, 2),
        "notes": "Linear extrapolation from per-step throughput; assumes same GPU count.",
    }


def validate_flags(
    script: str | list[str],
    flags: list[str],
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Check that each flag in `flags` is recognized by `<script> --help`."""
    import re as _re
    import shlex
    import subprocess

    cmd = shlex.split(script) if isinstance(script, str) else list(script)
    cmd.append("--help")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    help_text = (result.stdout or "") + (result.stderr or "")

    present, missing = [], []
    for raw in flags:
        flag = raw.lstrip("-")
        if _re.search(rf"--{_re.escape(flag)}\b", help_text):
            present.append(raw)
        else:
            missing.append(raw)
    return {"present": present, "missing": missing, "help_text": help_text}


# ---------------------------------------------------------------------------
# Experiment reports
# ---------------------------------------------------------------------------

def _run_id(value: str) -> str:
    """Extract a W&B run ID from a run URL, entity/project/run_id path, or raw ID."""
    from urllib.parse import urlparse

    text = value.strip().rstrip("/")
    parsed = urlparse(text)
    path = parsed.path if parsed.scheme and parsed.netloc else text
    return path.rstrip("/").rsplit("/", 1)[-1]


def _run_comparer(wr: Any) -> Any:
    """Construct a RunComparer across SDK versions."""
    if not hasattr(wr, "RunComparer"):
        raise RuntimeError(
            "The installed W&B Reports SDK does not expose RunComparer. "
            "Upgrade wandb with the workspaces extra."
        )
    try:
        return wr.RunComparer(diff_only=True)
    except TypeError:
        return wr.RunComparer(diff_only="on")


def _md_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _safe_report_title(title: str) -> str:
    """Strip `/` from report titles.

    wandb preserves the literal `/` in the report URL slug, producing an extra
    path segment that `wr.Report.from_url()` cannot parse back.
    """
    return title.replace("/", " - ")


def create_comparison_report(
    project: str,
    title: str,
    runs: dict[str, str],
    decision_metrics: list[str],
    health_metrics: list[str] | None = None,
    description: str | None = None,
    header_md: str | None = None,
    focused_table_md: str | None = None,
    x_axis: str = "train/global_step",
    draft: bool = True,
) -> str:
    """Create a pinned W&B Report comparing experiment runs.

    This lives in wandb-driven-dev because comparison dashboards are part of
    the experiment workflow. wbagent remains responsible for querying run data.

    Args:
        project: "entity/project".
        title: Report title.
        runs: Mapping of label -> wandb run URL, entity/project/run_id path, or run ID.
        decision_metrics: Metric keys plotted first.
        health_metrics: Additional health metric keys.
        description: Optional report description.
        header_md: Optional markdown block before the panel grid.
        focused_table_md: Optional markdown table with selected config/summary
            columns for the runs.
        x_axis: x-axis key for every line plot.
        draft: Save as draft.

    Returns:
        URL of the saved report.
    """
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.expr as expr

    entity, proj = project.split("/", 1)
    run_ids = [_run_id(url_or_id) for url_or_id in runs.values()]

    if header_md is None:
        lines = ["**Decision metrics:** " + ", ".join(f"`{m}`" for m in decision_metrics)]
        if health_metrics:
            lines.append("**Health metrics:** " + ", ".join(f"`{m}`" for m in health_metrics))
        lines.append("**Runs:**")
        for label, url_or_id in runs.items():
            lines.append(f"- `{label}` -> {url_or_id}")
        header_md = "\n\n".join(lines)

    runset = wr.Runset(
        entity=entity,
        project=proj,
        name="Selected runs",
        filters=[expr.Metric("name").isin(run_ids)],
    )

    # Visual-first: one PanelGrid per section (H2) rather than one big grid.
    blocks: list[Any] = [
        wr.H1(text=title),
        wr.MarkdownBlock(text=header_md),
    ]
    if focused_table_md:
        blocks.append(wr.MarkdownBlock(text=focused_table_md))

    decision_panels: list[Any] = [
        wr.LinePlot(title=m, x=x_axis, y=[m]) for m in decision_metrics
    ]
    if decision_metrics:
        decision_panels.append(
            wr.BarPlot(title="Final decision metrics by run", metrics=decision_metrics)
        )
    blocks.append(wr.H2(text="Decision metrics"))
    blocks.append(wr.PanelGrid(runsets=[runset], panels=decision_panels))

    if health_metrics:
        blocks.append(wr.H2(text="Stability / health"))
        blocks.append(
            wr.PanelGrid(
                runsets=[runset],
                panels=[wr.LinePlot(title=m, x=x_axis, y=[m]) for m in health_metrics],
            )
        )

    blocks.append(wr.H2(text="Run comparison"))
    blocks.append(wr.PanelGrid(runsets=[runset], panels=[_run_comparer(wr)]))

    report = wr.Report(
        entity=entity,
        project=proj,
        title=_safe_report_title(title),
        description=description or title,
        width="fluid",
        blocks=blocks,
    )
    report.save(draft=draft)
    return report.url


def create_experiment_report(
    project: str,
    slug: str,
    decision_metrics: list[str],
    runs: dict[str, str],
    health_metrics: list[str] | None = None,
    question: str | None = None,
    falsifier: str | None = None,
    report_columns: list[str] | None = None,
    report_column_values: dict[str, dict[str, Any]] | None = None,
    date: str | None = None,
    x_axis: str = "train/global_step",
    draft: bool = True,
) -> str:
    """Create a wandb Report for a wandb-driven-dev experiment.

    Builds the standard markdown header (slug, date, question, falsifier,
    role-labelled run links) and delegates panel + Runset construction to
    `create_comparison_report`.

    Args:
        project: "entity/project".
        slug: Experiment slug (matches the `exp/<slug>` tag).
        decision_metrics: Decision metric keys from plan.md `## Metrics`.
        runs: Mapping of role -> wandb run URL/id.
        health_metrics: Health metric keys from plan.md `## Metrics`.
        question: One-line experiment question copied from plan.md.
        falsifier: One-line falsifier copied from plan.md.
        report_columns: Selected config/summary columns to show in a focused
            comparison table.
        report_column_values: Mapping of role -> selected column values.
        date: Date string for header (default: today UTC).
        x_axis: x-axis key for every panel.
        draft: Save as draft (default True).

    Returns:
        URL of the saved report.
    """
    date = date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

    lines = [
        f"**Slug:** `{slug}` &nbsp;·&nbsp; **Date:** {date}",
        f"**Filter:** `tag:exp/{slug}` (smokes excluded)",
    ]
    if question:
        lines.append(f"**Question:** {question}")
    if falsifier:
        lines.append(f"**Falsifier:** {falsifier}")
    lines.append("**Decision metrics:** " + ", ".join(f"`{m}`" for m in decision_metrics))
    if health_metrics:
        lines.append("**Health metrics:** " + ", ".join(f"`{m}`" for m in health_metrics))
    lines.append("**Runs:**")
    for role, url in runs.items():
        lines.append(f"- `{role}` -> {url}")

    focused_table_md = None
    if report_columns:
        table_lines = [
            "**Focused columns:** "
            + ", ".join(f"`{column}`" for column in report_columns),
            "",
            "| Run | " + " | ".join(_md_table_cell(column) for column in report_columns) + " |",
            "|---|" + "---|" * len(report_columns),
        ]
        values = report_column_values or {}
        for role in runs:
            row_values = values.get(role, {})
            cells = [role, *[_md_table_cell(row_values.get(column, "-")) for column in report_columns]]
            table_lines.append("| " + " | ".join(cells) + " |")
        focused_table_md = "\n".join(table_lines)

    return create_comparison_report(
        project=project,
        title=f"Experiment {slug}",
        runs=runs,
        decision_metrics=decision_metrics,
        health_metrics=health_metrics,
        description=question or f"Dashboard for experiment `{slug}`.",
        header_md="\n\n".join(lines),
        focused_table_md=focused_table_md,
        x_axis=x_axis,
        draft=draft,
    )
