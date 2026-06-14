"""Create the project-standard W&B experiment comparison report.

This script recreates the nn_cfd W&B template report from exact run IDs. It
queries W&B for run summaries/configs, builds the Relative L2 table, and creates
the fixed `wandb_workspaces.reports.v2` panel grid used by the template.

Example:
    uv run python .claude/skills/experiment-report/scripts/create_experiment_report.py \
        --baseline_run_id a9300aedc8926440 \
        --experiment_run_ids 854f115d43294ece qz4xm8cq ch930ho8 \
        --title "cape domain residual variants"
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import simple_parsing as sp
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel


ENTITY = "milieu"
PROJECT = "nn_cfd"
REPORT_WIDTH = "fluid"

SUMMARY_METRICS = {
    "Overall": "eval/overall_rel_L2_pct",
    "Surf Cps": "eval/surface/cps_rel_L2_pct",
    "Vol Cps": "eval/volume/cps_rel_L2_pct",
    "WSS_x": "eval/surface/wallshear_x_normalised_rel_L2_pct",
    "WSS_y": "eval/surface/wallshear_y_normalised_rel_L2_pct",
    "WSS_z": "eval/surface/wallshear_z_normalised_rel_L2_pct",
    "Vol_V_x": "eval/volume/velocity_normalised_x_rel_L2_pct",
    "Vol_V_y": "eval/volume/velocity_normalised_y_rel_L2_pct",
    "Vol_V_z": "eval/volume/velocity_normalised_z_rel_L2_pct",
}


@dataclass
class Args:
    """Arguments for creating an nn_cfd experiment comparison W&B Report."""

    baseline_run_id: str
    experiment_run_ids: list[str] = field(default_factory=list)
    entity: str = ENTITY
    project: str = PROJECT
    title: str | None = None
    report_date: str | None = None
    report_description: str = "Generated nn_cfd experiment comparison report."
    human_result: str | None = None
    hypothesis: str | None = None
    core_changes: str | None = None
    pr_url: str | None = None
    disable_baseline_in_panels: bool = True
    created_at_max: str | None = None


def run_link(entity: str, project: str, run_id: str) -> str:
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def run_name(run) -> str:
    return run.display_name or run.name or run.id


def report_title(args: Args, baseline) -> str:
    name = args.title or f"Experiment Comparison: {run_name(baseline)}"
    if re.match(r"^\[\d{4}-\d{2}-\d{2}\] ", name):
        return name
    report_date = args.report_date or date.today().isoformat()
    date.fromisoformat(report_date)
    return f"[{report_date}] {name}"


def model_config(run) -> dict:
    model = run.config.get("model")
    if isinstance(model, dict):
        return model
    return run.config


def model_value(run, key: str):
    model = model_config(run)
    if key in model:
        return model[key]
    return run.config.get(key)


def model_yaml(run, keys: list[str]) -> str:
    payload = {key: model_value(run, key) for key in keys}
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).strip()


def input_features_yaml(run) -> str:
    return model_yaml(
        run,
        [
            "model_class",
            "coord_features",
            "surface_features",
            "volume_features",
            "global_features",
        ],
    )


def targets_yaml(run) -> str:
    return model_yaml(run, ["surface_output_fields", "volume_output_fields"])


def val_set_label(run) -> str:
    split_path = run.config.get("train_val_split_path")
    if split_path:
        return Path(split_path).parent.name
    processed_dir = run.config.get("processed_dir")
    if processed_dir:
        return Path(processed_dir).name
    return "unknown"


def metric(run, key: str) -> float:
    value = run.summary_metrics.get(key)
    if value is None:
        raise ValueError(f"Run {run.id} is missing required summary metric {key}")
    return float(value)


def fmt_value(value: float) -> str:
    return f"{value:.2f}"


def fmt_delta(value: float, baseline: float) -> str:
    return f"{((value - baseline) / baseline) * 100:+.2f}%"


def md_cell(value: str) -> str:
    return value.replace("|", "\\|")


def md_link_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def run_name_link(entity: str, project: str, run, label: str) -> str:
    return f"[{md_link_label(label)}]({run_link(entity, project, run.id)})"


def results_table(entity: str, project: str, baseline, experiments: list) -> str:
    runs = [baseline, *experiments]
    baseline_surf = metric(baseline, SUMMARY_METRICS["Surf Cps"])
    baseline_vol = metric(baseline, SUMMARY_METRICS["Vol Cps"])
    headers = [
        "Name",
        "Overall",
        "Surf Cps",
        "Surf Cps Delta vs Baseline",
        "Vol Cps",
        "Vol Cps Delta vs Baseline",
        "WSS_x",
        "WSS_y",
        "WSS_z",
        "Vol_V_x",
        "Vol_V_y",
        "Vol_V_z",
        "ID",
        "Val Set",
    ]
    aligns = [
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---",
        "---",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for index, run in enumerate(runs):
        is_baseline = index == 0
        name_label = run_name(run)
        if is_baseline:
            name_label += " (baseline)"
        linked_name = run_name_link(entity, project, run, name_label)
        surf = metric(run, SUMMARY_METRICS["Surf Cps"])
        vol = metric(run, SUMMARY_METRICS["Vol Cps"])
        row = [
            linked_name,
            fmt_value(metric(run, SUMMARY_METRICS["Overall"])),
            fmt_value(surf),
            "&mdash;" if is_baseline else fmt_delta(surf, baseline_surf),
            fmt_value(vol),
            "&mdash;" if is_baseline else fmt_delta(vol, baseline_vol),
            fmt_value(metric(run, SUMMARY_METRICS["WSS_x"])),
            fmt_value(metric(run, SUMMARY_METRICS["WSS_y"])),
            fmt_value(metric(run, SUMMARY_METRICS["WSS_z"])),
            fmt_value(metric(run, SUMMARY_METRICS["Vol_V_x"])),
            fmt_value(metric(run, SUMMARY_METRICS["Vol_V_y"])),
            fmt_value(metric(run, SUMMARY_METRICS["Vol_V_z"])),
            f"`{run.id}`",
            f"`{val_set_label(run)}`",
        ]
        if is_baseline:
            row = [f"**{cell}**" for cell in row]
        lines.append("| " + " | ".join(md_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def setup_markdown(args: Args, baseline, experiments: list) -> str:
    experiment_lines = "\n".join(
        f"- [`{run_name(run)}`]({run_link(args.entity, args.project, run.id)}) (`{run.id}`)"
        for run in experiments
    )
    core_changes = args.core_changes or (
        "Baseline and experiment runs selected by exact W&B run ID.\n\n"
        f"- Baseline: [`{run_name(baseline)}`]({run_link(args.entity, args.project, baseline.id)}) (`{baseline.id}`)\n"
        f"{experiment_lines}"
    )
    pr_url = args.pr_url or "< Link to the PR where the experiment is being documented >"
    return (
        "# Experiment Setup\n\n"
        "#### Core changes\n\n"
        f"{core_changes}\n\n"
        "##### PR Number\n\n"
        f"{pr_url}\n\n"
        "#### Input features\n\n"
        "Taken from the baseline run W&B config.\n\n"
        "```yaml\n"
        f"{input_features_yaml(baseline)}\n"
        "```\n\n"
        "#### Targets\n\n"
        "Taken from the baseline run W&B config.\n\n"
        "```yaml\n"
        f"{targets_yaml(baseline)}\n"
        "```"
    )


def line_plot(wr, metric_name: str, x: int, y: int, w: int, h: int, range_y=(None, None)):
    return wr.LinePlot(
        x=None,
        y=[wr.Metric(metric_name)],
        range_y=range_y,
        groupby="None",
        legend_fields=["run:displayName"],
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def bar_plot(wr, metric_name: str, x: int, y: int, w: int, h: int, range_x):
    return wr.BarPlot(
        metrics=[wr.Metric(metric_name)],
        orientation="v",
        range_x=range_x,
        max_runs_to_show=100,
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def media_browser(wr, title: str, media_key: str, x: int, y: int, w: int, h: int):
    return wr.MediaBrowser(
        title=title,
        num_columns=4,
        media_keys=[media_key],
        mode="grid",
        grid_x_axis="run",
        grid_y_axis="step",
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def template_panels(wr) -> list:
    return [
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_z_rel_L2_pct", 0, 184, 8, 9),
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_y_rel_L2_pct", 16, 184, 8, 9),
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_x_rel_L2_pct", 8, 184, 8, 9),
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_rel_L2_pct", 12, 53, 12, 10, (0.0, 26.0)),
        line_plot(wr, "val_rel_L2_pct/volume/cps_rel_L2_pct", 12, 41, 12, 12, (0.0, 16.0)),
        line_plot(wr, "val_rel_L2_pct/surface/wallshear_z_normalised_rel_L2_pct", 8, 175, 8, 9),
        line_plot(wr, "val_rel_L2_pct/surface/wallshear_x_normalised_rel_L2_pct", 0, 63, 8, 9, (0.0, 30.0)),
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_x_rel_L2_pct", 8, 63, 8, 9, (0.0, 40.0)),
        line_plot(wr, "val_rel_L2_pct/surface/wallshear_y_normalised_rel_L2_pct", 16, 63, 8, 9, (0.0, 30.0)),
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_y_rel_L2_pct", 0, 72, 8, 9, (0.0, 40.0)),
        line_plot(wr, "val_rel_L2_pct/surface/wallshear_z_normalised_rel_L2_pct", 8, 72, 8, 9, (0.0, 30.0)),
        line_plot(wr, "val_rel_L2_pct/volume/velocity_normalised_z_rel_L2_pct", 16, 72, 8, 9, (0.0, 40.0)),
        line_plot(wr, "val_rel_L2_pct/surface/wallshear_x_normalised_rel_L2_pct", 0, 175, 8, 9),
        line_plot(wr, "val_rel_L2_pct/surface/wallshear_normalised_rel_L2_pct", 0, 53, 12, 10, (0.0, 26.0)),
        line_plot(wr, "val_rel_L2_pct/surface/cps_rel_L2_pct", 0, 41, 12, 12, (0.0, 16.0)),
        wr.MarkdownPanel(markdown="# Front Wing Bottom", layout=wr.Layout(x=0, y=96, w=10, h=3)),
        wr.MarkdownPanel(markdown="# Floor", layout=wr.Layout(x=0, y=120, w=10, h=3)),
        wr.MarkdownPanel(markdown="# Front Wing Top", layout=wr.Layout(x=0, y=81, w=11, h=3)),
        wr.RunComparer(diff_only=True, layout=wr.Layout(x=0, y=157, w=24, h=18)),
        bar_plot(wr, "summary:eval/surface/cps_rel_L2_pct", 0, 0, 11, 9, (0.0, 10.0)),
        bar_plot(wr, "summary:eval/volume/cps_rel_L2_pct", 11, 0, 10, 9, (0.0, 10.0)),
        bar_plot(wr, "summary:eval/surface/wallshear_normalised_rel_L2_pct", 0, 9, 11, 10, (0.0, 19.0)),
        bar_plot(wr, "summary:eval/surface/wallshear_x_normalised_rel_L2_pct", 0, 19, 8, 11, (0.0, 22.0)),
        bar_plot(wr, "summary:eval/volume/velocity_normalised_x_rel_L2_pct", 0, 30, 8, 11, (0.0, 16.0)),
        bar_plot(wr, "summary:eval/surface/wallshear_y_normalised_rel_L2_pct", 8, 19, 8, 11, (None, 22.0)),
        bar_plot(wr, "summary:eval/volume/velocity_normalised_y_rel_L2_pct", 8, 30, 8, 11, (None, 16.0)),
        bar_plot(wr, "summary:eval/surface/wallshear_z_normalised_rel_L2_pct", 16, 19, 7, 11, (0.0, 22.0)),
        bar_plot(wr, "summary:eval/volume/velocity_normalised_z_rel_L2_pct", 16, 30, 7, 11, (None, 16.0)),
        bar_plot(wr, "summary:eval/volume/velocity_normalised_rel_L2_pct", 11, 9, 10, 10, (0.0, 19.0)),
        media_browser(wr, "Surf Cps Error - Front Wing Top", "views/surface_cps_error/front_wing_top", 0, 99, 24, 21),
        media_browser(
            wr,
            "Surf Cps Error - Front Wing Bottom",
            "views/surface_cps_error/front_wing_bottom",
            0,
            120,
            24,
            21,
        ),
        media_browser(wr, "Surf Cps Error - Floor", "views/surface_cps_error/floor", 0, 141, 24, 34),
    ]


def build_report(args: Args, baseline, experiments: list):
    import wandb_workspaces.expr as expr
    import wandb_workspaces.reports.v2 as wr

    run_ids = [args.baseline_run_id, *args.experiment_run_ids]
    filters = [expr.Metric("name").isin(run_ids)]
    if args.created_at_max:
        filters.append(expr.Metric("CreatedTimestamp") <= args.created_at_max)
    runset = wr.Runset(
        entity=args.entity,
        project=args.project,
        name="Run set",
        filters=filters,
        order=[wr.OrderBy(name=wr.Metric("CreatedTimestamp"), ascending=True)],
        run_settings={
            args.baseline_run_id: wr.RunSettings(disabled=args.disable_baseline_in_panels)
        },
        hidden_columns=["run:name"],
    )
    title = report_title(args, baseline)
    blocks = [
        wr.MarkdownBlock(
            text="# TL;DR Results\n\n\n"
            + (args.human_result or "< Final result to be written by a human, no AI >")
        ),
        wr.MarkdownBlock(
            text="# Hypothesis\n\n\n"
            + (args.hypothesis or "< The hypothesis being tested... >")
        ),
        wr.P(text=""),
        wr.MarkdownBlock(text=setup_markdown(args, baseline, experiments)),
        wr.P(text=""),
        wr.MarkdownBlock(
            text="# Results\n\n"
            "### Relative L2 Error\n\n"
            "Results table comparing experiment(s) to the baseline run. Values are percentages, lower is better.\n\n"
            + results_table(args.entity, args.project, baseline, experiments)
        ),
        wr.P(text=""),
        wr.P(text=""),
        wr.PanelGrid(
            runsets=[runset],
            hide_run_sets=False,
            active_runset=0,
            panels=template_panels(wr),
        ),
    ]
    return wr.Report(
        entity=args.entity,
        project=args.project,
        title=title,
        description=args.report_description,
        width=REPORT_WIDTH,
        blocks=blocks,
    )


def enforce_full_width(report) -> None:
    report.width = REPORT_WIDTH


def report_spec_json(report) -> str:
    enforce_full_width(report)
    model = report._to_model()
    payload = json.loads(model.spec.model_dump_json(by_alias=True, exclude_none=True))
    payload["width"] = REPORT_WIDTH
    return json.dumps(payload)


def saved_report_width(report) -> str:
    return report.__class__.from_url(report.url, as_model=True).spec.width


def validate_full_width(report) -> None:
    width = saved_report_width(report)
    if width != REPORT_WIDTH:
        raise RuntimeError(
            f"Expected saved report width={REPORT_WIDTH!r}, got {width!r}: {report.url}"
        )


def save_report(report) -> None:
    from wandb.sdk.lib.service.service_connection import WandbApiFailedError

    enforce_full_width(report)
    try:
        report.save(draft=False)
        validate_full_width(report)
        return
    except WandbApiFailedError as exc:
        if "relogin required" not in str(exc):
            raise
        import wandb_workspaces.reports.v2.interface as report_interface

        model = report._to_model()
        model.spec.width = REPORT_WIDTH
        response = report_interface.execute_graphql(
            report_interface._get_api(),
            report_interface.gql.upsert_view,
            {
                "id": None if not model.id else model.id,
                "name": report_interface.internal._generate_name()
                if not model.name
                else model.name,
                "entityName": model.project.entity_name,
                "projectName": model.project.name,
                "description": model.description,
                "displayName": model.display_name,
                "type": "runs",
                "spec": report_spec_json(report),
            },
        )
        new_model = report_interface.internal.ReportViewspec.model_validate(
            response["upsertView"]["view"]
        )
        report.id = new_model.id
        validate_full_width(report)


def main() -> None:
    args = sp.parse(Args)
    if not args.experiment_run_ids:
        raise ValueError("Pass at least one --experiment_run_ids value")

    load_dotenv(Path(".env"))

    import wandb

    console = Console()
    console.rule("W&B experiment report")
    api = wandb.Api(timeout=120)
    baseline = api.run(f"{args.entity}/{args.project}/{args.baseline_run_id}")
    experiments = [
        api.run(f"{args.entity}/{args.project}/{run_id}")
        for run_id in args.experiment_run_ids
    ]

    report = build_report(args, baseline, experiments)
    save_report(report)
    console.print(
        Panel.fit(
            f"[bold]Report saved[/bold]\n{report.url}\n\n"
            "mode: published\n"
            f"runs: {', '.join([args.baseline_run_id, *args.experiment_run_ids])}",
            title="experiment-report",
        )
    )


if __name__ == "__main__":
    main()
