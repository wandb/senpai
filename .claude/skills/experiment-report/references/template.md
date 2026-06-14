# nn_cfd Experiment Comparison Report Template

Source template:
`https://wandb.ai/milieu/nn_cfd/reports/-DO-NOT-DELETE-EDIT-Template-Experiment-Comparison-Report--VmlldzoxNzE4ODUxNg`

Inspected on 2026-06-11 with `wandb_workspaces.reports.v2.Report.from_url`.

## Top-Level Report

- `entity`: `milieu`
- `project`: `nn_cfd`
- `title`: `[YYYY-MM-DD] NAME`. Use the report's first-created date in the
  prefix, then a short human-readable experiment name. Avoid `/` because it
  creates awkward report URL paths.
- `description`: short report purpose.
- `width`: source template currently round-trips as `readable`, but generated
  experiment reports must override this to `fluid` (full-width) and verify the
  saved model spec has `width="fluid"` with
  `Report.from_url(url, as_model=True).spec.width`.
- Always publish: `report.save(draft=False)`.

## Block Order

Use these blocks in this order:

1. `wr.MarkdownBlock`: `# TL;DR Results`
2. `wr.MarkdownBlock`: `# Hypothesis`
3. empty `wr.P`
4. `wr.MarkdownBlock`: `# Experiment Setup`
5. empty `wr.P`
6. `wr.MarkdownBlock`: `# Results` with the Relative L2 Error table
7. empty `wr.P`
8. empty `wr.P`
9. one `wr.PanelGrid` containing the runset and all panels below

Use `wr.MarkdownBlock` for every markdown table. `wr.P` renders markdown
literally.

## Runset

Use exact run IDs:

```python
runset = wr.Runset(
    entity=entity,
    project=project,
    name="Run set",
    filters=[expr.Metric("name").isin(run_ids)],
    order=[wr.OrderBy(name=wr.Metric("CreatedTimestamp"), ascending=True)],
    run_settings={baseline_run_id: wr.RunSettings(disabled=True)},
    hidden_columns=["run:name"],
)
```

The inspected template stored a UI filter equivalent to:

```text
Metric("ID") in ["a9300aedc8926440", "854f115d43294ece", "qz4xm8cq", "ch930ho8"]
Metric("CreatedTimestamp") <= "2026-06-11T11:39:52.109Z"
```

Do not add the timestamp cap for normal reports; exact run IDs are more
accurate. Add it only when intentionally cloning a historical snapshot.

Use `wr.PanelGrid(runsets=[runset], hide_run_sets=False, active_runset=0, panels=panels)`.

## Results Table Metrics

Read these summary keys:

| Column | Summary key |
| --- | --- |
| Overall | `eval/overall_rel_L2_pct` |
| Surf Cps | `eval/surface/cps_rel_L2_pct` |
| Vol Cps | `eval/volume/cps_rel_L2_pct` |
| WSS_x | `eval/surface/wallshear_x_normalised_rel_L2_pct` |
| WSS_y | `eval/surface/wallshear_y_normalised_rel_L2_pct` |
| WSS_z | `eval/surface/wallshear_z_normalised_rel_L2_pct` |
| Vol_V_x | `eval/volume/velocity_normalised_x_rel_L2_pct` |
| Vol_V_y | `eval/volume/velocity_normalised_y_rel_L2_pct` |
| Vol_V_z | `eval/volume/velocity_normalised_z_rel_L2_pct` |

Format all metric cells to two decimals. Compute Surf Cps and Vol Cps deltas
against the baseline as `(variant - baseline) / baseline * 100`.

Every `Name` cell must be a plain markdown link to the W&B run URL:

```markdown
[run-display-name](https://wandb.ai/milieu/nn_cfd/runs/<run_id>)
```

Use `[run-display-name (baseline)](...)` for the baseline row. Escape square
brackets in run names before placing them inside the link label. Avoid code
formatting inside the link label because some W&B markdown table renderers make
that harder to scan or fail to show it as a normal link.

## Experiment Setup

Split model inputs and targets into separate YAML blocks:

````markdown
#### Input features

Taken from the baseline run W&B config.

```yaml
model_class: transolver_cape
coord_features: []
surface_features:
- normal_x
volume_features:
- sdf
global_features: []
```

#### Targets

Taken from the baseline run W&B config.

```yaml
surface_output_fields:
- cps
volume_output_fields:
- cps
```
````

Never say target fields are included in the input/features block.

## Panel Layout

The grid is 24 columns wide. Preserve positions and heights.

### Bar Charts

All bar charts use `orientation="v"` and `max_runs_to_show=100`.

| Metric | range_x | Layout |
| --- | --- | --- |
| `summary:eval/surface/cps_rel_L2_pct` | `(0.0, 10.0)` | `x=0, y=0, w=11, h=9` |
| `summary:eval/volume/cps_rel_L2_pct` | `(0.0, 10.0)` | `x=11, y=0, w=10, h=9` |
| `summary:eval/surface/wallshear_normalised_rel_L2_pct` | `(0.0, 19.0)` | `x=0, y=9, w=11, h=10` |
| `summary:eval/volume/velocity_normalised_rel_L2_pct` | `(0.0, 19.0)` | `x=11, y=9, w=10, h=10` |
| `summary:eval/surface/wallshear_x_normalised_rel_L2_pct` | `(0.0, 22.0)` | `x=0, y=19, w=8, h=11` |
| `summary:eval/surface/wallshear_y_normalised_rel_L2_pct` | `(None, 22.0)` | `x=8, y=19, w=8, h=11` |
| `summary:eval/surface/wallshear_z_normalised_rel_L2_pct` | `(0.0, 22.0)` | `x=16, y=19, w=7, h=11` |
| `summary:eval/volume/velocity_normalised_x_rel_L2_pct` | `(0.0, 16.0)` | `x=0, y=30, w=8, h=11` |
| `summary:eval/volume/velocity_normalised_y_rel_L2_pct` | `(None, 16.0)` | `x=8, y=30, w=8, h=11` |
| `summary:eval/volume/velocity_normalised_z_rel_L2_pct` | `(None, 16.0)` | `x=16, y=30, w=7, h=11` |

### Main Line Plots

All line plots use `x=None`, `groupby="None"`, and
`legend_fields=["run:displayName"]`.

| Metric | range_y | Layout |
| --- | --- | --- |
| `val_rel_L2_pct/surface/cps_rel_L2_pct` | `(0.0, 16.0)` | `x=0, y=41, w=12, h=12` |
| `val_rel_L2_pct/volume/cps_rel_L2_pct` | `(0.0, 16.0)` | `x=12, y=41, w=12, h=12` |
| `val_rel_L2_pct/surface/wallshear_normalised_rel_L2_pct` | `(0.0, 26.0)` | `x=0, y=53, w=12, h=10` |
| `val_rel_L2_pct/volume/velocity_normalised_rel_L2_pct` | `(0.0, 26.0)` | `x=12, y=53, w=12, h=10` |
| `val_rel_L2_pct/surface/wallshear_x_normalised_rel_L2_pct` | `(0.0, 30.0)` | `x=0, y=63, w=8, h=9` |
| `val_rel_L2_pct/volume/velocity_normalised_x_rel_L2_pct` | `(0.0, 40.0)` | `x=8, y=63, w=8, h=9` |
| `val_rel_L2_pct/surface/wallshear_y_normalised_rel_L2_pct` | `(0.0, 30.0)` | `x=16, y=63, w=8, h=9` |
| `val_rel_L2_pct/volume/velocity_normalised_y_rel_L2_pct` | `(0.0, 40.0)` | `x=0, y=72, w=8, h=9` |
| `val_rel_L2_pct/surface/wallshear_z_normalised_rel_L2_pct` | `(0.0, 30.0)` | `x=8, y=72, w=8, h=9` |
| `val_rel_L2_pct/volume/velocity_normalised_z_rel_L2_pct` | `(0.0, 40.0)` | `x=16, y=72, w=8, h=9` |

### Media Sections

Use `wr.MarkdownPanel` headings:

- `# Front Wing Top`: `x=0, y=81, w=11, h=3`
- `# Front Wing Bottom`: `x=0, y=96, w=10, h=3`
- `# Floor`: `x=0, y=120, w=10, h=3`

Use media browsers with `num_columns=4`, `mode="grid"`, `grid_x_axis="run"`,
and `grid_y_axis="step"`:

| Title | Media key | Layout |
| --- | --- | --- |
| `Surf Cps Error - Front Wing Top` | `views/surface_cps_error/front_wing_top` | `x=0, y=99, w=24, h=21` |
| `Surf Cps Error - Front Wing Bottom` | `views/surface_cps_error/front_wing_bottom` | `x=0, y=120, w=24, h=21` |
| `Surf Cps Error - Floor` | `views/surface_cps_error/floor` | `x=0, y=141, w=24, h=34` |

### Run Comparer

Add `wr.RunComparer(diff_only=True, layout=wr.Layout(x=0, y=157, w=24, h=18))`.

### Extra Bottom Line Plots

The template keeps three additional line plots below the main report area:

| Metric | range_y | Layout |
| --- | --- | --- |
| `val_rel_L2_pct/surface/wallshear_x_normalised_rel_L2_pct` | `(None, None)` | `x=0, y=175, w=8, h=9` |
| `val_rel_L2_pct/surface/wallshear_z_normalised_rel_L2_pct` | `(None, None)` | `x=8, y=175, w=8, h=9` |
| `val_rel_L2_pct/volume/velocity_normalised_z_rel_L2_pct` | `(None, None)` | `x=0, y=184, w=8, h=9` |
| `val_rel_L2_pct/volume/velocity_normalised_x_rel_L2_pct` | `(None, None)` | `x=8, y=184, w=8, h=9` |
| `val_rel_L2_pct/volume/velocity_normalised_y_rel_L2_pct` | `(None, None)` | `x=16, y=184, w=8, h=9` |
