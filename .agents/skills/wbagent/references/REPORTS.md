# W&B Reports — recipe, skeleton, filters, and authoring

Full reference for `wandb_workspaces.reports.v2` (the supported replacement
for the deprecated `wandb.apis.reports`). This file is the complete guide
for authoring W&B reports: the recipe that shapes query strategy through
final chat reply, a runnable skeleton, Runset filter patterns, and the
rendering gotchas that crash or degrade the report UI.

`wandb[workspaces]` is preinstalled in the sandbox — no install needed.

Required imports:

```python
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.expr as expr
```

A report's plots come from `Runset` objects inside a `PanelGrid`. The Runset's
filters decide which runs appear.

## Recipe

1. **Decide the question the report answers** — project overview, what's
   working/not working, a specific finding, or what to try next. Don't open
   with a generic dashboard.
2. **Query bounded evidence first, then find the project's comparison axis.**
   Gather counts, time range, run-state breakdown, BOTH best AND worst
   examples (not just the winner), and rates for the surface you're on (runs,
   traces, evals, artifacts). Then look at what the project varies — sweeps,
   groups, tags, distinct config values, scorer/op names — and treat that as
   the primary comparison axis of the report. **When the project has sweeps
   or groups, aggregate by sweep_id / group** (per-group run counts, crash
   counts, medians, best/worst values) — not just per individual run. Use
   the concrete project names (run names, group names, scorer names, op
   names), not generic metric keys. Label anything you couldn't verify as
   inconclusive instead of guessing.
3. **Visual-first, 3-6 panels** carrying the narrative. Pick panels that
   answer the question. For summaries: distribution, timeline, quality view;
   for comparisons (A/B, eval-vs-eval, baseline-vs-new): paired panels,
   diff/disagreement tables, delta views. Add deeper views only when the
   underlying data exists.
4. **Every major claim needs evidence.** Comparative claims — rankings,
   trends, anomalies, groupings, outliers — must point to a panel, table, or
   figure; otherwise phrase them as a hypothesis. Don't rely on `**bold**` or
   `-` bullets inside `wr.P()` for emphasis — they may render literally. Use
   `H1`/`H2`/`H3`.
5. **Build one narrative arc**: setup → evidence → finding → recommendation.
   The report is not done until you have (a) named the finding in plain
   language tied to a specific panel, and (b) proposed one concrete next step
   in the form **"Because [observed pattern], run [next experiment] to test
   [hypothesis]"** — not generic "compare X and Y" advice. Cut anything that
   doesn't advance the arc, including filler and repeated points.
6. **Final chat reply mirrors the report.** After `report.save(draft=True)`,
   the final user-facing message must restate the top 1-2 findings, key
   caveats, and the concrete next experiment — not just the URL. Users skim
   the chat first; if the headline isn't there, the report's value is hidden.

## Report skeleton

Runnable template for the recipe above. Uses multiple `PanelGrid` blocks
across sections (visual-first) rather than one large grid.

```python
import os
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.expr as expr

entity, project = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]
api = wandb.Api(timeout=120)

# Query evidence with helpers/SDK first.

runset = wr.Runset(
    entity=entity, project=project, name="Finished runs",
    filters=[expr.Metric("State") == "finished"],
)
blocks = [
    wr.H1(text="Short, specific research title"),
    wr.H2(text="Thesis / highlights"),
    wr.P(text="One or two plain-language conclusions with numbers."),
    wr.H2(text="Primary metric"),
    wr.PanelGrid(runsets=[runset], panels=[
        wr.LinePlot(title="Primary metric over steps", x="_step", y=["METRIC"]),
        wr.BarPlot(title="Final primary metric by run", metrics=["METRIC"]),
    ]),
    wr.H2(text="Stability / health"),
    wr.PanelGrid(runsets=[runset], panels=[
        wr.LinePlot(title="Loss / grad norm over steps", x="_step", y=["STABILITY_METRIC"]),
    ]),
    wr.H2(text="Notes and caveats"),
    wr.P(text="List inconclusive evidence plainly."),
]

report = wr.Report(
    entity=entity, project=project,
    title="Short, specific research title",
    description="Report generated from W&B project data.",
    width="fluid", blocks=blocks,
)
report.save(draft=True)
print(f"Report saved: {report.url}")
```

Use multiple `PanelGrid` blocks (one per section), not one big grid — distinct
grids per `H2` make the report read as visual-first rather than prose with one
chart dump.

`report.save()` is mutating; only call it with `draft=True` unless the user
has explicitly approved publishing. Always print `report.url` so the user can
open it.

## Runset filters — structured (preferred)

`filters` accepts a list of `expr.FilterExpr` (ANDed together). Always prefer
this over raw filter strings; it validates, never crashes the UI, and survives
field renames.

```python
runset = wr.Runset(
    entity=entity, project=project, name="Top GPT-4 runs",
    filters=[
        expr.Config("model") == "gpt-4",
        expr.Summary("accuracy") >= 0.9,
        expr.Metric("State") == "finished",
    ],
)
```

Constructors:

- `expr.Config("<key>")` — a hyperparameter from `run.config`.
- `expr.Summary("<key>")` — a final metric from `run.summary`.
- `expr.Metric("<system_field>")` — backend run fields. The most useful are
  `Metric("State")`, `Metric("Name")` (display name), `Metric("name")` (run ID
  — note lowercase, this is the backend field).
- `expr.Tags()` — supports `.isin([...])` for tag-based selection.

Operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `.isin([...])`.

## Explicit run selection by ID

The backend field for run ID is `name` (lowercase). Do NOT use `ID`.

```python
runset = wr.Runset(
    entity=entity, project=project, name="Selected runs",
    filters=[expr.Metric("name").isin(["abc123", "def456"])],
)
```

For "top N by metric": query the IDs first with `wandb.Api()`, then pass them in.

```python
api = wandb.Api(timeout=120)
sample = api.runs(f"{entity}/{project}", per_page=200)[:200]
top_ids = [r.id for r in sorted(
    sample, key=lambda r: r.summary.get("accuracy", -1), reverse=True
)[:5]]
runset = wr.Runset(
    entity=entity, project=project,
    filters=[expr.Metric("name").isin(top_ids)],
)
```

## Tag-based selection

Use `expr.Tags()` instead of the legacy `query="tags:..."` syntax.

```python
runset = wr.Runset(
    entity=entity, project=project,
    filters=[expr.Tags().isin(["healthy_baseline", "exploding_gradients"])],
)
```

## Ordering

```python
runset = wr.Runset(
    entity=entity, project=project,
    order=[wr.OrderBy(name="CreatedTimestamp", ascending=False)],
)
```

## RunComparer panel

`wr.RunComparer` renders a side-by-side config/summary diff table across the
runs in the grid. It is the most compact "what differs between these runs"
view for baseline-vs-variant comparisons. Use `diff_only=True` so the panel
shows only the fields that actually differ.

```python
wr.PanelGrid(runsets=[runset], panels=[wr.RunComparer(diff_only=True)])
```

## `query` is a regex on run name only

`query` mirrors the W&B UI search box and is **only** a regex over the run
display name. It does not understand `tags:...`, `config:...`, or any field
syntax. Use structured `filters` for everything else; reach for `query` only
when you actually want a name regex (e.g. `query="healthy_baseline|exploding_gradients"`).

If you do use a string filter, preflight it with `expr.expr_to_filters(...)`
and confirm every leaf filter has a `key` before passing it to a Runset.

## Dot-path warning

Never put dot-paths in filter strings: `"config.lr"`, `"summary.loss"`,
`"tags.foo"` all parse to missing keys and can crash the report UI. Always use
`expr.Config("lr")`, `expr.Summary("loss")`, `expr.Tags()`, etc.

## Width parameter

```python
report = wr.Report(
    entity=entity, project=project,
    title="Project analysis",
    description="Summary of recent runs",
    width="fluid",
    blocks=[...],
)
```

For research-style reports, use `width="fluid"` unless there is a reason to
keep the page narrow. Narrow reports make panel grids look tiny and can bury
the visual evidence below long prose.

## Rendering and visual-report gotchas

- `wr.P(text=...)` does NOT render markdown — `**bold**`, `-` bullets,
  backticks, and pipe-character tables (`| col | col |`) all render literally.
  Use `H1`/`H2`/`H3` for headings, panel grids for plots, and
  `wr.MarkdownBlock(text=...)` for any markdown content **including tables**.
  Stuffing a `md_table()` string into `wr.P()` is a known production failure
  mode; pass it to `wr.MarkdownBlock()` instead.
- Put the thesis/highlights before the first panel, but keep them short. A good
  report usually has 3-5 highlighted findings and then lets figures/tables carry
  the detail.
- If the text mentions clusters, basins, outliers, unstable zones, best/worst
  regions, or sweep winners, the report should include a visual/table that
  supports that claim. Otherwise label the statement as a hypothesis or a
  follow-up analysis.
- Use "inconclusive" notes for missing dataset/objective metadata, missing
  target metrics, non-comparable metric families, absent Weave traces, or absent
  system metrics. Do not convert missing data into a confident claim.
- **Avoid `/` in report titles.** wandb preserves the literal `/` in the
  report URL slug, which produces an extra path segment that
  `wr.Report.from_url()` cannot parse back. Use ` — ` (em-dash), `:`, or `_`
  to separate concepts in titles instead.
