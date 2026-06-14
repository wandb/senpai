---
name: experiment-report
description: "Create consistent nn_cfd W&B experiment comparison reports from selected run IDs using wandb and wandb_workspaces. Use this whenever the user asks to report experiment results, create a W&B Report, compare CFD training/eval runs, document baseline-vs-variant outcomes, reproduce the project experiment report template, or publish a leaderboard-style report for milieu/nn_cfd."
---

# Experiment Report

Use this skill to turn selected nn_cfd W&B runs into the project-standard
experiment comparison report. The canonical template is the W&B report
`[DO NOT DELETE/EDIT] Template Experiment Comparison Report`.

## Workflow

1. Identify the baseline run first. If the user did not provide it, ask for the
   baseline run ID or run name before creating the report.
2. Identify the experiment run IDs. Prefer exact run IDs over names, groups, or
   regex searches.
3. Query W&B summaries/configs with `wbagent` helpers or the W&B SDK. Use
   bounded metric keys; do not scan broad histories unless the report needs a
   curve-specific claim.
4. Read `references/template.md` before making a report by hand. It records the
   report width, runset settings, panel positions, plot axis limits, media
   keys, and expected block order from the template.
5. Prefer `scripts/create_experiment_report.py` for consistency. It builds the
   report from run IDs, validates the summary metrics needed by the table, and
   recreates the template panel grid.
6. Always publish the report. Do not leave experiment reports as W&B drafts.
7. Name every report `[YYYY-MM-DD] NAME`, where the date is the report's first
   creation date. Preserve that date on later edits because the W&B Reports UI
   shows the last modified date, not the original creation date.

## Report Command

Run from the repo root:

```bash
uv run python .claude/skills/experiment-report/scripts/create_experiment_report.py \
  --baseline_run_id a9300aedc8926440 \
  --experiment_run_ids 854f115d43294ece qz4xm8cq ch930ho8 \
  --title "cape domain residual variants"
```

Useful optional arguments:

- `--human_result`: fills the TL;DR block. If absent, keep the template's human
  placeholder.
- `--hypothesis`: fills the hypothesis block. If absent, keep the template
  placeholder.
- `--core_changes`: describes the tested changes in Experiment Setup.
- `--pr_url`: adds the PR link.
- `--report_date`: pins the `[YYYY-MM-DD]` prefix when recreating or updating a
  report whose first-created date is not today.
- `--created_at_max`: adds the template-style CreatedTimestamp runset cap when
  cloning a historical report snapshot. Prefer exact run IDs without this cap
  for normal reports.

## Report Standards

- Use `wandb_workspaces.reports.v2`, not deprecated `wandb.apis.reports`.
- Use `width="fluid"` so reports render full-width. The report builder must
  force this width immediately before save/upsert and verify the saved report
  model spec still has `width="fluid"`; do not silently accept W&B's default
  `readable` width. Check `Report.from_url(url, as_model=True).spec.width`
  because the hydrated `Report` object can fall back to its local default.
- Use `report.save(draft=False)`; this project does not keep hanging draft
  experiment reports.
- Use report titles of the form `[YYYY-MM-DD] NAME`. The script will add
  today's date if `--title` omits the bracketed prefix.
- Filter the runset by exact run ID with `expr.Metric("name").isin(run_ids)`.
  The W&B UI may display this as `Metric("ID")`, but the structured expression
  should use lowercase `name`.
- Sort by `CreatedTimestamp` ascending.
- Hide the `run:name` column.
- Set `RunSettings(disabled=True)` for the baseline by default, matching the
  template's plot behavior, while still keeping the baseline in the markdown
  results table.
- Use `wr.MarkdownBlock` for tables and markdown headings. Do not put markdown
  tables inside `wr.P`.
- In Experiment Setup, render `Input features` and `Targets` as separate YAML
  blocks. Never bury target fields inside an input/features block or say that
  targets are "included above."
- Keep the Results table to two decimal places and include run links, run IDs,
  and validation-set labels.
- In the Results table, every run name must be a plain markdown link to its W&B
  run URL. Escape square brackets in run names before using them as link
  labels, and do not rely on unlabeled IDs or placeholder text.
- State missing evidence plainly. Do not invent PR links, hypotheses, or human
  final verdicts.

## Resources

- `scripts/create_experiment_report.py`: deterministic report builder for the
  template.
- `references/template.md`: exact template anatomy and panel layout.
