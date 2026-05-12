# %% [markdown]
# # 03 - Single-Agent Student Loop
#
# Learning objective:
# Implement a minimal student-agent loop over one assignment and produce a
# structured result comment. The loop mirrors SENPAI's student role boundaries
# without mutating GitHub or running a GPU job.

# %%
from pathlib import Path
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.llm import anthropic_message
from common.notebook import checkpoint, h1, h2, repo_note, show_json
from common.senpai_protocol import Assignment, merge_decision, senpai_result_line, terminal_result_errors
from common.wandb_utils import recent_runs


# %%
h1("03 - Single-Agent Student Loop")
config = load_config(require=True)

# %%
h2("Assignment object")
assignment = Assignment(
    number=2417,
    title="DrivAerML EMA warmup for surface pressure",
    student="ws-fern",
    advisor_branch=config.advisor_branch,
    hypothesis=(
        "EMA warmup with best-checkpoint restore will smooth validation behavior "
        "and reduce held-out surface-pressure relative-L2."
    ),
    metric_name="test_primary/surface_pressure_rel_l2_pct",
    baseline_value=6.24,
)
show_json({
    "number": assignment.number,
    "title": assignment.title,
    "labels": assignment.labels,
    "hypothesis": assignment.hypothesis,
    "metric": assignment.metric_name,
    "baseline": assignment.baseline_value,
})

# %%
h2("Read live W&B context")
runs = recent_runs(config, limit=3)
show_json(runs)

# %%
h2("Simulate final metrics for the teaching loop")
observed = {
    "wandb_run_id": runs[0]["id"] if runs else "workshop-run-001",
    "primary_metric": assignment.metric_name,
    "primary_value": 6.31,
    "validation_metric": "full_val_primary/surface_pressure_rel_l2_pct",
    "validation_value": 4.42,
}
decision = merge_decision(observed["primary_value"], assignment.baseline_value)
show_json({"observed": observed, "baseline": assignment.baseline_value, "decision": decision})

# %%
h2("Student writes structured result comment")
result = {
    "terminal": True,
    "status": "complete",
    "pending_arms": False,
    "wandb_run_ids": [observed["wandb_run_id"]],
    "primary_metric": {"name": observed["primary_metric"], "value": observed["primary_value"]},
    "test_metric": {"name": observed["primary_metric"], "value": observed["primary_value"]},
}
errors = terminal_result_errors(result)
result_line = senpai_result_line(result)
analysis_prompt = f"""
You are the student agent. Write a concise PR result comment after this experiment.

Assignment: {assignment.hypothesis}
Baseline {assignment.metric_name}: {assignment.baseline_value}
Observed validation metric: {observed['validation_metric']} = {observed['validation_value']}
Observed test metric: {observed['primary_metric']} = {observed['primary_value']}
W&B run: {observed['wandb_run_id']}
Result marker: {result_line}

Be honest that test did not beat baseline.
"""
comment_body = anthropic_message(config, analysis_prompt, max_tokens=700)
print(result_line)
print()
print(comment_body)

# %%
h2("Write artifact")
path = write_artifact(
    "03_student_result.md",
    f"# Student Result Comment\n\n{result_line}\n\n{comment_body}\n",
)
checkpoint(f"Wrote {path}")
show_json({"result_errors": errors})

# %%
h2("What this teaches")
print(
    "A student agent is useful because it has a narrow role: implement, run, "
    "log, report. It should not invent hypotheses or decide merges."
)
repo_note(
    "system_instructions/CLAUDE-STUDENT.md",
    "plugins/senpai/skills/poll-for-work/SKILL.md",
    "plugins/senpai/skills/submit-experiment-results/SKILL.md",
    "k8s/entrypoint-student.sh",
)

# %%
h2("Staff-engineer gotcha")
print(
    "The student handoff is a protocol boundary. If `pending_arms` is true or "
    "final metrics are missing, the PR should remain WIP even if one arm looks promising."
)
