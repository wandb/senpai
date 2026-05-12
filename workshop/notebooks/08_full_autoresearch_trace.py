# %% [markdown]
# # 08 - Full Autoresearch Trace
#
# Learning objective:
# Pull the previous lessons into one end-to-end educational trace: hypothesis,
# assignment, student result, W&B evidence, advisor review, and decision.

# %%
from pathlib import Path
import json
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.llm import anthropic_message
from common.notebook import checkpoint, h1, h2, repo_note, show_json
from common.senpai_protocol import Assignment, merge_decision, senpai_result_line
from common.wandb_utils import local_trace_shape, recent_runs


# %%
h1("08 - Full Autoresearch Trace")
config = load_config(require=True)

# %%
h2("1. Advisor creates a hypothesis")
hypothesis_prompt = """
Create one concise CFD-surrogate experiment hypothesis for DrivAerML. It must
name a mechanism, a held-out test metric, and a falsifier. Keep it practical.
"""
hypothesis = anthropic_message(config, hypothesis_prompt, max_tokens=500)
print(hypothesis)

# %%
h2("2. Advisor packages the assignment")
assignment = Assignment(
    number=9001,
    title="Workshop capstone DrivAerML hypothesis",
    student="ws-fern",
    advisor_branch=config.advisor_branch,
    hypothesis=hypothesis,
    metric_name="test_primary/surface_pressure_rel_l2_pct",
    baseline_value=6.24,
)
assignment_packet = {
    "pr_number": assignment.number,
    "title": assignment.title,
    "labels": assignment.labels,
    "hypothesis": assignment.hypothesis,
    "metric": assignment.metric_name,
    "baseline": assignment.baseline_value,
}
show_json(assignment_packet)

# %%
h2("3. Student observes W&B evidence")
runs = recent_runs(config, limit=3)
run_id = runs[0]["id"] if runs else "workshop-run-001"
observed_metric = 6.31
student_result = {
    "terminal": True,
    "status": "complete",
    "pending_arms": False,
    "wandb_run_ids": [run_id],
    "primary_metric": {"name": assignment.metric_name, "value": observed_metric},
    "test_metric": {"name": assignment.metric_name, "value": observed_metric},
}
result_line = senpai_result_line(student_result)
print(result_line)

# %%
h2("4. Advisor verifies and decides")
decision = merge_decision(observed_metric, assignment.baseline_value)
review_prompt = f"""
You are the SENPAI advisor. Decide whether to merge, request changes, or close.

Assignment:
{json.dumps(assignment_packet, indent=2)}

Student result:
{result_line}

Baseline: {assignment.baseline_value}
Observed held-out test metric: {observed_metric}
Recent W&B run summaries:
{json.dumps(runs, indent=2, default=str)[:12000]}

Return a short advisor decision. Do not merge if the test metric is worse than baseline.
"""
advisor_decision = anthropic_message(config, review_prompt, max_tokens=700)
print(advisor_decision)

# %%
h2("5. Trace-shaped audit record")
trace = {
    "hypothesis": hypothesis,
    "assignment": assignment_packet,
    "student_result": student_result,
    "wandb_runs_consulted": runs,
    "weave_trace_shape": local_trace_shape(),
    "decision": decision,
    "advisor_decision_text": advisor_decision,
}
show_json({k: v for k, v in trace.items() if k != "wandb_runs_consulted"})

# %%
h2("Write capstone artifact")
path = write_artifact("08_full_autoresearch_trace.json", trace)
markdown = write_artifact(
    "08_full_autoresearch_trace.md",
    "# Full Autoresearch Trace\n\n"
    f"## Hypothesis\n\n{hypothesis}\n\n"
    f"## Assignment\n\n```json\n{json.dumps(assignment_packet, indent=2)}\n```\n\n"
    f"## Student Result\n\n`{result_line}`\n\n"
    f"## Advisor Decision\n\n{advisor_decision}\n",
)
checkpoint(f"Wrote {path}")
checkpoint(f"Wrote {markdown}")

# %%
h2("What this teaches")
print(
    "The full loop is credible because every step leaves a durable artifact: "
    "hypothesis, labels, W&B run, result marker, trace, and advisor decision."
)
repo_note(
    "README.md",
    "system_instructions/CLAUDE-ADVISOR.md",
    "system_instructions/CLAUDE-STUDENT.md",
    "papers/paper.md",
)

# %%
h2("Staff-engineer gotcha")
print(
    "The capstone intentionally does not mutate GitHub or launch Kubernetes. "
    "The workshop teaches the live reasoning and read paths first; mutations belong "
    "behind explicit operator gates."
)
