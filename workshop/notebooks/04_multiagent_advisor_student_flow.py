# %% [markdown]
# # 04 - Multiagent Advisor/Student Flow
#
# Learning objective:
# Model SENPAI's advisor/student/researcher separation and label-based queue
# state using live LLM reasoning over a small in-memory PR board.

# %%
from pathlib import Path
import json
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.github import routing_labels
from common.llm import anthropic_message
from common.notebook import checkpoint, h1, h2, repo_note, show_json


# %%
h1("04 - Multiagent Advisor/Student Flow")
config = load_config(require=True)
students = ["ws-fern", "ws-stark", "ws-robin", "ws-brook"]
show_json(routing_labels(config.advisor_branch, students))

# %%
h2("PR board")
pr_board = [
    {"number": 3101, "title": "cosine warmup surface loss", "labels": [config.advisor_branch, "student:ws-fern", "status:wip"], "draft": True, "metric": None},
    {"number": 3102, "title": "MQA attention audit", "labels": [config.advisor_branch, "student:ws-stark", "status:review"], "draft": False, "metric": 6.8},
    {"number": 3103, "title": "EMA warmup 0.999", "labels": [config.advisor_branch, "student:ws-robin", "status:review"], "draft": False, "metric": 6.1},
    {"number": 3104, "title": "volume loss 1.5x", "labels": [config.advisor_branch, "student:ws-brook", "status:wip"], "draft": True, "metric": None},
    {"number": 3105, "title": "stale Fourier features", "labels": [config.advisor_branch, "student:ws-fern", "status:wip"], "draft": True, "metric": None},
    {"number": 3106, "title": "missing route labels", "labels": ["student:ws-brook"], "draft": True, "metric": None},
]
show_json(pr_board)

# %%
h2("Programmatic queue analysis")
wip_by_student = {name: [] for name in students}
review_ready = []
malformed = []
for pr in pr_board:
    labels = set(pr["labels"])
    if "status:review" in labels:
        review_ready.append(pr["number"])
    for student in students:
        if f"student:{student}" in labels and "status:wip" in labels:
            wip_by_student[student].append(pr["number"])
    required_any_status = {"status:wip", "status:review"}
    if config.advisor_branch not in labels or not labels.intersection(required_any_status):
        malformed.append(pr["number"])

queue_summary = {
    "wip_by_student": wip_by_student,
    "idle_students": [name for name, prs in wip_by_student.items() if not prs],
    "duplicate_wip": {name: prs for name, prs in wip_by_student.items() if len(prs) > 1},
    "review_ready": review_ready,
    "malformed": malformed,
}
show_json(queue_summary)

# %%
h2("Advisor LLM recommendation")
prompt = f"""
You are the SENPAI advisor. Given this PR board and queue summary, recommend the next actions.

Baseline metric is 6.24 lower-is-better. PR metrics are held-out test metrics when present.

PR board:
{json.dumps(pr_board, indent=2)}

Queue summary:
{json.dumps(queue_summary, indent=2)}

Return a concise ordered action list. Do not merge without mentioning preflight.
"""
recommendation = anthropic_message(config, prompt, max_tokens=800)
print(recommendation)

# %%
h2("Write artifact")
path = write_artifact(
    "04_multiagent_queue.json",
    {"pr_board": pr_board, "queue_summary": queue_summary, "advisor_recommendation": recommendation},
)
checkpoint(f"Wrote {path}")

# %%
h2("What this teaches")
print(
    "SENPAI's multiagent system coordinates through labels and durable PR state. "
    "The advisor is a research-control role, while students are GPU execution roles."
)
repo_note(
    "system_instructions/CLAUDE-ADVISOR.md",
    "plugins/senpai/skills/survey-prs/SKILL.md",
    "plugins/senpai/skills/assign-experiment/SKILL.md",
    "k8s/launch_helpers.py",
)

# %%
h2("Staff-engineer gotcha")
print(
    "Multiagent correctness is mostly queue correctness: duplicate WIP, missing labels, "
    "and stale review states are distributed-systems bugs, not just model errors."
)
