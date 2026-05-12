# %% [markdown]
# # 05 - W&B And Weave As Memory
#
# Learning objective:
# Query live W&B run metadata and connect it to a trace-shaped Weave artifact.
# W&B is metric/config/provenance memory; Weave is agent/tool decision memory.

# %%
from pathlib import Path
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.notebook import checkpoint, h1, h2, repo_note, show_json
from common.wandb_utils import local_trace_shape, recent_runs, validate_project, wandb_api


# %%
h1("05 - W&B And Weave As Memory")
config = load_config(require=True)

# %%
h2("Validate W&B project")
project = validate_project(config)
show_json(project)

# %%
h2("Read compact recent runs")
runs = recent_runs(config, limit=5)
show_json(runs)

# %%
h2("Optional Weave trace check")
trace_result = local_trace_shape()
try:
    import weave

    client = weave.init(config.wandb_path)
    calls = list(client.get_calls(limit=3))
    trace_result = {
        "source": "live_weave",
        "call_count": len(calls),
        "calls": [
            {
                "id": str(getattr(call, "id", "")),
                "op_name": str(getattr(call, "op_name", "")),
                "started_at": str(getattr(call, "started_at", "")),
            }
            for call in calls
        ],
    }
except Exception as exc:
    trace_result = {
        "source": "local_trace_shape",
        "reason": f"{type(exc).__name__}: {exc}",
        "trace": local_trace_shape(),
    }
show_json(trace_result)

# %%
h2("Ledger interpretation")
ledger = {
    "hypothesis_truth": "GitHub PR body",
    "code_truth": "git commit and PR diff",
    "metric_truth": "W&B config, history, summary, artifacts",
    "agent_behavior_truth": "Weave trace",
    "workflow_truth": "PR labels and comments",
    "recent_runs_seen": len(runs),
    "trace_source": trace_result["source"],
}
show_json(ledger)

# %%
h2("Write artifact")
path = write_artifact(
    "05_wandb_weave_memory.json",
    {"project": project, "runs": runs, "trace_result": trace_result, "ledger": ledger},
)
checkpoint(f"Wrote {path}")

# %%
h2("What this teaches")
print(
    "In autoresearch, observability is memory. W&B captures training evidence; "
    "Weave captures the agent/tool path. The advisor should reconcile both before deciding."
)
repo_note(
    ".agents/skills/wandb-primary/SKILL.md",
    "k8s/install-weave-cc-plugin.sh",
    "tests/test_docker_image.py",
    "papers/icml/AGENT_FAILURE_MODES1.2.md",
)

# %%
h2("Staff-engineer gotcha")
print(
    "Do not dump raw run histories into model context. Summarize first, then ask the model "
    "to reason over compact evidence."
)
