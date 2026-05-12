# %% [markdown]
# # 02 - Tool Calls And Executable Contracts
#
# Learning objective:
# Build a tiny read-only tool loop and teach why SENPAI moves fragile workflow
# mechanics into structured helpers.

# %%
from pathlib import Path
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.github import get_branch, get_repo
from common.llm import Tool, choose_tool, run_chosen_tool, summarize_with_llm
from common.notebook import checkpoint, h1, h2, repo_note, show_json
from common.senpai_protocol import parse_senpai_result, senpai_result_line, terminal_result_errors
from common.wandb_utils import recent_runs


# %%
h1("02 - Tool Calls And Executable Contracts")
config = load_config(require=True)

# %%
h2("Define read-only tools")
tools = [
    Tool("github_repo", "Read configured target GitHub repository metadata.", lambda _: get_repo(config)),
    Tool("github_branch", "Read configured target branch metadata.", lambda _: get_branch(config)),
    Tool("wandb_recent_runs", "Read compact summaries of recent W&B runs.", lambda args: recent_runs(config, limit=int(args.get("limit", 3)))),
]
show_json([{"name": tool.name, "description": tool.description} for tool in tools])

# %%
h2("Let the LLM choose a tool")
task = "We need to know whether the configured target GitHub branch exists before assigning experiments."
choice = choose_tool(config, task, tools)
show_json(choice)
tool_result = run_chosen_tool(choice, tools)
show_json(tool_result)

# %%
h2("Summarize structured output")
summary = summarize_with_llm(config, "tool result", tool_result)
print(summary)

# %%
h2("Recreate the SENPAI-RESULT guard in teaching form")
candidate = {
    "terminal": True,
    "status": "complete",
    "pending_arms": False,
    "wandb_run_ids": ["workshop-run-001"],
    "primary_metric": {"name": "test_primary/surface_pressure_rel_l2_pct", "value": 6.31},
    "test_metric": {"name": "test_primary/surface_pressure_rel_l2_pct", "value": 6.31},
}
line = senpai_result_line(candidate)
print(line)
parsed = parse_senpai_result(line)
errors = terminal_result_errors(parsed)
show_json({"parsed": parsed, "errors": errors})

# %%
h2("Write artifact")
path = write_artifact(
    "02_tool_contracts.json",
    {
        "choice": choice,
        "tool_result": tool_result,
        "summary": summary,
        "senpai_result": parsed,
        "result_errors": errors,
    },
)
checkpoint(f"Wrote {path}")

# %%
h2("What this teaches")
print(
    "Tool calls are useful because they return structured state. Production "
    "autoresearch systems should encode workflow invariants in tools instead "
    "of asking the model to remember fragile command sequences."
)
repo_note(
    "plugins/senpai/scripts/senpai-gh.sh",
    "plugins/senpai/scripts/senpai-pr-guard.py",
    "plugins/senpai/skills/senpai-gh/SKILL.md",
)

# %%
h2("Staff-engineer gotcha")
print(
    "A tool that simply exposes raw shell is still a large action space. "
    "The better tool says: here is the safe state transition, and here are "
    "the invariants that must pass first."
)
