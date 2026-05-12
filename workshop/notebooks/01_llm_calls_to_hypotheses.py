# %% [markdown]
# # 01 - From LLM Calls To Falsifiable Hypotheses
#
# Learning objective:
# Use a live LLM call for ideation, then convert a plausible idea into an
# experiment assignment with metric, split, allowed files, and falsifier.

# %%
from pathlib import Path
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.llm import anthropic_message
from common.notebook import checkpoint, h1, h2, repo_note


# %%
h1("01 - From LLM Calls To Falsifiable Hypotheses")
config = load_config(require=True)

# %%
h2("Live LLM ideation")
problem_statement = """
We are improving a CFD surrogate for aerodynamic surface pressure. The system
uses a Transolver-style model, W&B logging, and strict validation/test metric
contracts. Suggest three experiment ideas, but do not claim any result.
"""
ideas = anthropic_message(config, problem_statement, max_tokens=700)
print(ideas)

# %%
h2("Turn one idea into an assignment")
assignment_prompt = f"""
Take one idea from the list below and rewrite it as a SENPAI-style experiment
assignment. Include:
- hypothesis
- mechanism
- primary metric
- split or benchmark contract
- allowed files
- minimal implementation instruction
- W&B logging requirement
- falsifying result

Ideas:
{ideas}
"""
assignment = anthropic_message(config, assignment_prompt, max_tokens=900)
print(assignment)

# %%
h2("Write artifact")
path = write_artifact(
    "01_hypothesis_assignment.md",
    f"# LLM Ideas\n\n{ideas}\n\n# Falsifiable Assignment\n\n{assignment}\n",
)
checkpoint(f"Wrote {path}")

# %%
h2("What this teaches")
print(
    "A text-only LLM call is useful for proposing directions, but the research "
    "state only becomes actionable after we add metric contracts, file boundaries, "
    "W&B logging, and a falsifier."
)
repo_note(
    "papers/appendix_sources/tandemfoil2_program.md",
    "papers/appendix_sources/drivaerml_program.md",
    ".claude/agents/researcher-agent.md",
)

# %%
h2("Staff-engineer gotcha")
print(
    "Never let a plausible hypothesis spend GPU time until it names the metric, "
    "the split, the command, and what result would change our mind."
)
