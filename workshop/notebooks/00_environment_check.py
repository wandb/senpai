# %% [markdown]
# # 00 - Environment Check
#
# Learning objective:
# Validate the live Anthropic, GitHub, and W&B credentials that power the rest
# of the runnable autoresearch workshop.

# %%
from pathlib import Path
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.github import get_branch, get_repo, repo_slug
from common.llm import anthropic_message
from common.notebook import checkpoint, h1, h2, repo_note, show_json
from common.wandb_utils import validate_project


# %%
h1("00 - Environment Check")
config = load_config(require=True)
show_json(config.redacted())

# %%
h2("Anthropic")
reply = anthropic_message(config, "Reply with exactly: OK", max_tokens=8)
checkpoint(f"Anthropic replied: {reply!r}")

# %%
h2("GitHub")
repo = get_repo(config)
branch = get_branch(config)
github_summary = {
    "repo": repo_slug(config.target_repo_url),
    "default_branch": repo.get("default_branch"),
    "configured_branch": config.target_repo_branch,
    "branch_sha": branch.get("commit", {}).get("sha"),
}
show_json(github_summary)

# %%
h2("W&B")
wandb_summary = validate_project(config)
show_json(wandb_summary)

# %%
h2("Write artifact")
artifact = {
    "config": config.redacted(),
    "anthropic_reply": reply,
    "github": github_summary,
    "wandb": wandb_summary,
}
path = write_artifact("environment_check.json", artifact)
checkpoint(f"Wrote {path}")

# %%
h2("What this teaches")
print(
    "The workshop is live-service-first: the LLM, GitHub, and W&B are not mocked in this check. "
    "Later notebooks can treat these services as the observation and memory layers for autoresearch."
)
repo_note("workshop/setup_credentials.py", "k8s/launch.py", "tests/test_docker_image.py")
