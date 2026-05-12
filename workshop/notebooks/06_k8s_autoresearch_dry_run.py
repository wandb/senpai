# %% [markdown]
# # 06 - Kubernetes Autoresearch Dry Run
#
# Learning objective:
# Render SENPAI launch resources without applying them, then inspect identity,
# routing, resources, data access, and observability fields.

# %%
from pathlib import Path
import subprocess
import sys

import yaml

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSHOP_ROOT.parent
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.notebook import checkpoint, h1, h2, repo_note, show_json


# %%
h1("06 - Kubernetes Autoresearch Dry Run")
config = load_config(require=True)

# %%
h2("Render dry-run resources")
command = [
    sys.executable,
    "k8s/launch.py",
    "--tag",
    "workshop",
    "--target_repo_url",
    config.target_repo_url,
    "--target_repo_branch",
    config.target_repo_branch,
    "--advisor_branch",
    config.advisor_branch,
    "--n_students",
    "2",
    "--student_prefix",
    "ws",
    "--gpus_per_student",
    "1",
    "--advisor",
    "--dry_run",
]
result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
if result.returncode != 0:
    print(result.stderr)
    raise SystemExit(result.returncode)
dry_run_output = result.stdout
print(dry_run_output[:6000])

# %%
h2("Parse YAML documents")
yaml_text = "\n".join(line for line in dry_run_output.splitlines() if not line.startswith("--- Secret:") and not line.startswith("--- Student:") and not line.startswith("--- Advisor"))
docs = [doc for doc in yaml.safe_load_all(yaml_text) if isinstance(doc, dict)]
summary = []
for doc in docs:
    metadata = doc.get("metadata", {})
    summary.append({
        "kind": doc.get("kind"),
        "name": metadata.get("name"),
        "labels": metadata.get("labels", {}),
    })
show_json(summary)

# %%
h2("Classify launch fields")
classification = {
    "research_identity": ["research-tag=workshop", f"target_repo={config.target_repo_url}", f"advisor_branch={config.advisor_branch}"],
    "routing": ["ADVISOR_BRANCH", "student labels", "status:wip", "status:review"],
    "resources": ["nvidia.com/gpu", "cpu", "memory", "/dev/shm"],
    "data_access": ["PVC_MOUNT_PATH", "PVC_CLAIM_NAME"],
    "observability": ["WANDB_ENTITY", "WANDB_PROJECT", "weave_project derived at runtime"],
    "secret_dependent": ["GITHUB_TOKEN", "ANTHROPIC_API_KEY", "EXA_API_KEY", "WANDB_API_KEY"],
}
show_json(classification)

# %%
h2("Write artifacts")
path = write_artifact("06_dry_run_output.txt", dry_run_output)
parsed_path = write_artifact("06_dry_run_summary.json", {"resources": summary, "classification": classification})
checkpoint(f"Wrote {path}")
checkpoint(f"Wrote {parsed_path}")

# %%
h2("What this teaches")
print(
    "A SENPAI launch is not just deployment. It establishes research identity, "
    "student routing, GPU economics, data access, and observability wiring."
)
repo_note(
    "k8s/launch.py",
    "k8s/launch_helpers.py",
    "k8s/advisor-deployment.yaml",
    "k8s/student-deployment.yaml",
)

# %%
h2("Staff-engineer gotcha")
print(
    "The dry-run path is the right classroom demo. It exercises the launch contract "
    "without creating secrets, deployments, PRs, or GPU workloads."
)
