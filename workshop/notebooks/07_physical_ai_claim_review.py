# %% [markdown]
# # 07 - Physical-AI Claim Review
#
# Learning objective:
# Use benchmark contracts to classify physical-AI claims as defensible,
# proxy-only, or misleading.

# %%
from pathlib import Path
import json
import sys

WORKSHOP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSHOP_ROOT.parent
if str(WORKSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOP_ROOT))

from common.config import load_config, write_artifact
from common.llm import anthropic_message
from common.notebook import checkpoint, h1, h2, repo_note, show_json


# %%
h1("07 - Physical-AI Claim Review")
config = load_config(require=True)

# %%
h2("Load benchmark contract excerpts")
contract_paths = [
    REPO_ROOT / "analysis" / "AIRFRANS_BENCHMARK.md",
    REPO_ROOT / "analysis" / "DRIVAERML_BENCHMARK.md",
    REPO_ROOT / "analysis" / "TANDEMFOILSET_BENCHMARK.md",
]
contracts = {
    path.name: "\n".join(path.read_text().splitlines()[:90])
    for path in contract_paths
}
show_json({name: text[:500] + "..." for name, text in contracts.items()})

# %%
h2("Claims to classify")
claims = [
    "Our DrivAerML validation surface-pressure relative-L2 improved from 4.62% to 4.42%, so we beat AB-UPT.",
    "On DrivAerML public 400/34/50, our held-out test surface-pressure relative-L2 is 6.24%, still behind AB-UPT 3.82%.",
    "TandemFoilSet surface-pressure MAE of 24.58 is directly comparable to the original paper Table 6 full-field MSE.",
    "On AirfRANS, surface MSE improved, but volume MSE remains worse than SpiderSolver, so this is not a full benchmark win.",
]
show_json(claims)

# %%
h2("LLM classification")
prompt = f"""
Classify each physical-AI claim as one of:
- defensible
- proxy_only
- misleading

Use the benchmark contract excerpts below. For each claim, include a one-sentence correction.

Contracts:
{json.dumps(contracts, indent=2)[:18000]}

Claims:
{json.dumps(claims, indent=2)}
"""
review = anthropic_message(config, prompt, max_tokens=1200)
print(review)

# %%
h2("Write artifact")
path = write_artifact(
    "07_physical_ai_claim_review.md",
    f"# Physical-AI Claim Review\n\n{review}\n",
)
checkpoint(f"Wrote {path}")

# %%
h2("What this teaches")
print(
    "Physical-AI autoresearch needs claim governance. The same scalar can mean "
    "different things under different split, normalization, target, and aggregation contracts."
)
repo_note(
    "analysis/AIRFRANS_BENCHMARK.md",
    "analysis/DRIVAERML_BENCHMARK.md",
    "analysis/TANDEMFOILSET_BENCHMARK.md",
)

# %%
h2("Staff-engineer gotcha")
print(
    "A completed training run can still produce a misleading claim. The metric contract "
    "is part of the system boundary."
)
