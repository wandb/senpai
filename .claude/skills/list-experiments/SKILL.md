---
name: list-experiments
description: Use this skill whenever you need to list all of the experiment ideas tried and in progress for this research programme. It outputs 3 files organized by usefulness — merged winners, a compact results table, and full details for deep dives. Use when generating new experimental ideas to check what has already been tried.
---

# List Experiments

Run the script below to fetch all experiment PRs from the advisor branch and organize them into 3 files.

The `BASE_BRANCH` should be set to the advisor branch (e.g. `noam`). Check the `$ADVISOR_BRANCH` env var or the PR base branch.

## Code

```python
import subprocess, json, os, re
from datetime import datetime

FIELDS = "number,title,body,state,labels,url,mergedAt"
BASE_BRANCH = os.environ.get("ADVISOR_BRANCH", "noam")

def fetch():
    cmd = ["gh", "pr", "list", "--json", FIELDS, "--limit", "10000",
           "--base", BASE_BRANCH, "--state", "all"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout)

def extract_section(body, header):
    match = re.search(rf'(^## {header}.*?)(?=^## |\Z)', body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None

def extract_val_loss(body):
    """Try to find val/loss or val_loss number from results."""
    if not body:
        return None
    patterns = [
        r'val[/_]loss[:\s]*\*?\*?(\d+\.\d+)',
        r'\*\*val/loss\*\*[:\s|]*\*?\*?(\d+\.\d+)',
        r'val/loss[:\s|]+(\d+\.\d+)',
    ]
    for p in patterns:
        m = re.search(p, body, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def has_results(pr):
    body = pr.get("body", "") or ""
    return "## Results" in body and "_To be filled" not in body

def labels_str(pr):
    return ", ".join(l["name"] for l in pr.get("labels", []))

def hypothesis_oneline(body):
    """Extract first sentence of hypothesis."""
    sec = extract_section(body or "", "Hypothesis")
    if not sec:
        return ""
    text = sec.replace("## Hypothesis\n", "").strip()
    # First sentence
    m = re.match(r'(.+?[.!])\s', text)
    return m.group(1) if m else text[:120]

# --- Fetch ---
prs = fetch()
merged = sorted([p for p in prs if p.get("mergedAt")], key=lambda p: p["mergedAt"])
with_results = [p for p in prs if has_results(p) and not p.get("mergedAt")]
no_results = [p for p in prs if not has_results(p)]

ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
outdir = "experiment_log"
os.makedirs(outdir, exist_ok=True)

# --- File 1: Merged winners (chronological) ---
merged_path = f"{outdir}/merged_{ts}.md"
with open(merged_path, "w") as f:
    f.write(f"# Merged Experiments ({len(merged)} winners)\n\n")
    f.write("These PRs beat baseline and were merged, in chronological order.\n\n")
    for pr in merged:
        body = pr.get("body", "") or ""
        hyp = extract_section(body, "Hypothesis") or ""
        results = extract_section(body, "Results") or ""
        val = extract_val_loss(results)
        val_str = f" | val_loss: {val}" if val else ""
        f.write(f"## #{pr['number']}: {pr['title']}{val_str}\n\n")
        if hyp:
            f.write(f"{hyp}\n\n")
        if results:
            f.write(f"{results}\n\n")
        f.write("---\n\n")

# --- File 2: Compact results table ---
table_path = f"{outdir}/results_table_{ts}.md"
with open(table_path, "w") as f:
    f.write(f"# Experiment Results Table\n\n")
    f.write(f"Total: {len(prs)} PRs | Merged: {len(merged)} | ")
    f.write(f"Ran (not merged): {len(with_results)} | Never ran: {len(no_results)}\n\n")

    f.write("## Merged\n\n")
    f.write("| PR | Title | val_loss |\n|---|---|---|\n")
    for pr in merged:
        val = extract_val_loss(pr.get("body", ""))
        val_str = f"{val:.4f}" if val else "—"
        f.write(f"| #{pr['number']} | {pr['title']} | {val_str} |\n")

    f.write(f"\n## Ran but not merged ({len(with_results)})\n\n")
    f.write("| PR | Title | val_loss | Hypothesis (short) |\n|---|---|---|---|\n")
    for pr in sorted(with_results, key=lambda p: p["number"], reverse=True):
        body = pr.get("body", "") or ""
        val = extract_val_loss(body)
        val_str = f"{val:.4f}" if val else "—"
        hyp = hypothesis_oneline(body)
        f.write(f"| #{pr['number']} | {pr['title']} | {val_str} | {hyp[:80]} |\n")

    f.write(f"\n## Never ran ({len(no_results)})\n\n")
    f.write("| PR | Title | Hypothesis (short) |\n|---|---|---|\n")
    for pr in sorted(no_results, key=lambda p: p["number"], reverse=True):
        hyp = hypothesis_oneline(pr.get("body", ""))
        f.write(f"| #{pr['number']} | {pr['title']} | {hyp[:80]} |\n")

# --- File 3: Full details (only PRs that ran) ---
full_path = f"{outdir}/full_details_{ts}.md"
with open(full_path, "w") as f:
    f.write(f"# Full Experiment Details ({len(merged) + len(with_results)} experiments)\n\n")
    for pr in merged + sorted(with_results, key=lambda p: p["number"], reverse=True):
        body = pr.get("body", "").strip() or "_No description._"
        state = "MERGED" if pr.get("mergedAt") else "CLOSED"
        f.write(f"# #{pr['number']}: {pr['title']} [{state}]\n\n{body}\n\n---\n\n")

print(f"Saved {len(prs)} PRs ({len(merged)} merged, {len(with_results)} ran, {len(no_results)} never ran):")
print(f"  {merged_path}         — merged winners with hypothesis + results")
print(f"  {table_path}   — compact table of all experiments")
print(f"  {full_path}   — full PR bodies (only experiments that ran)")
```
