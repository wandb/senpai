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

BASE_BRANCH = os.environ.get("ADVISOR_BRANCH", "noam")

def get_repo_info():
    r = subprocess.run(["gh", "repo", "view", "--json", "owner,name"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    info = json.loads(r.stdout)
    return info["owner"]["login"], info["name"]

def fetch():
    """Fetch PRs with comments via GraphQL (single API call, paginated)."""
    owner, name = get_repo_info()
    prs = []
    cursor = None
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = f'''{{
          repository(owner: "{owner}", name: "{name}") {{
            pullRequests(first: 100, baseRefName: "{BASE_BRANCH}", states: [OPEN, CLOSED, MERGED]{after}) {{
              pageInfo {{ hasNextPage endCursor }}
              nodes {{
                number title body state mergedAt url
                labels(first: 10) {{ nodes {{ name }} }}
                comments(first: 100) {{ nodes {{ body }} }}
              }}
            }}
          }}
        }}'''
        r = subprocess.run(["gh", "api", "graphql", "-f", f"query={query}"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        data = json.loads(r.stdout)["data"]["repository"]["pullRequests"]
        for node in data["nodes"]:
            prs.append({
                "number": node["number"],
                "title": node["title"],
                "body": node["body"] or "",
                "state": node["state"],
                "mergedAt": node["mergedAt"],
                "url": node["url"],
                "labels": [{"name": l["name"]} for l in node["labels"]["nodes"]],
                "comments": [c["body"] for c in node["comments"]["nodes"]],
            })
        if not data["pageInfo"]["hasNextPage"]:
            break
        cursor = data["pageInfo"]["endCursor"]
    return prs

def extract_section(text, header):
    """Extract a markdown section (# or ##) by header name."""
    match = re.search(rf'(^#{{1,2}} {header}.*?)(?=^#{{1,2}} |\Z)', text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None

def find_results_text(pr):
    """Search body then comments (newest first) for the Results section."""
    body = pr.get("body", "") or ""
    section = extract_section(body, "Results")
    if section and "_To be filled" not in section:
        return section
    for comment in reversed(pr.get("comments", [])):
        section = extract_section(comment, "Results")
        if section and "_To be filled" not in section:
            return section
    return None

def extract_val_loss(text):
    if not text:
        return None
    patterns = [
        r'val[/_]loss[:\s]*\*?\*?(\d+\.\d+)',
        r'\*\*val/loss\*\*[:\s|]*\*?\*?(\d+\.\d+)',
        r'val/loss[:\s|]+(\d+\.\d+)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def has_results(pr):
    return find_results_text(pr) is not None

def labels_str(pr):
    return ", ".join(l["name"] for l in pr.get("labels", []))

def hypothesis_oneline(body):
    """Extract first sentence of hypothesis."""
    sec = extract_section(body or "", "Hypothesis")
    if not sec:
        return ""
    text = re.sub(r'^#{1,2} Hypothesis\n', '', sec).strip()
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
        results = find_results_text(pr) or ""
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
        results = find_results_text(pr)
        val = extract_val_loss(results)
        val_str = f"{val:.4f}" if val else "—"
        f.write(f"| #{pr['number']} | {pr['title']} | {val_str} |\n")

    f.write(f"\n## Ran but not merged ({len(with_results)})\n\n")
    f.write("| PR | Title | val_loss | Hypothesis (short) |\n|---|---|---|---|\n")
    for pr in sorted(with_results, key=lambda p: p["number"], reverse=True):
        body = pr.get("body", "") or ""
        results = find_results_text(pr)
        val = extract_val_loss(results)
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
        f.write(f"# #{pr['number']}: {pr['title']} [{state}]\n\n{body}\n\n")
        results = find_results_text(pr)
        if results and results not in body:
            f.write(f"{results}\n\n")
        f.write("---\n\n")

print(f"Saved {len(prs)} PRs ({len(merged)} merged, {len(with_results)} ran, {len(no_results)} never ran):")
print(f"  {merged_path}         — merged winners with hypothesis + results")
print(f"  {table_path}   — compact table of all experiments")
print(f"  {full_path}   — full PR bodies (only experiments that ran)")
```
