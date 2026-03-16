---
name: list-experiments
description: Use this skill whenever you need to list all of the experiment ideas tried and in progress for this research programme. It outputs 2 files; a summary file that lists high-level detail of all the experiments and an full experiment list of all experiments including instructions and results. Its useful while in the process of generating new experimental ideas to try by checking what experiments have already been considered.
---

# Experiment List Sill

Use this code or a close modification of it to download a list of our experiment results from a particular branch. It downloads 2 files:

1. A summary file with the experiment metadata, hypothesis and results
2. The full experiment details which includes the abvoe as well as the full instructions (code etc) and baseline details given to the agent that executed the experiment

## Experiment List Code

```python
import subprocess, json, sys, os, re
from datetime import datetime

FIELDS = "number,title,body,state,labels,url"
BASE_BRANCH = <branch-fetch-PRs-from>  # only fetch PRs targeting this branch

def fetch(n=None):
    cmd = ["gh", "pr", "view", n, "--json", FIELDS] if n else ["gh", "pr", "list", "--json", FIELDS, "--limit", "10000", "--base", BASE_BRANCH]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: sys.exit(r.stderr)
    return json.loads(r.stdout)

def extract_section(body, header):
    match = re.search(rf'(^## {header}.*?)(?=^## |\Z)', body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None

def meta(pr):
    labels = ', '.join(f'`{l["name"]}`' for l in pr.get('labels', [])) or '_none_'
    return f"# PR {pr['number']}: {pr['title']}\n| URL | State | Labels |\n|-----|-------|--------|\n| {pr['url']} | `{pr['state']}` | {labels} |"

def to_md_full(pr):
    body = pr.get('body', '').strip() or '_No description._'
    return f"{meta(pr)}\n\n{body}\n\n---\n"

def to_md_summary(pr):
    body = pr.get('body', '') or ''
    sections = [extract_section(body, h) for h in ('Hypothesis', 'Results')]
    extra = "".join(f"\n\n{s}" for s in sections if s)
    return f"{meta(pr)}{extra}\n\n---\n"

n = sys.argv[1] if len(sys.argv) > 1 else None
ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
outdir = "experiment_log"
os.makedirs(outdir, exist_ok=True)

data = fetch(n)
items = data if isinstance(data, list) else [data]

full_path = f"{outdir}/all_experiments_{ts}.md"
summary_path = f"{outdir}/experiments_summary_{ts}.md"

with open(full_path, "w") as f:
    f.write("\n" + "\n".join(to_md_full(p) for p in items))

with open(summary_path, "w") as f:
    f.write("# Experiments Summary\n\n" + "\n".join(to_md_summary(p) for p in items))

print(f"Saved {len(items)} PR(s) to:\n  {full_path}\n  {summary_path}")
```