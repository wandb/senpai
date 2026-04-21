# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

---
name: rlm
description: >
  Recursive Language Model (RLM) helper for writing parallel async LLM pipelines.
  Use this skill whenever the task involves processing many items with an LLM in
  parallel: fan-out extraction, batch classification, bulk annotation, parallel
  summarisation, or any "map each item → LLM call → aggregate results" workflow.
  Triggers for: "process all X in parallel", "use RLM", "fan out LLM calls",
  "batch LLM", "parallel Codex calls", "process hundreds of PRs / files / rows
  with Codex", "RLM pattern", "recursive language model", "async LLM", "sub-LLM".
  Also use proactively whenever a script needs to call the Anthropic API in a loop
  over a list of items — replace the loop with this pattern instead.
---

# RLM (Recursive Language Model) Skill

**Based on:** [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) — Zhang, Kraska & Khattab (MIT, 2025)
**Reference implementation:** `tools/update_wandb_run_metadata.py` — extracts W&B run IDs from 1,678 PR bodies using 30 parallel sub-LLM calls.

---

## Core concept

Instead of one LLM call over a huge context, the **root** (this script/agent) splits
the input into atomic items and fans out to parallel **sub-LLMs** — one call per item.
Each sub-LLM operates on a small, focused context and returns structured JSON.
The root aggregates the results.

```
Root orchestrator (this script)
├── item_0  →  sub-LLM call  →  result_0
├── item_1  →  sub-LLM call  →  result_1   (all fired concurrently,
├── item_2  →  sub-LLM call  →  result_2    capped by semaphore)
│   ...
└── item_N  →  sub-LLM call  →  result_N
```

The four strategies from the paper — use the one that fits:

| Strategy | When to use | Sub-LLM prompt shape |
|---|---|---|
| **Peek** | Inspect a sample before committing to a strategy | "Here are 3 rows — what fields exist?" |
| **Grep** | Narrow a large corpus to relevant items first | "Does this chunk contain X? Answer yes/no" |
| **Partition + Map** | Process every item independently | "Extract Y from this item. Return JSON." |
| **Summarise** | Compress each chunk before a second aggregation pass | "Summarise the key facts in 3 sentences." |

For most senpai tasks (PR analysis, run tagging, metric extraction) **Partition + Map** is the right choice.

---

## Boilerplate template

Copy this and fill in the three marked sections.

```python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0

"""<One-line description of what this script does.>

RLM pattern: root fans out to up to LLM_CONCURRENCY parallel sub-LLM calls,
one per item. Each sub-LLM returns structured JSON.

Usage: uv run tools/<script_name>.py
"""

import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

# ---------------------------------------------------------------------------
# Config — adjust as needed
# ---------------------------------------------------------------------------

MODEL = "Codex-sonnet-4-6"       # sub-LLM model
LLM_CONCURRENCY = 30              # max parallel API calls (safe limit for Sonnet)
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# ① SUB-LLM PROMPT — fill this in
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
<Task description for the sub-LLM. Be explicit about the output format.>

Item:
{item_text}

Respond ONLY with valid JSON, no other text:
{{"field1": "...", "field2": [...]}}
If nothing found: {{"field1": null, "field2": []}}"""


# ---------------------------------------------------------------------------
# ② SUB-LLM CALL — one per item
# ---------------------------------------------------------------------------

async def process_item(
    item: dict,                        # dict with at minimum an "id" key
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
) -> dict:
    """Call the sub-LLM for a single item. Returns item dict merged with result."""
    # ③ EXTRACT the text to pass to the LLM from your item dict
    item_text = item.get("body") or item.get("text") or str(item)

    if not item_text.strip():
        return {**item, "llm_result": None}

    async with sem:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=512,                    # increase for longer responses
            messages=[{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(item_text=item_text[:12_000]),
            }],
        )

    parsed = json.loads(response.content[0].text.strip())
    return {**item, "llm_result": parsed}


# ---------------------------------------------------------------------------
# ROOT ORCHESTRATOR — fan-out + aggregate
# ---------------------------------------------------------------------------

async def run_rlm(items: list[dict]) -> list[dict]:
    """Fan out sub-LLM calls over all items, return results in completion order."""
    client = AsyncAnthropic()
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    tasks = [process_item(item, client, sem) for item in items]
    return await atqdm.gather(*tasks, desc=f"RLM ({MODEL})", total=len(tasks))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # ③ LOAD YOUR ITEMS — replace with your actual data source
    items = [
        {"id": 1, "body": "first item text ..."},
        {"id": 2, "body": "second item text ..."},
    ]
    print(f"Processing {len(items)} items with RLM...")
    results = asyncio.run(run_rlm(items))

    # ③ AGGREGATE — replace with your actual post-processing
    for r in results:
        if r.get("llm_result"):
            print(r["id"], r["llm_result"])


if __name__ == "__main__":
    main()
```

---

## Rate limits and concurrency

| Model | Safe `LLM_CONCURRENCY` | Notes |
|---|---|---|
| `Codex-sonnet-4-6` | 30 | Default — good balance of speed and cost |
| `Codex-haiku-4-5` | 50 | Use for cheap, high-volume sub-calls |
| `Codex-opus-4-7` | 5 | Heavy model; use only for root-level synthesis |

Use a **second semaphore** for downstream rate-limited APIs (e.g. W&B, GitHub):

```python
WANDB_CONCURRENCY = 10   # W&B API is stricter than Anthropic

async def update_downstream(item: dict, api, sem: asyncio.Semaphore) -> bool:
    async with sem:
        try:
            return await asyncio.to_thread(_sync_api_call, item)
        except Exception as e:
            print(f"\n  [skip] {item['id']}: {e}", file=sys.stderr)
            return False

async def run_updates(items: list[dict], api) -> int:
    sem = asyncio.Semaphore(WANDB_CONCURRENCY)
    tasks = [update_downstream(item, api, sem) for item in items]
    results = await atqdm.gather(*tasks, desc="Updating downstream", total=len(tasks))
    return sum(results)
```

---

## Prompt engineering for sub-LLMs

Sub-LLM prompts should be:

1. **Extractive, not generative** — "extract X" not "describe X". Constrain the search space.
2. **JSON-only output** — always end with `Respond ONLY with valid JSON`. Avoids markdown fences.
3. **Schema explicit in the prompt** — show the exact keys and types expected.
4. **Null/empty case handled** — always include the empty-result example so the model doesn't hallucinate.
5. **Item context in the prompt header** — include `PR #NNN: title` or similar so the model can orient itself.

**Validation pattern** — always validate LLM output before use:

```python
def validate_run_ids(raw: list) -> list[str]:
    """Example: validate 8-char W&B run IDs."""
    return [
        r for r in raw
        if isinstance(r, str)
        and len(r) == 8
        and r.isalnum()
        and any(c.isdigit() for c in r)
    ]
```

---

## Two-pass pattern: regex first, LLM second

For extraction tasks where a regex can find most cases, use it as a cheap first pass and
only invoke LLM where needed. This cuts cost by ~80% on typical PR corpora:

```python
def regex_pass(items: list[dict]) -> dict[int, list[str]]:
    return {item["id"]: RUN_ID_RE.findall(item.get("body", "")) for item in items}

async def llm_pass(items: list[dict]) -> dict[int, list[str]]:
    results = await run_rlm(items)
    return {r["id"]: r["llm_result"].get("run_ids", []) for r in results}

def merge(regex: dict, llm: dict, ids: list[int]) -> dict[int, list[str]]:
    """Prefer LLM result; fall back to regex."""
    return {i: (llm.get(i) or regex.get(i, [])) for i in ids}
```

---

## Multi-role classification

When a single item has multiple extracted entities that need different labels (e.g. one PR
has a "final" run and several "pre-merge" runs), ask the sub-LLM to classify each entity:

```python
PROMPT_TEMPLATE = """\
Extract all W&B run IDs from this PR body and classify each one.

Roles:
- "final": the best/featured run reported in the Results section (at most one)
- "pre-merge": earlier or preliminary runs mentioned in the body

PR #{pr_number}: {pr_title}

{pr_body}

Respond ONLY with valid JSON:
{{"runs": [{{"run_id": "abc12345", "role": "final"}}, {{"run_id": "xyz98765", "role": "pre-merge"}}]}}
If no run IDs: {{"runs": []}}"""
```

---

## Error handling philosophy

- **Don't wrap sub-LLM calls in try/except** — let `asyncio.gather` propagate LLM errors.
  A malformed JSON response from one call failing the whole batch surfaces the bug immediately.
- **Do wrap downstream side-effects** (W&B updates, DB writes, file writes) in a narrow
  try/except — these fail for external reasons (404, rate limit) unrelated to the logic.
- **json.loads will raise on bad JSON** — this is intentional. If you need resilience,
  add a Peek pass first to validate the model reliably returns JSON before the main run.

---

## Worked example in this repo

**`tools/update_wandb_run_metadata.py`** — full production example:
- **Input**: 1,678 GitHub PRs (closed + merged)
- **Partition + Map**: 30 parallel `Codex-sonnet-4-6` calls extract W&B run IDs + classify role
- **Downstream action**: 10-concurrent W&B config updates via `asyncio.to_thread`
- **Result**: 1,255 W&B runs tagged with `gh_pr_branch`, `gh_pr_status`, `gh_pr_url`, `gh_pr_run_stage`
- **Wall time**: ~4 min LLM pass + ~2 min W&B pass for 1,678 items

Key excerpt:

```python
async def run_llm_pass(prs: list[dict], client: AsyncAnthropic) -> dict[int, list[dict]]:
    sem = asyncio.Semaphore(LLM_CONCURRENCY)   # 30
    tasks = [extract_runs_llm(pr, client, sem) for pr in prs]
    results = await atqdm.gather(*tasks, desc="RLM: extracting run IDs", total=len(tasks))
    return {pr["number"]: runs for pr, runs in zip(prs, results)}
```

---

## When NOT to use RLM

| Situation | Better approach |
|---|---|
| Single item or < 5 items | Direct LLM call, no async needed |
| Items have dependencies (B needs A's output) | Sequential chain, not fan-out |
| Need the LLM to reason across all items together | Single call with all items in context |
| Pure regex/heuristic suffices | Skip LLM entirely |
| Reading a codebase for patterns | Use Codex's built-in Grep/Read tools |
