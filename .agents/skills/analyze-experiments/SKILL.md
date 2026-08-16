---
name: analyze-experiments
description: >
  Operator-side analysis of historical ML experiment PRs in Senpai research tracks.
  Use this skill whenever the user asks to: analyze experiments, categorize PRs, bucket
  experiments, summarize what's been tried, understand experiment history, review merged
  vs closed results, or asks "what experiments have we run / worked / failed".
  Also triggers for: "pull the latest experiments", "what's been tried so far",
  "category breakdown of PRs", "which experiments succeeded", "noam track analysis".
  When a branch name is mentioned (e.g. "noam branch", "on the noam branch"), pass it
  as the base branch to scope the fetch to just those PRs.
---

# Analyze Experiments Skill

This is a human/operator analysis guide, not a live advisor or student skill.
It fetches fresh experiment PR data via the list-experiments skill, categorizes
each PR using parallel readers, and produces a 5-section report: full catalogue,
category breakdown with merge rates, merged-only wins, closed-only failures,
and key narratives.

---

## Step 1: Fetch PR data via list-experiments

Read the list-experiments skill at `.agents/skills/list-experiments/SKILL.md`
and run its Python code with the following configuration:

**If a base branch was specified** (e.g. "noam", "yan", "main") — use it directly:
```python
BASE_BRANCH = "noam"   # replace with the branch the user mentioned
# use the fetch command from list-experiments as-is (it uses --base BASE_BRANCH)
```

**If no branch was specified** — fetch all senpai-labelled experiments instead. Replace the `fetch()` function body with:
```python
cmd = ["gh", "pr", "list", "--repo", "wandb/senpai", "--json", FIELDS,
       "--limit", "10000", "--state", "all", "--label", "senpai"]
```

Run the script and note the `experiments_summary_<ts>.md` path printed — that's your input for Steps 2–3.

---

## Step 2: Find split points for parallel reading

The summary file can be large (thousands of lines). Divide it into 4 roughly equal chunks:

```bash
# Get total PR count and line positions for every ~25% boundary
grep -n "^# PR" <summary_path> | awk 'BEGIN{getline l1; print l1} NR==int(total*0.25) || NR==int(total*0.5) || NR==int(total*0.75) {print} END{print}' total=$(grep -c "^# PR" <summary_path>)

# Simpler: just get all PR line numbers and pick 3 evenly-spaced split points
grep -n "^# PR" <summary_path> | awk -v n=$(grep -c "^# PR" <summary_path>) 'NR==1||NR==int(n/4)||NR==int(n/2)||NR==int(3*n/4)||NR==n{print NR, $0}'
```

Note the line numbers for the 4 batch boundaries.

---

## Step 3: Launch 4 parallel Explore agents

Send all 4 in a single message. Each agent reads its assigned line range from the summary
file and returns one row per PR. The agents need to read both the title AND the `## Results`
section to assign an accurate outcome — title-based keywords set the category, the results
section determines whether it worked.

Give each agent these instructions (substituting LINE_START, LINE_END, PR range):

> Read `<summary_path>` from line LINE_START to LINE_END.
> For every PR in this range return exactly one row:
> `PR #NNN | STATE | Category | emoji | 1-line outcome`
>
> **STATE**: MERGED / CLOSED / OPEN (from the `| State |` header table in each PR block)
>
> **Category** — pick exactly one based on the primary change being tested:
> 1. **Loss function** — loss type (L1, MSE, Huber, cosine sim, asymmetric surf/vol split)
> 2. **LR / optimizer** — learning rate value, warmup, scheduler, weight decay, β params, optimizer choice
> 3. **Model architecture** — depth (n_layers), width (n_hidden), n_heads, mlp_ratio, output head design, preprocess MLP structure
> 4. **Initialization** — weight init strategy (Xavier, Kaiming, orthogonal, learnable placeholders, init scale/gain)
> 5. **Training efficiency** — bf16 autocast, batch size, gradient accumulation, volume node subsampling, epoch budget tricks
> 6. **Regularization** — dropout, SWA, EMA, target noise magnitude/schedule, gradient clipping, spectral norm
> 7. **Physics / normalization** — Cp normalization, per-sample normalization, domain-aware scaling, split surf/vol stats
> 8. **Loss weighting** — surf_weight value or schedule, per-channel weights, domain re-weighting
> 9. **Feature engineering** — slice count/structure, attention temperature, spatial/positional encoding (RFF, NeRF)
> 10. **Inference fix** — fp32 for OOD splits, NaN guards, denormalization fixes, clamping
>
> **emoji**: ✓ improved primary metric (mae_surf_p on val_in_dist), ✗ hurt it, ~ inconclusive/noise-level, — no results yet
>
> **1-line outcome**: state what changed, the key metric delta (e.g. "mae_surf_p 119→103 (-14%)"), and the reason it succeeded or failed. If Results section is empty, write "No results".
>
> Return ONLY the data rows — no headers, no commentary.

---

## Step 4: Synthesize into report

Combine all rows and produce a 5-section markdown report:

**Section 1 — Full PR Catalogue**
Table: `| PR | State | Category | Result | Outcome |`
All PRs sorted by number. OPEN PRs included but marked `—`.

**Section 2 — Category Breakdown (completed PRs only)**
Table: `| Category | Total | Merged | Closed | Merge Rate | Merged PRs |`
Compute merge rate = MERGED ÷ (MERGED + CLOSED). Sort descending by merge rate.
OPEN PRs excluded from totals and rates.

**Section 3 — Merged PRs Only**
For each category that has merges: list the merged PRs and a sentence on what the wins had in common.

**Section 4 — Closed PRs — Failure Patterns**
Table: `| Category | Closed | Hard ✗ | Inconclusive ~ | Showed promise ✓ (not merged) |`
Call out which categories had PRs that showed improvement but still weren't merged.

**Section 5 — Key Narratives**
3–5 bullet points on cross-cutting patterns: what reliably works, what never works,
what the current productive frontier looks like (recent OPEN PRs).

---

## Notes

- **OPEN PRs**: include in catalogue with `—` outcome; exclude from all merge-rate calculations.
- **Primary metric**: `mae_surf_p` on `val_in_dist`. When that's unavailable use any reported split. Lower is better.
- **Metric era shift**: Cp normalization (around PR #392 in the noam track) changed mae_surf_p from ~80–200 Pa to ~20–50 Pa range. Don't compare raw numbers across this boundary — only relative improvement within each era matters.
- **Initialization vs Model architecture**: If a PR changes *only* the init strategy (Xavier, orthogonal, Kaiming, learnable init), use **Initialization**. If it changes both init and something structural, use whichever is the primary tested hypothesis.
