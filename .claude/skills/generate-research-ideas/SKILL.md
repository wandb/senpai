---
name: generate-research-ideas
description: Use this skill to generate fresh research hypotheses for idle students. Launches a sub-agent (Opus model) that reviews all past experiments via the list-experiments skill, then produces a ranked list of new ideas to test. Use when students need new work assigned.
---

# Generate Research Ideas

Launch a sub-agent powered by the Opus model to review past experiments and generate new hypotheses.

## Sub-agent instructions

Give the sub-agent these instructions plus any additional context you have (current baseline, recent trends, what has/hasn't worked):

1. Read `program.md` for the full context and goals. The key metric is surface MAE (especially pressure).

2. Review what has been tried already:
   - Use the `list-experiments` skill to download all experiment details
   - Every PR is an experiment idea and result — some PRs contain multiple related trials
   - Pay attention to what worked, what didn't, and why

3. After thorough review, generate new ideas:
   - Think creatively across machine learning, computer science, mathematics, optimization, and systems design
   - Connect modern ML research to older ideas (Schmidhuber-style cross-pollination)
   - Target surface accuracy (the most important metric)
   - Each idea must be distinct from what's been tried or is currently in-flight

4. Return a **ranked list** of the most promising new ideas, each with:
   - A clear hypothesis statement
   - Specific, actionable instructions for implementation in `train.py`
   - Why this idea is worth trying (connection to past results or theoretical reasoning)

## Prioritization

Not all ideas are equal. Prefer:
1. Ideas targeting **surface accuracy** (most important metric)
2. Low-complexity, high-impact changes (loss formulation, learning rate schedules)
3. Architectural changes only after simpler levers are exhausted
4. Avoid duplicating ideas already in-flight — check current open PRs

## Plateau protocol

If 5+ consecutive experiments show no improvement, escalate:
1. **Change strategy tier** — hyperparams -> architecture -> loss/data -> fundamentally different
2. **Revisit first principles** — what does the model struggle with? What pattern do failures share?
3. **Think bigger** — what techniques from aerodynamics, physics, math, CS haven't been tried?
4. **Try bold ideas** — plateaus are permission for bigger swings

A plateau is never a completion signal. It is a map telling you where not to look, which makes it an asset.
