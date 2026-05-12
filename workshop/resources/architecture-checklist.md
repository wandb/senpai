# Autoresearch Architecture Checklist

Use this checklist during the closing design review or as a take-home handout.

## Research Contract

- What is the target problem?
- What files may agents edit?
- What files are protected?
- What command runs the experiment?
- What is the wall-clock or epoch budget?
- What metric decides merge, request changes, or close?
- Is the metric validation, test, proxy, or paper-facing?
- What result would falsify the hypothesis?

## Agent Roles

- Who creates hypotheses?
- Who implements hypotheses?
- Who can spend GPU time?
- Who reviews results?
- Who can merge?
- Who can close work?
- How do agents ask humans questions?
- How do humans steer agents without hidden manual tuning?

## Tool Contracts

- Which actions are exposed as raw shell?
- Which actions are wrapped in idempotent helpers?
- Do wrappers enforce workflow invariants?
- Do tools return structured outputs?
- Do tools fail loudly on unsafe state?
- Are known CLI footguns hidden from agents?
- Are waits, monitors, and wakeups first-class patterns?

## State And Ledger

- What is the authoritative hypothesis record?
- What is the authoritative code record?
- What is the authoritative metric record?
- What is the authoritative workflow state?
- What is the agent/tool trace record?
- Which summaries are caches rather than truth?
- How are PR, W&B, Weave, local logs, and pod state reconciled?

## W&B And Weave

- Does every training run log config, command, git commit, and primary metrics?
- Are final metrics finite and present?
- Are histories summarized before entering model context?
- Are system metrics available for GPU utilization and memory?
- Are agent traces captured in Weave?
- Is Weave runtime registration tested?
- Can a human follow PR -> W&B run -> Weave trace -> decision?

## Fleet Runtime

- How are advisor and student roles deployed?
- Which pods require GPUs?
- Where does data live?
- Are PVC mounts explicit?
- Are secrets separated by purpose?
- What happens if a pod restarts?
- What happens if a student has no work?
- What happens if a student has duplicate work?
- What happens if training outlives the agent session?

## Monitoring And Cost

- What events wake the model?
- Are raw logs reduced before they hit context?
- Are monitor events rate-limited or milestone-gated?
- Is there a maximum event count?
- Are unchanged statuses suppressed?
- Is polling done outside the LLM when no reasoning is needed?
- Can monitoring cost be attributed to sessions and tasks?

## Physical-AI Claim Discipline

- Are benchmark splits named precisely?
- Are validation and test metrics labeled correctly?
- Are proxy metrics separated from paper-facing metrics?
- Are normalization and aggregation rules documented?
- Are physical target fields reported separately when needed?
- Are comparison baselines truly apples-to-apples?
- Is runnable distinguished from scientifically defensible?

## Launch Readiness

Do not launch a real autoresearch fleet until the answer is yes:

- The target repo is separate from the runner repo.
- The branch and label routing scheme is unique to this launch.
- The W&B project is ready.
- Weave tracing is verified or intentionally disabled.
- Credentials are preflighted.
- The dry-run manifest has been reviewed.
- Teardown command is known.
- The first experiment can run as a bounded smoke test.
- Humans know how to issue steering through the approved channel.

Closing rule:

**Every autonomous action should leave a durable artifact that a human and an agent can inspect later.**
