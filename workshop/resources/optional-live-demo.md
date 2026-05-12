# Optional Live Demo Path

The default workshop is live-service-first for reads, but safe for mutations. This guide gives facilitator-friendly demo options.

## Mode A: Slidev + Offline Fixtures

Use when network access is uncertain.

Open:

- `workshop/slides.md`
- `workshop/resources/artifact-pack/fake-pr-body.md`
- `workshop/resources/artifact-pack/wandb-run-summary.json`
- `workshop/resources/artifact-pack/weave-trace-excerpt.json`
- `workshop/resources/artifact-pack/dry-run-launch-redacted.yaml`

Message:

> The artifact trail is the teaching object: hypothesis, code, metric, trace, decision.

## Mode B: Live Read-Only Notebooks

Use after credentials validate:

```bash
uv run python workshop/setup_credentials.py --check-only
uv run python workshop/notebooks/00_environment_check.py
uv run python workshop/notebooks/01_llm_calls_to_hypotheses.py
uv run python workshop/notebooks/02_tool_calls_and_contracts.py
```

Message:

> Live reads give the model current evidence; mutations remain gated.

## Mode C: K8s Dry-Run

Use the dry-run notebook:

```bash
uv run python workshop/notebooks/06_k8s_autoresearch_dry_run.py
```

Or show the direct command:

```bash
python k8s/launch.py \
  --tag workshop \
  --target_repo_url "$TARGET_REPO_URL" \
  --target_repo_branch "$TARGET_REPO_BRANCH" \
  --advisor_branch "$ADVISOR_BRANCH" \
  --n_students 2 \
  --student_prefix ws \
  --gpus_per_student 1 \
  --advisor \
  --dry_run
```

Message:

> Dry-run is the correct live demo for deployment mechanics: it exercises the launch contract without creating secrets, deployments, or GPU workloads.

## Risk Register

| Risk | Fallback |
| --- | --- |
| Credentials unavailable | Use Mode A |
| W&B project inaccessible | Use artifact pack and explain expected live shape |
| Node/Slidev unavailable | Present from `slides.md` in editor |
| Network unreliable | Use offline fixtures |
| Terminal output too long | Use `workshop/resources/artifact-pack/dry-run-launch-redacted.yaml` |
| Audience asks for real mutation | Explain mutation gates and defer to follow-up ops session |

## Facilitator Rule

Never run non-dry-run Kubernetes launch commands or GitHub mutations during the public workshop. The workshop teaches the control system first; real fleet launch belongs in a prepared follow-up session.
