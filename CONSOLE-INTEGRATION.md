# SENPAI Console integration (Phase 2 change-set)

The [SENPAI Console](https://github.com/wandb/senpai-console) is a separate repo that
monitors, guides, and steers this fleet. This change-set is the small, coordinated
set of changes that must land **here** so the console can index the fleet reliably
and steer agents near-live. Everything here is **additive and inert by default** —
it does not change current behaviour unless the console wires up the new env vars.

## Runtime reality (important)

The console build plan describes an OpenHands "agent-server" (Surface B) for live
watch+steer. **This fleet runs Claude Code** (`k8s/run-senpai-claude.sh`), not
OpenHands, so Phase 2 is implemented in terms of the Claude-Code runtime:

| Build-plan concept | This fleet's mechanism |
| --- | --- |
| Live event stream (`/sockets/events/...`) | `SENPAI_EVENT_DIR` — Claude Code's `--output-format stream-json` is mirrored to a stable per-iteration event file the console tails. |
| `send_message()` mid-run steer | `SENPAI_CONSOLE_INBOX_DIR` — the console drops a directive file; the entrypoint drains it into the **next heartbeat's** prompt (near-live). |
| `pause` / `interrupt` | Not natively available under `claude -p`; the existing watchdogs + GitHub-mediated controls remain the mechanism. |

## What changed

1. **Markers (P0-S2).** `system_instructions/CLAUDE-ADVISOR.md` now instructs the
   advisor to write `SENPAI-EXP` (lineage/queue/taste, in each assignment PR body)
   and `SENPAI-ADVISOR` (heartbeat status). `CLAUDE-STUDENT.md` notes the experiment
   id comes from the PR's `SENPAI-EXP`. These give the console durable lineage/queue/
   status without scraping. Keep them single-line valid JSON — the console degrades
   loudly (logs) on malformed markers, never silently.

2. **Supervisor role.** `system_instructions/CLAUDE-SUPERVISOR.md` — the console's
   control agent (sensors, alerts, charts, code-explain, steering; `act_safe`
   default). It runs in the **console backend** today (`backend/senpai_console/
   supervisor/`); an in-fleet pod is a future option that would need a dedicated
   `entrypoint-supervisor.sh`.

3. **Live event mirroring (`SENPAI_EVENT_DIR`).** `k8s/run-senpai-claude.sh` mirrors
   the stream-json into `$SENPAI_EVENT_DIR/event-*.jsonl` when the var is set, so the
   console ingests a stable event log instead of scraping iteration logs.

4. **Console → agent inbox (`SENPAI_CONSOLE_INBOX_DIR`).** `k8s/senpai-console-inbox.sh`
   provides `senpai_drain_inbox`; the advisor/student entrypoints drain pending
   directive files into that iteration's prompt (under a `# Console directives`
   header) and archive them. This is how a Supervisor steer or a "researcher edited
   `<doc>`" ping reaches the agent near-live. An inbox directive also wakes the
   advisor even when nothing else is actionable.

5. **Launch plumbing.** `k8s/launch.py` gains `--event_dir` and `--console_inbox_dir`,
   passed through to both advisor and student ConfigMaps (default empty = disabled).

## Enable it

```bash
python k8s/launch.py --tag <tag> --target_repo_url <url> --advisor true \
  --event_dir /mnt/new-pvc/senpai-events/<tag> \
  --console_inbox_dir /mnt/new-pvc/senpai-inbox/<tag>
```

Point the console's `OH_STATE_DIR` at the event dir and its inbox writer at the
inbox dir (both on the shared PVC the console can also mount). Leave them unset to
keep the fleet exactly as it is today.

## Tests

- `bash k8s/test-senpai-console-inbox.sh` — unit-tests `senpai_drain_inbox`
  (inert-when-unset, drains + archives, second drain empty).
- `bash -n` passes on all modified scripts; `python k8s/launch.py --dry_run true …`
  renders the new keys into every ConfigMap.

**Not cluster-tested:** the entrypoint wiring and event mirroring are additive and
inert by default, but have not been run on a live cluster — verify on a scratch tag
before relying on them.
