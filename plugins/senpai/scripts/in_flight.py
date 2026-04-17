#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Track and harvest backgrounded experiment batches across student sessions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("WORKDIR", Path(__file__).resolve().parents[3]))
IN_FLIGHT_DIR = ROOT / ".senpai" / "in_flight"
ACTIVE_STATES = {"pending", "queued", "running", "starting"}
TERMINAL_STATES = {"crashed", "failed", "finished", "killed", "preempted"}


def note(message: str) -> None:
    """Write a harness status line to stderr."""
    print(message, file=sys.stderr)


def sh(*args: str, cwd: Path = ROOT) -> str:
    """Run a command and return stripped stdout, raising on failure."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"{' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def senpai_gh(*args: str) -> str:
    """Invoke a function from senpai-gh.sh and return its stdout."""
    return sh(
        "bash",
        "-lc",
        'source "$1/plugins/senpai/scripts/senpai-gh.sh" && shift && "$@"',
        "bash",
        str(ROOT),
        *args,
    )


def now_iso() -> str:
    """Return the current UTC timestamp in compact ISO-8601 form."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def session_id() -> str:
    """Return the current session identifier if one is available."""
    return (
        os.environ.get("CLAUDE_SESSION_ID")
        or os.environ.get("CODEX_SESSION_ID")
        or os.environ.get("CLAUDE_RUN_ID")
        or "unknown"
    )


def new_tag() -> str:
    """Generate a unique harness W&B tag for the current PR."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()
    return f"senpai-inflight-pr{int(senpai_gh('current_pr_number'))}-{stamp}-{uuid.uuid4().hex[:8]}"


def iter_entries():
    """Yield parsed in-flight tracking entries from disk."""
    if not IN_FLIGHT_DIR.exists():
        return
    for path in sorted(IN_FLIGHT_DIR.glob("*.json")):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            note(f"HARNESS: skipping malformed in-flight entry {path.name}: {exc}")


def record_entry(wandb_tag: str, expected_runs: int) -> None:
    """Persist an in-flight background batch for a later session to harvest."""
    if not wandb_tag:
        raise ValueError("wandb_tag must be non-empty")
    if expected_runs < 1:
        raise ValueError("expected_runs must be >= 1")

    senpai_gh("require_clean_tracked_worktree")
    senpai_gh("require_pushed_head")
    IN_FLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = IN_FLIGHT_DIR / f"{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {
                "pr": int(senpai_gh("current_pr_number")),
                "wandb_entity": os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                "wandb_project": os.environ.get("WANDB_PROJECT", "senpai-v1"),
                "wandb_tag": wandb_tag,
                "expected_runs": expected_runs,
                "launched_at": now_iso(),
                "launched_by_session": session_id(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    note(f"HARNESS: tracked in-flight background experiment tag={wandb_tag} expected_runs={expected_runs}")


def clear_entries(pr: int) -> None:
    """Delete all in-flight tracking entries for a PR."""
    for path, entry in iter_entries() or ():
        if entry.get("pr") == pr:
            path.unlink(missing_ok=True)
            note(f"HARNESS: cleared in-flight entry for PR #{pr}: {path.name}")


def load_runs(entry: dict):
    """Fetch W&B runs matching the tracked harness tag."""
    import wandb

    runs = list(
        wandb.Api().runs(
            f"{entry['wandb_entity']}/{entry['wandb_project']}",
            filters={"tags": {"$in": [entry["wandb_tag"]]}},
        )
    )
    return sorted(runs, key=lambda run: ((getattr(run, "name", None) or ""), getattr(run, "id", "")))


def build_comment(entry: dict, runs: list) -> str:
    """Build the generic harness-authored PR comment for harvested runs."""
    lines = [
        "HARNESS: Senpai in-flight harvester",
        "",
        "This comment was posted programmatically by the senpai harness.",
        "No LLM model authored this summary.",
        "",
        f"- Tracked W&B tag: `{entry['wandb_tag']}`",
        f"- Expected runs: {entry['expected_runs']}",
        f"- Launch time: {entry['launched_at']}",
        f"- Launch session: `{entry['launched_by_session']}`",
        "",
        "The senpai harness is not interpreting experiment-specific metrics or configs here.",
        "Review the linked W&B runs directly for task-specific results.",
        "",
        "The senpai harness is marking this PR ready for advisor review programmatically now.",
        "",
        "| Run | State |",
        "|-----|-------|",
    ]
    for run in runs:
        lines.append(
            f"| [{getattr(run, 'name', getattr(run, 'id', 'run'))}]({run.url})"
            f" | {getattr(run, 'state', 'unknown')}"
            " |"
        )
    return "\n".join(lines)


def post_comment(pr: int, body: str) -> None:
    """Post a markdown PR comment via the shared senpai-gh retry path."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        senpai_gh("comment_pr_with_file", str(pr), path)
    finally:
        Path(path).unlink(missing_ok=True)


def mark_ready(pr: int) -> None:
    """Mark a PR ready for advisor review via the shared senpai-gh helper."""
    senpai_gh("mark_ready_for_review", str(pr))


def harvest() -> None:
    """Harvest any completed in-flight entries and finish their PR handoff."""
    for path, entry in iter_entries() or ():
        try:
            runs = load_runs(entry)
            expected_runs = int(entry["expected_runs"])
            if len(runs) < expected_runs:
                note(
                    f"HARNESS: waiting on PR #{entry['pr']} tag={entry['wandb_tag']} "
                    f"({len(runs)}/{expected_runs} runs observed)"
                )
                continue

            states = {getattr(run, "state", "unknown") or "unknown" for run in runs}
            if states & ACTIVE_STATES:
                note(f"HARNESS: PR #{entry['pr']} tag={entry['wandb_tag']} still running: {sorted(states)}")
                continue
            if not states.issubset(TERMINAL_STATES):
                note(f"HARNESS: PR #{entry['pr']} tag={entry['wandb_tag']} has unknown run states {sorted(states)}")
                continue

            post_comment(entry["pr"], build_comment(entry, runs))
            mark_ready(entry["pr"])
            path.unlink(missing_ok=True)
            note(f"HARNESS: harvested PR #{entry['pr']} tag={entry['wandb_tag']}")
        except Exception as exc:  # noqa: BLE001
            note(f"HARNESS: failed to harvest {path.name}: {exc}")


def main() -> int:
    """Parse CLI arguments and run the requested in-flight action."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("new-tag", help="Print a fresh harness W&B tag for the current PR")
    record = subparsers.add_parser("record", help="Record an in-flight backgrounded batch")
    record.add_argument("--wandb-tag", required=True)
    record.add_argument("--expected-runs", required=True, type=int)
    clear = subparsers.add_parser("clear", help="Delete in-flight entries for a PR")
    clear.add_argument("--pr", required=True, type=int)
    subparsers.add_parser("harvest", help="Post harvested results for completed backgrounded batches")
    args = parser.parse_args()

    if args.command == "new-tag":
        print(new_tag())
    elif args.command == "record":
        record_entry(args.wandb_tag, args.expected_runs)
    elif args.command == "clear":
        clear_entries(args.pr)
    else:
        harvest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
