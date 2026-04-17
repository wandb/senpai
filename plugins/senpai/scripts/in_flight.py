#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Track and harvest backgrounded experiment batches across student sessions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = Path(os.environ.get("WORKDIR", SCRIPT_PATH.parents[3]))
IN_FLIGHT_REL = Path(".senpai") / "in_flight"
ACTIVE_STATES = {"pending", "queued", "running", "starting"}
TERMINAL_STATES = {"crashed", "failed", "finished", "killed", "preempted"}
SESSION_ENV_VARS = ("CLAUDE_SESSION_ID", "CLAUDE_RUN_ID", "CODEX_SESSION_ID")


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def utcnow_iso() -> str:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def repo_root(value: str | None) -> Path:
    return Path(value).resolve() if value else DEFAULT_REPO_ROOT


def in_flight_dir(root: Path) -> Path:
    return root / IN_FLIGHT_REL


def run_capture(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"{' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def current_branch(root: Path) -> str:
    branch = run_capture(["git", "branch", "--show-current"], cwd=root)
    if not branch:
        raise RuntimeError("unable to determine current git branch")
    return branch


def current_pr_number(root: Path) -> int:
    pr = run_capture(["gh", "pr", "view", "--json", "number", "--jq", ".number"], cwd=root)
    if not pr:
        raise RuntimeError("no open PR for current branch")
    return int(pr)


def current_head(root: Path) -> str:
    return run_capture(["git", "rev-parse", "HEAD"], cwd=root)


def ensure_recordable_launch_state(root: Path) -> str:
    tracked_changes = run_capture(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=root,
    )
    if tracked_changes:
        raise RuntimeError(
            "backgrounded harness tracking requires a clean tracked worktree; "
            "commit your experiment code first"
        )

    try:
        upstream_head = run_capture(["git", "rev-parse", "@{u}"], cwd=root)
    except RuntimeError as exc:
        raise RuntimeError(
            "backgrounded harness tracking requires the PR branch to be pushed first"
        ) from exc

    head = current_head(root)
    if head != upstream_head:
        raise RuntimeError(
            "backgrounded harness tracking requires the current HEAD commit to already be pushed; "
            "run `git push origin $(git branch --show-current)` first"
        )
    return head


def session_id() -> str:
    for name in SESSION_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return "unknown"


def new_tag(root: Path) -> str:
    pr = current_pr_number(root)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()
    suffix = uuid.uuid4().hex[:8]
    return f"senpai-inflight-pr{pr}-{stamp}-{suffix}"


def record_entry(root: Path, wandb_tag: str, expected_runs: int) -> Path:
    if not wandb_tag:
        raise ValueError("wandb_tag must be non-empty")
    if expected_runs < 1:
        raise ValueError("expected_runs must be >= 1")

    git_commit = ensure_recordable_launch_state(root)
    entry = {
        "schema": 1,
        "pr": current_pr_number(root),
        "branch": current_branch(root),
        "git_commit": git_commit,
        "wandb_entity": os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        "wandb_project": os.environ.get("WANDB_PROJECT", "senpai-v1"),
        "wandb_tag": wandb_tag,
        "expected_runs": expected_runs,
        "launched_at": utcnow_iso(),
        "launched_by_session": session_id(),
    }
    out_dir = in_flight_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    stderr(
        f"HARNESS: tracked in-flight background experiment for PR #{entry['pr']} "
        f"tag={wandb_tag} expected_runs={expected_runs} -> {path}"
    )
    return path


def clear_entries(root: Path, pr: int) -> int:
    count = 0
    for path, entry in load_entries(root):
        if entry.get("pr") != pr:
            continue
        path.unlink(missing_ok=True)
        count += 1
        stderr(f"HARNESS: cleared in-flight entry for PR #{pr}: {path.name}")
    return count


def load_entries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    directory = in_flight_dir(root)
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            stderr(f"HARNESS: skipping malformed in-flight entry {path.name}: {exc}")
            continue
        out.append((path, entry))
    return out


def load_runs(entry: dict[str, Any]) -> list[Any]:
    import wandb

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{entry['wandb_entity']}/{entry['wandb_project']}",
            filters={"tags": {"$in": [entry["wandb_tag"]]}},
        )
    )
    runs.sort(key=lambda run: ((getattr(run, "name", None) or ""), getattr(run, "id", "")))
    return runs


def format_summary_value(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_comment(entry: dict[str, Any], runs: list[Any]) -> str:
    lines = [
        "HARNESS: Senpai in-flight harvester",
        "",
        "This comment was posted programmatically by the senpai harness.",
        "No LLM model authored this summary.",
        "",
        "The original launching session ended before it could report these backgrounded runs.",
        "",
        f"- Tracked W&B tag: `{entry['wandb_tag']}`",
        f"- Expected runs: {entry['expected_runs']}",
        f"- Code commit: `{entry['git_commit']}`",
        f"- Launch time: {entry['launched_at']}",
        f"- Launch session: `{entry['launched_by_session']}`",
        "",
        "The senpai harness is marking this PR ready for advisor review programmatically now.",
        "",
        "### Harvested runs",
        "",
        "| Run | State | Seed | Steps | val/loss |",
        "|-----|-------|------|------:|---------:|",
    ]
    for run in runs:
        summary = getattr(run, "summary", {}) or {}
        config = getattr(run, "config", {}) or {}
        lines.append(
            "| "
            f"[{getattr(run, 'name', getattr(run, 'id', 'run'))}]({run.url}) | "
            f"{getattr(run, 'state', 'unknown')} | "
            f"{format_summary_value(config.get('seed'), digits=0)} | "
            f"{format_summary_value(summary.get('_step'), digits=0)} | "
            f"{format_summary_value(summary.get('val/loss'))} |"
        )
    return "\n".join(lines)


def post_pr_comment(root: Path, pr: int, body: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        tmp_path = Path(handle.name)
    try:
        run_capture(
            ["gh", "pr", "comment", str(pr), "--body-file", str(tmp_path)],
            cwd=root,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def mark_ready_for_review(root: Path, pr: int) -> None:
    command = (
        'source "$1/plugins/senpai/scripts/senpai-gh.sh" '
        '&& mark_ready_for_review "$2"'
    )
    run_capture(
        ["bash", "-lc", command, "bash", str(root), str(pr)],
        cwd=root,
    )


def harvest(root: Path) -> int:
    harvested = 0
    entries = load_entries(root)
    if not entries:
        return harvested

    for path, entry in entries:
        try:
            runs = load_runs(entry)
            if len(runs) < int(entry["expected_runs"]):
                stderr(
                    f"HARNESS: waiting on PR #{entry['pr']} tag={entry['wandb_tag']} "
                    f"({len(runs)}/{entry['expected_runs']} runs observed)"
                )
                continue

            states = {getattr(run, "state", "unknown") or "unknown" for run in runs}
            if states & ACTIVE_STATES:
                stderr(f"HARNESS: PR #{entry['pr']} tag={entry['wandb_tag']} still running: {sorted(states)}")
                continue
            if not states.issubset(TERMINAL_STATES):
                stderr(
                    f"HARNESS: PR #{entry['pr']} tag={entry['wandb_tag']} has unknown run states "
                    f"{sorted(states)}; leaving entry in place"
                )
                continue

            body = build_comment(entry, runs)
            post_pr_comment(root, int(entry["pr"]), body)
            mark_ready_for_review(root, int(entry["pr"]))
            path.unlink(missing_ok=True)
            harvested += 1
            stderr(
                f"HARNESS: harvested PR #{entry['pr']} tag={entry['wandb_tag']} "
                f"with {len(runs)} terminal runs"
            )
        except Exception as exc:  # noqa: BLE001
            stderr(f"HARNESS: failed to harvest {path.name}: {exc}")
    return harvested


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None, help="Override the repo root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("new-tag", help="Print a fresh harness W&B tag for the current PR")

    record_parser = subparsers.add_parser("record", help="Record an in-flight backgrounded batch")
    record_parser.add_argument("--wandb-tag", required=True, help="Harness tracking tag shared by the batch")
    record_parser.add_argument("--expected-runs", required=True, type=int, help="How many runs must finish before harvest")

    clear_parser = subparsers.add_parser("clear", help="Delete in-flight entries for a PR")
    clear_parser.add_argument("--pr", required=True, type=int, help="PR number to clear")

    subparsers.add_parser("harvest", help="Post harvested results for completed backgrounded batches")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = repo_root(args.repo_root)

    if args.command == "new-tag":
        print(new_tag(root))
        return 0
    if args.command == "record":
        record_entry(root, args.wandb_tag, args.expected_runs)
        return 0
    if args.command == "clear":
        clear_entries(root, args.pr)
        return 0
    if args.command == "harvest":
        harvest(root)
        return 0
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
