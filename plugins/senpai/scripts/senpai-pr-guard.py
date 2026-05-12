#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import json
import re
import sys


HOLD_RE = re.compile(r"SENPAI-HOLD|hold the merge|do not merge|don't merge|hold .*until", re.I)


def stamp(comment):
    return comment.get("createdAt") or comment.get("submittedAt") or comment.get("updatedAt") or ""


def result_markers(comments, errors):
    markers = []
    for comment in comments:
        for line in (comment.get("body") or "").splitlines():
            if "SENPAI-RESULT:" not in line:
                continue
            try:
                markers.append((stamp(comment), json.loads(line.split("SENPAI-RESULT:", 1)[1].strip())))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid SENPAI-RESULT JSON at {stamp(comment) or 'unknown time'}: {exc}")
    return sorted(markers)


def terminal_result_error(markers):
    if not markers:
        return "No terminal SENPAI-RESULT marker found. The student must post final structured results before merge."
    _, result = markers[-1]
    if not result.get("terminal"):
        return "Latest SENPAI-RESULT is not terminal=true."
    if result.get("pending_arms") or result.get("pending_runs"):
        return "Latest SENPAI-RESULT still reports pending arms/runs."
    return None


def refuse(num, errors):
    print(f"SENPAI-MERGE-REFUSED: PR #{num} is not safe to merge.", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    print("Next steps:", file=sys.stderr)
    print("- If work is still running or held, leave the PR in WIP/hold and wait.", file=sys.stderr)
    print("- If work is done, ask the student for a terminal SENPAI-RESULT marker and resubmit.", file=sys.stderr)
    raise SystemExit(1)


def require_terminal_result(num, comments):
    errors = []
    markers = result_markers(comments, errors)
    error = terminal_result_error(markers)
    if error:
        errors.append(error)
    if errors:
        print(f"SENPAI-RESULT-REQUIRED: PR #{num} cannot be marked ready.", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print('Add one line like: SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path-to-jsonl-or-summary>"]}', file=sys.stderr)
        raise SystemExit(1)
    print(f"SENPAI-RESULT: terminal marker found on PR #{num} at {markers[-1][0] or 'unknown time'}")


def merge_preflight(num, advisor_branch, pr, comments):
    labels = {label["name"] for label in pr["labels"]}
    errors = []

    if pr["state"] != "OPEN":
        errors.append(f"PR state is {pr['state']}, expected OPEN.")
    if pr["isDraft"]:
        errors.append("PR is still draft.")
    if "status:review" not in labels:
        errors.append("PR is missing status:review.")
    if "status:wip" in labels:
        errors.append("PR still has status:wip; active assignments must not be merged.")
    if advisor_branch and advisor_branch not in labels:
        errors.append(f"PR is missing advisor branch label {advisor_branch}.")
    if not any(label.startswith("student:") for label in labels):
        errors.append("PR is missing a student:<name> label.")
    for label in ("status:hold", "status:blocked", "status:needs-rebase"):
        if label in labels:
            errors.append(f"PR has {label}.")
    if pr["mergeStateStatus"] == "DIRTY" or pr["mergeable"] == "CONFLICTING":
        errors.append("GitHub reports merge conflicts; send the PR back for rebase.")

    markers = result_markers(comments, errors)
    error = terminal_result_error(markers)
    if error:
        errors.append(error)

    holds = sorted(
        (stamp(comment), (comment.get("body") or "").splitlines()[0][:160])
        for comment in comments
        if HOLD_RE.search(comment.get("body") or "")
    )
    if holds and (not markers or holds[-1][0] > markers[-1][0]):
        errors.append(f"Latest hold comment is newer than the terminal result ({holds[-1][0] or 'unknown time'}: {holds[-1][1]!r}).")

    if errors:
        refuse(num, errors)
    print(f"SENPAI-MERGE-PREFLIGHT: PR #{num} passed merge preflight.")


def main():
    mode = sys.argv[1]
    if mode == "require-terminal-result":
        require_terminal_result(sys.argv[2], json.load(sys.stdin))
        return

    raw_pr, raw_comments = sys.stdin.buffer.read().split(b"\0", 1)
    merge_preflight(sys.argv[2], sys.argv[3], json.loads(raw_pr), json.loads(raw_comments))


if __name__ == "__main__":
    main()
