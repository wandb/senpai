#!/bin/bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# senpai-gh.sh — GitHub CLI primitives for the senpai research workflow.
#
# Source this file; don't execute it directly.
#
#   source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
#
# Every `gh` call here targets the problem-package repo via the GH_REPO
# env var (e.g. "morganmcg1/tandemfoil2"), which the pod sets at startup.
# No --repo flag needed, and cwd doesn't matter.
#
# WHY THIS EXISTS:
# The GitHub CLI's `gh pr edit --remove-label X --add-label Y` silently
# strips *all other labels* from the PR. The REST API's individual
# DELETE + POST calls are the only safe way to swap a single label.
# This library wraps that pattern so nobody has to remember (or get
# bitten by) the footgun.

# ---------------------------------------------------------------------------
# Retry helper: up to 6 attempts with 15s backoff, then fail loudly.
# ---------------------------------------------------------------------------
gh_retry() {
    local attempt
    for attempt in 1 2 3 4 5 6; do
        "$@" && return 0
        echo "gh_retry: attempt $attempt failed, retrying in 15s..." >&2
        sleep 15
    done
    return 1
}

# ---------------------------------------------------------------------------
# Label operations
# ---------------------------------------------------------------------------

# Atomically swap one label for another on a PR/issue.
#   swap_gh_pr_label <number> <remove_label> <add_label>
# Uses $GH_REPO for the repo slug (set by the pod entrypoint).
swap_gh_pr_label() {
    local num="$1" remove="$2" add="$3"

    # DELETE the old label — retry transient failures, tolerate 404 (already gone).
    local attempt err
    for attempt in 1 2 3 4 5 6; do
        err=$(gh api "repos/${GH_REPO}/issues/${num}/labels/${remove}" \
            --method DELETE --silent 2>&1) && break
        echo "$err" | grep -q "404" && break
        echo "swap_gh_pr_label: DELETE attempt $attempt failed, retrying in 15s..." >&2
        [ "$attempt" -eq 6 ] && return 1
        sleep 15
    done

    # POST the new label (gh_retry gives 6 attempts on transient failure).
    gh_retry gh api "repos/${GH_REPO}/issues/${num}/labels" \
        -f "labels[]=${add}" --method POST --silent
}

# ---------------------------------------------------------------------------
# Compound actions — advisor
# ---------------------------------------------------------------------------

# Send a PR back to its student: comment, convert to draft, swap review→wip.
#   send_pr_back_to_student_with_comment <number> <comment_body>
send_pr_back_to_student_with_comment() {
    local num="$1" body="$2"
    gh_retry gh pr comment "$num" --body "$body"
    gh_retry gh pr ready "$num" --undo
    swap_gh_pr_label "$num" "status:review" "status:wip"
}

# Close a dead-end PR: comment explaining why, close, delete remote branch.
#   close_pr_with_comment <number> <reason>
close_pr_with_comment() {
    local num="$1" reason="$2"
    gh_retry gh pr comment "$num" --body "ADVISOR: Closing PR #${num} because ${reason}."
    gh_retry gh pr close "$num" --delete-branch
}

# Create an assignment PR through the one path that verifies student routing.
#
# Assignment pickup is label-based: student pods only see work when the PR has
# the advisor branch label, student:<name>, and status:wip. Raw gh pr create
# calls can leave behind an unroutable PR if any of that metadata is omitted.
#
# On success, this prints the created PR URL after confirming the PR is a draft,
# targets the requested base/head, and has every required routing label.
# On failure, it prints the specific missing or mismatched invariant to stderr
# and returns nonzero so the advisor can fix the assignment before students idle.
#   create_assignment_pr_from_file <student> <head-branch> <title> <body-file> [base-branch]
create_assignment_pr_from_file() {
    local student="$1" head_branch="$2" title="$3" body_file="$4" base_branch="${5:-${ADVISOR_BRANCH:-}}"
    local pr_url num details

    if [ -z "$student" ] || [ -z "$head_branch" ] || [ -z "$title" ] || [ -z "$body_file" ] || [ -z "$base_branch" ]; then
        echo "create_assignment_pr_from_file: usage: <student> <head-branch> <title> <body-file> [base-branch]" >&2
        return 2
    fi
    if [ ! -f "$body_file" ]; then
        echo "create_assignment_pr_from_file: body file not found: $body_file" >&2
        return 2
    fi

    pr_url=$(gh_retry gh pr create --draft \
        --title "$title" \
        --body-file "$body_file" \
        --label "$base_branch" \
        --label "student:$student" \
        --label "status:wip" \
        --base "$base_branch" \
        --head "$head_branch")

    num=$(printf '%s' "$pr_url" | sed -n 's#.*/pull/\([0-9][0-9]*\).*#\1#p')
    if [ -z "$num" ]; then
        echo "create_assignment_pr_from_file: could not parse PR number from: $pr_url" >&2
        return 1
    fi

    details=$(gh_retry gh pr view "$num" --json number,baseRefName,headRefName,labels,isDraft)
    python3 - "$student" "$base_branch" "$head_branch" "$details" <<'PY' || return
import json
import sys

student, base_branch, head_branch, payload = sys.argv[1:5]
pr = json.loads(payload)
labels = {label.get("name", "") for label in pr.get("labels", [])}

errors = []
if pr.get("baseRefName") != base_branch:
    errors.append(f"expected base {base_branch}, got {pr.get('baseRefName')}")
if pr.get("headRefName") != head_branch:
    errors.append(f"expected head {head_branch}, got {pr.get('headRefName')}")
if not pr.get("isDraft"):
    errors.append("expected draft PR")
for label in (base_branch, f"student:{student}", "status:wip"):
    if label not in labels:
        errors.append(f"missing label {label}")

if errors:
    for error in errors:
        print(f"create_assignment_pr_from_file: PR #{pr.get('number')} {error}", file=sys.stderr)
    raise SystemExit(1)
PY

    printf '%s\n' "$pr_url"
}

# ---------------------------------------------------------------------------
# Compound actions — student
# ---------------------------------------------------------------------------

# Mark a PR as ready for advisor review: mark ready + swap wip→review.
#   mark_ready_for_review <number>
mark_ready_for_review() {
    local num="$1"
    gh_retry gh pr ready "$num"
    swap_gh_pr_label "$num" "status:wip" "status:review"  # swap_gh_pr_label uses gh_retry internally
}

# Create a real assignment branch before opening a PR.
#   create_assignment_branch <student> <hypothesis-slug>
create_assignment_branch() {
    local student="$1" slug="$2" branch="${student}/${slug}"
    git checkout "$ADVISOR_BRANCH"
    git pull origin "$ADVISOR_BRANCH"
    git checkout -b "$branch"
    git commit --allow-empty -m "assign ${student}: ${slug}"
    git push -u origin "$branch"
    git rev-list --count "${ADVISOR_BRANCH}..HEAD" | grep -vq '^0$'
}

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

json_len() { python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))"; }
json_join() { python3 -c "import sys,json; print(','.join(json.loads(sys.stdin.read())))"; }
json_numbers() { python3 -c "import sys,json; print(','.join(f'#{i[\"number\"]}' for i in json.loads(sys.stdin.read())))"; }

# Print the maximum updatedAt timestamp from one or more JSON arrays.
# Returns empty string if all arrays are empty.  Usage:
#   max_updated_at "$JSON_BLOB1" "$JSON_BLOB2" ...
max_updated_at() {
    python3 -c "
import json, sys
items = [i for blob in sys.argv[1:] if blob for i in json.loads(blob)]
ts = [i['updatedAt'] for i in items if 'updatedAt' in i]
print(max(ts) if ts else '')
" "$@"
}

# Merge the JSON values emitted by `gh api --paginate --jq '[...]' into one
# array. `gh` applies --jq once per page, so a multi-page call emits multiple
# top-level arrays.
json_merge_arrays_from_stream() {
    python3 -c '
import json
import sys

text = sys.stdin.read()
decoder = json.JSONDecoder()
idx = 0
items = []
while idx < len(text):
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        break
    value, idx = decoder.raw_decode(text, idx)
    if isinstance(value, list):
        items.extend(value)
    else:
        items.append(value)
print(json.dumps(items))
'
}

gh_api_paginated_array() {
    local path="$1" jq_expr="$2" raw
    raw=$(gh_retry gh api --paginate "$path" --jq "$jq_expr") || return
    printf '%s\n' "$raw" | json_merge_arrays_from_stream
}

# ---------------------------------------------------------------------------
# PR reads
# ---------------------------------------------------------------------------

# REST-backed PR reads avoid `gh pr view --comments`, whose GraphQL query can
# require org scopes that repo-only launch tokens do not have.
pr_body() {
    local num="$1"
    gh_retry gh api "repos/${GH_REPO}/pulls/${num}" \
        --jq '{number,title,headRefName:.head.ref,baseRefName:.base.ref,isDraft:.draft,body}'
}

pr_issue_comments() {
    local num="$1"
    gh_api_paginated_array "repos/${GH_REPO}/issues/${num}/comments?per_page=100" \
        '[.[] | {kind:"issue",author:.user.login,createdAt:.created_at,updatedAt:.updated_at,body}]'
}

pr_reviews() {
    local num="$1"
    gh_api_paginated_array "repos/${GH_REPO}/pulls/${num}/reviews?per_page=100" \
        '[.[] | {kind:"review",author:.user.login,state,submittedAt:.submitted_at,body}]'
}

pr_review_comments() {
    local num="$1"
    gh_api_paginated_array "repos/${GH_REPO}/pulls/${num}/comments?per_page=100" \
        '[.[] | {kind:"inline",author:.user.login,path,line,createdAt:.created_at,updatedAt:.updated_at,body}]'
}

pr_all_comments() {
    local num="$1" issues reviews inline
    issues=$(pr_issue_comments "$num") || return
    reviews=$(pr_reviews "$num") || return
    inline=$(pr_review_comments "$num") || return
    printf '%s\0%s\0%s' "$issues" "$reviews" "$inline" | python3 -c '
import json
import sys

blobs = sys.stdin.buffer.read().split(b"\0")
print(json.dumps([item for blob in blobs if blob for item in json.loads(blob)]))
'
}

issue_body() {
    local num="$1"
    gh_retry gh api "repos/${GH_REPO}/issues/${num}" \
        --jq '{number,title,state,author:.user.login,createdAt:.created_at,updatedAt:.updated_at,body}'
}

issue_comments() {
    local num="$1"
    gh_api_paginated_array "repos/${GH_REPO}/issues/${num}/comments?per_page=100" \
        '[.[] | {kind:"issue",author:.user.login,createdAt:.created_at,updatedAt:.updated_at,body}]'
}

issue_with_comments() {
    local num="$1" issue comments
    issue=$(issue_body "$num") || return
    comments=$(issue_comments "$num") || return
    printf '%s\0%s' "$issue" "$comments" | python3 -c '
import json
import sys

issue, comments = sys.stdin.buffer.read().split(b"\0", 1)
print(json.dumps({"issue": json.loads(issue), "comments": json.loads(comments)}))
'
}

# Summarize recent training logs after a sparse wakeup. This is deliberately
# not a streaming watcher: per-epoch Monitor callbacks reload too much context.
training_log_status() {
    SENPAI_LOG_STATUS_LINES="${SENPAI_LOG_STATUS_LINES:-2000}" python3 - "$@" <<'PY'
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

ERROR_RE = re.compile(r"Traceback|RuntimeError|Exception|CUDA out of memory|out of memory|OOM|NaN|Killed|FAILED", re.I)
DONE_RE = re.compile(r"best_test_metrics|Best model at epoch|Training complete|Finished|DONE", re.I)
EPOCH_RE = re.compile(r'"epoch"\s*:|(^|\s)Epoch\s+\d+', re.I)
tail_lines = int(os.environ["SENPAI_LOG_STATUS_LINES"])

rows = []
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.exists():
        rows.append({"path": str(path), "state": "not_started"})
        continue
    lines = [line.rstrip() for line in deque(path.open(encoding="utf-8", errors="replace"), maxlen=tail_lines)]
    if not lines:
        rows.append({"path": str(path), "state": "running_no_metric"})
        continue
    errors = [line for line in lines if ERROR_RE.search(line)]
    done = [line for line in lines if DONE_RE.search(line)]
    epochs = [line for line in lines if EPOCH_RE.search(line)]
    state = "failed" if errors else "complete" if done else "metric_seen" if epochs else "running_no_metric"
    rows.append({
        "path": str(path),
        "state": state,
        "last_epoch": epochs[-1] if epochs else None,
        "latest_events": (errors or done)[-3:],
        "scanned_tail_lines": len(lines),
    })

print(json.dumps(rows, indent=2))
PY
}

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

# GitHub CLI list commands default to 30 items, which silently truncates busy
# research branches. Use an explicit high cap so advisor triage sees the full
# queue when dozens of PRs are in flight.
GH_LIST_LIMIT="${GH_LIST_LIMIT:-999}"

# List human-created GitHub Issues addressed to a role (+ team issues).
# Returns a JSON array, deduplicated by issue number.
# Optional second arg: ISO timestamp — only return issues updated after it.
#   check_gh_issues <role_label> [since]
#   e.g. check_gh_issues "noam" "2026-04-01T12:00:00Z"
check_gh_issues() {
    local role="$1" since="${2:-}"
    local role_issues team_issues
    role_issues=$(gh_retry gh issue list --label "human" --label "$role" --state open \
        --limit "$GH_LIST_LIMIT" \
        --json number,title,updatedAt,comments)
    team_issues=$(gh_retry gh issue list --label "human" --label "team" --state open \
        --limit "$GH_LIST_LIMIT" \
        --json number,title,updatedAt,comments)
    printf '[%s,%s]' "$role_issues" "$team_issues" | python3 -c "
import json, sys
a, b = json.loads(sys.stdin.read())
since = sys.argv[1]
seen = set()
merged = []
for i in a + b:
    if i['number'] not in seen:
        seen.add(i['number'])
        if not since or i.get('updatedAt', '') > since:
            merged.append(i)
print(json.dumps(merged))
" "$since"
}

# List PRs that are ready for advisor review on a given branch.
# Returns a JSON array.
# Optional second arg: ISO timestamp — only return PRs updated after it.
#   list_ready_for_review_prs <branch> [since]
list_ready_for_review_prs() {
    local branch="$1" since="${2:-}"
    local prs
    prs=$(gh_retry gh pr list --label "$branch" --label "status:review" \
        --limit "$GH_LIST_LIMIT" \
        --json number,title,headRefName,labels,updatedAt)
    if [ -z "$since" ]; then
        printf '%s' "$prs"
    else
        printf '%s' "$prs" | python3 -c "
import json, sys
prs = json.loads(sys.stdin.read())
print(json.dumps([p for p in prs if p.get('updatedAt', '') > sys.argv[1]]))
" "$since"
    fi
}

# List all open PRs on a branch (any status).
# Returns a JSON array.
#   list_all_prs <branch>
list_all_prs() {
    local branch="$1"
    gh_retry gh pr list --label "$branch" \
        --limit "$GH_LIST_LIMIT" \
        --json number,title,state,labels,headRefName,updatedAt,isDraft
}

# List WIP PRs assigned to a specific student on the current advisor branch.
# Returns a JSON array.
#   student_poll_for_work <student_name> [advisor_branch]
student_poll_for_work() {
    local name="$1" branch="${2:-${ADVISOR_BRANCH:-}}"
    if [ -z "$branch" ]; then
        echo "student_poll_for_work: missing advisor branch" >&2
        return 2
    fi
    gh_retry gh pr list --label "$branch" --label "student:${name}" --label "status:wip" \
        --limit "$GH_LIST_LIMIT" \
        --json number,title,headRefName,updatedAt,body
}

# Compute which students are idle (have no status:wip PR).
# Expects a comma-separated student list and the advisor branch.
# Returns a JSON array of idle student names.
#   list_idle_students <student_names_csv> <branch>
list_idle_students() {
    local students_csv="$1" branch="$2"
    local all_prs
    all_prs=$(gh_retry gh pr list --label "$branch" --label "status:wip" \
        --limit "$GH_LIST_LIMIT" \
        --json labels)
    printf '%s' "$all_prs" | python3 -c "
import json, sys
students = [s.strip() for s in sys.argv[1].split(',') if s.strip()]
prs = json.loads(sys.stdin.read())
busy = set()
for pr in prs:
    for label in pr.get('labels', []):
        name = label.get('name', '')
        if name.startswith('student:'):
            busy.add(name.split(':', 1)[1])
print(json.dumps([s for s in students if s not in busy]))
" "$students_csv"
}
