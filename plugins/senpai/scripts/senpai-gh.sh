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

# List WIP PRs assigned to a specific student.
# Returns a JSON array.
#   student_poll_for_work <student_name>
student_poll_for_work() {
    local name="$1"
    gh_retry gh pr list --label "student:${name}" --label "status:wip" \
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
