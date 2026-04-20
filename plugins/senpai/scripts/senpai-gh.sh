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
# Functions auto-detect {owner}/{repo} from the git remote so callers
# never need to hardcode it.
#
# WHY THIS EXISTS:
# The GitHub CLI's `gh pr edit --remove-label X --add-label Y` silently
# strips *all other labels* from the PR. The REST API's individual
# DELETE + POST calls are the only safe way to swap a single label.
# This library wraps that pattern so nobody has to remember (or get
# bitten by) the footgun.

# ---------------------------------------------------------------------------
# Internal: repo slug cache
# ---------------------------------------------------------------------------
_SENPAI_REPO=""

# Print owner/repo (e.g. "wandb/senpai"), cached after first call.
print_gh_repo() {
    if [ -z "$_SENPAI_REPO" ]; then
        _SENPAI_REPO=$(git remote get-url origin 2>/dev/null \
            | sed -E 's|.*github\.com[:/]||; s|\.git$||')
    fi
    echo "$_SENPAI_REPO"
}

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
swap_gh_pr_label() {
    local num="$1" remove="$2" add="$3"
    local repo
    repo=$(print_gh_repo)

    # DELETE the old label — retry transient failures, tolerate 404 (already gone).
    local attempt err
    for attempt in 1 2 3 4 5 6; do
        err=$(gh api "repos/${repo}/issues/${num}/labels/${remove}" \
            --method DELETE --silent 2>&1) && break
        echo "$err" | grep -q "404" && break
        echo "swap_gh_pr_label: DELETE attempt $attempt failed, retrying in 15s..." >&2
        [ "$attempt" -eq 6 ] && return 1
        sleep 15
    done

    # POST the new label (gh_retry gives 6 attempts on transient failure).
    gh_retry gh api "repos/${repo}/issues/${num}/labels" \
        -f "labels[]=${add}" --method POST --silent
}

# Ensure a PR/issue has a label. Safe to call even if the label already exists.
#   ensure_gh_pr_label <number> <label>
ensure_gh_pr_label() {
    local num="$1" label="$2"
    local repo
    repo=$(print_gh_repo)
    gh_retry gh api "repos/${repo}/issues/${num}/labels" \
        -f "labels[]=${label}" --method POST --silent >/dev/null
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

# Create a draft assignment PR from a prepared body file, then verify that
# the routing labels and base branch invariants are present.
#   create_assignment_pr_from_file <student> <head_branch> <title> <body_file> [base_branch]
create_assignment_pr_from_file() {
    local student="$1" head_branch="$2" title="$3" body_file="$4" base_branch="${5:-${ADVISOR_BRANCH:-}}"
    local pr_url num details

    if [ -z "$student" ] || [ -z "$head_branch" ] || [ -z "$title" ] || [ -z "$body_file" ]; then
        echo "create_assignment_pr_from_file: usage: <student> <head_branch> <title> <body_file> [base_branch]" >&2
        return 2
    fi
    if [ ! -f "$body_file" ]; then
        echo "create_assignment_pr_from_file: body file not found: $body_file" >&2
        return 2
    fi
    if [ -z "$base_branch" ]; then
        echo "create_assignment_pr_from_file: base branch is required" >&2
        return 2
    fi

    pr_url=$(gh_retry gh pr create --draft \
        --title "$title" \
        --body-file "$body_file" \
        --base "$base_branch" \
        --head "$head_branch")

    num=$(printf '%s' "$pr_url" | sed -n 's#.*/pull/\([0-9][0-9]*\).*#\1#p')
    [ -n "$num" ] || num=$(gh_retry gh pr view "$head_branch" --json number --jq '.number')

    ensure_gh_pr_label "$num" "$base_branch"
    ensure_gh_pr_label "$num" "student:$student"
    ensure_gh_pr_label "$num" "status:wip"

    details=$(gh_retry gh pr view "$num" --json number,baseRefName,headRefName,labels,isDraft)
    python3 - "$student" "$base_branch" "$head_branch" "$details" <<'PY'
import json, sys

student, base_branch, head_branch, payload = sys.argv[1:5]
pr = json.loads(payload)
labels = {label.get("name", "") for label in pr.get("labels", [])}

errors = []
if pr.get("baseRefName") != base_branch:
    errors.append(f"expected base branch {base_branch}, got {pr.get('baseRefName')}")
if pr.get("headRefName") != head_branch:
    errors.append(f"expected head branch {head_branch}, got {pr.get('headRefName')}")
if not pr.get("isDraft"):
    errors.append("expected assignment PR to be draft")
for required in (base_branch, f"student:{student}", "status:wip"):
    if required not in labels:
        errors.append(f"missing required label {required}")

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

_list_open_prs_for_base_branch() {
    local branch="$1"
    gh_retry gh pr list --base "$branch" --state open --limit 200 \
        --json number,title,state,labels,headRefName,baseRefName,updatedAt,isDraft
}

# List human-created GitHub Issues addressed to a role (+ team issues).
# Returns a JSON array, deduplicated by issue number.
# Optional second arg: ISO timestamp — only return issues updated after it.
#   check_gh_issues <role_label> [since]
#   e.g. check_gh_issues "noam" "2026-04-01T12:00:00Z"
check_gh_issues() {
    local role="$1" since="${2:-}"
    local role_issues team_issues
    role_issues=$(gh_retry gh issue list --label "human" --label "$role" --state open \
        --json number,title,updatedAt,comments)
    team_issues=$(gh_retry gh issue list --label "human" --label "team" --state open \
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
    prs=$(_list_open_prs_for_base_branch "$branch")
    if [ -z "$since" ]; then
        printf '%s' "$prs" | python3 -c "
import json, sys
prs = json.loads(sys.stdin.read())
print(json.dumps([
    p for p in prs
    if 'status:review' in [label.get('name', '') for label in p.get('labels', [])]
]))
"
    else
        printf '%s' "$prs" | python3 -c "
import json, sys
prs = json.loads(sys.stdin.read())
print(json.dumps([
    p for p in prs
    if 'status:review' in [label.get('name', '') for label in p.get('labels', [])]
    and p.get('updatedAt', '') > sys.argv[1]
]))
" "$since"
    fi
}

# List all open PRs on a branch (any status).
# Returns a JSON array.
#   list_all_prs <branch>
list_all_prs() {
    local branch="$1"
    _list_open_prs_for_base_branch "$branch"
}

# List WIP PRs assigned to a specific student.
# Returns a JSON array.
#   student_poll_for_work <student_name>
student_poll_for_work() {
    local name="$1" branch="${2:-${ADVISOR_BRANCH:-}}"
    local prs
    prs=$(_list_open_prs_for_base_branch "$branch")
    python3 - "$name" "$prs" <<'PY'
import json, sys

target, payload = sys.argv[1:3]
prs = json.loads(payload)

def labels(pr):
    return [label.get("name", "") for label in pr.get("labels", [])]

def owner(pr):
    names = labels(pr)
    student_labels = [name.split(":", 1)[1] for name in names if name.startswith("student:")]
    head = pr.get("headRefName", "")
    prefix = head.split("/", 1)[0] if head else ""
    if len(student_labels) == 1:
        return student_labels[0]
    if len(student_labels) > 1:
        if target in student_labels:
            return target
        if prefix in student_labels:
            return prefix
        return ""
    return prefix

out = []
for pr in prs:
    pr_labels = labels(pr)
    if "status:wip" not in pr_labels:
        continue
    if owner(pr) == target:
        out.append(pr)

print(json.dumps(out))
PY
}

# List routing anomalies that affect a student's work discovery.
# Returns a JSON array of warning strings.
#   student_poll_for_work_warnings <student_name> [advisor_branch]
student_poll_for_work_warnings() {
    local name="$1" branch="${2:-${ADVISOR_BRANCH:-}}"
    local prs
    prs=$(_list_open_prs_for_base_branch "$branch")
    python3 - "$name" "$prs" <<'PY'
import json, sys

target, payload = sys.argv[1:3]
prs = json.loads(payload)
warnings = []

for pr in prs:
    labels = [label.get("name", "") for label in pr.get("labels", [])]
    student_labels = [label for label in labels if label.startswith("student:")]
    student_names = [label.split(":", 1)[1] for label in student_labels]
    status_labels = [label for label in labels if label.startswith("status:")]
    head = pr.get("headRefName", "")
    prefix = head.split("/", 1)[0] if head else ""

    relevant = prefix == target or target in student_names
    if not relevant:
        continue

    if f"student:{target}" not in labels and prefix == target:
        warnings.append(
            f"PR #{pr['number']} is missing student:{target}; routing recovered from head branch {head}"
        )
    if len(student_labels) > 1:
        warnings.append(
            f"PR #{pr['number']} has multiple student labels: {', '.join(student_labels)}"
        )
    if len(status_labels) == 0:
        warnings.append(
            f"PR #{pr['number']} has no status label; student pickup may stall until advisor repair"
        )
    elif len(status_labels) > 1:
        warnings.append(
            f"PR #{pr['number']} has multiple status labels: {', '.join(status_labels)}"
        )

print(json.dumps(warnings))
PY
}

# Compute which students are idle (have no status:wip PR).
# Expects a comma-separated student list and the advisor branch.
# Returns a JSON array of idle student names.
#   list_idle_students <student_names_csv> <branch>
list_idle_students() {
    local students_csv="$1" branch="$2"
    local all_prs
    all_prs=$(_list_open_prs_for_base_branch "$branch")
    printf '%s' "$all_prs" | python3 -c "
import json, sys
students = [s.strip() for s in sys.argv[1].split(',') if s.strip()]
prs = json.loads(sys.stdin.read())
busy = set()
for pr in prs:
    labels = [label.get('name', '') for label in pr.get('labels', [])]
    if 'status:wip' not in labels:
        continue
    student_labels = [name.split(':', 1)[1] for name in labels if name.startswith('student:')]
    head = pr.get('headRefName', '')
    prefix = head.split('/', 1)[0] if head else ''
    if len(student_labels) == 1:
        busy.add(student_labels[0])
    elif len(student_labels) > 1:
        if prefix in students:
            busy.add(prefix)
        else:
            busy.update(name for name in student_labels if name in students)
    elif prefix in students:
        busy.add(prefix)
print(json.dumps([s for s in students if s not in busy]))
" "$students_csv"
}

# Repair missing or mismatched assignment metadata for open PRs on a branch.
# Returns a JSON array of warning strings describing the repairs or anomalies.
#   reconcile_assignment_prs <student_names_csv> <advisor_branch>
reconcile_assignment_prs() {
    local students_csv="$1" branch="$2"
    local prs actions
    prs=$(_list_open_prs_for_base_branch "$branch")
    actions=$(python3 - "$students_csv" "$branch" "$prs" <<'PY'
import json, sys

students = {s.strip() for s in sys.argv[1].split(",") if s.strip()}
branch = sys.argv[2]
prs = json.loads(sys.argv[3])

def emit(kind, number, arg1="", arg2="", warning=""):
    print("\x1f".join([kind, str(number), arg1, arg2, warning]))

for pr in prs:
    number = pr["number"]
    labels = [label.get("name", "") for label in pr.get("labels", [])]
    student_labels = [label for label in labels if label.startswith("student:")]
    student_names = [label.split(":", 1)[1] for label in student_labels]
    status_labels = [label for label in labels if label.startswith("status:")]
    head = pr.get("headRefName", "")
    prefix = head.split("/", 1)[0] if head else ""
    inferred = prefix if prefix in students else ""

    if len(student_labels) == 0 and inferred:
        emit(
            "add",
            number,
            f"student:{inferred}",
            "",
            f"PR #{number} missing student label; restored student:{inferred} from head branch {head}",
        )
    elif len(student_labels) == 1 and inferred and student_names[0] != inferred:
        emit(
            "swap",
            number,
            f"student:{student_names[0]}",
            f"student:{inferred}",
            f"PR #{number} student label mismatched head branch {head}; swapped to student:{inferred}",
        )
    elif len(student_labels) > 1:
        emit(
            "warn",
            number,
            "",
            "",
            f"PR #{number} has multiple student labels: {', '.join(student_labels)}",
        )

    if branch not in labels:
        emit(
            "add",
            number,
            branch,
            "",
            f"PR #{number} missing branch label {branch}; restored standard label",
        )

    if len(status_labels) == 0:
        expected = "status:wip" if pr.get("isDraft") else "status:review"
        emit(
            "add",
            number,
            expected,
            "",
            f"PR #{number} missing status label; restored {expected}",
        )
    elif len(status_labels) > 1:
        emit(
            "warn",
            number,
            "",
            "",
            f"PR #{number} has multiple status labels: {', '.join(status_labels)}",
        )
PY
)

    local warnings=()
    while IFS=$'\x1f' read -r action num arg1 arg2 warning; do
        [ -z "$action" ] && continue
        case "$action" in
            add)
                ensure_gh_pr_label "$num" "$arg1"
                ;;
            swap)
                swap_gh_pr_label "$num" "$arg1" "$arg2"
                ;;
            warn)
                ;;
        esac
        echo "WARNING: $warning" >&2
        warnings+=("$warning")
    done <<< "$actions"

    python3 - <<'PY' "${warnings[@]}"
import json, sys
print(json.dumps(sys.argv[1:]))
PY
}
