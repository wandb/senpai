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

SENPAI_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Retry helpers.
# ---------------------------------------------------------------------------
senpai_run_with_timeout() {
    local timeout_seconds="${SENPAI_GH_TIMEOUT_SECONDS:-120}"
    if command -v timeout >/dev/null 2>&1; then
        local kill_after="${SENPAI_GH_TIMEOUT_KILL_AFTER_SECONDS:-30}"
        timeout -k "$kill_after" "$timeout_seconds" "$@"
    else
        "$@"
    fi
}

senpai_random_jitter_s() {
    local max="${1:-0}"
    case "$max" in
        ''|*[!0-9]*) max=0 ;;
    esac
    [ "$max" -le 0 ] && { printf '0\n'; return; }

    python3 - "$max" <<'PY' 2>/dev/null || printf '0\n'
import random
import sys

print(random.randint(0, int(sys.argv[1])))
PY
}

senpai_sleep_with_jitter() {
    local base="${1:-0}" jitter="${2:-0}" extra total
    case "$base" in
        ''|*[!0-9]*) base=0 ;;
    esac
    extra=$(senpai_random_jitter_s "$jitter")
    total=$((base + extra))
    [ "$total" -gt 0 ] && sleep "$total"
}

gh_rate_limit_backoff_s() {
    local err_file="$1" reset now wait max jitter
    grep -qi "API rate limit exceeded\\|rate limit" "$err_file" || { printf '0\n'; return; }

    reset=$(gh api rate_limit --jq '.resources.core.reset' 2>/dev/null || true)
    case "$reset" in
        ''|*[!0-9]*) printf '0\n'; return ;;
    esac

    now=$(date +%s)
    wait=$((reset - now + 5))
    [ "$wait" -le 0 ] && { printf '0\n'; return; }

    max="${SENPAI_GH_RATE_LIMIT_MAX_SLEEP_SECONDS:-900}"
    case "$max" in
        ''|*[!0-9]*) max=900 ;;
    esac
    [ "$wait" -gt "$max" ] && wait="$max"

    jitter=$(senpai_random_jitter_s "${SENPAI_GH_RATE_LIMIT_JITTER_S:-30}")
    printf '%s\n' "$((wait + jitter))"
}

gh_retry() {
    local attempt status out err backoff
    for attempt in 1 2 3 4 5 6; do
        out=$(mktemp "${TMPDIR:-/tmp}/senpai-gh-out.XXXXXX") || return
        err=$(mktemp "${TMPDIR:-/tmp}/senpai-gh-err.XXXXXX") || { rm -f "$out"; return; }

        senpai_run_with_timeout "$@" >"$out" 2>"$err"
        status=$?
        if [ "$status" -eq 0 ]; then
            cat "$out"
            rm -f "$out" "$err"
            return 0
        fi
        cat "$err" >&2
        if [ "$attempt" -eq 6 ]; then
            echo "gh_retry: attempt $attempt failed with status $status; giving up" >&2
            rm -f "$out" "$err"
            return "$status"
        fi

        backoff=$(gh_rate_limit_backoff_s "$err")
        if [ "${backoff:-0}" -gt 0 ]; then
            echo "gh_retry: rate limit detected on attempt $attempt, retrying in ${backoff}s..." >&2
            rm -f "$out" "$err"
            sleep "$backoff"
        else
            echo "gh_retry: attempt $attempt failed with status $status, retrying in 15s..." >&2
            rm -f "$out" "$err"
            senpai_sleep_with_jitter 15 "${SENPAI_GH_RETRY_JITTER_S:-5}"
        fi
    done
    return 1
}

comment_on_pr() {
    local num="$1" body="$2" tmp status
    if [ -z "$num" ]; then
        echo "comment_on_pr: usage: <pr-number> <body>" >&2
        return 2
    fi

    tmp=$(mktemp "${TMPDIR:-/tmp}/senpai-gh-comment.XXXXXX") || return
    printf '%s' "$body" > "$tmp"
    gh_retry gh pr comment "$num" --repo "$GH_REPO" --body-file "$tmp"
    status=$?
    rm -f "$tmp"
    return "$status"
}

poll_or_empty() {
    local label="$1"
    shift
    "$@" || {
        echo "WARN: $label failed; treating as empty for this iteration" >&2
        printf '[]\n'
        return 1
    }
}

install_senpai_git_guard() {
    local workdir="$1" target_workdir="$2" credential_file="$3"
    if [ -z "$workdir" ] || [ -z "$target_workdir" ] || [ -z "$credential_file" ]; then
        echo "install_senpai_git_guard: usage: <workdir> <target-workdir> <credential-file>" >&2
        return 2
    fi

    git remote set-url --push origin DISABLED
    git config remote.origin.pushurl DISABLED
    git config --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf 2>/dev/null || true

    export TARGET_WORKDIR="$target_workdir"
    export SENPAI_REAL_GIT="${SENPAI_REAL_GIT:-$(command -v git)}"

    mkdir -p .git/hooks "$workdir/git-guard-bin"
    cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
echo "ERROR: refusing to push from the senpai runner repo; use the cloned target repo instead." >&2
exit 1
EOF
    chmod +x .git/hooks/pre-push

    cat > "$workdir/git-guard-bin/git" <<'EOF'
#!/bin/sh
real_git="${SENPAI_REAL_GIT:-/usr/bin/git}"
if [ "$1" = "push" ]; then
    top="$("$real_git" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${TARGET_WORKDIR:-}" ] && [ "$top" != "${TARGET_WORKDIR%/}" ]; then
        echo "ERROR: refusing git push outside target repo; cwd=$(pwd), top=${top:-none}, target=$TARGET_WORKDIR" >&2
        exit 2
    fi
fi
exec "$real_git" "$@"
EOF
    chmod +x "$workdir/git-guard-bin/git"
    export PATH="$workdir/git-guard-bin:$PATH"

    printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > "$credential_file"
    chmod 600 "$credential_file"
    git config --global credential.helper "store --file=$credential_file"
}

install_senpai_target_git_guard() {
    local target_workdir="$1"
    if [ -z "$target_workdir" ] || [ ! -d "$target_workdir/.git" ]; then
        echo "install_senpai_target_git_guard: target repo not found: ${target_workdir:-<missing>}" >&2
        return 2
    fi

    mkdir -p "$target_workdir/.git/hooks"
    cat > "$target_workdir/.git/hooks/pre-push" <<'EOF'
#!/bin/sh
while read -r local_ref _ remote_ref _; do
    [ "$remote_ref" = "refs/heads/$ADVISOR_BRANCH" ] || continue
    [ "$SENPAI_ROLE" != "student" ] || {
        echo "SENPAI-GIT-GUARD: students must not push $ADVISOR_BRANCH" >&2
        exit 2
    }
    [ "$SENPAI_ROLE" != "advisor" ] || [ "$local_ref" = "refs/heads/$ADVISOR_BRANCH" ] || {
        echo "SENPAI-GIT-GUARD: advisor must push $ADVISOR_BRANCH from local $ADVISOR_BRANCH, not ${local_ref:-<unknown>}" >&2
        exit 2
    }
done
EOF
    chmod +x "$target_workdir/.git/hooks/pre-push"
}

require_target_repo() {
    local origin
    origin=$(git config --get remote.origin.url || true)
    case "$origin" in
        *"${GH_REPO}"*|*"${TARGET_REPO_URL:-__unset__}"*) return 0 ;;
    esac
    echo "ERROR: refusing git operation outside target repo; cwd=$(pwd), origin=${origin:-none}, target=${GH_REPO}" >&2
    return 2
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
    local attempt err_file err backoff
    for attempt in 1 2 3 4 5 6; do
        err_file=$(mktemp "${TMPDIR:-/tmp}/senpai-gh-err.XXXXXX") || return
        if senpai_run_with_timeout gh api "repos/${GH_REPO}/issues/${num}/labels/${remove}" \
            --method DELETE --silent 2>"$err_file"; then
            rm -f "$err_file"
            break
        fi
        err=$(cat "$err_file")
        cat "$err_file" >&2
        if echo "$err" | grep -q "404"; then
            rm -f "$err_file"
            break
        fi
        backoff=$(gh_rate_limit_backoff_s "$err_file")
        rm -f "$err_file"
        [ "$attempt" -eq 6 ] && return 1
        if [ "${backoff:-0}" -gt 0 ]; then
            echo "swap_gh_pr_label: rate limit detected on DELETE attempt $attempt, retrying in ${backoff}s..." >&2
            sleep "$backoff"
        else
            echo "swap_gh_pr_label: DELETE attempt $attempt failed, retrying in 15s..." >&2
            senpai_sleep_with_jitter 15 "${SENPAI_GH_RETRY_JITTER_S:-5}"
        fi
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
    comment_on_pr "$num" "$body"
    gh_retry gh pr ready "$num" --repo "$GH_REPO" --undo
    swap_gh_pr_label "$num" "status:review" "status:wip"
}

# Require a terminal structured result comment before a student can hand a PR
# to the advisor. This makes "ready for review" a durable workflow state rather
# than a bare label flip.
#   senpai_require_terminal_result <number>
senpai_require_terminal_result() {
    local num="$1" comments
    comments=$(pr_all_comments "$num") || return
    printf '%s' "$comments" | python3 "$SENPAI_SCRIPT_DIR/senpai-pr-guard.py" require-terminal-result "$num"
}

# Refuse unsafe merges before `senpai:merge-winner` can squash a PR. This guard
# checks GitHub workflow state plus the latest structured student result marker.
# It intentionally does not query or mutate W&B; runs already record their git
# commit SHA, and the PR result comment remains the workflow contract.
#   senpai_merge_winner_preflight <number> [problem-dir]
senpai_merge_winner_preflight() {
    local num="$1" pr comments
    if [ -z "$num" ]; then
        echo "senpai_merge_winner_preflight: usage: <pr-number> [problem-dir]" >&2
        return 2
    fi

    pr=$(gh_retry gh pr view "$num" --repo "$GH_REPO" \
        --json number,title,state,isDraft,labels,baseRefName,headRefName,mergeStateStatus,mergeable,files) || return
    comments=$(pr_all_comments "$num") || return

    printf '%s\0%s' "$pr" "$comments" |
        python3 "$SENPAI_SCRIPT_DIR/senpai-pr-guard.py" merge-preflight "$num" "${ADVISOR_BRANCH:-}"
}

# Close a dead-end PR: comment explaining why and close it. Keep the remote
# branch so humans can reopen/recover assignments without reconstructing refs.
#   close_pr_with_comment <number> <reason>
close_pr_with_comment() {
    local num="$1" reason="$2"

    comment_on_pr "$num" "ADVISOR: Closing PR #${num} because ${reason}."
    gh_retry gh pr close "$num" --repo "$GH_REPO"
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
    local existing existing_count pr_url num details

    if [ -z "$student" ] || [ -z "$head_branch" ] || [ -z "$title" ] || [ -z "$body_file" ] || [ -z "$base_branch" ]; then
        echo "create_assignment_pr_from_file: usage: <student> <head-branch> <title> <body-file> [base-branch]" >&2
        return 2
    fi
    if [ ! -f "$body_file" ]; then
        echo "create_assignment_pr_from_file: body file not found: $body_file" >&2
        return 2
    fi

    existing=$(student_poll_for_work "$student" "$base_branch")
    existing_count=$(printf '%s' "$existing" | json_len)
    if [ "$existing_count" -gt 0 ]; then
        echo "create_assignment_pr_from_file: student:${student} already has active status:wip PR(s): $(printf '%s' "$existing" | json_numbers)" >&2
        return 1
    fi

    pr_url=$(gh_retry gh pr create --repo "$GH_REPO" --draft \
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

    details=$(gh_retry gh pr view "$num" --repo "$GH_REPO" --json number,baseRefName,headRefName,labels,isDraft)
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
    senpai_require_terminal_result "$num" || return
    gh_retry gh pr ready "$num" --repo "$GH_REPO"
    swap_gh_pr_label "$num" "status:wip" "status:review"  # swap_gh_pr_label uses gh_retry internally
}

# Create a real assignment branch before opening a PR.
#   create_assignment_branch <student> <hypothesis-slug>
#   create_assignment_branch <student>/<hypothesis-slug>
create_assignment_branch() {
    local student="$1" slug="${2:-}"
    if [ -z "$slug" ] && [[ "$student" == */* ]]; then
        slug="${student#*/}"
        student="${student%%/*}"
    fi
    if [ -z "$student" ] || [ -z "$slug" ]; then
        echo "create_assignment_branch: usage: <student> <hypothesis-slug> or <student>/<hypothesis-slug>" >&2
        return 2
    fi
    local branch="${student}/${slug}"
    require_target_repo || return
    git checkout "$ADVISOR_BRANCH" || return
    git pull origin "$ADVISOR_BRANCH" || return
    git checkout -b "$branch" || return
    git commit --allow-empty -m "assign ${student}: ${slug}" || return
    git push -u origin "$branch" || return
    git rev-list --count "${ADVISOR_BRANCH}..HEAD" | grep -vq '^0$'
}

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

json_len() { python3 -c "import sys,json; print(len(json.loads(sys.stdin.read())))"; }
json_join() { python3 -c "import sys,json; print(','.join(json.loads(sys.stdin.read())))"; }
json_numbers() { python3 -c "import sys,json; print(','.join(f'#{i[\"number\"]}' for i in json.loads(sys.stdin.read())))"; }
json_advisor_action_summary() {
    python3 -c '
import json
import sys

items = json.loads(sys.stdin.read())
parts = []
for item in items:
    reasons = ",".join(item.get("reasons", []))
    detail = ""
    if item.get("unknownStudentLabels"):
        detail = " unknown={}".format("|".join(item["unknownStudentLabels"]))
    parts.append("#{}[{}{}]".format(item["number"], reasons, detail))
print(",".join(parts))
'
}

# Print the maximum updatedAt timestamp from one or more JSON arrays.
# Returns empty string if all arrays are empty.  Usage:
#   max_updated_at "$JSON_BLOB1" "$JSON_BLOB2" ...
max_updated_at() {
    printf '%s\0' "$@" | python3 -c '
import json
import sys

items = []
for blob in sys.stdin.buffer.read().split(b"\0"):
    if blob:
        items.extend(json.loads(blob))
ts = [i["updatedAt"] for i in items if "updatedAt" in i]
print(max(ts) if ts else "")
'
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

github_urlencode() {
    python3 - "$1" <<'PY'
from urllib.parse import quote
import sys

print(quote(sys.argv[1], safe=""))
PY
}

rest_pull_details_from_numbers() {
    python3 -c '
import json
import subprocess
import sys

repo = sys.argv[1]
items = json.load(sys.stdin)
out = []


def gh_label(label):
    return {
        "id": label.get("node_id") or str(label.get("id", "")),
        "name": label.get("name", ""),
        "description": label.get("description"),
        "color": label.get("color", ""),
    }


def mergeable_value(pr):
    value = pr.get("mergeable")
    if value is True:
        return "MERGEABLE"
    if value is False:
        return "CONFLICTING"
    return "UNKNOWN"


def merge_state_status(pr):
    return (pr.get("mergeable_state") or "unknown").upper()


for item in items:
    num = item["number"]
    res = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{num}"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        sys.exit(res.returncode)
    pr = json.loads(res.stdout)
    out.append({
        "number": pr["number"],
        "title": pr["title"],
        "state": pr["state"].upper(),
        "labels": [gh_label(label) for label in pr.get("labels", [])],
        "headRefName": pr["head"]["ref"],
        "baseRefName": pr["base"]["ref"],
        "updatedAt": pr["updated_at"],
        "isDraft": pr.get("draft", False),
        "body": pr.get("body") or "",
        "mergeStateStatus": merge_state_status(pr),
        "mergeable": mergeable_value(pr),
    })

print(json.dumps(out))
' "$GH_REPO"
}

rest_labeled_pull_details() {
    local labels="$1" labels_q issues
    labels_q=$(github_urlencode "$labels")
    issues=$(gh_api_paginated_array "repos/${GH_REPO}/issues?state=open&labels=${labels_q}&per_page=100" \
        '[.[] | select(has("pull_request")) | {number}]') || return
    printf '%s' "$issues" | rest_pull_details_from_numbers
}

rest_base_pull_details() {
    local branch="$1" branch_q pulls
    branch_q=$(github_urlencode "$branch")
    pulls=$(gh_api_paginated_array "repos/${GH_REPO}/pulls?state=open&base=${branch_q}&per_page=100" \
        '[.[] | {number}]') || return
    printf '%s' "$pulls" | rest_pull_details_from_numbers
}

student_pod_and_target_path() {
    local student="$1" tag="${RESEARCH_TAG:-}" pod
    [ -n "$student" ] && [ -n "$tag" ] || return 1
    command -v kubectl >/dev/null 2>&1 || return 1

    pod=$(
        kubectl get pods -l "app=senpai,research-tag=${tag},student=${student}" \
            -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' 2>/dev/null |
            head -1
    ) || return 1
    if [ -n "$pod" ]; then
        printf '%s\t%s\n' "$pod" "/workspace/senpai/target"
        return 0
    fi

    kubectl get pods -l "app=senpai,role=student,research-tag=${tag}" -o json 2>/dev/null |
        STUDENT_TO_FIND="$student" python3 -c '
import json
import os
import sys

student = os.environ["STUDENT_TO_FIND"]
data = json.load(sys.stdin)
for item in data.get("items", []):
    if item.get("status", {}).get("phase") != "Running":
        continue
    metadata = item.get("metadata", {})
    annotations = metadata.get("annotations", {})
    names = [
        name.strip()
        for name in annotations.get("senpai/student-names", "").split(",")
        if name.strip()
    ]
    if student in names:
        print("{}\t/workspace/senpai-{}/target".format(metadata.get("name", ""), student))
        sys.exit(0)
sys.exit(1)
'
}

student_pr_looks_live() {
    local student="$1" head_ref="$2" tag="${RESEARCH_TAG:-}"
    local pod_info pod target_path grouped
    [ -n "$student" ] && [ -n "$head_ref" ] && [ -n "$tag" ] || return 1
    command -v kubectl >/dev/null 2>&1 || return 1

    pod_info=$(student_pod_and_target_path "$student") || return 1
    pod=${pod_info%%$'\t'*}
    target_path=${pod_info#*$'\t'}
    [ -n "$pod" ] && [ -n "$target_path" ] || return 1
    grouped=0
    case "$target_path" in
        /workspace/senpai-*/target) grouped=1 ;;
    esac

    kubectl exec "$pod" -- sh -lc '
        target_path="$1"
        expected_branch="$2"
        grouped="$3"
        branch=$(git -C "$target_path" branch --show-current 2>/dev/null || true)
        [ "$branch" = "$expected_branch" ] || exit 1

        # A stale GitHub timestamp is not actionable if the pod is still doing
        # useful work on this exact PR branch. Count either active training,
        # active GPU use, or Claude Code editing an uncommitted checkout.
        if [ "$grouped" = "1" ]; then
            pytrain=0
            gpu=0
        else
            pytrain=$(ps -eo comm=,args= | awk '\''$1 ~ /^python/ && /train[.]py/ {n++} END{print n+0}'\'')
            gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '\''NF{n++} END{print n+0}'\'')
        fi
        claude=$(ps -eo comm=,args= | awk '\''$1 ~ /^claude$/ || /[ /]claude( |$)/ {n++} END{print n+0}'\'')
        dirty=$(git -C "$target_path" status --porcelain 2>/dev/null | wc -l | tr -d " ")

        [ "$pytrain" -gt 0 ] || [ "$gpu" -gt 0 ] || { [ "$claude" -gt 0 ] && [ "${dirty:-0}" -gt 0 ]; }
    ' sh "$target_path" "$head_ref" "$grouped" >/dev/null 2>&1
}

suppress_live_stale_wips() {
    local actions="$1" live_numbers="" num student head_ref
    while IFS=$'\t' read -r num student head_ref; do
        [ -n "$num" ] || continue
        if student_pr_looks_live "$student" "$head_ref"; then
            live_numbers+="${num}"$'\n'
        fi
    done < <(printf '%s' "$actions" | python3 -c '
import json
import sys

for pr in json.load(sys.stdin):
    if "stale_wip" not in pr.get("reasons", []):
        continue
    students = [
        label.get("name", "").removeprefix("student:")
        for label in pr.get("labels", [])
        if label.get("name", "").startswith("student:")
    ]
    if len(students) == 1:
        print("{}\t{}\t{}".format(pr["number"], students[0], pr.get("headRefName", "")))
')

    printf '%s\0%s' "$actions" "$live_numbers" | python3 -c '
import json
import sys

raw_actions, raw_live = sys.stdin.buffer.read().split(b"\0", 1)
live = {int(line) for line in raw_live.decode().splitlines() if line.strip()}
filtered = []
for pr in json.loads(raw_actions):
    if pr["number"] in live:
        # Only remove the stale timestamp reason. Other advisor-action reasons
        # such as blocked, duplicate assignment, or rebase trouble remain live.
        pr["reasons"] = [reason for reason in pr.get("reasons", []) if reason != "stale_wip"]
    if pr.get("reasons"):
        filtered.append(pr)
print(json.dumps(filtered))
'
}

# List active student pods whose training process does not match an open WIP PR.
# Returns a JSON array of human-readable warnings for the advisor prompt.
#   list_student_pod_anomalies <student_names_csv> <branch>
list_student_pod_anomalies() {
    local students_csv="$1" branch="$2" tag="${RESEARCH_TAG:-}"
    local all_prs open_heads anomalies="" student pod snapshot current_branch pytrain gpu expected_heads pod_info target_path grouped
    # This helper runs inside advisor pods. Local/dev shells without Kubernetes
    # context should stay quiet rather than making every advisor poll noisy.
    [ -n "$tag" ] && command -v kubectl >/dev/null 2>&1 || { echo "[]"; return 0; }

    all_prs=$(rest_labeled_pull_details "${branch},status:wip") || return
    # Map each open WIP assignment to the branch the student's pod should be on.
    open_heads=$(printf '%s' "$all_prs" | python3 -c '
import json
import sys

for pr in json.load(sys.stdin):
    head = pr.get("headRefName", "")
    for label in pr.get("labels", []):
        name = label.get("name", "")
        if name.startswith("student:"):
            student = name.split(":", 1)[1]
            print(f"{student}\t{head}")
')

    IFS=',' read -r -a students <<< "$students_csv"
    for student in "${students[@]}"; do
        student="${student//[[:space:]]/}"
        [ -n "$student" ] || continue

        pod_info=$(student_pod_and_target_path "$student") || continue
        pod=${pod_info%%$'\t'*}
        target_path=${pod_info#*$'\t'}
        [ -n "$pod" ] && [ -n "$target_path" ] || continue
        grouped=0
        case "$target_path" in
            /workspace/senpai-*/target) grouped=1 ;;
        esac

        snapshot=$(kubectl exec "$pod" -- sh -lc '
            target_path="$1"
            grouped="$2"
            branch=$(git -C "$target_path" branch --show-current 2>/dev/null || true)
            # Only active training/GPU use is a zombie risk. Claude editing a
            # checkout is handled by stale-WIP suppression above, not here.
            if [ "$grouped" = "1" ]; then
                pytrain=0
                gpu=0
            else
                pytrain=$(ps -eo comm=,args= | awk '\''$1 ~ /^python/ && /train[.]py/ {n++} END{print n+0}'\'')
                gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '\''NF{n++} END{print n+0}'\'')
            fi
            printf "%s\t%s\t%s\n" "$branch" "$pytrain" "$gpu"
        ' sh "$target_path" "$grouped" 2>/dev/null) || continue

        current_branch=${snapshot%%$'\t'*}
        snapshot=${snapshot#*$'\t'}
        pytrain=${snapshot%%$'\t'*}
        gpu=${snapshot#*$'\t'}
        if [ "${pytrain:-0}" -eq 0 ] && [ "${gpu:-0}" -eq 0 ]; then
            continue
        fi

        expected_heads=$(printf '%s' "$open_heads" | awk -F '\t' -v wanted="$student" '$1 == wanted {heads = heads sep $2; sep = " or "} END {print heads}')
        if [ -z "$expected_heads" ]; then
            anomalies+="${student}: active training on branch ${current_branch:-unknown} but this student has no open status:wip PR on ${branch}; possible zombie run after PR closure; inspect or stop the pod before assigning new work; pytrain=${pytrain:-0}; gpu=${gpu:-0}"$'\n'
            continue
        fi
        if ! printf '%s' "$open_heads" | awk -F '\t' -v wanted="$student" -v current="$current_branch" '$1 == wanted && $2 == current {found=1} END {exit !found}'; then
            anomalies+="${student}: active training on branch ${current_branch:-unknown} but open status:wip assignment expects ${expected_heads}; possible stale checkout or wrong pod assignment; reconcile or stop the pod before assigning new work; pytrain=${pytrain:-0}; gpu=${gpu:-0}"$'\n'
        fi
    done

    printf '%s' "$anomalies" | python3 -c '
import json
import sys

print(json.dumps([line for line in sys.stdin.read().splitlines() if line]))
'
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

# List human-created GitHub Issues addressed to a role (+ team issues).
# Returns a JSON array, deduplicated by issue number.
# Optional second arg: ISO timestamp — only return issues updated after it.
#   check_gh_issues <role_label> [since]
#   e.g. check_gh_issues "noam" "2026-04-01T12:00:00Z"
check_gh_issues() {
    local role="$1" since="${2:-}"
    local role_issues team_issues
    if [ "${SENPAI_ENABLE_HUMAN_ISSUES:-true}" = "false" ]; then
        printf '[]\n'
        return
    fi
    role_issues=$(gh_api_paginated_array \
        "repos/${GH_REPO}/issues?state=open&labels=$(github_urlencode "human,${role}")&per_page=100" \
        '[.[] | select(has("pull_request") | not) | {number,title,updatedAt:.updated_at}]')
    team_issues=$(gh_api_paginated_array \
        "repos/${GH_REPO}/issues?state=open&labels=$(github_urlencode "human,team")&per_page=100" \
        '[.[] | select(has("pull_request") | not) | {number,title,updatedAt:.updated_at}]')
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
# Optional second arg is accepted for backward compatibility but intentionally
# ignored: review-ready PRs are level-triggered until the advisor resolves them.
#   list_ready_for_review_prs <branch> [since]
list_ready_for_review_prs() {
    local branch="$1"
    rest_labeled_pull_details "${branch},status:review" | python3 -c '
import json
import sys

print(json.dumps([
    {
        "number": pr["number"],
        "title": pr["title"],
        "headRefName": pr["headRefName"],
        "labels": pr["labels"],
        "updatedAt": pr["updatedAt"],
    }
    for pr in json.load(sys.stdin)
]))
'
}

# List all open PRs on a branch (any status).
# Returns a JSON array.
#   list_all_prs <branch>
list_all_prs() {
    local branch="$1"
    rest_labeled_pull_details "$branch" | python3 -c '
import json
import sys

print(json.dumps([
    {
        "number": pr["number"],
        "title": pr["title"],
        "state": pr["state"],
        "labels": pr["labels"],
        "headRefName": pr["headRefName"],
        "updatedAt": pr["updatedAt"],
        "isDraft": pr["isDraft"],
    }
    for pr in json.load(sys.stdin)
]))
'
}

# List open PRs requiring advisor action, even when they have not been updated
# since the last heartbeat.
# Returns a JSON array with a `reasons` list on each PR.
#   list_prs_requiring_advisor_action <branch> [stale_wip_seconds] [student_names_csv]
list_prs_requiring_advisor_action() {
    local branch="$1" stale_seconds="${2:-${SENPAI_STALE_WIP_SECONDS:-7200}}" students_csv="${3:-${STUDENT_NAMES:-}}"
    local prs comments_by_pr num comments row actions
    prs=$(rest_base_pull_details "$branch")
    comments_by_pr=""
    while IFS= read -r num; do
        [ -z "$num" ] && continue
        comments=$(pr_issue_comments "$num") || return
        row=$(printf '%s\0%s' "$num" "$comments" | python3 -c '
import json
import sys

num, comments = sys.stdin.buffer.read().split(b"\0", 1)
print(json.dumps({"number": int(num), "comments": json.loads(comments)}))
')
        comments_by_pr+="${row}"$'\n'
    done < <(printf '%s' "$prs" | python3 -c 'import json,sys; print("\n".join(str(pr["number"]) for pr in json.loads(sys.stdin.read())))')

    actions=$(printf '%s\0%s' "$prs" "$comments_by_pr" | python3 -c '
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

branch = sys.argv[1]
stale_seconds = int(sys.argv[2])
known_students = {s.strip() for s in sys.argv[3].split(",") if s.strip()}
now = datetime.now(timezone.utc)
raw_prs, raw_comments = sys.stdin.buffer.read().split(b"\0", 1)
prs = json.loads(raw_prs)
comments_by_pr = {}
for line in raw_comments.decode().splitlines():
    if line.strip():
        row = json.loads(line)
        comments_by_pr[row["number"]] = row["comments"]

def parse_ts(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def label_names(pr):
    return {label.get("name", "") for label in pr.get("labels", [])}

student_wips = defaultdict(list)
metadata = {}
for pr in prs:
    labels = label_names(pr)
    students = sorted(label.removeprefix("student:") for label in labels if label.startswith("student:"))
    unknown_students = [student for student in students if known_students and student not in known_students]
    routed_students = [student for student in students if not known_students or student in known_students]
    metadata[pr["number"]] = (labels, students, routed_students, unknown_students)
    if "status:wip" in labels:
        for student in routed_students:
            student_wips[student].append(pr["number"])

duplicate_wips = {
    number
    for numbers in student_wips.values()
    if len(numbers) > 1
    for number in numbers
}

conflict_re = re.compile(r"merge conflict|rebase conflict|cannot automatically merge|can.t automatically merge|conflicts? with", re.I)
prs_requiring_advisor_action = []
for pr in prs:
    labels, students, routed_students, unknown_students = metadata[pr["number"]]
    reasons = []
    if branch not in labels:
        reasons.append("missing_branch_label")
    if not students:
        reasons.append("missing_student_label")
    if unknown_students:
        reasons.append("unknown_student_label")
    if students and not routed_students:
        reasons.append("unroutable_student_label")
    if pr["number"] in duplicate_wips:
        reasons.append("duplicate_student_wip")
    if "status:wip" in labels:
        age = (now - parse_ts(pr["updatedAt"])).total_seconds()
        if age > stale_seconds:
            reasons.append("stale_wip")
    if "status:needs-rebase" in labels:
        reasons.append("needs_rebase")
    if "status:blocked" in labels:
        reasons.append("blocked_wip")
    if pr.get("mergeStateStatus") in {"BEHIND", "DIRTY"} or pr.get("mergeable") == "CONFLICTING":
        reasons.append("needs_rebase")
    comment_text = "\n".join(comment.get("body", "") for comment in comments_by_pr.get(pr["number"], []))
    if conflict_re.search(comment_text):
        reasons.append("merge_conflict_comment")
    if pr.get("isDraft") and "status:review" in labels:
        reasons.append("draft_but_claimed_mergeable")
    if reasons:
        item = dict(pr)
        item["reasons"] = sorted(set(reasons), key=reasons.index)
        if unknown_students:
            item["unknownStudentLabels"] = [f"student:{student}" for student in unknown_students]
        prs_requiring_advisor_action.append(item)

print(json.dumps(prs_requiring_advisor_action))
' "$branch" "$stale_seconds" "$students_csv")
    suppress_live_stale_wips "$actions"
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
    rest_labeled_pull_details "${branch},student:${name},status:wip" | python3 -c '
import json
import sys

print(json.dumps([
    {
        "number": pr["number"],
        "title": pr["title"],
        "headRefName": pr["headRefName"],
        "updatedAt": pr["updatedAt"],
        "body": pr["body"],
    }
    for pr in json.load(sys.stdin)
]))
'
}

# Compute which students are idle (have no status:wip PR).
# Expects a comma-separated student list and the advisor branch.
# Returns a JSON array of idle student names.
#   list_idle_students <student_names_csv> <branch>
list_idle_students() {
    local students_csv="$1" branch="$2"
    local all_prs
    all_prs=$(rest_labeled_pull_details "${branch},status:wip")
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
