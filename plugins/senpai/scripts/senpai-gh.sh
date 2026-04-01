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
# Label operations
# ---------------------------------------------------------------------------

# Atomically swap one label for another on a PR/issue.
#   swap_gh_pr_label <number> <remove_label> <add_label>
swap_gh_pr_label() {
    local num="$1" remove="$2" add="$3"
    local repo
    repo=$(print_gh_repo)
    # DELETE may 404 if the label isn't present — that's fine.
    gh api "repos/${repo}/issues/${num}/labels/${remove}" \
        --method DELETE --silent 2>/dev/null || true
    gh api "repos/${repo}/issues/${num}/labels" \
        -f "labels[]=${add}" --method POST --silent
}

# ---------------------------------------------------------------------------
# Compound actions — advisor
# ---------------------------------------------------------------------------

# Send a PR back to its student: comment, convert to draft, swap review→wip.
#   send_pr_back_to_student_with_comment <number> <comment_body>
send_pr_back_to_student_with_comment() {
    local num="$1" body="$2"
    gh pr comment "$num" --body "$body"
    gh pr ready "$num" --undo
    swap_gh_pr_label "$num" "status:review" "status:wip"
}

# Close a dead-end PR: comment explaining why, close, delete remote branch.
#   close_pr_with_comment <number> <reason>
close_pr_with_comment() {
    local num="$1" reason="$2"
    gh pr comment "$num" --body "ADVISOR: Closing PR #${num} because ${reason}."
    gh pr close "$num" --delete-branch
}

# ---------------------------------------------------------------------------
# Compound actions — student
# ---------------------------------------------------------------------------

# Mark a PR as ready for advisor review: mark ready + swap wip→review.
#   mark_ready_for_review <number>
mark_ready_for_review() {
    local num="$1"
    gh pr ready "$num"
    swap_gh_pr_label "$num" "status:wip" "status:review"
}

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

# List human-created GitHub Issues addressed to a role (+ team issues).
# Returns a JSON array, deduplicated by issue number.
#   check_gh_issues <role_label>
#   e.g. check_gh_issues "noam"  OR  check_gh_issues "student:frieren"
check_gh_issues() {
    local role="$1"
    local role_issues team_issues
    role_issues=$(gh issue list --label "human" --label "$role" --state open \
        --json number,title,updatedAt,comments 2>/dev/null || echo "[]")
    team_issues=$(gh issue list --label "human" --label "team" --state open \
        --json number,title,updatedAt,comments 2>/dev/null || echo "[]")
    python3 -c "
import json, sys
a = json.loads('''$role_issues''')
b = json.loads('''$team_issues''')
seen = set()
merged = []
for i in a + b:
    if i['number'] not in seen:
        seen.add(i['number'])
        merged.append(i)
print(json.dumps(merged))
"
}

# List PRs that are ready for advisor review on a given branch.
# Returns a JSON array.
#   list_ready_for_review_prs <branch>
list_ready_for_review_prs() {
    local branch="$1"
    gh pr list --label "$branch" --label "status:review" \
        --json number,title,headRefName,labels
}

# List all open PRs on a branch (any status).
# Returns a JSON array.
#   list_all_prs <branch>
list_all_prs() {
    local branch="$1"
    gh pr list --label "$branch" \
        --json number,title,state,labels,headRefName,isDraft
}

# List WIP PRs assigned to a specific student.
# Returns a JSON array.
#   student_poll_for_work <student_name>
student_poll_for_work() {
    local name="$1"
    gh pr list --label "student:${name}" --label "status:wip" \
        --json number,title,headRefName,body
}

# Compute which students are idle (have no status:wip PR).
# Expects a comma-separated student list and the advisor branch.
# Returns a JSON array of idle student names.
#   list_idle_students <student_names_csv> <branch>
list_idle_students() {
    local students_csv="$1" branch="$2"
    local all_prs
    all_prs=$(gh pr list --label "$branch" --label "status:wip" \
        --json labels 2>/dev/null || echo "[]")
    python3 -c "
import json, sys
students = [s.strip() for s in '''$students_csv'''.split(',') if s.strip()]
prs = json.loads('''$all_prs''')
busy = set()
for pr in prs:
    for label in pr.get('labels', []):
        name = label.get('name', '')
        if name.startswith('student:'):
            busy.add(name.split(':', 1)[1])
print(json.dumps([s for s in students if s not in busy]))
"
}
