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
senpai_repo() {
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
#   senpai_label_swap <number> <remove_label> <add_label>
senpai_label_swap() {
    local num="$1" remove="$2" add="$3"
    local repo
    repo=$(senpai_repo)
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
#   senpai_send_back <number> <comment_body>
senpai_send_back() {
    local num="$1" body="$2"
    gh pr comment "$num" --body "$body"
    gh pr ready "$num" --undo
    senpai_label_swap "$num" "status:review" "status:wip"
}

# Close a dead-end PR: comment explaining why, close, delete remote branch.
#   senpai_close_pr <number> <reason>
senpai_close_pr() {
    local num="$1" reason="$2"
    gh pr comment "$num" --body "ADVISOR: Closing PR #${num} because ${reason}."
    gh pr close "$num" --delete-branch
}

# ---------------------------------------------------------------------------
# Compound actions — student
# ---------------------------------------------------------------------------

# Mark a PR as ready for advisor review: mark ready + swap wip→review.
#   senpai_mark_review <number>
senpai_mark_review() {
    local num="$1"
    gh pr ready "$num"
    senpai_label_swap "$num" "status:wip" "status:review"
}

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

# List human-created GitHub Issues addressed to a role (+ team issues).
# Returns a JSON array, deduplicated by issue number.
#   senpai_check_issues <role_label>
#   e.g. senpai_check_issues "noam"  OR  senpai_check_issues "student:frieren"
senpai_check_issues() {
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
#   senpai_list_review_prs <branch>
senpai_list_review_prs() {
    local branch="$1"
    gh pr list --label "$branch" --label "status:review" \
        --json number,title,headRefName,labels
}

# List all open PRs on a branch (any status).
# Returns a JSON array.
#   senpai_list_all_prs <branch>
senpai_list_all_prs() {
    local branch="$1"
    gh pr list --label "$branch" \
        --json number,title,state,labels,headRefName,isDraft
}

# List WIP PRs assigned to a specific student.
# Returns a JSON array.
#   senpai_poll_work <student_name>
senpai_poll_work() {
    local name="$1"
    gh pr list --label "student:${name}" --label "status:wip" \
        --json number,title,headRefName,body
}

# Compute which students are idle (have no status:wip PR).
# Expects a comma-separated student list and the advisor branch.
# Prints one idle student name per line.
#   senpai_idle_students <student_names_csv> <branch>
senpai_idle_students() {
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
for s in students:
    if s not in busy:
        print(s)
"
}
