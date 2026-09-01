#!/bin/bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

# Bootstrap-only Git guards. Model-facing GitHub writes use typed tools.

install_senpai_git_guard() {
    local workdir="$1" askpass_file="$2"
    if [ -z "$workdir" ] || [ -z "$askpass_file" ]; then
        echo "install_senpai_git_guard: usage: <workdir> <askpass-file>" >&2
        return 2
    fi

    git remote set-url --push origin DISABLED
    git config remote.origin.pushurl DISABLED
    git config --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf 2>/dev/null || true

    mkdir -p .git/hooks
    cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
echo "ERROR: refusing to push from the senpai runner repo; use the cloned target repo instead." >&2
exit 1
EOF
    chmod +x .git/hooks/pre-push

    cat > "$askpass_file" <<'EOF'
#!/bin/sh
case "$1" in
    *Username*) printf '%s\n' x-access-token ;;
    *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
esac
EOF
    chmod 700 "$askpass_file"
    export GIT_ASKPASS="$askpass_file"
    export GIT_TERMINAL_PROMPT=0
    git config --global --unset-all credential.helper 2>/dev/null || true
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
refuse_push() {
    echo "SENPAI-GIT-GUARD: refusing $1" >&2
    exit 2
}

advisor_owns_student_ref() {
    candidate="$1"
    previous_ifs="$IFS"
    IFS=,
    for student in ${STUDENT_NAMES:-}; do
        case "$candidate" in
            "refs/heads/$student/"*)
                IFS="$previous_ifs"
                return 0
                ;;
        esac
    done
    IFS="$previous_ifs"
    return 1
}

source_is_branch_tip() {
    local_ref="$1"
    local_sha="$2"
    branch_ref="$3"
    [ "$local_ref" = "$branch_ref" ] && return 0
    [ "$local_ref" = "$local_sha" ] || return 1
    branch_sha="$(git rev-parse --verify "$branch_ref" 2>/dev/null)" || return 1
    [ "$branch_sha" = "$local_sha" ]
}

while read -r local_ref local_sha remote_ref _; do
    [ "$local_ref" != "(delete)" ] || refuse_push "branch deletion"
    case "$SENPAI_ROLE:$remote_ref" in
        "advisor:refs/heads/$ADVISOR_BRANCH")
            source_is_branch_tip \
                "$local_ref" "$local_sha" "refs/heads/$ADVISOR_BRANCH" ||
                refuse_push "$ADVISOR_BRANCH from ${local_ref:-<unknown>}"
            ;;
        advisor:refs/heads/*)
            advisor_owns_student_ref "$remote_ref" ||
                refuse_push "advisor write to $remote_ref"
            ;;
        "student:refs/heads/$STUDENT_NAME/"*)
            source_is_branch_tip "$local_ref" "$local_sha" "$remote_ref" ||
                refuse_push "$remote_ref from ${local_ref:-<unknown>}"
            ;;
        *) refuse_push "$SENPAI_ROLE write to $remote_ref" ;;
    esac
done
EOF
    chmod +x "$target_workdir/.git/hooks/pre-push"
}
