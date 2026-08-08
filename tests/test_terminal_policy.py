# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

from pathlib import Path

import pytest

from senpai_agent.hooks import terminal_policy

WORKSPACE = Path("/workspace")


def is_allowed(command: str) -> bool:
    return terminal_policy(command, "student", WORKSPACE).allowed


@pytest.mark.parametrize(
    "command",
    [
        "git push origin experiment",
        "git config alias.ship push",
        "env -S 'command gh pr merge 17 --squash'",
        "gh issue comment 12 --body done",
        "gh api --methodPATCH repos/wandb/senpai/pulls/17",
        "gh api -Fbody=@comment.md repos/wandb/senpai/issues/17",
        "gh pr checks 17 --watch",
        "gh workflow run deploy.yml",
        "curl -XPOST https://API.GitHub.com/repos/wandb/senpai/pulls/17",
        "curl https://api.github.com/repos/wandb/senpai/issues/17 -d state=closed",
    ],
)
def test_policy_denies_recognized_publication_and_github_mutations(command: str):
    assert is_allowed(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "git status --short",
        "git add model.py",
        "git commit -m 'record experiment'",
        "git -c color.ui=always diff --stat",
        "git config user.name",
        "git config user.email 'Senpai Student'",
        "git config --get remote.origin.url",
        "git remote -v",
        "git show HEAD | git patch-id --stable",
        "git rebase research && git show HEAD | git patch-id --stable",
        "gh pr view 17 --json title",
        "gh api repos/wandb/senpai/pulls/17",
        "env GH_HOST=github.com gh repo view wandb/senpai",
        "curl https://api.github.com/repos/wandb/senpai/pulls/17",
    ],
)
def test_policy_allows_explicit_local_git_and_read_only_github_commands(
    command: str,
):
    assert is_allowed(command) is True


def test_one_shot_git_alias_cannot_hide_a_push():
    assert is_allowed("git -c alias.ship=push ship origin experiment") is False


@pytest.mark.parametrize(
    "command",
    [
        "git ship origin experiment",
        "git send-pack origin refs/heads/experiment",
    ],
)
def test_unknown_git_subcommands_cannot_bypass_branch_publication(command: str):
    assert is_allowed(command) is False


def test_compound_command_denial_names_the_rejected_git_subcommand():
    decision = terminal_policy(
        "git rebase research && git push origin experiment",
        "student",
        WORKSPACE,
    )

    assert decision.allowed is False
    assert "git push" in decision.reason
    assert "git rebase" not in decision.reason


@pytest.mark.parametrize(
    "command",
    [
        "python train.py --epochs 10",
        "uv run python scripts/train_model.py",
        "torchrun --nproc-per-node 4 train.py",
        "./train_baseline.py --debug",
        "for id in 1 2; do python train.py --epochs 10; done",
    ],
)
def test_policy_denies_recognized_training_launches(command: str):
    assert is_allowed(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q tests/test_train.py",
        "python -c 'print(\"train.py\")'",
        "python train.py --help",
        "timeout 120 python train.py --help 2>&1 | grep epochs",
        "tail -n 50 training.log",
        "for id in run-a run-b; do grep -n \"$id\" results.log; done",
    ],
)
def test_policy_allows_bounded_training_inspection(command: str):
    assert is_allowed(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "python train.py --help --epochs 10",
        "python -m package.train --help --epochs 10",
        "torchrun --help train.py",
    ],
)
def test_help_flags_do_not_hide_training_arguments(command: str):
    assert is_allowed(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "tail -f training.log",
        "timeout 3600 tail -f training.log",
        "setsid sleep 3600",
        "for (( ; ; )); do echo waiting; done",
        "while true; do echo waiting; done",
    ],
)
def test_policy_denies_foreground_polling(command: str):
    assert is_allowed(command) is False


def test_quoted_file_heredoc_treats_restricted_words_as_literal_data():
    command = """cat > /tmp/notes.txt <<'EOF'
git push origin experiment
gh pr merge 17 --squash
python train.py --epochs 10
sleep 300
EOF
"""

    assert is_allowed(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "bash <<'EOF'\ngit push origin experiment\nEOF",
        "cat <<'EOF' | sh\ngit push origin experiment\nEOF",
        "cat > >(sh) <<'EOF'\ngit push origin experiment\nEOF",
        "cat() { sh; }; cat >/tmp/a <<'EOF'\ngit push origin experiment\nEOF",
    ],
)
def test_executable_heredocs_remain_subject_to_policy(command: str):
    assert is_allowed(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "git push origin experiment; cat <<'EOF'\ntext\nEOF",
        "cat <<'EOF'\ntext\nEOF\ngit push origin experiment",
        "cat <<'EOF'; python train.py --epochs 10\ntext\nEOF",
        "printf '%s\\n' \"<<'EOF'\"\ngit push origin experiment\nEOF",
    ],
)
def test_commands_around_literal_heredocs_remain_subject_to_policy(command: str):
    assert is_allowed(command) is False


def test_eval_cannot_hide_a_push_inside_a_nested_heredoc():
    command = (
        'eval "$(cat "$(echo x >/tmp/y; echo -)" <<\'EOF\'\n'
        "git push origin experiment\nEOF\n)\""
    )

    assert is_allowed(command) is False
