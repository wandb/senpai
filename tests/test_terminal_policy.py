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


@pytest.mark.parametrize(
    "command",
    [
        'echo "$(git push origin experiment)"',
        "echo `python train.py --epochs 10`",
        "cat <(gh issue comment 12 --body done)",
        "(setsid sleep 3600)",
        "{ git push origin experiment; }",
        "publish() { git push origin experiment; }; publish",
        "case x in x) git push origin experiment;; esac",
    ],
)
def test_nested_shell_execution_cannot_hide_restricted_commands(command: str):
    assert is_allowed(command) is False


def test_policy_allows_nested_shell_execution_when_every_command_is_safe():
    assert is_allowed('echo "$(date -u)"') is True


@pytest.mark.parametrize(
    "command",
    [
        "$(printf git) push origin experiment",
        "$'git' push origin experiment",
        "g$''it push origin experiment",
        "G=git; $G push origin experiment",
        "command $(printf git) push origin experiment",
        "env $(printf git) push origin experiment",
        "/usr/bin/g?t push origin experiment",
    ],
)
def test_dynamic_executable_names_cannot_hide_restricted_commands(command: str):
    assert is_allowed(command) is False


def test_alias_expansion_cannot_defer_restricted_command_parsing():
    command = "\n".join(
        [
            "shopt -s expand_aliases",
            "alias ship='git push'",
            "ship origin experiment",
        ]
    )

    assert is_allowed(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "[[ -v 'a[$(git push origin experiment)]' ]]",
        "printf -v 'a[$(git push origin experiment)]' x",
        "x='a[$(git push origin experiment)]'; echo $((x))",
        "x='a[$(git push origin experiment)]'; ((x))",
        (
            "x='a[$(git push origin experiment)]'; "
            "for ((i=x; i<1; i++)); do :; done"
        ),
        "x='a[$(git push origin experiment)]'; let x",
        "x='a[$(git push origin experiment)]'; declare -i y=x",
        (
            "declare -n ref='a[$(git push origin experiment)]'; "
            'echo "$ref"'
        ),
        "read 'a[$(git push origin experiment)]' <<< value",
        "declare 'a[$(git push origin experiment)]=x'",
        "typeset 'a[$(git push origin experiment)]=x'",
        "f() { local 'a[$(git push origin experiment)]=x'; }; f",
        "a=(x); unset 'a[$(git push origin experiment)]'",
        "mapfile -C 'git push origin experiment' -c 1 values",
        "mapfile -t PROMPT_COMMAND < commands.txt",
        "readarray -t PROMPT_COMMAND < commands.txt",
        "getopts a PROMPT_COMMAND -a",
        "x=PROMPT_COMMAND; mapfile -t \"$x\" < commands.txt",
        "x=PROMPT_COMMAND; readarray -t \"$x\" < commands.txt",
        "x=PROMPT_COMMAND; getopts a \"$x\" -a",
        "for PROMPT_COMMAND in 'git push origin experiment'; do :; done",
        "select PROMPT_COMMAND in 'git push origin experiment'; do break; done",
        "hash -p /usr/bin/git ship; ship push origin experiment",
        "trap 'git push origin experiment' EXIT",
        "fc -s status=push",
        "history -s 'git push origin experiment'",
        "jobs -x git push origin experiment",
        "bind -x '\"\\C-x\":git push origin experiment'",
        "complete -C 'git push origin experiment' git",
        "time git push origin experiment",
        "time python train.py --epochs 10",
        "time sleep 3600",
        "coproc git push origin experiment",
        "coproc date -u",
        "exec -a audit git push origin experiment",
        "env -a audit git push origin experiment",
        "env --argv0 audit git push origin experiment",
        "env --argv0=audit git push origin experiment",
        "source ./commands.sh",
        ". ./commands.sh",
        "BASH_ENV=/tmp/commands bash -c 'date -u'",
        "BASH_ENV=<(printf 'git push origin experiment\\n') bash -c 'date -u'",
        "env BASH_ENV=/tmp/commands bash -c 'date -u'",
        "export BASH_ENV=/tmp/commands; bash -c 'date -u'",
        "ENV=/tmp/commands sh -c 'date -u'",
        "ZDOTDIR=/tmp/commands zsh -c 'date -u'",
        "SHELLOPTS=xtrace PS4='$(git push origin experiment)' bash -c date",
        "PROMPT_COMMAND='git push origin experiment'",
        "PROMPT_COMMAND+='; git push origin experiment'",
        "export PROMPT_COMMAND='git push origin experiment'",
        "declare PROMPT_COMMAND='git push origin experiment'",
        "readonly PROMPT_COMMAND='git push origin experiment'",
        "PS0='$(git push origin experiment)'",
        "MAILCHECK=0 MAILPATH='/tmp/mail?$(git push origin experiment)'",
        "x='$(git push origin experiment)'; echo \"${x@P}\"",
        "set -a; for BASH_ENV in x; do bash -c 'date -u'; done",
        (
            "set -o allexport; set -- -x; getopts x BASH_ENV; "
            "bash -c 'date -u'"
        ),
        "set -ae; for BASH_ENV in x; do bash -c 'date -u'; done",
        "bash -ac 'for BASH_ENV in x; do bash -c date; done'",
        "v=a; set -$v; for BASH_ENV in x; do bash -c 'date -u'; done",
        (
            "v=allexport; set -o $v; for BASH_ENV in x; "
            "do bash -c 'date -u'; done"
        ),
        "v=ac; bash -$v 'for BASH_ENV in x; do bash -c date; done'",
        "set -{a,e}; for BASH_ENV in x; do bash -c date; done",
        "bash -lc 'date -u'",
        "bash --rcfile=/tmp/commands -ic 'date -u'",
        "zsh -f -c 'date -u'",
    ],
)
def test_shell_argument_reevaluation_cannot_hide_restricted_commands(command: str):
    assert is_allowed(command) is False


def test_shell_aliases_cannot_load_startup_files(tmp_path: Path):
    (tmp_path / "shell-runner").symlink_to("/bin/bash")

    assert terminal_policy(
        "./shell-runner -lc 'date -u'",
        "student",
        tmp_path,
    ).allowed is False
    assert is_allowed("rbash -lc 'date -u'") is False


@pytest.mark.parametrize(
    "command",
    [
        "(date -u)",
        "{ date -u; }",
        "report() { date -u; }; report",
        "case x in x) date -u;; esac",
    ],
)
def test_policy_preserves_safe_static_nested_commands(command: str):
    assert is_allowed(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "export CUDA_VISIBLE_DEVICES=0",
        "f() { local x=value; }; f",
        "declare -a values",
        "typeset x=value",
        "read value",
        "read value < input.txt",
        "readonly VERSION=1",
        "export RESULT=ok > output.txt",
        "unset value",
        "mapfile values",
        "hash -r",
        "time -p date -u",
        "exec -a audit date -u",
        "env --argv0 audit date -u",
        "set +a",
        "set +o allexport",
        "set -o nounset",
        "set -- one two",
        "bash -c 'date -u'",
    ],
)
def test_policy_preserves_safe_shell_variable_builtins(command: str):
    assert is_allowed(command) is True
