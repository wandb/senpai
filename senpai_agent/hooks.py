from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from senpai_agent.training import training_result_paths


QUEUED_FEEDBACK_MARKER = "queued-feedback-pending"


def queued_feedback_marker(state_dir: Path) -> Path:
    return state_dir / QUEUED_FEEDBACK_MARKER


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


_SHELL_SEPARATOR_CHARACTERS = frozenset(";&|\n")
_REDIRECTION_CHARACTERS = frozenset("<>&|")
_SHELL_BODY_PREFIXES = {"do", "elif", "else", "if", "then"}
_GH_READ_ONLY = {
    "auth": {"status"},
    "issue": {"list", "status", "view"},
    "pr": {"checks", "diff", "list", "status", "view"},
    "repo": {"list", "view"},
    "run": {"list", "view"},
    "search": {"commits", "issues", "prs", "repos"},
    "workflow": {"list", "view"},
}
_TRAIN_LAUNCHERS = {"accelerate", "deepspeed", "torchrun"}
_TRAIN_SCRIPT = re.compile(r"^train[^/]*[.]py$")
_HELP_FLAGS = {"-h", "--help"}
_REDIRECTION_OPERATORS = {"<", ">", ">>", "<>", ">|", "<&", ">&", "&>", "&>>"}
_GIT_TERMINAL_COMMANDS = {
    "add",
    "apply",
    "bisect",
    "blame",
    "branch",
    "cat-file",
    "check-ignore",
    "check-ref-format",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "describe",
    "diff",
    "diff-tree",
    "fetch",
    "for-each-ref",
    "gc",
    "grep",
    "hash-object",
    "help",
    "init",
    "log",
    "ls-files",
    "ls-remote",
    "ls-tree",
    "merge",
    "merge-base",
    "merge-tree",
    "mv",
    "name-rev",
    "notes",
    "patch-id",
    "range-diff",
    "rebase",
    "reflog",
    "reset",
    "restore",
    "rev-list",
    "rev-parse",
    "revert",
    "rm",
    "shortlog",
    "show",
    "show-ref",
    "sparse-checkout",
    "stash",
    "status",
    "submodule",
    "switch",
    "tag",
    "verify-commit",
    "verify-tag",
    "version",
    "worktree",
}


def _without_literal_file_heredocs(command: str) -> str:
    """Remove literal heredocs written directly to files by ``cat``."""
    if "<<" not in command:
        return command

    import tree_sitter_bash
    from tree_sitter import Language, Parser

    source = command.encode()
    tree = Parser(Language(tree_sitter_bash.language())).parse(source)
    if tree.root_node.has_error:
        return command
    if any(
        node.type == "function_definition"
        for node in _descendants(tree.root_node)
    ):
        return command

    policy_source = bytearray(source)
    nodes = [tree.root_node]
    while nodes:
        node = nodes.pop()
        if node.type == "heredoc_redirect" and _is_literal_cat_file_sink(
            node, source
        ):
            start = next(
                child for child in node.children if child.type == "heredoc_start"
            )
            delimiter = source[start.start_byte : start.end_byte]
            if any(mark in delimiter for mark in b"'\"\\"):
                for child in node.children:
                    if child.type in {"heredoc_body", "heredoc_end"}:
                        for position in range(child.start_byte, child.end_byte):
                            if policy_source[position] not in b"\r\n":
                                policy_source[position] = ord(" ")
        nodes.extend(node.children)
    return policy_source.decode()


def _is_literal_cat_file_sink(node: object, source: bytes) -> bool:
    parent = node.parent
    if parent is None or parent.type != "redirected_statement":
        return False
    body = parent.child_by_field_name("body")
    name = body.child_by_field_name("name") if body is not None else None
    if name is None or source[name.start_byte : name.end_byte] != b"cat":
        return False

    if any(child.type == "pipeline" for child in node.children):
        return False
    redirects = [
        redirect
        for owner in (parent, node)
        for redirect in owner.children_by_field_name("redirect")
        if redirect.type == "file_redirect"
    ]
    stdout_redirects = [
        redirect
        for redirect in redirects
        if _redirects_stdout(redirect, source)
    ]
    if not stdout_redirects:
        return False
    redirect = max(stdout_redirects, key=lambda candidate: candidate.start_byte)
    return _redirects_stdout_to_literal_file(redirect, source)


def _redirects_stdout(redirect: object, source: bytes) -> bool:
    operator = next(
        (child.type for child in redirect.children if child.type in {">", ">>", ">|"}),
        None,
    )
    descriptor = redirect.child_by_field_name("descriptor")
    descriptor_text = (
        source[descriptor.start_byte : descriptor.end_byte]
        if descriptor is not None
        else b"1"
    )
    return operator is not None and descriptor_text == b"1"


def _redirects_stdout_to_literal_file(redirect: object, source: bytes) -> bool:
    destination = redirect.child_by_field_name("destination")
    if destination is None:
        return False
    if destination.type not in {"word", "raw_string", "string"}:
        return False
    if destination.type == "word" and destination.named_child_count:
        return False
    if destination.type == "string" and any(
        child.type != "string_content" for child in destination.named_children
    ):
        return False
    target = source[destination.start_byte : destination.end_byte].strip(b"'\"")
    return target not in {b"-", b"/dev/stdout", b"/dev/stderr"} and not target.startswith(
        (b"/dev/fd/", b"/proc/")
    )


def _descendants(root: object) -> list[object]:
    nodes = [root]
    for node in nodes:
        nodes.extend(node.children)
    return nodes


def _command_segments(command: str) -> list[list[str]]:
    command = _without_literal_file_heredocs(command)
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>\n")
    lexer.commenters = ""
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lexer:
        if set(token) <= _SHELL_SEPARATOR_CHARACTERS:
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _is_redirection(token: str) -> bool:
    characters = set(token)
    return bool(characters & {"<", ">"}) and characters <= _REDIRECTION_CHARACTERS


def _is_assignment(token: str) -> bool:
    if "=" not in token or token.startswith(("/", "./")):
        return False
    name, _, _value = token.partition("=")
    return name.replace("_", "").isalnum()


def _program_index(tokens: list[str]) -> int | None:
    """Locate the command word after any leading assignments and redirections."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _is_redirection(token):
            index += 2  # the operator and its target
        elif (
            token.isdigit()
            and index + 1 < len(tokens)
            and _is_redirection(tokens[index + 1])
        ):
            index += 1  # a file descriptor prefix such as the 2 in 2>&1
        elif _is_assignment(token):
            index += 1
        else:
            return index
    return None


def _gh_policy(tokens: list[str], index: int) -> PolicyDecision:
    arguments = tokens[index + 1 :]
    noun_index = next(
        (
            position
            for position, value in enumerate(arguments)
            if value == "api" or value in _GH_READ_ONLY
        ),
        None,
    )
    if noun_index is None:
        if any(value in {"--help", "--version"} for value in arguments):
            return PolicyDecision(True)
        return PolicyDecision(
            False,
            "Only explicitly read-only gh commands may use the terminal.",
        )

    noun = arguments[noun_index]
    remaining = arguments[noun_index + 1 :]
    if noun == "api":
        method = "GET"
        has_body = False
        for position, value in enumerate(remaining):
            if value in {"-X", "--method"} and position + 1 < len(remaining):
                method = remaining[position + 1].upper()
            elif value.startswith("-X") and len(value) > 2:
                method = value[2:].upper()
            elif value.startswith("--method="):
                method = value.partition("=")[2].upper()
            elif value.startswith("--method") and len(value) > len("--method"):
                method = value.removeprefix("--method").upper()
            elif value in {
                "-f",
                "-F",
                "--field",
                "--raw-field",
                "--input",
            } or value.startswith(
                (
                    "-f=",
                    "-F=",
                    "-f",
                    "-F",
                    "--field=",
                    "--raw-field=",
                    "--input=",
                )
            ):
                has_body = True
        if method != "GET" or has_body:
            return PolicyDecision(
                False,
                "Use a typed Senpai GitHub tool for mutating GitHub API calls.",
            )
        return PolicyDecision(True)

    operation = next((value for value in remaining if not value.startswith("-")), "")
    if operation not in _GH_READ_ONLY[noun]:
        return PolicyDecision(
            False,
            f"Use a typed Senpai GitHub tool for `gh {noun}` mutations.",
        )
    if noun == "pr" and operation == "checks" and any(
        value == "--watch" or value.startswith("--watch=") for value in remaining
    ):
        return PolicyDecision(
            False,
            "Do not model-poll GitHub checks; let the controller deliver updates.",
        )
    return PolicyDecision(True)


def _wrapper_command(
    arguments: list[str],
    *,
    value_options: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    position = 0
    while position < len(arguments):
        value = arguments[position]
        if value == "--":
            return arguments[position + 1 :]
        if value in value_options:
            position += 2
            continue
        if value.startswith("-") or _program_index([value]) is None:
            position += 1
            continue
        return arguments[position:]
    return []


def _env_command(arguments: list[str]) -> list[str]:
    for position, value in enumerate(arguments):
        if value in {"-S", "--split-string"} and position + 1 < len(arguments):
            return shlex.split(arguments[position + 1]) + arguments[position + 2 :]
        if value.startswith("--split-string="):
            return shlex.split(value.partition("=")[2]) + arguments[position + 1 :]
    return _wrapper_command(
        arguments,
        value_options={"-u", "--unset", "-C", "--chdir"},
    )


def _shell_command(arguments: list[str]) -> str | None:
    for position, value in enumerate(arguments):
        is_command_option = value == "-c" or (
            value.startswith("-") and not value.startswith("--") and "c" in value[1:]
        )
        if is_command_option and position + 1 < len(arguments):
            return arguments[position + 1]
    return None


def _curl_policy(tokens: list[str], index: int) -> PolicyDecision:
    arguments = tokens[index + 1 :]
    if not any("api.github.com" in value.casefold() for value in arguments):
        return PolicyDecision(True)

    method = "GET"
    for position, value in enumerate(arguments):
        if value in {"-X", "--request"} and position + 1 < len(arguments):
            method = arguments[position + 1].upper()
        elif value.startswith(("-X", "--request=")):
            method = value.removeprefix("-X").partition("=")[-1].upper()
        elif value in {
            "-d",
            "--data",
            "--data-ascii",
            "--data-binary",
            "--data-raw",
            "--json",
        } or value.startswith(
            (
                "-d",
                "--data=",
                "--data-ascii=",
                "--data-binary=",
                "--data-raw=",
                "--json=",
            )
        ):
            if method == "GET":
                method = "POST"
        elif (
            value in {"-T", "--upload-file"}
            or value.startswith(("-T", "--upload-file="))
        ) and method == "GET":
            method = "PUT"
    if method != "GET":
        return PolicyDecision(
            False,
            "Use a typed Senpai GitHub tool for mutating GitHub API calls.",
        )
    return PolicyDecision(True)


def _python_launches_training(tokens: list[str], index: int) -> bool:
    arguments = tokens[index + 1 :]
    if "-c" in arguments:
        return False
    if "-m" in arguments:
        position = arguments.index("-m")
        if position + 1 < len(arguments):
            module = arguments[position + 1]
            is_training_module = (
                "train" in module.lower() or module == "torch.distributed.run"
            )
            return is_training_module and not _help_only(arguments[position + 2 :])
    script_position = next(
        (
            position
            for position, value in enumerate(arguments)
            if not value.startswith("-")
        ),
        None,
    )
    if script_position is None:
        return False
    script = arguments[script_position]
    return bool(_TRAIN_SCRIPT.fullmatch(Path(script).name)) and not _help_only(
        arguments[script_position + 1 :]
    )


def _help_only(arguments: list[str]) -> bool:
    command_arguments: list[str] = []
    position = 0
    while position < len(arguments):
        operator_position = position + int(arguments[position].isdecimal())
        if (
            operator_position < len(arguments)
            and arguments[operator_position] in _REDIRECTION_OPERATORS
        ):
            target_position = operator_position + 1
            if target_position >= len(arguments):
                return False
            position = target_position + 1
            continue
        command_arguments.append(arguments[position])
        position += 1
    return len(command_arguments) == 1 and command_arguments[0] in _HELP_FLAGS


def _timeout_command(arguments: list[str]) -> list[str]:
    position = 0
    while position < len(arguments) and arguments[position].startswith("-"):
        option = arguments[position]
        if option == "--":
            position += 1
            break
        if option in {"-k", "--kill-after", "-s", "--signal"}:
            position += 2
        else:
            position += 1
    if position >= len(arguments):
        return []
    return arguments[position + 1 :]


def _git_policy(arguments: list[str]) -> PolicyDecision:
    command_line = _wrapper_command(
        arguments,
        value_options={
            "-C",
            "-c",
            "--config-env",
            "--git-dir",
            "--namespace",
            "--work-tree",
        },
    )
    if not command_line:
        return PolicyDecision(True)
    command, *options = command_line
    if command == "config":
        read_only = {"--get", "--get-all", "--get-regexp", "--list", "-l"}
        if read_only & set(options):
            return PolicyDecision(True)
        if (
            len(options) in {1, 2}
            and options[0] in {"user.name", "user.email"}
            and all(not value.startswith("-") for value in options)
        ):
            return PolicyDecision(True)
    elif command == "remote":
        operation = next(
            (value for value in options if not value.startswith("-")),
            None,
        )
        if operation in {None, "get-url", "show"}:
            return PolicyDecision(True)
    elif command in _GIT_TERMINAL_COMMANDS:
        return PolicyDecision(True)
    return PolicyDecision(
        False,
        f"Terminal use of `git {command}` is not allowed; use explicit local or "
        "read-only Git commands, and use the typed Senpai tool for branch "
        "publication.",
    )


def _segment_policy(tokens: list[str]) -> PolicyDecision:
    index = _program_index(tokens)
    if index is None:
        return PolicyDecision(True)
    program = Path(tokens[index]).name

    arguments = tokens[index + 1 :]
    if program in _SHELL_BODY_PREFIXES:
        return _segment_policy(arguments)
    if program == "env":
        command = _env_command(arguments)
        return _segment_policy(command) if command else PolicyDecision(True)
    if program in {"command", "exec", "nohup"}:
        command = _wrapper_command(arguments)
        return _segment_policy(command) if command else PolicyDecision(True)
    if program == "timeout":
        command = _timeout_command(arguments)
        return _segment_policy(command) if command else PolicyDecision(True)
    if program == "setsid":
        command = _wrapper_command(arguments)
        return _segment_policy(command) if command else PolicyDecision(True)
    if program in {"bash", "dash", "sh", "zsh"}:
        command = _shell_command(arguments)
        if command is not None:
            return terminal_policy(command, "", Path.cwd())
    if program == "eval":
        return PolicyDecision(
            False,
            "Do not use eval to execute commands hidden from Senpai's policy.",
        )

    if program == "git":
        return _git_policy(arguments)
    if program == "gh":
        return _gh_policy(tokens, index)
    if program == "curl":
        return _curl_policy(tokens, index)

    if program == "uv" and arguments[:1] == ["run"]:
        return _segment_policy(tokens[index + 2 :])
    if program.startswith("python") and _python_launches_training(tokens, index):
        return PolicyDecision(
            False,
            "Use run_training so timeouts, logs, status, and W&B IDs are supervised.",
        )
    if (
        program in _TRAIN_LAUNCHERS or _TRAIN_SCRIPT.fullmatch(program)
    ) and not _help_only(arguments):
        return PolicyDecision(
            False,
            "Use run_training so timeouts, logs, status, and W&B IDs are supervised.",
        )

    if program == "for":
        if any("((" in argument for argument in arguments):
            return PolicyDecision(
                False,
                "Do not run potentially unbounded foreground loops; use Senpai "
                "events or status tools.",
            )
        return PolicyDecision(True)
    if program in {"sleep", "watch", "while", "until"}:
        return PolicyDecision(
            False,
            "Do not run foreground polling loops; use Senpai events or status tools.",
        )
    if program == "tail" and any(
        argument == "--follow" or argument.startswith("-") and "f" in argument[1:]
        for argument in tokens[index + 1 :]
    ):
        return PolicyDecision(
            False,
            "Do not stream logs; use get_training_status for bounded updates.",
        )
    return PolicyDecision(True)


def terminal_policy(
    command: str,
    role: str,
    workspace: Path,
) -> PolicyDecision:
    del role, workspace
    try:
        segments = _command_segments(command)
    except ValueError as error:
        return PolicyDecision(
            False,
            f"Senpai could not parse this command ({error}). Balance the quotes, "
            "or quote the heredoc delimiter (<<'EOF') so its body is literal text.",
        )
    for segment in segments:
        decision = _segment_policy(segment)
        if not decision.allowed:
            return decision
    return PolicyDecision(True)


def _stop_policy(
    role: str,
    working_dir: Path,
    state_dir: Path | None,
    *,
    require_clean_workspace: bool = True,
) -> PolicyDecision:
    if role != "student" or not (working_dir / ".git").exists():
        return PolicyDecision(True)
    if state_dir is not None:
        running = {
            path.stem
            for path in training_result_paths(state_dir / "training")
            if json.loads(path.read_text()).get("state") == "running"
        }
        monitored = {
            path.stem for path in (state_dir / "training" / "monitors").glob("*.json")
        }
        unmonitored = running - monitored
        if unmonitored:
            return PolicyDecision(
                False,
                "Training is still running without the terminal monitor that "
                "run_training normally registers; call monitor_training to "
                "repair it before finishing: "
                f"{', '.join(sorted(unmonitored))}",
            )
    if require_clean_workspace:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=working_dir,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        if status.strip():
            return PolicyDecision(
                False,
                "Commit the exact implementation before training or discard "
                "incidental assignment changes before finishing.",
            )
    return PolicyDecision(True)


def _emit(decision: PolicyDecision) -> int:
    output = {"decision": "allow" if decision.allowed else "deny"}
    if decision.reason:
        output["reason"] = decision.reason
        output["additionalContext"] = decision.reason
    print(json.dumps(output))
    return 0 if decision.allowed else 2


def hook_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    command = (argv or sys.argv[1:])[0]
    try:
        event = json.loads(sys.stdin.read() or "{}")
        working_dir = Path(event.get("working_dir") or os.getcwd()).resolve()
        role = env.get("SENPAI_ROLE", "")
        if command == "pre-tool-use":
            tool_input = event.get("tool_input") or {}
            return _emit(terminal_policy(str(tool_input["command"]), role, working_dir))
        if command == "stop":
            state_dir_value = env.get("SENPAI_OPENHANDS_STATE_DIR")
            state_dir = Path(state_dir_value).resolve() if state_dir_value else None
            queued_feedback = (
                state_dir is not None
                and queued_feedback_marker(state_dir).is_file()
            )
            return _emit(
                _stop_policy(
                    role,
                    working_dir,
                    state_dir,
                    require_clean_workspace=not queued_feedback,
                )
            )
        if command == "session-end":
            return _emit(PolicyDecision(True))
        raise ValueError(f"unknown hook command: {command}")
    except Exception:  # noqa: BLE001
        return _emit(
            PolicyDecision(False, "Senpai safety policy could not be evaluated.")
        )


if __name__ == "__main__":
    raise SystemExit(hook_main())
