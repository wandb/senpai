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

from senpai_agent.git_transport import GIT_EXECUTABLE
from senpai_agent.secrets import SHELL_STARTUP_ENV_NAMES
from senpai_agent.training import training_result_paths


QUEUED_FEEDBACK_MARKER = "queued-feedback-pending"


def queued_feedback_marker(state_dir: Path) -> Path:
    return state_dir / QUEUED_FEEDBACK_MARKER


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""


_SHELL_SEPARATOR_CHARACTERS = frozenset(";&|\n")
_SHELL_BODY_PREFIXES = {"do", "elif", "else", "if", "then"}
_SHELL_REEVALUATORS = {
    ".",
    "alias",
    "bind",
    "compgen",
    "complete",
    "compopt",
    "eval",
    "fc",
    "history",
    "source",
    "trap",
}
_SHELL_EXPANSION_MARKS = frozenset("$`*?[<({")
_SHELL_PROGRAMS = {
    "bash": "bash",
    "dash": "dash",
    "rbash": "bash",
    "sh": "sh",
    "zsh": "zsh",
}
_SHELL_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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
_REDIRECTION_OPERATORS = {
    "<",
    "<<",
    "<<<",
    ">",
    ">>",
    "<>",
    ">|",
    "<&",
    ">&",
    "&>",
    "&>>",
}
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


def _bash_commands(command: str) -> list[str] | None:
    import tree_sitter_bash
    from tree_sitter import Language, Parser

    source = command.encode()
    tree = Parser(Language(tree_sitter_bash.language())).parse(source)
    if tree.root_node.has_error:
        return None
    nodes = _descendants(tree.root_node)
    if any(
        node.type == "arithmetic_expansion"
        or node.type == "c_style_for_statement"
        or (
            node.type == "expansion"
            and source[node.start_byte : node.end_byte].endswith(b"@P}")
        )
        or (
            node.type == "compound_statement"
            and source[node.start_byte : node.end_byte].lstrip().startswith(b"((")
        )
        for node in nodes
    ):
        return None
    return [
        source[node.start_byte : node.end_byte].decode()
        for node in nodes
        if node.type in {"command", "declaration_command"}
    ]


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


def _program_index(tokens: list[str]) -> int | None:
    for index, token in enumerate(tokens):
        if not token.startswith(("/", "./")) and _assignment_name(token) is not None:
            continue
        return index
    return None


def _assignment_name(value: str) -> str | None:
    name, separator, _value = value.partition("=")
    if not separator:
        return None
    name = name.removesuffix("+")
    return name if _SHELL_VARIABLE_NAME.fullmatch(name) else None


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
        value_options={"-a", "--argv0", "-u", "--unset", "-C", "--chdir"},
    )


def _shell_command(arguments: list[str]) -> str | None:
    for position, value in enumerate(arguments):
        is_command_option = value == "-c" or (
            value.startswith("-") and not value.startswith("--") and "c" in value[1:]
        )
        if is_command_option and position + 1 < len(arguments):
            return arguments[position + 1]
    return None


def _enables_shell_allexport(arguments: Sequence[str]) -> bool:
    position = 0
    while position < len(arguments):
        argument = arguments[position]
        if argument == "--":
            return False
        if argument == "-o":
            if (
                position + 1 < len(arguments)
                and arguments[position + 1] == "allexport"
            ):
                return True
            position += 2
            continue
        if argument.startswith("-") and not argument.startswith("--"):
            options = argument[1:]
            if "a" in options:
                return True
            if "c" in options:
                return False
        position += 1
    return False


def _has_dynamic_set_arguments(arguments: Sequence[str]) -> bool:
    for argument in arguments:
        if argument == "--":
            return False
        if any(mark in argument for mark in _SHELL_EXPANSION_MARKS):
            return True
    return False


def _has_dynamic_shell_options(arguments: Sequence[str]) -> bool:
    position = 0
    while position < len(arguments):
        argument = arguments[position]
        if argument == "--":
            return False
        if any(mark in argument for mark in _SHELL_EXPANSION_MARKS):
            return True
        if argument == "-o":
            if position + 1 >= len(arguments):
                return False
            if any(
                mark in arguments[position + 1]
                for mark in _SHELL_EXPANSION_MARKS
            ):
                return True
            position += 2
            continue
        if argument.startswith("-") and not argument.startswith("--"):
            if "c" in argument[1:]:
                return False
            position += 1
            continue
        return False
    return False


def _uses_shell_startup_files(arguments: Sequence[str]) -> bool:
    for argument in arguments:
        if argument == "--":
            return False
        reads_named_file = argument in {"--init-file", "--login", "--rcfile"}
        reads_named_file = reads_named_file or argument.startswith(
            ("--init-file=", "--rcfile=")
        )
        if reads_named_file:
            return True
        if argument.startswith("-") and not argument.startswith("--"):
            options = argument[1:]
            if {"i", "l"} & set(options):
                return True
            if "c" in options:
                return False
    return False


def _time_command(arguments: list[str]) -> list[str] | None:
    while arguments[:1] == ["-p"]:
        arguments = arguments[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or arguments[0].startswith("-"):
        return None
    return arguments


def _declaration_policy(program: str, arguments: list[str]) -> PolicyDecision:
    options = ""
    separator = False
    values: list[str] = []
    for argument in _without_redirections(arguments):
        if argument == "--":
            separator = True
            continue
        if not separator and argument[:1] in {"-", "+"}:
            options += argument[1:]
            continue
        values.append(argument)
    if {"i", "n"} & set(options):
        return PolicyDecision(
            False,
            "Do not use shell arithmetic or nameref variables.",
        )
    return _variable_name_policy(program, values)


def _read_policy(arguments: list[str]) -> PolicyDecision:
    names: list[str] = []
    arguments = _without_redirections(arguments)
    position = 0
    while position < len(arguments):
        argument = arguments[position]
        if argument == "--":
            names.extend(arguments[position + 1 :])
            break
        if argument == "-a":
            if position + 1 < len(arguments):
                names.append(arguments[position + 1])
            position += 2
            continue
        if argument in {"-d", "-i", "-n", "-N", "-p", "-t", "-u"}:
            position += 2
            continue
        if argument.startswith("-"):
            position += 1
            continue
        names.extend(arguments[position:])
        break
    return _variable_name_policy("read", names)


def _mapfile_target(arguments: Sequence[str]) -> str | None:
    arguments = _without_redirections(arguments)
    value_options = {"-C", "-O", "-c", "-d", "-n", "-s", "-u"}
    position = 0
    while position < len(arguments):
        argument = arguments[position]
        if argument == "--":
            return arguments[position + 1] if position + 1 < len(arguments) else None
        if argument in value_options:
            position += 2
            continue
        if argument.startswith("-"):
            position += 1
            continue
        return argument
    return None


def _without_redirections(arguments: Sequence[str]) -> list[str]:
    command_arguments: list[str] = []
    position = 0
    while position < len(arguments):
        operator_position = position + int(arguments[position].isdecimal())
        if (
            operator_position < len(arguments)
            and arguments[operator_position] in _REDIRECTION_OPERATORS
            and operator_position + 1 < len(arguments)
        ):
            position = operator_position + 2
            continue
        command_arguments.append(arguments[position])
        position += 1
    return command_arguments


def _variable_name_policy(
    program: str,
    arguments: Sequence[str],
) -> PolicyDecision:
    for argument in arguments:
        name = _assignment_name(argument)
        if name is None and "=" not in argument:
            name = argument
        if name in SHELL_STARTUP_ENV_NAMES:
            return PolicyDecision(
                False,
                f"Do not alter shell startup variable {name}.",
            )
        if name is None or _SHELL_VARIABLE_NAME.fullmatch(name) is None:
            return PolicyDecision(
                False,
                f"Do not pass dynamic variable names to {program}.",
            )
    return PolicyDecision(True)


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


def _shell_program(program: str, workspace: Path) -> str | None:
    shell = _SHELL_PROGRAMS.get(Path(program).name)
    if shell is not None or "/" not in program:
        return shell
    path = Path(program)
    if not path.is_absolute():
        path = workspace / path
    try:
        return _SHELL_PROGRAMS.get(path.resolve(strict=True).name)
    except OSError:
        return None


def _segment_policy(tokens: list[str], workspace: Path) -> PolicyDecision:
    index = _program_index(tokens)
    assignment_tokens = tokens if index is None else tokens[:index]
    for token in assignment_tokens:
        name = _assignment_name(token)
        if name in SHELL_STARTUP_ENV_NAMES:
            return PolicyDecision(
                False,
                f"Do not alter shell startup variable {name}.",
            )
    if index is None:
        return PolicyDecision(True)
    program_token = tokens[index]
    program = "." if program_token == "." else Path(program_token).name
    if any(mark in program for mark in ("$", "`", "*", "?", "\n", "\r")) or (
        "[" in program and program not in {"[", "[["}
    ):
        return PolicyDecision(
            False,
            "Do not construct executable names with shell expansion.",
        )

    arguments = tokens[index + 1 :]
    if program in _SHELL_BODY_PREFIXES:
        return _segment_policy(arguments, workspace)
    if program == "env":
        decision = _variable_name_policy(
            "env",
            [
                argument
                for argument in arguments
                if "=" in argument
                and _SHELL_VARIABLE_NAME.fullmatch(argument.partition("=")[0])
            ],
        )
        if not decision.allowed:
            return decision
        command = _env_command(arguments)
        return _segment_policy(command, workspace) if command else PolicyDecision(True)
    if program == "exec":
        command = _wrapper_command(arguments, value_options={"-a"})
        return _segment_policy(command, workspace) if command else PolicyDecision(True)
    if program in {"builtin", "command", "nohup"}:
        command = _wrapper_command(arguments)
        return _segment_policy(command, workspace) if command else PolicyDecision(True)
    if program == "timeout":
        command = _timeout_command(arguments)
        return _segment_policy(command, workspace) if command else PolicyDecision(True)
    if program == "setsid":
        command = _wrapper_command(arguments)
        return _segment_policy(command, workspace) if command else PolicyDecision(True)
    shell = _shell_program(program_token, workspace)
    if shell is not None:
        if shell == "zsh" or _uses_shell_startup_files(arguments):
            return PolicyDecision(
                False,
                "Do not launch nested shells that load startup files.",
            )
        if _has_dynamic_shell_options(arguments):
            return PolicyDecision(
                False,
                "Do not construct nested-shell options with expansion.",
            )
        if _enables_shell_allexport(arguments):
            return PolicyDecision(
                False,
                "Do not enable automatic export in nested shells.",
            )
        command = _shell_command(arguments)
        if command is not None:
            return terminal_policy(command, "", workspace)
    if program == "time":
        command = _time_command(arguments)
        if command is None:
            return PolicyDecision(False, "Senpai could not parse `time` safely.")
        return _segment_policy(command, workspace)
    if program == "coproc":
        return PolicyDecision(False, "Do not launch asynchronous shell coprocesses.")
    if program in _SHELL_REEVALUATORS:
        return PolicyDecision(
            False,
            "Do not use shell constructs that reinterpret commands or arguments.",
        )
    if program == "let":
        return PolicyDecision(False, "Do not use shell arithmetic evaluation.")
    if program == "set":
        if _has_dynamic_set_arguments(arguments):
            return PolicyDecision(
                False,
                "Do not construct shell options with expansion.",
            )
        if _enables_shell_allexport(arguments):
            return PolicyDecision(
                False,
                "Do not enable automatic export of shell variables.",
            )
    if program in {"declare", "local", "typeset"}:
        return _declaration_policy(program, arguments)
    if program in {"export", "readonly", "unset"}:
        return _declaration_policy(program, arguments)
    if program == "read":
        return _read_policy(arguments)
    if program == "getopts" and len(arguments) >= 2:
        decision = _variable_name_policy(program, arguments[1:2])
        if not decision.allowed:
            return decision
    if program in {"mapfile", "readarray"}:
        target = _mapfile_target(arguments)
        if target is not None:
            decision = _variable_name_policy(program, [target])
            if not decision.allowed:
                return decision
    if program in {"mapfile", "readarray"} and any(
        argument == "-C" or argument.startswith("-C")
        for argument in arguments
    ):
        return PolicyDecision(False, "Do not use shell callback evaluation.")
    if program == "hash" and any(
        argument == "-p" or argument.startswith("-p")
        for argument in arguments
    ):
        return PolicyDecision(False, "Do not bind alternate executable names.")
    if program == "jobs" and any(
        argument.startswith("-") and "x" in argument[1:]
        for argument in arguments
    ):
        return PolicyDecision(False, "Do not use shell command runners.")
    if program == "shopt" and "expand_aliases" in arguments:
        return PolicyDecision(False, "Do not enable shell alias expansion.")
    if program == "printf" and "-v" in arguments:
        return PolicyDecision(False, "Do not use printf to evaluate variable names.")
    if program in {"[", "[[", "test"} and "-v" in arguments:
        return PolicyDecision(False, "Do not evaluate dynamic variable names.")

    if program == "git":
        return _git_policy(arguments)
    if program == "gh":
        return _gh_policy(tokens, index)
    if program == "curl":
        return _curl_policy(tokens, index)

    if program == "uv" and arguments[:1] == ["run"]:
        return _segment_policy(tokens[index + 2 :], workspace)
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

    if program in {"for", "select"}:
        decision = _variable_name_policy(program, arguments[:1])
        if not decision.allowed:
            return decision
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
    del role
    nested_commands = _bash_commands(command)
    if nested_commands is None:
        return PolicyDecision(
            False,
            "Senpai could not parse this shell command safely.",
        )
    for nested_command in nested_commands:
        for segment in _command_segments(nested_command):
            decision = _segment_policy(segment, workspace)
            if not decision.allowed:
                return decision
    for segment in _command_segments(command):
        decision = _segment_policy(segment, workspace)
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
            [GIT_EXECUTABLE, "status", "--porcelain"],
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
