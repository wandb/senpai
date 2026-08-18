"""Isolated Git subprocess and credential transport helpers."""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.secrets import scrub_github_credentials


GIT_EXECUTABLE = "/usr/bin/git"
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PROXY_ENVIRONMENT = (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)
_GIT_TRANSPORT_ENVIRONMENT = (
    "GIT_ASKPASS",
    "GIT_CONFIG_PARAMETERS",
    "GIT_PROXY_COMMAND",
    "GIT_SSL_CAINFO",
    "GIT_SSL_CAPATH",
    "GIT_SSL_NO_VERIFY",
    "SSH_ASKPASS",
)


class GitWorkflowPreconditionError(RuntimeError):
    """Local or remote Git state does not permit a safe push."""


def github_repository_url(repo: str) -> str:
    """Return the only authenticated GitHub transport URL Senpai permits."""

    if not _GITHUB_REPOSITORY.fullmatch(repo) or any(
        part in {".", ".."} for part in repo.split("/")
    ):
        raise ValueError("repo must use a safe owner/name form")
    return f"https://github.com/{repo}.git"


@contextmanager
def isolated_bare_repository() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="senpai-git-network-") as directory:
        repository = Path(directory) / "objects.git"
        run_git(Path(directory), "init", "--bare", str(repository))
        yield repository


@contextmanager
def staged_commit(workspace: Path, commit_sha: str) -> Iterator[Path]:
    object_directory = Path(
        run_git(
            workspace,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "objects",
        )
    ).resolve()
    if not object_directory.is_dir() or "\n" in str(object_directory):
        raise GitWorkflowPreconditionError("workspace Git object directory is invalid")
    with isolated_bare_repository() as repository:
        alternates = repository / "objects" / "info" / "alternates"
        alternates.write_text(f"{object_directory}\n", encoding="utf-8")
        run_git(repository, "update-ref", "refs/senpai/local", commit_sha)
        staged_sha = run_git(repository, "rev-parse", "refs/senpai/local^{commit}")
        if staged_sha != commit_sha:
            raise GitWorkflowPreconditionError(
                f"staged commit is {staged_sha}, expected {commit_sha}"
            )
        yield repository


def remote_head(
    workspace: Path,
    remote: str,
    branch: str,
    *,
    token: SecretStr | None,
) -> str:
    result = run_git(
        workspace,
        "ls-remote",
        "--refs",
        remote,
        f"refs/heads/{branch}",
        token=token,
    )
    return result.split(maxsplit=1)[0] if result else ""


def run_git(
    workspace: Path,
    *arguments: str,
    input_text: str | None = None,
    token: SecretStr | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    environment = git_process_env(token)
    if extra_env:
        environment.update(extra_env)
    completed = subprocess.run(
        [GIT_EXECUTABLE, *arguments],
        cwd=workspace,
        text=True,
        input=input_text,
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitWorkflowPreconditionError(
            f"git {' '.join(arguments[:2])} failed: {detail[:1000]}"
        )
    return completed.stdout.strip()


def git_process_env(token: SecretStr | None) -> dict[str, str]:
    """Build an isolated Git environment with optional in-memory GitHub auth."""

    _validate_token(token)
    environment = dict(os.environ)
    scrub_github_credentials(environment)
    for name in tuple(environment):
        if name.startswith("GIT_") or name in _PROXY_ENVIRONMENT:
            environment.pop(name)
    for name in _GIT_TRANSPORT_ENVIRONMENT:
        environment.pop(name, None)

    configuration: list[tuple[str, str]] = []
    if token is not None:
        credential = base64.b64encode(
            f"x-access-token:{token.get_secret_value()}".encode()
        ).decode()
        configuration.append(
            (
                "http.https://github.com/.extraHeader",
                f"Authorization: Basic {credential}",
            )
        )
    configuration.extend(
        (
            ("credential.helper", ""),
            ("core.hooksPath", os.devnull),
            ("http.proxy", ""),
            ("http.https://github.com/.proxy", ""),
            ("http.sslVerify", "true"),
            ("http.https://github.com/.sslVerify", "true"),
            ("http.followRedirects", "false"),
            ("http.https://github.com/.followRedirects", "false"),
            ("http.extraHeader", ""),
        )
    )
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "file:https",
            "GIT_CONFIG_COUNT": str(len(configuration)),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    for index, (key, value) in enumerate(configuration):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


def _validate_token(token: SecretStr | None) -> None:
    if token is not None and not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if token is not None and not token.get_secret_value().strip():
        raise ValueError("token must not be empty")
