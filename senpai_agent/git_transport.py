"""Isolated Git subprocess and credential transport helpers."""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.secrets import scrub_github_credentials


GIT_EXECUTABLE = "/usr/bin/git"
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
PROXY_ENVIRONMENT = (
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
    shallow = Path(
        run_git(
            workspace,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "shallow",
        )
    )
    with isolated_bare_repository() as repository:
        alternates = repository / "objects" / "info" / "alternates"
        alternates.write_text(f"{object_directory}\n", encoding="utf-8")
        # Shallow grafts travel with the objects; without them history walks
        # fail at the clone boundary. Grafts only remove parent edges.
        if shallow.is_file():
            shutil.copyfile(shallow, repository / "shallow")
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
    expected_ref = f"refs/heads/{branch}"
    result = run_git(
        workspace,
        "ls-remote",
        "--refs",
        remote,
        expected_ref,
        token=token,
    )
    # ls-remote patterns match ref-name suffixes as well as full names.
    for line in result.splitlines():
        sha, ref = line.split(maxsplit=1)
        if ref == expected_ref:
            return sha
    return ""


def run_git(
    workspace: Path,
    *arguments: str,
    input_text: str | None = None,
    token: SecretStr | None = None,
    extra_env: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
    timeout: float | None = None,
) -> str:
    if token is not None and environment is not None:
        raise ValueError("pass a token or a complete environment, not both")
    environment = (
        dict(environment) if environment is not None else git_process_env(token)
    )
    if extra_env:
        environment.update(extra_env)
    try:
        completed = subprocess.run(
            [GIT_EXECUTABLE, "-c", f"safe.directory={workspace.resolve()}", *arguments],
            cwd=workspace,
            text=True,
            input=input_text,
            capture_output=True,
            env=environment,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise GitWorkflowPreconditionError(
            f"git is not installed at {GIT_EXECUTABLE}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise GitWorkflowPreconditionError(
            f"git {' '.join(arguments[:2])} timed out after {timeout:g} s"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitWorkflowPreconditionError(
            f"git {' '.join(arguments[:2])} failed: {detail[:1000]}"
        )
    return completed.stdout.strip()


def git_process_env(token: SecretStr | None) -> dict[str, str]:
    """Build an isolated Git environment with optional in-memory GitHub auth."""

    _validate_token(token)
    environment = _scrubbed_environment()

    configuration: list[tuple[str, str]] = []
    if token is not None:
        configuration.append(
            ("http.https://github.com/.extraHeader", _authorization_header(token))
        )
    configuration.extend(
        (
            ("credential.helper", ""),
            ("core.fsmonitor", "false"),
            ("core.hooksPath", os.devnull),
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


def isolated_git_environment() -> dict[str, str]:
    """Build a Git environment with no inherited transport, proxy or config."""

    environment = _scrubbed_environment()
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def authenticated_git_environment(token: SecretStr) -> dict[str, str]:
    """Carry the GitHub header over https only, never over a local path."""

    _validate_token(token)
    environment = isolated_git_environment()
    configuration = (
        # Borrowed checkout objects must not make Git run for-each-ref inside
        # the checkout while this environment holds the header.
        ("core.alternateRefsCommand", "true"),
        ("credential.helper", ""),
        ("http.proxy", ""),
        ("http.https://github.com/.proxy", ""),
        ("http.sslVerify", "true"),
        ("http.https://github.com/.sslVerify", "true"),
        ("http.followRedirects", "false"),
        ("http.https://github.com/.followRedirects", "false"),
        ("http.extraHeader", ""),
        ("http.https://github.com/.extraHeader", _authorization_header(token)),
    )
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "https",
            "GIT_CONFIG_COUNT": str(len(configuration)),
        }
    )
    for index, (key, value) in enumerate(configuration):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


def file_git_environment() -> dict[str, str]:
    """Carry no credential and reach nothing but the local filesystem."""

    environment = isolated_git_environment()
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": os.devnull,
        }
    )
    return environment


def fetch_refs(
    workspace: Path,
    *,
    remote: str,
    refs: Sequence[tuple[str, str]],
    token: SecretStr | None,
    have: str | None = None,
    timeout: float = 300,
) -> None:
    """Copy remote refs into a checkout without handing that checkout the token.

    An authenticated copy runs in two hops: an https-only fetch into a throwaway
    bare repository that holds the Authorization header, then a file-only fetch
    from that repository into the checkout. The checkout's Git is never given a
    credential and can never reach the network, so it cannot become a token
    oracle for whatever reads the worktree.

    ``have`` names a commit the checkout already holds. The throwaway repository
    then borrows the checkout's objects and negotiates from that commit, so only
    the missing history crosses the network.
    """

    if not refs:
        raise ValueError("fetch_refs requires at least one ref")
    _validate_token(token)
    if token is None:
        run_git(
            workspace,
            "fetch",
            "--no-tags",
            "--atomic",
            remote,
            *(f"+{source}:{destination}" for source, destination in refs),
            timeout=timeout,
        )
        return

    repository = (
        staged_commit(workspace, have) if have else isolated_bare_repository()
    )
    with repository as staging:
        staged = tuple(
            (source, f"refs/senpai/transfer/{index}", destination)
            for index, (source, destination) in enumerate(refs)
        )
        run_git(
            staging,
            "fetch",
            "--no-tags",
            "--atomic",
            remote,
            *(f"+{source}:{transfer}" for source, transfer, _ in staged),
            environment=authenticated_git_environment(token),
            timeout=timeout,
        )
        run_git(
            workspace,
            "fetch",
            "--no-tags",
            "--atomic",
            str(staging),
            *(f"+{transfer}:{destination}" for _, transfer, destination in staged),
            environment=file_git_environment(),
            timeout=timeout,
        )


def _scrubbed_environment() -> dict[str, str]:
    environment = dict(os.environ)
    scrub_github_credentials(environment)
    for name in tuple(environment):
        if name.startswith("GIT_") or name in PROXY_ENVIRONMENT:
            environment.pop(name)
    for name in _GIT_TRANSPORT_ENVIRONMENT:
        environment.pop(name, None)
    return environment


def _authorization_header(token: SecretStr) -> str:
    credential = base64.b64encode(
        f"x-access-token:{token.get_secret_value()}".encode()
    ).decode()
    return f"Authorization: Basic {credential}"


def _validate_token(token: SecretStr | None) -> None:
    if token is not None and not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")
    if token is not None and not token.get_secret_value().strip():
        raise ValueError("token must not be empty")
