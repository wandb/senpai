"""Target-checkout reconciliation for student assignments."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from senpai_agent.git_transport import (
    FETCH_TIMEOUT_SECONDS,
    GIT_EXECUTABLE,
    fetch_refs,
    git_process_env,
)
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.PROMPTS import WORKSPACE_DIVERGENCE_PROMPT
from senpai_agent.supervisor import ProgressLease


# Time for a timed-out Git command to be killed and its error raised before
# the supervisor would treat the reconcile lease as overdue.
RECONCILE_LEASE_GRACE_SECONDS = 30
_HEAD_REF = "refs/senpai/assignment/head"
_BASE_REF = "refs/senpai/assignment/base"
_BASE_TIP_REF = "refs/senpai/assignment/base-tip"
_OBJECT_ID = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
_UNTRACKED_CONTENT_BUDGET = 1_048_576
_UNTRACKED_FILE_LIMIT = 1_024


@dataclass(frozen=True, slots=True)
class _Assignment:
    head_ref: str
    head_sha: str
    base_ref: str
    base_sha: str

    @classmethod
    def from_event(cls, event: ControllerEvent) -> _Assignment:
        assignment = cls(
            head_ref=str(event.payload["head_ref"]),
            head_sha=str(event.payload["head_sha"]),
            base_ref=str(event.payload["base_ref"]),
            base_sha=str(event.payload["base_sha"]),
        )
        for name, value in (
            ("head_sha", assignment.head_sha),
            ("base_sha", assignment.base_sha),
        ):
            if _OBJECT_ID.fullmatch(value) is None:
                raise ValueError(f"assignment {name} must be a full Git object ID")
        return assignment


class WorkspaceDivergence(RuntimeError):
    """The checkout has local history or dirty work worth preserving."""

    def __init__(
        self,
        *,
        head_ref: str,
        expected_head: str,
        local_head: str,
        base_ref: str | None = None,
        base_sha: str | None = None,
        current_branch: str | None = None,
        worktree_state: str = "",
    ):
        fingerprint = hashlib.sha256(
            "\0".join(
                (
                    head_ref,
                    expected_head,
                    local_head,
                    base_ref or "",
                    base_sha or "",
                    current_branch or "",
                    worktree_state,
                )
            ).encode()
        ).hexdigest()
        self.event = ControllerEvent(
            kind="workspace_diverged",
            dedupe_key=f"workspace_diverged:{fingerprint}",
            payload={
                "head_ref": head_ref,
                "expected_remote_head": expected_head,
                "preserved_local_head": local_head,
                "base_ref": base_ref,
                "base_sha": base_sha,
                "current_branch": current_branch,
                "worktree_fingerprint": hashlib.sha256(
                    worktree_state.encode()
                ).hexdigest(),
                "instructions": WORKSPACE_DIVERGENCE_PROMPT,
            },
        )
        super().__init__(
            f"preserved workspace conflict for assignment {head_ref}: "
            f"local {local_head}, remote {expected_head}"
        )


class StudentWorkspaceReconciler:
    """Hydrate an assignment and check it out without discarding local work.

    With a progress lease, each Git command renews the ``reconcile`` lease for
    its own timeout. A slow fetch is then bounded by that timeout, however many
    commands the reconcile runs, and the lease still expires if the controller
    hangs between commands.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        repo: str | None = None,
        token: SecretStr | None = None,
        progress: ProgressLease | None = None,
    ):
        if token is not None and repo is None:
            raise ValueError("authenticated reconciliation requires a GitHub repo")
        if repo is not None and (
            len(repo.split("/")) != 2 or not all(repo.split("/"))
        ):
            raise ValueError("repo must use owner/name form")
        self.workspace = workspace
        self.token = token
        self.progress = progress
        self.remote = f"https://github.com/{repo}.git" if repo else "origin"

    def __call__(self, events: Sequence[ControllerEvent]) -> None:
        assignment_event = next(
            (
                event
                for event in events
                if event.kind in {"student_assignment", "student_pr_feedback"}
            ),
            None,
        )
        if assignment_event is None:
            return

        assignment = _Assignment.from_event(assignment_event)
        fetched_head = self._hydrate(assignment)
        current_branch = self._git("branch", "--show-current") or None
        worktree_state = self._worktree_state()
        if current_branch != assignment.head_ref and worktree_state:
            raise WorkspaceDivergence(
                head_ref=assignment.head_ref,
                expected_head=assignment.head_sha,
                local_head=self._git("rev-parse", "HEAD"),
                base_ref=assignment.base_ref,
                base_sha=assignment.base_sha,
                current_branch=current_branch,
                worktree_state=worktree_state,
            )

        local_ref = f"refs/heads/{assignment.head_ref}"
        branch_exists = self._run(
            "show-ref",
            "--verify",
            "--quiet",
            local_ref,
            check=False,
        ).returncode == 0
        if not branch_exists:
            if fetched_head != assignment.head_sha:
                raise WorkspaceDivergence(
                    head_ref=assignment.head_ref,
                    expected_head=fetched_head,
                    local_head=self._git("rev-parse", "HEAD"),
                    base_ref=assignment.base_ref,
                    base_sha=assignment.base_sha,
                    current_branch=current_branch,
                    worktree_state=worktree_state,
                )
            self._run("checkout", "-b", assignment.head_ref, _HEAD_REF)
            return

        local_head = self._git("rev-parse", local_ref)
        # A stale event may outlive successful student pushes. Resume only when
        # the assigned, remote, and preserved local heads form one forward chain.
        if not self._is_ancestor(
            assignment.head_sha,
            fetched_head,
        ) or not self._is_ancestor(fetched_head, local_head):
            raise WorkspaceDivergence(
                head_ref=assignment.head_ref,
                expected_head=fetched_head,
                local_head=local_head,
                base_ref=assignment.base_ref,
                base_sha=assignment.base_sha,
                current_branch=current_branch,
                worktree_state=worktree_state,
            )
        self._run("checkout", assignment.head_ref)

    def _hydrate(self, assignment: _Assignment) -> str:
        self._git("check-ref-format", "--branch", assignment.head_ref)
        self._git("check-ref-format", "--branch", assignment.base_ref)
        self._fetch_refs(
            (f"refs/heads/{assignment.head_ref}", _HEAD_REF),
            (f"refs/heads/{assignment.base_ref}", _BASE_TIP_REF),
        )
        fetched_head = self._git("rev-parse", _HEAD_REF)

        if not self._commit_exists(assignment.base_sha):
            try:
                self._fetch_refs((assignment.base_sha, _BASE_REF))
            except RuntimeError as error:
                raise RuntimeError(
                    f"assignment base {assignment.base_ref}@{assignment.base_sha} "
                    "is unavailable from the configured GitHub repository"
                ) from error
            if not self._commit_exists(assignment.base_sha):
                raise RuntimeError(
                    f"assignment base {assignment.base_ref}@{assignment.base_sha} "
                    "is unavailable from the configured GitHub repository"
                )
        self._run("update-ref", _BASE_REF, assignment.base_sha)
        return fetched_head

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return (
            self._run(
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
                check=False,
            ).returncode
            == 0
        )

    def _fetch_refs(self, *refs: tuple[str, str]) -> None:
        self._renew_lease(FETCH_TIMEOUT_SECONDS)
        fetch_refs(
            self.workspace,
            remote=self.remote,
            refs=refs,
            token=self.token,
            timeout=FETCH_TIMEOUT_SECONDS,
        )

    def _renew_lease(self, timeout: float) -> None:
        if self.progress is not None:
            self.progress.update(
                "reconcile",
                timeout + RECONCILE_LEASE_GRACE_SECONDS,
            )

    def _commit_exists(self, sha: str) -> bool:
        return self._run(
            "cat-file",
            "-e",
            f"{sha}^{{commit}}",
            check=False,
        ).returncode == 0

    def _worktree_state(self) -> str:
        status = self._git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if not status:
            return ""
        return "\0".join(
            (
                status,
                self._git("diff", "--binary"),
                self._git("diff", "--cached", "--binary"),
                self._untracked_state(),
            )
        )

    def _untracked_state(self) -> str:
        raw_paths = self._run(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).stdout
        paths = sorted(path for path in raw_paths.split("\0") if path)
        state = [
            f"paths={hashlib.sha256(raw_paths.encode()).hexdigest()}:{len(paths)}"
        ]
        budget = _UNTRACKED_CONTENT_BUDGET
        for relative in paths[:_UNTRACKED_FILE_LIMIT]:
            path = self.workspace / relative
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                state.append(f"{relative}:missing")
                continue
            digest = "metadata-only"
            if path.is_symlink():
                digest = hashlib.sha256(
                    path.readlink().as_posix().encode()
                ).hexdigest()
            elif path.is_file() and metadata.st_size <= budget:
                with path.open("rb") as file:
                    content = file.read(budget + 1)
                if len(content) <= budget:
                    digest = hashlib.sha256(content).hexdigest()
                    budget -= len(content)
            state.append(
                f"{relative}:{metadata.st_mode}:{metadata.st_size}:"
                f"{metadata.st_mtime_ns}:{digest}"
            )
        return "\0".join(state)

    def _git(self, *arguments: str) -> str:
        return self._run(*arguments).stdout.strip()

    def _run(
        self,
        *arguments: str,
        check: bool = True,
        environment: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        self._renew_lease(timeout)
        return self._run_at(
            self.workspace,
            *arguments,
            check=check,
            environment=environment or git_process_env(None),
            timeout=timeout,
        )

    @staticmethod
    def _run_at(
        workspace: Path,
        *arguments: str,
        check: bool = True,
        environment: dict[str, str],
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [GIT_EXECUTABLE, "-c", f"safe.directory={workspace.resolve()}", *arguments],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=environment,
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"git {' '.join(arguments[:2])} failed: {detail[:1000]}"
            )
        return completed

