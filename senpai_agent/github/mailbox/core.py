"""Level-triggered GitHub mailbox for advisor and student controllers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlencode

from pydantic import SecretStr

from senpai_agent.github.http import GitHubReadError, GitHubReader
from senpai_agent.mailbox import ControllerEvent
from .advisor import advisor_events
from .ledger import acknowledge_feedback
from .student import student_events
from .values import (
    DEFAULT_FEEDBACK_BATCH_BYTES,
    DEFAULT_FEEDBACK_BATCH_EVENTS,
    FeedbackBinding,
)


class GitHubMailbox:
    """Read level-triggered Senpai work from GitHub PRs and Issues."""

    def __init__(
        self,
        *,
        repo: str,
        token: SecretStr,
        role: Literal["advisor", "student"],
        advisor_branch: str,
        students: Sequence[str] = (),
        student_name: str | None = None,
        stale_wip_seconds: int = 7200,
        api_url: str = "https://api.github.com",
        trusted_actor: str | None = None,
        human_issues_enabled: bool = True,
        feedback_path: Path | None = None,
        feedback_batch_events: int = DEFAULT_FEEDBACK_BATCH_EVENTS,
        feedback_batch_bytes: int = DEFAULT_FEEDBACK_BATCH_BYTES,
    ):
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("repo must use owner/name form")
        if role == "student" and not student_name:
            raise ValueError("student mailbox requires student_name")
        if feedback_batch_events <= 0 or feedback_batch_bytes <= 0:
            raise ValueError("feedback batch limits must be positive")
        self.repo = repo
        self.role = role
        self.advisor_branch = advisor_branch
        self.students = tuple(student for student in students if student)
        self.student_name = student_name
        self.stale_wip_seconds = stale_wip_seconds
        self.human_issues_enabled = human_issues_enabled
        self.feedback_path = feedback_path
        self.feedback_batch_events = feedback_batch_events
        self.feedback_batch_bytes = feedback_batch_bytes
        self._memory_feedback: dict[str, FeedbackBinding] = {}
        self._pull_comment_cache: dict[
            int,
            list[dict[str, object]] | GitHubReadError,
        ] = {}
        self._github = GitHubReader(
            token,
            api_url=api_url,
            trusted_actor=trusted_actor,
        )

    def poll(self) -> tuple[ControllerEvent, ...]:
        self._pull_comment_cache.clear()
        pulls = self._pulls()
        issues = self._issues() if self.human_issues_enabled else ()
        if self.role == "advisor":
            return advisor_events(self, pulls, issues)
        return student_events(self, pulls, issues)

    def acknowledge(self, dedupe_keys: Sequence[str]) -> None:
        """Mark persisted feedback delivered after a successful controller turn."""
        acknowledge_feedback(self, dedupe_keys)

    def _issue_comments(
        self,
        issue: Mapping[str, object],
    ) -> list[dict[str, object]]:
        comments_url = issue.get("comments_url")
        return self._github.objects(str(comments_url)) if comments_url else []

    def _pull_comments(self, number: int) -> list[dict[str, object]]:
        if number not in self._pull_comment_cache:
            try:
                comments = self._github.objects(
                    f"/repos/{self.repo}/issues/{number}/comments?per_page=100"
                )
                self._pull_comment_cache[number] = comments
            except GitHubReadError as error:
                self._pull_comment_cache[number] = error
        comments = self._pull_comment_cache[number]
        if isinstance(comments, GitHubReadError):
            raise comments
        return comments

    def _pulls(self) -> list[dict[str, object]]:
        query = urlencode(
            {
                "state": "open",
                "base": self.advisor_branch,
                "per_page": 100,
            }
        )
        return self._github.objects(f"/repos/{self.repo}/pulls?{query}")

    def _has_write_permission(self, login: str) -> bool:
        permission = self._github.get(
            f"/repos/{self.repo}/collaborators/{quote(login, safe='')}/permission"
        )
        if not isinstance(permission, dict) or not isinstance(
            permission.get("permission"), str
        ):
            raise GitHubReadError("GitHub returned an invalid collaborator permission")
        return permission["permission"] in {"admin", "maintain", "write"}

    def _issues(self) -> list[dict[str, object]]:
        query = urlencode(
            {
                "state": "open",
                "labels": "human",
                "per_page": 100,
            }
        )
        return [
            issue
            for issue in self._github.objects(f"/repos/{self.repo}/issues?{query}")
            if "pull_request" not in issue
        ]
