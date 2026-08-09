"""Credential-scoped pull-request retrieval tool."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Self

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import Action, Observation, ToolDefinition, ToolExecutor
from pydantic import BaseModel, ConfigDict, Field

from senpai_agent.github import PRRetrievalResult, get_prs

from .runtime import (
    GitHubCredentials,
    current_github_credentials,
    tool_annotations,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class GetPRsAction(Action):
    """Retrieve complete context for a bounded set of pull requests."""

    repo: str = Field(
        min_length=3,
        description="GitHub repository in owner/name form.",
    )
    numbers: tuple[int, ...] = Field(
        default=(),
        description="Explicit positive PR numbers to include.",
    )
    date_range: tuple[str | date, str | date] | None = Field(
        default=None,
        description="Optional inclusive PR creation-date range.",
    )
    search: str | None = Field(
        default=None,
        description="Optional GitHub issue-search terms or qualifiers.",
    )
    max_inline_prs: int = Field(
        default=5,
        ge=0,
        description=(
            "Maximum PRs returned inline. Do not set this above 5 unless "
            "explicitly necessary; prefer the returned artifact path."
        ),
    )


class PRManifestObservation(BaseModel):
    """Compact identity for one retrieved pull request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    head_sha: str
    url: str


class GetPRsObservation(Observation):
    """Inline pull-request context or a bounded external artifact reference."""

    manifest: tuple[PRManifestObservation, ...]
    markdown: str | None = None
    path: str | None = None

    @classmethod
    def from_result(cls, result: PRRetrievalResult) -> Self:
        return cls(
            manifest=tuple(
                PRManifestObservation(
                    number=entry.number,
                    title=entry.title,
                    head_sha=entry.head_sha,
                    url=entry.url,
                )
                for entry in result.manifest
            ),
            markdown=result.markdown,
            path=str(result.path) if result.path is not None else None,
        )

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        if self.markdown is not None:
            return [TextContent(text=self.markdown)]

        manifest = "\n".join(
            f"- #{entry.number} `{entry.head_sha}` {entry.title} ({entry.url})"
            for entry in self.manifest
        )
        return [
            TextContent(
                text=(
                    f"Full PR context is stored at: {self.path}\n"
                    f"Compact manifest:\n{manifest}"
                )
            )
        ]


class _GetPRsExecutor(ToolExecutor[GetPRsAction, GetPRsObservation]):
    def __init__(
        self,
        get_prs_fn: Callable[..., PRRetrievalResult],
        *,
        credentials: GitHubCredentials | None,
        artifact_dir: Path,
        target_workspace: Path,
    ):
        self.get_prs = get_prs_fn
        self.credentials = credentials
        self.artifact_dir = artifact_dir
        self.target_workspace = target_workspace

    def __call__(
        self,
        action: GetPRsAction,
        conversation: LocalConversation | None = None,
    ) -> GetPRsObservation:
        if self.credentials is not None and action.repo != self.credentials.repo:
            raise PermissionError(
                "requested repository does not match configured GitHub credentials"
            )
        auth = {"token": self.credentials.token} if self.credentials is not None else {}
        result = self.get_prs(
            action.repo,
            numbers=action.numbers,
            date_range=action.date_range,
            search=action.search,
            max_inline_prs=action.max_inline_prs,
            artifact_dir=self.artifact_dir,
            target_workspace=self.target_workspace,
            **auth,
        )
        return GetPRsObservation.from_result(result)


class GetPRsTool(ToolDefinition[GetPRsAction, GetPRsObservation]):
    """Read complete pull-request context without exposing GitHub credentials."""

    name = "get_prs"

    @classmethod
    def create(
        cls,
        conv_state: object | None = None,
        *,
        get_prs_fn: Callable[..., PRRetrievalResult] = get_prs,
        state_dir: str | Path | None = None,
        workspace: str | Path | None = None,
    ) -> Sequence[Self]:
        credentials = (
            current_github_credentials() if get_prs_fn is get_prs else None
        )
        if get_prs_fn is get_prs and credentials is None:
            raise RuntimeError(
                "configure GitHub credentials before initializing get_prs"
            )
        if workspace is None:
            if conv_state is None:
                raise ValueError("get_prs requires its OpenHands workspace")
            workspace = Path(conv_state.workspace.working_dir)
        target_workspace = Path(workspace).resolve()
        artifact_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else Path(tempfile.gettempdir()).resolve() / "senpai-pr-artifacts"
        )
        if artifact_dir == target_workspace or artifact_dir.is_relative_to(
            target_workspace
        ):
            raise ValueError("get_prs state_dir must be outside the target workspace")
        return [
            cls(
                description=(
                    "Retrieve complete PR bodies, comments, reviews, and inline "
                    "comments by number, date range, and/or search. Large results "
                    "are returned as one external Markdown artifact."
                ),
                action_type=GetPRsAction,
                observation_type=GetPRsObservation,
                annotations=tool_annotations("Get pull requests", read_only=True),
                executor=_GetPRsExecutor(
                    get_prs_fn,
                    credentials=credentials,
                    artifact_dir=artifact_dir,
                    target_workspace=target_workspace,
                ),
            )
        ]
