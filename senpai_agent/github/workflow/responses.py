"""Typed GitHub API responses and workflow snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError

from senpai_agent.github.workflow.errors import ReconciliationError
from senpai_agent.models import AssignmentRecord, Contract, ExperimentResult


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    json_body: object | None = None
    headers: tuple[tuple[str, str], ...] = ()

    def header(self, name: str) -> str | None:
        normalized = name.casefold()
        return next(
            (value for key, value in self.headers if key.casefold() == normalized),
            None,
        )


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class PullRequestSnapshot:
    number: int
    node_id: str
    url: str
    title: str
    body: str
    base_ref: str
    head_ref: str
    head_sha: str
    labels: tuple[str, ...]
    draft: bool
    state: Literal["open", "closed"]
    merged: bool
    mergeable: bool | None
    merge_commit_sha: str | None


@dataclass(frozen=True, slots=True)
class SubmitResultPreflight:
    snapshot: PullRequestSnapshot
    assignment: AssignmentRecord


@dataclass(frozen=True, slots=True)
class MutationResult:
    changed: bool
    resource_url: str
    state: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class IssueComment:
    id: int
    body: str
    url: str
    author: str
    author_type: str
    author_association: str


@dataclass(frozen=True, slots=True)
class HumanIssueMessage:
    body: str
    author: str
    author_type: str
    author_association: str


@dataclass(frozen=True, slots=True)
class ResultComment:
    comment: IssueComment
    result: ExperimentResult


class GitHubResponse(Contract):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=False,
    )


RequiredString = Annotated[StrictStr, Field(min_length=1)]
PositiveInteger = Annotated[StrictInt, Field(gt=0)]


class GitHubRef(GitHubResponse):
    ref: RequiredString


class GitHubHead(GitHubRef):
    sha: RequiredString


class GitObject(GitHubResponse):
    sha: RequiredString


class GitRefResponse(GitHubResponse):
    ref: RequiredString
    object: GitObject


class GitHubLabel(GitHubResponse):
    name: RequiredString


class GitHubUser(GitHubResponse):
    login: RequiredString


class GitHubAuthor(GitHubUser):
    type: RequiredString


class PullRequestResponse(GitHubResponse):
    number: PositiveInteger
    node_id: RequiredString
    html_url: RequiredString
    title: RequiredString
    body: StrictStr | None
    base: GitHubRef
    head: GitHubHead
    labels: tuple[GitHubLabel, ...]
    draft: StrictBool
    state: Literal["open", "closed"]
    merged: StrictBool
    mergeable: StrictBool | None
    merge_commit_sha: StrictStr | None

    def snapshot(self) -> PullRequestSnapshot:
        return PullRequestSnapshot(
            number=self.number,
            node_id=self.node_id,
            url=self.html_url,
            title=self.title,
            body=self.body or "",
            base_ref=self.base.ref,
            head_ref=self.head.ref,
            head_sha=self.head.sha,
            labels=tuple(sorted({label.name for label in self.labels})),
            draft=self.draft,
            state=self.state,
            merged=self.merged,
            mergeable=self.mergeable,
            merge_commit_sha=self.merge_commit_sha,
        )


class IssueCommentResponse(GitHubResponse):
    id: PositiveInteger
    body: StrictStr
    html_url: StrictStr
    user: GitHubAuthor
    author_association: RequiredString

    def comment(self) -> IssueComment:
        return IssueComment(
            id=self.id,
            body=self.body,
            url=self.html_url,
            author=self.user.login,
            author_type=self.user.type,
            author_association=self.author_association,
        )


class IssueResponse(GitHubResponse):
    id: PositiveInteger
    body: StrictStr | None
    state: StrictStr
    labels: tuple[GitHubLabel, ...]
    user: GitHubAuthor
    author_association: RequiredString
    pull_request: dict[str, object] | None = None


class NumberedResponse(GitHubResponse):
    number: PositiveInteger


class IssueSearchResponse(NumberedResponse):
    labels: tuple[GitHubLabel, ...]
    pull_request: dict[str, object] | None = None


class DraftPullRequestResponse(GitHubResponse):
    id: RequiredString
    is_draft: StrictBool = Field(alias="isDraft")


class DraftMutationResponse(GitHubResponse):
    pull_request: DraftPullRequestResponse = Field(alias="pullRequest")


def validated_response[ResponseT: GitHubResponse](
    model: type[ResponseT],
    value: object,
    name: str,
) -> ResponseT:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise ReconciliationError(f"GitHub returned invalid {name}") from error
