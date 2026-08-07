"""GitHub transport, snapshots, and primitive desired-state mutations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

from pydantic import SecretStr

from senpai_agent.github.workflow.errors import (
    GitHubAPIError,
    GitHubTransportError,
    ReconciliationError,
)
from senpai_agent.github.workflow.responses import (
    DraftMutationResponse,
    GitRefResponse,
    HttpResponse,
    HttpTransport,
    PullRequestResponse,
    PullRequestSnapshot,
    validated_response,
)
from senpai_agent.github.workflow.transport import UrllibTransport
from senpai_agent.github.workflow.validation import (
    positive_number,
    require_assignment_identity,
    require_head,
    validate_labels,
)

if TYPE_CHECKING:
    from senpai_agent.models import AssignmentRecord


class WorkflowCore:
    """Small desired-state client for Senpai pull-request transitions."""

    __slots__ = (
        "_api_url",
        "_assignment_lifecycle_lock",
        "_repo",
        "_role",
        "_token",
        "_transport",
        "_trusted_actor",
    )

    def __init__(
        self,
        repo: str,
        token: SecretStr,
        *,
        role: Literal["advisor", "student"],
        transport: HttpTransport | None = None,
        api_url: str = "https://api.github.com",
        trusted_actor: str | None = None,
    ):
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("repo must use owner/name form")
        if not isinstance(token, SecretStr):
            raise TypeError("token must be a SecretStr")
        if not token.get_secret_value().strip():
            raise ValueError("token must not be empty")
        if role not in {"advisor", "student"}:
            raise ValueError("role must be advisor or student")
        if trusted_actor is not None and not trusted_actor.strip():
            raise ValueError("trusted actor must not be empty")

        self._repo = repo
        self._role = role
        self._token = token
        self._transport = transport or UrllibTransport()
        self._api_url = api_url.rstrip("/")
        self._trusted_actor = trusted_actor
        self._assignment_lifecycle_lock = RLock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(repo={self._repo!r}, api_url={self._api_url!r})"

    @property
    def repo(self) -> str:
        return self._repo

    @property
    def role(self) -> Literal["advisor", "student"]:
        return self._role

    @contextmanager
    def serialized_assignment_mutation(self) -> Iterator[None]:
        """Serialize a coupled local mutation with this workflow's transitions.

        This closes races among Senpai operations sharing this workflow instance.
        GitHub's merge API has no base-SHA compare-and-swap; external writers still
        require strict branch protection or a merge queue.
        """

        with self._assignment_lifecycle_lock:
            yield

    def __getstate__(self) -> None:
        raise TypeError("GitHubWorkflow cannot be serialized")

    def pull_request(self, number: int) -> PullRequestSnapshot:
        number = positive_number(number)
        response = self._request(
            "GET",
            f"/repos/{self._repo}/pulls/{number}",
            expected_statuses={200},
        )
        snapshot = validated_response(
            PullRequestResponse,
            response.json_body,
            "pull request",
        ).snapshot()
        if snapshot.number != number:
            raise ReconciliationError("GitHub returned the wrong pull request")
        return snapshot

    def _branch_head_sha(self, branch: str) -> str:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/git/ref/heads/{quote(branch, safe='')}",
            expected_statuses={200},
        )
        git_ref = validated_response(
            GitRefResponse,
            response.json_body,
            "git reference",
        )
        expected_ref = f"refs/heads/{branch}"
        if git_ref.ref != expected_ref:
            raise ReconciliationError(
                f"GitHub returned git reference {git_ref.ref!r}, "
                f"expected {expected_ref!r}"
            )
        return git_ref.object.sha

    def _pull_at_head(
        self,
        number: int,
        expected_head_sha: str,
    ) -> PullRequestSnapshot:
        snapshot = self.pull_request(number)
        require_head(snapshot, expected_head_sha)
        return snapshot

    def _assigned_pull_at_head(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> tuple[PullRequestSnapshot, AssignmentRecord]:
        snapshot = self._pull_at_head(number, expected_head_sha)
        assignment = require_assignment_identity(
            snapshot,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        return snapshot, assignment

    def _set_draft(
        self,
        snapshot: PullRequestSnapshot,
        *,
        draft: bool,
    ) -> bool:
        if snapshot.draft is draft:
            return False
        mutation = (
            "convertPullRequestToDraft" if draft else "markPullRequestReadyForReview"
        )
        response = self._mutate(
            "POST",
            "/graphql",
            json_body={
                "query": (
                    f"mutation($pullRequestId: ID!) {{ {mutation}("
                    "input: {pullRequestId: $pullRequestId}) { "
                    "pullRequest { id isDraft } } }"
                ),
                "variables": {"pullRequestId": snapshot.node_id},
            },
            expected_statuses={200},
        )
        if response is None:
            return True
        if not isinstance(response.json_body, dict):
            raise ReconciliationError("GitHub returned invalid GraphQL response")
        if response.json_body.get("errors"):
            raise ReconciliationError(
                f"GitHub GraphQL {mutation} mutation returned errors"
            )
        data = response.json_body.get("data")
        mutation_payload = data.get(mutation) if isinstance(data, dict) else None
        mutation_result = validated_response(
            DraftMutationResponse,
            mutation_payload,
            f"GraphQL {mutation} result",
        )
        if mutation_result.pull_request.is_draft is not draft:
            raise ReconciliationError(
                f"GitHub GraphQL {mutation} returned the wrong draft state"
            )
        return True

    def _set_labels(
        self,
        number: int,
        snapshot: PullRequestSnapshot,
        *,
        add: set[str],
        remove: set[str],
    ) -> tuple[bool, tuple[str, ...]]:
        validate_labels(add | remove)
        if overlap := add & remove:
            raise ValueError(
                "labels cannot be both added and removed: "
                + ", ".join(sorted(overlap))
            )
        desired = tuple(sorted((set(snapshot.labels) | add) - remove))
        if snapshot.labels == desired:
            return False, desired
        self._mutate(
            "PUT",
            f"/repos/{self._repo}/issues/{number}/labels",
            json_body={"labels": list(desired)},
            expected_statuses={200},
        )
        return True, desired

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: object | None = None,
        expected_statuses: set[int],
    ) -> HttpResponse:
        absolute_url = self._url(url)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token.get_secret_value()}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        response = self._transport.request(
            method,
            absolute_url,
            headers=headers,
            json_body=json_body,
        )
        if response.status_code not in expected_statuses:
            raise GitHubAPIError(method, absolute_url, response.status_code)
        return response

    def _mutate(
        self,
        method: str,
        url: str,
        *,
        json_body: object,
        expected_statuses: set[int],
    ) -> HttpResponse | None:
        """Issue a mutation; an ambiguous transport failure is verified by caller."""

        try:
            return self._request(
                method,
                url,
                json_body=json_body,
                expected_statuses=expected_statuses,
            )
        except GitHubTransportError:
            return None

    def _url(self, value: str) -> str:
        if value.startswith(("https://", "http://")):
            return value
        return f"{self._api_url}/{value.lstrip('/')}"
