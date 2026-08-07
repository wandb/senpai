"""Trusted marker comments and immutable experiment results."""

from collections.abc import Callable

from senpai_agent.github.http import next_link
from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    StaleResearchBaseError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    IssueComment,
    IssueCommentResponse,
    ResultComment,
    validated_response,
)
from senpai_agent.github.workflow.text import role_prefixed_comment
from senpai_agent.github.workflow.validation import (
    distinct_results,
    positive_number,
    require_result_identity,
)
from senpai_agent.models import (
    ExperimentResult,
    ResearchBaseAcceptanceRecord,
    ResultMarkerError,
    experiment_result_digest,
    parse_research_base_acceptance_markers,
    parse_result_markers,
    render_result_comment,
)


class CommentsMixin:
    __slots__ = ()

    def _upsert_marker_comment(
        self,
        number: int,
        *,
        marker: str,
        body: str,
    ) -> tuple[bool, IssueComment]:
        return self._upsert_comment(
            number,
            body=body,
            matches=lambda: self._marker_comments(number, marker),
            subject=f"comments for marker {marker!r}",
            desired_state="marker comment",
        )

    def _upsert_result_comment(
        self,
        number: int,
        *,
        result: ExperimentResult,
    ) -> tuple[bool, IssueComment]:
        assignment_id = result.assignment.assignment_id
        body = role_prefixed_comment(render_result_comment(result), self._role)
        existing = self._result_comments(number, assignment_id)
        distinct = distinct_results(existing)
        if len(distinct) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple distinct result markers for "
                f"{assignment_id!r}"
            )
        if distinct:
            published = distinct[0]
            if experiment_result_digest(published) == experiment_result_digest(result):
                comments = {match.comment.id: match.comment for match in existing}
                changed = False
                for comment in comments.values():
                    if comment.body == body:
                        continue
                    self._mutate(
                        "PATCH",
                        f"/repos/{self._repo}/issues/comments/{comment.id}",
                        json_body={"body": body},
                        expected_statuses={200},
                    )
                    changed = True
                if not changed:
                    return False, existing[0].comment
                verified = self._result_comments(number, assignment_id)
                if not verified or any(
                    match.comment.body != body for match in verified
                ):
                    raise ReconciliationError(
                        "GitHub did not normalize the terminal result comment"
                    )
                return True, verified[0].comment
            if (
                published.assignment.revision_id == result.assignment.revision_id
                and published.assignment.expected_head_sha
                == result.assignment.expected_head_sha
            ):
                raise WorkflowPreconditionError(
                    "a published terminal result is immutable at the same "
                    "assignment revision and head; use a new revision or head"
                )
            comments = {match.comment.id: match.comment for match in existing}
            for comment in comments.values():
                self._mutate(
                    "PATCH",
                    f"/repos/{self._repo}/issues/comments/{comment.id}",
                    json_body={"body": body},
                    expected_statuses={200},
                )
        else:
            self._mutate(
                "POST",
                f"/repos/{self._repo}/issues/{number}/comments",
                json_body={"body": body},
                expected_statuses={201},
            )

        verified = self._result_comments(number, assignment_id)
        verified_distinct = distinct_results(verified)
        if (
            len(verified_distinct) != 1
            or experiment_result_digest(verified_distinct[0])
            != experiment_result_digest(result)
        ):
            raise ReconciliationError(
                "GitHub did not reach the requested terminal result"
            )
        return True, verified[0].comment

    def _upsert_comment(
        self,
        number: int,
        *,
        body: str,
        matches: Callable[[], tuple[IssueComment, ...]],
        subject: str,
        desired_state: str,
    ) -> tuple[bool, IssueComment]:
        body = role_prefixed_comment(body, self._role)
        existing = matches()
        if len(existing) > 1:
            raise ReconciliationError(f"GitHub contains multiple {subject}")
        if existing and existing[0].body == body:
            return False, existing[0]
        if existing:
            method = "PATCH"
            path = f"/repos/{self._repo}/issues/comments/{existing[0].id}"
            expected_statuses = {200}
        else:
            method = "POST"
            path = f"/repos/{self._repo}/issues/{number}/comments"
            expected_statuses = {201}
        self._mutate(
            method,
            path,
            json_body={"body": body},
            expected_statuses=expected_statuses,
        )
        verified = matches()
        if len(verified) != 1 or verified[0].body != body:
            raise ReconciliationError(
                f"GitHub did not reach the requested {desired_state}"
            )
        return True, verified[0]

    def _result_comments(
        self,
        number: int,
        assignment_id: str,
    ) -> tuple[ResultComment, ...]:
        matches: list[ResultComment] = []
        trusted_actor = self._actor()
        for comment in self._comments(number):
            if comment.author.casefold() != trusted_actor.casefold():
                continue
            for line in comment.body.splitlines():
                try:
                    results = parse_result_markers(line)
                except ResultMarkerError:
                    continue
                matches.extend(
                    ResultComment(comment=comment, result=result)
                    for result in results
                    if result.assignment.assignment_id == assignment_id
                )
        return tuple(matches)

    def _research_base_acceptances(
        self,
        number: int,
    ) -> tuple[ResearchBaseAcceptanceRecord, ...]:
        trusted_actor = self._actor()
        acceptances: list[ResearchBaseAcceptanceRecord] = []
        for comment in self._comments(number):
            if comment.author.casefold() != trusted_actor.casefold():
                continue
            try:
                acceptances.extend(
                    parse_research_base_acceptance_markers(comment.body)
                )
            except ValueError:
                continue
        return tuple(acceptances)

    def _require_research_base_acceptance(
        self,
        number: int,
        expected: ResearchBaseAcceptanceRecord,
    ) -> None:
        if expected not in self._research_base_acceptances(number):
            raise StaleResearchBaseError(
                "the current result has no durable acceptance for research base "
                f"{expected.base_ref}@{expected.accepted_base_sha}"
            )

    def _require_result(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> ExperimentResult:
        matches = self._result_comments(number, assignment_id)
        if not matches:
            raise WorkflowPreconditionError(
                f"schema-valid terminal result for {assignment_id!r} is missing"
            )
        distinct = distinct_results(matches)
        if len(distinct) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple distinct result markers for "
                f"{assignment_id!r}"
            )
        result = distinct[0]
        require_result_identity(
            result,
            repo=self._repo,
            number=number,
            expected_head_sha=expected_head_sha,
        )
        return result

    def _marker_comments(
        self,
        number: int,
        marker: str,
    ) -> tuple[IssueComment, ...]:
        trusted_actor = self._actor()
        return tuple(
            comment
            for comment in self._comments(number)
            if comment.author.casefold() == trusted_actor.casefold()
            and marker in comment.body.splitlines()
        )

    def _comments(self, number: int) -> tuple[IssueComment, ...]:
        number = positive_number(number)
        url: str | None = f"/repos/{self._repo}/issues/{number}/comments?per_page=100"
        comments: list[IssueComment] = []
        visited: set[str] = set()
        while url is not None:
            absolute_url = self._url(url)
            if absolute_url in visited:
                raise ReconciliationError("GitHub comment pagination contains a cycle")
            visited.add(absolute_url)
            response = self._request("GET", absolute_url, expected_statuses={200})
            if not isinstance(response.json_body, list):
                raise ReconciliationError("GitHub returned invalid paginated comments")
            comments.extend(
                validated_response(
                    IssueCommentResponse,
                    raw_comment,
                    "issue comment",
                ).comment()
                for raw_comment in response.json_body
            )
            url = next_link(response.header("Link"))
            if url is not None and not url.startswith(f"{self._api_url}/"):
                raise ReconciliationError(
                    "GitHub pagination returned an unexpected origin"
                )
        return tuple(comments)
