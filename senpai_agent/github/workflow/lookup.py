"""GitHub lookups that resolve assignments, actors, and human messages."""

from urllib.parse import urlencode

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    GitHubUser,
    HumanIssueMessage,
    IssueResponse,
    IssueSearchResponse,
    NumberedResponse,
    PullRequestSnapshot,
    validated_response,
)
from senpai_agent.models import AssignmentRecord


class LookupMixin:
    __slots__ = ()

    def _assignment_pull_requests(
        self,
        assignment: AssignmentRecord,
    ) -> tuple[PullRequestSnapshot, ...]:
        owner = self._repo.split("/", 1)[0]
        query = urlencode(
            {
                "state": "all",
                "head": f"{owner}:{assignment.head_ref}",
                "base": assignment.base_ref,
                "per_page": 100,
            }
        )
        response = self._request(
            "GET",
            f"/repos/{self._repo}/pulls?{query}",
            expected_statuses={200},
        )
        if not isinstance(response.json_body, list):
            raise ReconciliationError(
                "GitHub returned invalid assignment PR search results"
            )
        return tuple(
            self.pull_request(
                validated_response(
                    NumberedResponse,
                    item,
                    "assignment pull request",
                ).number
            )
            for item in response.json_body
        )

    def _active_student_assignment_numbers(
        self,
        student: str,
    ) -> tuple[int, ...]:
        query = urlencode(
            {
                "state": "open",
                "labels": f"student:{student}",
                "per_page": 100,
            }
        )
        response = self._request(
            "GET",
            f"/repos/{self._repo}/issues?{query}",
            expected_statuses={200},
        )
        if not isinstance(response.json_body, list):
            raise ReconciliationError(
                "GitHub returned invalid active assignment results"
            )
        issues = tuple(
            validated_response(
                IssueSearchResponse,
                item,
                "active assignment",
            )
            for item in response.json_body
        )
        return tuple(
            issue.number
            for issue in issues
            if issue.pull_request is not None
            and {"status:wip", "status:review"}
            & {label.name for label in issue.labels}
        )

    def _human_issue(
        self,
        number: int,
        *,
        audience_labels: set[str],
    ) -> IssueResponse:
        response = self._request(
            "GET",
            f"/repos/{self._repo}/issues/{number}",
            expected_statuses={200},
        )
        issue = validated_response(IssueResponse, response.json_body, "issue")
        if issue.pull_request is not None:
            raise WorkflowPreconditionError(
                "human messages must use an issue, not a pull request"
            )
        if issue.state != "open":
            raise WorkflowPreconditionError("human issue must be open")
        labels = {label.name for label in issue.labels}
        if "human" not in labels:
            raise WorkflowPreconditionError("human issue must retain the human label")
        if not labels.intersection(audience_labels):
            raise WorkflowPreconditionError(
                "human issue must retain a team or current-role audience label"
            )
        return issue

    def _human_message(
        self,
        number: int,
        *,
        issue: IssueResponse,
        human_message_id: int,
    ) -> HumanIssueMessage:
        if issue.id == human_message_id:
            return HumanIssueMessage(
                body=issue.body or "",
                author=issue.user.login if issue.user else "",
                author_type=issue.user.type if issue.user else "",
                author_association=issue.author_association,
            )
        match = next(
            (
                comment
                for comment in self._comments(number)
                if comment.id == human_message_id
            ),
            None,
        )
        if match is None:
            raise WorkflowPreconditionError(
                f"human message ID {human_message_id} is not present on issue #{number}"
            )
        return HumanIssueMessage(
            body=match.body,
            author=match.author,
            author_type=match.author_type,
            author_association=match.author_association,
        )

    def _actor(self) -> str:
        if self._trusted_actor is None:
            response = self._request("GET", "/user", expected_statuses={200})
            self._trusted_actor = validated_response(
                GitHubUser,
                response.json_body,
                "authenticated user",
            ).login
        return self._trusted_actor
