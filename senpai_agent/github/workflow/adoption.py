"""Adopt one pre-existing pull request into the typed assignment lifecycle."""

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import MutationResult, PullRequestSnapshot
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.github.workflow.validation import require_open
from senpai_agent.models import (
    AssignmentRecord,
    authoritative_marker_line,
    parse_assignment_markers,
    render_assignment_marker,
)


class AdoptionMixin:
    __slots__ = ()

    def adopt_assignment(
        self,
        number: int,
        assignment: AssignmentRecord,
    ) -> MutationResult:
        """Attach typed identity to one exact pre-existing assignment PR."""

        with self._assignment_lifecycle_lock:
            return self._adopt_assignment(number, assignment)

    def _adopt_assignment(
        self,
        number: int,
        assignment: AssignmentRecord,
    ) -> MutationResult:
        if self._role != "advisor":
            raise WorkflowPreconditionError("only an advisor may adopt assignments")
        if assignment.repo != self._repo:
            raise WorkflowPreconditionError(
                "assignment repository does not match the GitHub workflow"
            )

        current = self.pull_request(number)
        if self._is_existing_adoption(current, assignment):
            return MutationResult(
                changed=False,
                resource_url=current.url,
                state="assignment_adopted",
                version=current.head_sha,
            )

        matches = self._assignment_pull_requests(assignment)
        if len(matches) != 1 or matches[0].number != number:
            raise WorkflowPreconditionError(
                "assignment branch must have exactly one matching pull request"
            )
        conflicts = tuple(
            active
            for active in self._active_student_assignment_numbers(assignment.student)
            if active != number
        )
        if conflicts:
            raise WorkflowPreconditionError(
                f"student:{assignment.student} already has active assignment "
                f"PR(s): {', '.join(f'#{active}' for active in conflicts)}"
            )

        before = self._pull_at_head(number, assignment.head_sha)
        self._require_adoptable_assignment(before, assignment)
        latest = self._pull_at_head(number, assignment.head_sha)
        self._require_adoptable_assignment(latest, assignment)
        if latest.body != before.body:
            raise ReconciliationError(
                "assignment pull request body changed during adoption"
            )

        rendered_body = marker_body(
            render_assignment_marker(assignment),
            latest.body,
        )
        self._mutate(
            "PATCH",
            f"/repos/{self._repo}/pulls/{number}",
            json_body={"body": rendered_body},
            expected_statuses={200},
        )
        after = self.pull_request(number)
        if not self._is_existing_adoption(after, assignment):
            raise ReconciliationError(
                "GitHub did not persist the adopted assignment marker"
            )
        return MutationResult(
            changed=True,
            resource_url=after.url,
            state="assignment_adopted",
            version=after.head_sha,
        )

    def _is_existing_adoption(
        self,
        snapshot: PullRequestSnapshot,
        assignment: AssignmentRecord,
    ) -> bool:
        if snapshot.author.casefold() != self._actor().casefold():
            raise WorkflowPreconditionError(
                "assignment pull request must be authored by the authenticated actor"
            )
        if snapshot.base_ref != assignment.base_ref:
            raise WorkflowPreconditionError(
                f"pull request base is {snapshot.base_ref!r}, "
                f"expected {assignment.base_ref!r}"
            )
        if snapshot.head_ref != assignment.head_ref:
            raise WorkflowPreconditionError(
                f"pull request head is {snapshot.head_ref!r}, "
                f"expected {assignment.head_ref!r}"
            )

        try:
            markers = parse_assignment_markers(snapshot.body)
        except ValueError as error:
            raise WorkflowPreconditionError(
                f"pull request contains an invalid assignment marker: {error}"
            ) from error
        protocol_lines = [
            line
            for line in snapshot.body.splitlines()
            if line.startswith("<!-- senpai-")
        ]
        if markers:
            if markers != (assignment,) or protocol_lines != [
                render_assignment_marker(assignment)
            ]:
                raise WorkflowPreconditionError(
                    "pull request must contain exactly the requested assignment marker"
                )
            return True
        if protocol_lines:
            raise WorkflowPreconditionError(
                "markerless assignment body must not contain other Senpai markers"
            )
        return False

    def _require_adoptable_assignment(
        self,
        snapshot: PullRequestSnapshot,
        assignment: AssignmentRecord,
    ) -> None:
        require_open(snapshot)
        if self._is_existing_adoption(snapshot, assignment):
            raise ReconciliationError(
                "assignment marker appeared during adoption; retry the operation"
            )
        if not snapshot.draft:
            raise WorkflowPreconditionError(
                "assignment pull request must be draft before adoption"
            )
        if not snapshot.body.strip():
            raise WorkflowPreconditionError(
                "assignment pull request body must not be empty"
            )

        labels = set(snapshot.labels)
        student_labels = {label for label in labels if label.startswith("student:")}
        if student_labels != {f"student:{assignment.student}"}:
            raise WorkflowPreconditionError(
                "assignment pull request must have exactly its requested student label"
            )
        status_labels = {label for label in labels if label.startswith("status:")}
        if status_labels != {"status:wip"}:
            raise WorkflowPreconditionError(
                "assignment pull request must have status:wip as its only status"
            )
        if assignment.base_ref not in labels:
            raise WorkflowPreconditionError(
                "assignment pull request must retain its configured base label"
            )
        if any(
            comment.author.casefold() == self._actor().casefold()
            and authoritative_marker_line(comment.body).startswith(
                ("<!-- senpai-result:", "<!-- senpai-disposition:")
            )
            for comment in self._comments(snapshot.number)
        ):
            raise WorkflowPreconditionError(
                "cannot adopt a pull request with terminal Senpai evidence"
            )
