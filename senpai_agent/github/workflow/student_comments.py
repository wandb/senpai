"""Student-authored progress and question comments on active assignments."""

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import MutationResult, PullRequestSnapshot
from senpai_agent.github.workflow.text import marker_body, role_prefixed_comment
from senpai_agent.github.workflow.validation import (
    require_current_revision,
    require_open,
)
from senpai_agent.models import (
    AssignmentRecord,
    AssignmentCommentRecord,
    render_assignment_comment_marker,
)


def _require_assigned_routing(
    snapshot: PullRequestSnapshot,
    assignment: AssignmentRecord,
) -> None:
    labels = set(snapshot.labels)
    student_labels = {label for label in labels if label.startswith("student:")}
    if student_labels != {f"student:{assignment.student}"}:
        raise WorkflowPreconditionError(
            "pull request must retain exactly its assigned student label"
        )
    if len(labels & {"status:wip", "status:review"}) != 1:
        raise WorkflowPreconditionError(
            "pull request must retain exactly one active assignment status"
        )


class StudentCommentMixin:
    __slots__ = ()

    def post_assignment_comment(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        student: str,
        comment_id: str,
        comment: str,
    ) -> MutationResult:
        """Post one idempotent student message to its current assignment."""

        if self._role != "student":
            raise PermissionError("post_assignment_comment requires a student workflow")
        with self._assignment_lifecycle_lock:
            return self._post_assignment_comment(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                student=student,
                comment_id=comment_id,
                comment=comment,
            )

    def _post_assignment_comment(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        student: str,
        comment_id: str,
        comment: str,
    ) -> MutationResult:
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(before)
        require_current_revision(assignment, revision_id)
        _require_assigned_routing(before, assignment)
        if assignment.student != student:
            raise PermissionError(
                f"assignment student {assignment.student!r} does not match this "
                f"runtime's student {student!r}"
            )

        comment_id = comment_id.strip()
        content = comment.strip()
        if not comment_id or not content:
            raise ValueError("comment_id and comment must not be empty")
        marker = render_assignment_comment_marker(
            AssignmentCommentRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=assignment.revision_id,
                student=assignment.student,
                comment_id=comment_id,
            )
        )
        rendered = marker_body(marker, content)
        existing = self._marker_comments(number, marker)
        if len(existing) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple comments for marker {marker!r}"
            )
        desired_body = role_prefixed_comment(rendered, self._role)
        if existing and existing[0].body != desired_body:
            raise WorkflowPreconditionError(
                "comment_id already identifies a different message; "
                "use a new comment_id"
            )

        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=rendered,
        )
        after, current_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(after)
        require_current_revision(current_assignment, revision_id)
        _require_assigned_routing(after, current_assignment)
        if current_assignment.student != student:
            raise ReconciliationError("assignment student changed while posting comment")
        return MutationResult(
            changed=changed,
            resource_url=verified.url,
            state="assignment_comment_posted",
            version=after.head_sha,
        )
