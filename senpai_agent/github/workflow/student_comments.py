"""Student-authored comments on current assignments."""

from senpai_agent.github.workflow.errors import ReconciliationError
from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.models import (
    AssignmentCommentRecord,
    render_assignment_comment_marker,
)


_COMMENT_STATUSES = frozenset({"status:wip", "status:review"})


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
            _before, assignment = self._routed_assignment_at_head(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                allowed_statuses=_COMMENT_STATUSES,
            )
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
            changed, verified = self._upsert_marker_comment(
                number,
                marker=marker,
                body=rendered,
                conflict_message=(
                    "comment_id already identifies a different message; "
                    "use a new comment_id"
                ),
                exact_conflict=True,
            )
            after, current_assignment = self._routed_assignment_at_head(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                allowed_statuses=_COMMENT_STATUSES,
            )
            if current_assignment.student != student:
                raise ReconciliationError(
                    "assignment student changed while posting comment"
                )
            return MutationResult(
                changed=changed,
                resource_url=verified.url,
                state="assignment_comment_posted",
                version=after.head_sha,
            )
