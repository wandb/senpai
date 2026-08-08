"""Revise assignments and send guidance within a revision."""

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    StaleAssignmentRevisionError,
    StaleResearchBaseError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.text import (
    marker_body,
    replace_assignment_marker,
    role_prefixed_comment,
)
from senpai_agent.github.workflow.validation import (
    require_active_assignment_routing,
    require_current_revision,
    require_exact_labels,
    require_open,
)
from senpai_agent.models import (
    AssignmentFeedbackRecord,
    RevisionRecord,
    parse_assignment_markers,
    render_assignment_feedback_marker,
    render_revision_marker,
)


class RevisionMixin:
    __slots__ = ()

    def request_revision(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        new_revision_id: str,
        expected_head_sha: str,
        required_base_sha: str,
        comment: str,
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._request_revision(
                number,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                new_revision_id=new_revision_id,
                expected_head_sha=expected_head_sha,
                required_base_sha=required_base_sha,
                comment=comment,
            )

    def _request_revision(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        new_revision_id: str,
        expected_head_sha: str,
        required_base_sha: str,
        comment: str,
    ) -> MutationResult:
        if current_revision_id == new_revision_id:
            raise ValueError("new_revision_id must differ from current_revision_id")
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(before)
        if assignment.revision_id == current_revision_id:
            live_base_sha = self._branch_head_sha(assignment.base_ref)
            if live_base_sha != required_base_sha:
                raise StaleResearchBaseError(
                    f"required research base {assignment.base_ref}@"
                    f"{required_base_sha} does not match live {live_base_sha}"
                )
        elif not (
            assignment.revision_id == new_revision_id
            and assignment.base_sha == required_base_sha
        ):
            raise StaleAssignmentRevisionError(
                f"assignment revision is {assignment.revision_id!r}, expected "
                f"current {current_revision_id!r} or applied {new_revision_id!r}"
            )
        conflicts = tuple(
            active_number
            for active_number in self._active_student_assignment_numbers(
                assignment.student
            )
            if active_number != number
        )
        if conflicts:
            raise WorkflowPreconditionError(
                f"student:{assignment.student} already has active assignment "
                f"PR(s): {', '.join(f'#{active_number}' for active_number in conflicts)}"
            )
        marker = render_revision_marker(
            RevisionRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=new_revision_id,
                requested_head_sha=expected_head_sha,
            )
        )
        rendered_comment = marker_body(marker, comment)
        marker_comments = self._marker_comments(number, marker)
        if len(marker_comments) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple comments for marker {marker!r}"
            )
        revised_assignment = assignment.model_copy(
            update={
                "revision_id": new_revision_id,
                "base_sha": required_base_sha,
            }
        )
        revised_body = replace_assignment_marker(before.body, revised_assignment)
        assignment_changed = revised_body != before.body
        if assignment_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{number}",
                json_body={"body": revised_body},
                expected_statuses={200},
            )
        current = self._pull_at_head(number, expected_head_sha)
        require_open(current)
        if parse_assignment_markers(current.body) != (revised_assignment,):
            raise ReconciliationError("GitHub did not update the assignment revision")
        marker_changed, _ = self._upsert_marker_comment(
            number,
            marker=marker,
            body=rendered_comment,
        )
        draft_changed = self._set_draft(current, draft=True)
        labels_changed, desired_labels = self._set_labels(
            number,
            current,
            add={"status:wip"},
            remove={"status:review"},
        )
        after = self._pull_at_head(number, expected_head_sha)
        require_open(after)
        if not after.draft:
            raise ReconciliationError(
                "GitHub did not convert the pull request to draft"
            )
        require_exact_labels(after, desired_labels)
        return MutationResult(
            changed=assignment_changed
            or marker_changed
            or draft_changed
            or labels_changed,
            resource_url=after.url,
            state="revision_requested",
            version=after.head_sha,
        )

    def send_assignment_feedback(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        feedback_id: str,
        comment: str,
    ) -> MutationResult:
        """Upsert guidance for the current assignment without starting a revision."""

        with self._assignment_lifecycle_lock:
            return self._send_assignment_feedback(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                feedback_id=feedback_id,
                comment=comment,
            )

    def _send_assignment_feedback(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        feedback_id: str,
        comment: str,
    ) -> MutationResult:
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(before)
        require_current_revision(assignment, revision_id)
        require_active_assignment_routing(before, assignment)
        feedback_id = feedback_id.strip()
        body = comment.strip()
        if not feedback_id or not body:
            raise ValueError("feedback_id and comment must not be empty")

        marker = render_assignment_feedback_marker(
            AssignmentFeedbackRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=assignment.revision_id,
                feedback_id=feedback_id,
            )
        )
        rendered_comment = marker_body(marker, body)
        existing = self._marker_comments(number, marker)
        if len(existing) > 1:
            raise ReconciliationError(
                f"GitHub contains multiple comments for marker {marker!r}"
            )
        desired_body = role_prefixed_comment(rendered_comment, self._role)
        if existing and role_prefixed_comment(existing[0].body, self._role) != desired_body:
            raise WorkflowPreconditionError(
                "feedback_id already identifies different guidance; "
                "use a new feedback_id"
            )
        changed, verified = self._upsert_marker_comment(
            number,
            marker=marker,
            body=rendered_comment,
        )
        after, current_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(after)
        require_current_revision(current_assignment, revision_id)
        require_active_assignment_routing(after, current_assignment)
        return MutationResult(
            changed=changed,
            resource_url=verified.url,
            state="assignment_feedback_upserted",
            version=after.head_sha,
        )
