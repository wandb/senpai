"""Create assignments and repair their GitHub routing state."""

from typing import Literal

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.github.workflow.validation import (
    require_assignment_result,
    require_assignment_snapshot,
    require_current_revision,
    require_label_update,
    require_open,
)
from senpai_agent.models import AssignmentRecord, render_assignment_marker


class AssignmentMixin:
    __slots__ = ()

    def create_assignment(
        self,
        assignment: AssignmentRecord,
        *,
        title: str,
        body: str,
    ) -> MutationResult:
        """Create or reconcile one typed draft assignment PR."""

        with self._assignment_lifecycle_lock:
            return self._create_assignment(assignment, title=title, body=body)

    def _create_assignment(
        self,
        assignment: AssignmentRecord,
        *,
        title: str,
        body: str,
    ) -> MutationResult:
        if assignment.repo != self._repo:
            raise WorkflowPreconditionError(
                "assignment repository does not match the GitHub workflow"
            )
        title = title.strip()
        body = body.strip()
        if not title or not body:
            raise ValueError("assignment title and body must not be empty")
        rendered_body = marker_body(render_assignment_marker(assignment), body)

        matches = self._assignment_pull_requests(assignment)
        if len(matches) > 1:
            raise ReconciliationError(
                "GitHub contains multiple PRs for the assignment branch"
            )
        active = self._active_student_assignment_numbers(assignment.student)
        current_number = matches[0].number if matches else None
        conflicts = tuple(number for number in active if number != current_number)
        if conflicts:
            raise WorkflowPreconditionError(
                f"student:{assignment.student} already has active assignment "
                f"PR(s): {', '.join(f'#{number}' for number in conflicts)}"
            )

        created = not matches
        if created:
            self._mutate(
                "POST",
                f"/repos/{self._repo}/pulls",
                json_body={
                    "title": title,
                    "body": rendered_body,
                    "head": assignment.head_ref,
                    "base": assignment.base_ref,
                    "draft": True,
                },
                expected_statuses={201},
            )
            matches = self._assignment_pull_requests(assignment)
            if len(matches) != 1:
                raise ReconciliationError(
                    "GitHub did not create exactly one assignment PR"
                )

        current = matches[0]
        require_open(current)
        require_assignment_snapshot(current, assignment)
        content_changed = current.title != title or current.body != rendered_body
        if content_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{current.number}",
                json_body={"title": title, "body": rendered_body},
                expected_statuses={200},
            )
            current = self.pull_request(current.number)
            require_assignment_snapshot(current, assignment)

        draft_changed = self._set_draft(current, draft=True)
        routing_labels = {
            assignment.base_ref,
            f"student:{assignment.student}",
            "status:wip",
        }
        remove = {
            label
            for label in current.labels
            if label.startswith(("student:", "status:"))
            and label not in routing_labels
        }
        labels_changed, desired_labels = self._set_labels(
            current.number,
            current,
            add=routing_labels,
            remove=remove,
        )
        after = self.pull_request(current.number)
        require_assignment_snapshot(after, assignment)
        if not after.draft:
            raise ReconciliationError("assignment pull request is not draft")
        if after.title != title or after.body != rendered_body:
            raise ReconciliationError(
                "assignment pull request content did not converge"
            )
        require_label_update(
            after,
            required=desired_labels,
            forbidden=remove,
        )
        return MutationResult(
            changed=created or content_changed or draft_changed or labels_changed,
            resource_url=after.url,
            state="assignment_created",
            version=after.head_sha,
        )

    def repair_assignment_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        working_state: Literal["wip", "review"],
        blockers: set[Literal["blocked", "hold", "needs-rebase"]],
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._repair_assignment_routing(
                number,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
                working_state=working_state,
                blockers=blockers,
            )

    def _repair_assignment_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        working_state: Literal["wip", "review"],
        blockers: set[Literal["blocked", "hold", "needs-rebase"]],
    ) -> MutationResult:
        if working_state not in ("wip", "review"):
            raise ValueError("working_state must be wip or review")
        invalid_blockers = blockers - {"blocked", "hold", "needs-rebase"}
        if invalid_blockers:
            raise ValueError(
                "unsupported assignment blocker(s): "
                + ", ".join(sorted(invalid_blockers))
            )
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(before)
        require_current_revision(assignment, current_revision_id)
        if working_state == "review":
            result = self._require_result(
                number,
                assignment_id=assignment_id,
                revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
            )
            require_assignment_result(before, result)
        protocol_labels = {
            assignment.base_ref,
            f"student:{assignment.student}",
            f"status:{working_state}",
            *(f"status:{blocker}" for blocker in blockers),
        }
        remove = {
            label
            for label in before.labels
            if label.startswith(("student:", "status:"))
            and label not in protocol_labels
        }
        expected_draft = working_state == "wip"
        draft_changed = self._set_draft(before, draft=expected_draft)
        labels_changed, desired = self._set_labels(
            number,
            before,
            add=protocol_labels,
            remove=remove,
        )
        if not draft_changed and not labels_changed:
            return MutationResult(
                changed=False,
                resource_url=before.url,
                state="assignment_routing_repaired",
                version=before.head_sha,
            )

        after, current_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_open(after)
        require_current_revision(current_assignment, current_revision_id)
        if after.draft is not expected_draft:
            raise ReconciliationError(
                "GitHub did not reach the requested assignment draft state"
            )
        require_label_update(after, required=desired, forbidden=remove)
        return MutationResult(
            changed=draft_changed or labels_changed,
            resource_url=after.url,
            state="assignment_routing_repaired",
            version=after.head_sha,
        )
