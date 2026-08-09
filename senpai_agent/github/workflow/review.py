"""Accept results on moved bases and close completed experiments."""

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.github.workflow.validation import (
    require_assignment_result,
    require_current_revision,
    require_exact_research_base,
    require_labels,
    require_open,
    require_unmerged,
)
from senpai_agent.models import (
    ResearchBaseAcceptanceRecord,
    experiment_result_digest,
    render_research_base_acceptance_marker,
)


class ReviewMixin:
    __slots__ = ()

    def accept_result_on_current_base(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        expected_current_base_sha: str,
        reason: str,
    ) -> MutationResult:
        """Durably approve one exact result against the exact live research base."""

        with self._assignment_lifecycle_lock:
            before, assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            require_open(before)
            require_current_revision(assignment, current_revision_id)
            if before.draft:
                raise WorkflowPreconditionError(
                    "cannot accept a result while its pull request is draft"
                )
            require_labels(before, required={"status:review"}, forbidden=set())
            result = self._require_result(
                number,
                assignment_id=assignment_id,
                revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
            )
            require_assignment_result(before, result)
            require_exact_research_base(
                assignment,
                live_base_sha=self._branch_head_sha(assignment.base_ref),
                expected_current_base_sha=expected_current_base_sha,
            )
            acceptance = ResearchBaseAcceptanceRecord(
                repo=self._repo,
                pr_number=number,
                assignment_id=assignment.assignment_id,
                revision_id=assignment.revision_id,
                result_head_sha=expected_head_sha,
                result_digest=experiment_result_digest(result),
                evaluated_base_sha=assignment.base_sha,
                base_ref=assignment.base_ref,
                accepted_base_sha=expected_current_base_sha,
            )
            marker = render_research_base_acceptance_marker(acceptance)
            changed, verified = self._upsert_marker_comment(
                number,
                marker=marker,
                body=marker_body(marker, reason),
            )

            after, current_assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            require_open(after)
            require_current_revision(current_assignment, current_revision_id)
            current_result = self._require_result(
                number,
                assignment_id=assignment_id,
                revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
            )
            require_assignment_result(after, current_result)
            require_exact_research_base(
                current_assignment,
                live_base_sha=self._branch_head_sha(current_assignment.base_ref),
                expected_current_base_sha=expected_current_base_sha,
            )
            self._require_research_base_acceptance(number, acceptance)
            return MutationResult(
                changed=changed,
                resource_url=verified.url,
                state="research_base_accepted",
                version=expected_current_base_sha,
            )

    def close_experiment(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        marker: str,
        reason: str,
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._close_experiment(
                number,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
                marker=marker,
                reason=reason,
            )

    def _close_experiment(
        self,
        number: int,
        *,
        assignment_id: str,
        current_revision_id: str,
        expected_head_sha: str,
        marker: str,
        reason: str,
    ) -> MutationResult:
        before, assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_current_revision(assignment, current_revision_id)
        require_unmerged(before)
        state_changed = before.state != "closed"
        if state_changed:
            self._mutate(
                "PATCH",
                f"/repos/{self._repo}/pulls/{number}",
                json_body={"state": "closed"},
                expected_statuses={200},
            )
        closed, closed_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_current_revision(closed_assignment, current_revision_id)
        require_unmerged(closed)
        if closed.state != "closed":
            raise ReconciliationError("GitHub did not close the pull request")
        rendered_body = marker_body(marker, reason)
        marker_changed, _ = self._upsert_marker_comment(
            number,
            marker=marker,
            body=rendered_body,
        )
        after, after_assignment = self._assigned_pull_at_head(
            number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        require_current_revision(after_assignment, current_revision_id)
        require_unmerged(after)
        if after.state != "closed":
            raise ReconciliationError("pull request reopened during reconciliation")
        return MutationResult(
            changed=marker_changed or state_changed,
            resource_url=after.url,
            state="experiment_closed",
            version=after.head_sha,
        )
