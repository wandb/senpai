"""Merge one reviewed experiment without losing its result evidence."""

from typing import Literal

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import MutationResult
from senpai_agent.github.workflow.validation import (
    require_assignment_result,
    require_current_revision,
    require_exact_research_base,
    require_labels,
    require_open,
    require_same_result,
)
from senpai_agent.models import ResearchBaseAcceptanceRecord, experiment_result_digest


class MergeMixin:
    __slots__ = ()

    def merge_experiment(
        self,
        number: int,
        *,
        expected_head_sha: str,
        assignment_id: str,
        current_revision_id: str,
        expected_current_base_sha: str,
        merge_method: Literal["merge", "squash", "rebase"] = "squash",
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._merge_experiment(
                number,
                expected_head_sha=expected_head_sha,
                assignment_id=assignment_id,
                current_revision_id=current_revision_id,
                expected_current_base_sha=expected_current_base_sha,
                merge_method=merge_method,
            )

    def _merge_experiment(
        self,
        number: int,
        *,
        expected_head_sha: str,
        assignment_id: str,
        current_revision_id: str,
        expected_current_base_sha: str,
        merge_method: Literal["merge", "squash", "rebase"],
    ) -> MutationResult:
        if merge_method not in ("merge", "squash", "rebase"):
            raise ValueError("merge_method must be merge, squash, or rebase")
        if not assignment_id.strip():
            raise ValueError("assignment_id must not be empty")
        before = self._pull_at_head(number, expected_head_sha)
        terminal_result = self._require_result(
            number,
            assignment_id=assignment_id,
            revision_id=current_revision_id,
            expected_head_sha=expected_head_sha,
        )
        assignment = require_assignment_result(before, terminal_result)
        require_current_revision(assignment, current_revision_id)
        if before.merged:
            if before.state != "closed":
                raise ReconciliationError(
                    "GitHub returned a merged pull request that is not closed"
                )
            if not before.merge_commit_sha:
                raise ReconciliationError(
                    "GitHub returned a merged pull request without a merge SHA"
                )
            return MutationResult(
                changed=False,
                resource_url=before.url,
                state="experiment_merged",
                version=before.merge_commit_sha,
            )

        require_open(before)
        if before.draft:
            raise WorkflowPreconditionError("cannot merge a draft pull request")
        require_labels(before, required={"status:review"}, forbidden=set())
        blocking_labels = {
            "status:blocked",
            "status:hold",
            "status:needs-rebase",
            "status:wip",
        }.intersection(before.labels)
        if blocking_labels:
            raise WorkflowPreconditionError(
                "cannot merge with blocking label(s): "
                + ", ".join(sorted(blocking_labels))
            )
        if before.mergeable is False:
            raise WorkflowPreconditionError(
                "cannot merge a pull request with a merge conflict"
            )
        if before.mergeable is None:
            raise WorkflowPreconditionError(
                "cannot merge while GitHub mergeability is unknown"
            )

        require_exact_research_base(
            assignment,
            live_base_sha=self._branch_head_sha(assignment.base_ref),
            expected_current_base_sha=expected_current_base_sha,
        )
        if assignment.base_sha != expected_current_base_sha:
            self._require_research_base_acceptance(
                number,
                ResearchBaseAcceptanceRecord(
                    repo=self._repo,
                    pr_number=number,
                    assignment_id=assignment.assignment_id,
                    revision_id=assignment.revision_id,
                    result_head_sha=expected_head_sha,
                    result_digest=experiment_result_digest(terminal_result),
                    evaluated_base_sha=assignment.base_sha,
                    base_ref=assignment.base_ref,
                    accepted_base_sha=expected_current_base_sha,
                ),
            )

        require_exact_research_base(
            assignment,
            live_base_sha=self._branch_head_sha(assignment.base_ref),
            expected_current_base_sha=expected_current_base_sha,
        )
        require_same_result(
            terminal_result,
            self._require_result(
                number,
                assignment_id=assignment_id,
                revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
            ),
            phase="immediately before merge",
        )

        self._mutate(
            "PUT",
            f"/repos/{self._repo}/pulls/{number}/merge",
            json_body={
                "sha": expected_head_sha,
                "merge_method": merge_method,
            },
            expected_statuses={200},
        )
        require_same_result(
            terminal_result,
            self._require_result(
                number,
                assignment_id=assignment_id,
                revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
            ),
            phase="immediately after merge",
        )
        after = self._pull_at_head(number, expected_head_sha)
        if not after.merged or after.state != "closed":
            raise ReconciliationError("GitHub did not merge the pull request")
        if not after.merge_commit_sha:
            raise ReconciliationError(
                "GitHub did not return the resulting merge commit SHA"
            )
        require_same_result(
            terminal_result,
            self._require_result(
                number,
                assignment_id=assignment_id,
                revision_id=current_revision_id,
                expected_head_sha=expected_head_sha,
            ),
            phase="final merge reconciliation",
        )
        return MutationResult(
            changed=True,
            resource_url=after.url,
            state="experiment_merged",
            version=after.merge_commit_sha,
        )
