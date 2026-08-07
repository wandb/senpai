"""Publish experiment results and recover stale concurrent submissions."""

from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    MutationResult,
    PullRequestSnapshot,
    SubmitResultPreflight,
)
from senpai_agent.github.workflow.validation import (
    require_assignment_result,
    require_exact_labels,
    require_open,
    require_result_identity,
)
from senpai_agent.models import ExperimentResult


class ResultMixin:
    __slots__ = ()

    def preflight_submit_result(
        self,
        number: int,
        *,
        branch: str,
        current_head_sha: str,
        expected_result_head_sha: str,
        result: ExperimentResult,
    ) -> SubmitResultPreflight:
        """Validate an assignment/result pair before mutating its Git branch."""

        snapshot = self._pull_at_head(number, current_head_sha)
        require_open(snapshot)
        if snapshot.head_ref != branch:
            raise WorkflowPreconditionError(
                f"pull request branch is {snapshot.head_ref!r}, expected {branch!r}"
            )
        require_result_identity(
            result,
            repo=self._repo,
            number=number,
            expected_head_sha=expected_result_head_sha,
        )
        assignment = require_assignment_result(snapshot, result)
        return SubmitResultPreflight(snapshot=snapshot, assignment=assignment)

    def submit_result(
        self,
        number: int,
        *,
        expected_head_sha: str,
        result: ExperimentResult,
    ) -> MutationResult:
        with self._assignment_lifecycle_lock:
            return self._submit_result(
                number,
                expected_head_sha=expected_head_sha,
                result=result,
            )

    def _submit_result(
        self,
        number: int,
        *,
        expected_head_sha: str,
        result: ExperimentResult,
    ) -> MutationResult:
        before = self._pull_at_head(number, expected_head_sha)
        require_open(before)
        require_result_identity(
            result,
            repo=self._repo,
            number=number,
            expected_head_sha=expected_head_sha,
        )
        require_assignment_result(before, result)
        result_changed, _ = self._upsert_result_comment(number, result=result)
        current = self._pull_at_head(number, expected_head_sha)
        require_open(current)
        require_assignment_result(current, result)
        ready_changed = self._set_draft(current, draft=False)
        labels_changed, desired_labels = self._set_labels(
            number,
            current,
            add={"status:review"},
            remove={"status:wip"},
        )
        after = self._pull_at_head(number, expected_head_sha)
        require_open(after)
        try:
            require_assignment_result(after, result)
        except StaleAssignmentRevisionError:
            self._recover_stale_submit_routing(
                number,
                assignment_id=result.assignment.assignment_id,
                expected_head_sha=expected_head_sha,
                stale_result=result,
            )
            raise
        if after.draft:
            raise ReconciliationError(
                "GitHub did not mark the pull request ready for review"
            )
        require_exact_labels(after, desired_labels)
        return MutationResult(
            changed=result_changed or ready_changed or labels_changed,
            resource_url=after.url,
            state="result_submitted",
            version=after.head_sha,
        )

    def _recover_stale_submit_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
        stale_result: ExperimentResult,
    ) -> None:
        """Undo stale review routing without clobbering newer valid evidence."""

        for _attempt in range(3):
            current, assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            require_open(current)
            if assignment.revision_id == stale_result.assignment.revision_id:
                return
            if self._has_result_for_snapshot(current, assignment_id):
                return

            self._set_draft(current, draft=True)
            current, refreshed = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            if refreshed != assignment:
                continue
            if self._has_result_for_snapshot(current, assignment_id):
                self._restore_current_result_review(
                    current,
                    assignment_id=assignment_id,
                    expected_head_sha=expected_head_sha,
                )
                return

            self._set_labels(
                number,
                current,
                add={"status:wip"},
                remove={"status:review"},
            )
            after, verified = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            if verified != assignment:
                continue
            if self._has_result_for_snapshot(after, assignment_id):
                self._restore_current_result_review(
                    after,
                    assignment_id=assignment_id,
                    expected_head_sha=expected_head_sha,
                )
                return
            if (
                after.draft
                and "status:wip" in after.labels
                and "status:review" not in after.labels
            ):
                return
            raise ReconciliationError(
                "GitHub did not restore the current assignment revision to WIP"
            )
        raise ReconciliationError(
            "assignment revision kept changing during stale-submit recovery"
        )

    def _has_result_for_snapshot(
        self,
        snapshot: PullRequestSnapshot,
        assignment_id: str,
    ) -> bool:
        for match in self._result_comments(snapshot.number, assignment_id):
            try:
                require_result_identity(
                    match.result,
                    repo=self._repo,
                    number=snapshot.number,
                    expected_head_sha=snapshot.head_sha,
                )
                require_assignment_result(snapshot, match.result)
            except WorkflowPreconditionError:
                continue
            return True
        return False

    def _restore_current_result_review(
        self,
        snapshot: PullRequestSnapshot,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> None:
        """Keep a concurrently submitted current-revision result reviewable."""

        self._set_draft(snapshot, draft=False)
        current, assignment = self._assigned_pull_at_head(
            snapshot.number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        if not self._has_result_for_snapshot(current, assignment_id):
            raise ReconciliationError(
                "current-revision result disappeared during stale-submit recovery"
            )
        self._set_labels(
            snapshot.number,
            current,
            add={"status:review"},
            remove={"status:wip"},
        )
        after, verified = self._assigned_pull_at_head(
            snapshot.number,
            assignment_id=assignment_id,
            expected_head_sha=expected_head_sha,
        )
        if verified != assignment or not self._has_result_for_snapshot(
            after, assignment_id
        ):
            raise ReconciliationError(
                "current-revision result changed during stale-submit recovery"
            )
