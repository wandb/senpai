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
    require_assignment_identity,
    require_assignment_result,
    require_label_update,
    require_open,
    require_result_identity,
)
from senpai_agent.models import AssignmentRecord, ExperimentResult


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
        try:
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
            require_assignment_result(after, result)
        except StaleAssignmentRevisionError:
            self._reconcile_current_result_routing(
                number,
                assignment_id=result.assignment.assignment_id,
                expected_head_sha=expected_head_sha,
            )
            raise
        if after.draft:
            raise ReconciliationError(
                "GitHub did not mark the pull request ready for review"
            )
        require_label_update(
            after,
            required=desired_labels,
            forbidden={"status:wip"},
        )
        return MutationResult(
            changed=result_changed or ready_changed or labels_changed,
            resource_url=after.url,
            state="result_submitted",
            version=after.head_sha,
        )

    def _reconcile_current_result_routing(
        self,
        number: int,
        *,
        assignment_id: str,
        expected_head_sha: str,
    ) -> tuple[PullRequestSnapshot, AssignmentRecord, bool, bool]:
        """Route the current assignment from its current durable result."""

        changed = False
        for _attempt in range(3):
            current, assignment = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            require_open(current)
            has_result = self._has_result_for_snapshot(current, assignment_id)
            changed |= self._set_draft(current, draft=not has_result)
            current, refreshed = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            require_open(current)
            if refreshed != assignment:
                continue
            if self._has_result_for_snapshot(current, assignment_id) is not has_result:
                continue

            labels_changed, _ = self._set_labels(
                number,
                current,
                add={"status:review" if has_result else "status:wip"},
                remove={"status:wip" if has_result else "status:review"},
            )
            changed |= labels_changed
            after, verified = self._assigned_pull_at_head(
                number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
            require_open(after)
            if verified != assignment:
                continue
            if self._has_result_for_snapshot(after, assignment_id) is not has_result:
                continue
            if has_result:
                if (
                    after.draft
                    or "status:review" not in after.labels
                    or "status:wip" in after.labels
                ):
                    raise ReconciliationError(
                        "current-revision result did not remain reviewable during "
                        "current-assignment routing"
                    )
            elif not (
                after.draft
                and "status:wip" in after.labels
                and "status:review" not in after.labels
            ):
                raise ReconciliationError(
                    "GitHub did not restore the current assignment revision to WIP"
                )
            return after, verified, has_result, changed
        raise ReconciliationError(
            "assignment or result kept changing during current-assignment routing"
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
    ) -> bool:
        """Keep a concurrently submitted current-revision result reviewable."""

        expected_assignment = require_assignment_identity(
            snapshot,
            repo=self._repo,
            assignment_id=assignment_id,
        )
        after, verified, has_result, changed = (
            self._reconcile_current_result_routing(
                snapshot.number,
                assignment_id=assignment_id,
                expected_head_sha=expected_head_sha,
            )
        )
        if (
            verified != expected_assignment
            or not has_result
            or after.draft
            or "status:review" not in after.labels
            or "status:wip" in after.labels
        ):
            raise ReconciliationError(
                "current-revision result did not remain reviewable during "
                "current-assignment routing"
            )
        return changed
