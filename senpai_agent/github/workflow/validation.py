"""Workflow invariants shared by GitHub operations."""

from senpai_agent.github.human_messages import (
    is_trusted_human_author,
    is_trusted_human_message,
)
from senpai_agent.github.workflow.errors import (
    PullHeadMismatchError,
    ReconciliationError,
    StaleAssignmentRevisionError,
    StaleResearchBaseError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    PullRequestSnapshot,
    ResultComment,
)
from senpai_agent.models import (
    AssignmentRecord,
    ExperimentResult,
    experiment_result_digest,
    parse_assignment_markers,
)


def positive_number(number: int) -> int:
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError("pull request number must be a positive integer")
    return number


def positive_message_id(message_id: int) -> int:
    if (
        isinstance(message_id, bool)
        or not isinstance(message_id, int)
        or message_id <= 0
    ):
        raise ValueError("human message ID must be a positive integer")
    return message_id


def require_trusted_human_message(
    *,
    author: str,
    author_type: str,
    association: str,
    body: str,
    actor: str,
) -> None:
    if is_trusted_human_message(
        author=author,
        author_type=author_type,
        association=association,
        body=body,
        actor=actor,
    ):
        return
    if not is_trusted_human_author(
        author_type=author_type,
        association=association,
    ):
        raise WorkflowPreconditionError(
            f"human message author {author!r} must be an OWNER, MEMBER, or "
            "COLLABORATOR User"
        )
    raise WorkflowPreconditionError(
        "human message must not be a Senpai protocol comment from the "
        "authenticated actor"
    )


def distinct_results(
    matches: tuple[ResultComment, ...],
) -> tuple[ExperimentResult, ...]:
    distinct: dict[str, ExperimentResult] = {}
    for match in matches:
        distinct.setdefault(experiment_result_digest(match.result), match.result)
    return tuple(distinct.values())


def require_same_result(
    expected: ExperimentResult,
    current: ExperimentResult,
    *,
    phase: str,
) -> None:
    if experiment_result_digest(expected) != experiment_result_digest(current):
        raise ReconciliationError(f"terminal result changed {phase}")


def require_head(snapshot: PullRequestSnapshot, expected_head_sha: str) -> None:
    if not expected_head_sha:
        raise ValueError("expected_head_sha must not be empty")
    if snapshot.head_sha != expected_head_sha:
        raise PullHeadMismatchError(
            f"pull request head SHA is {snapshot.head_sha}, "
            f"expected {expected_head_sha}"
        )


def require_result_identity(
    result: ExperimentResult,
    *,
    repo: str,
    number: int,
    expected_head_sha: str,
) -> None:
    assignment = result.assignment
    if assignment.repo != repo:
        raise WorkflowPreconditionError(
            "result repository does not match the GitHub workflow repository"
        )
    if assignment.pr_number != number:
        raise WorkflowPreconditionError(
            "result pull request number does not match the requested pull request"
        )
    if assignment.expected_head_sha != expected_head_sha:
        raise WorkflowPreconditionError(
            "result expected head SHA does not match the requested head SHA"
        )
    if result.commit_sha != expected_head_sha:
        raise WorkflowPreconditionError(
            "result commit does not match the pull request head SHA"
        )


def require_assignment_snapshot(
    snapshot: PullRequestSnapshot,
    assignment: AssignmentRecord,
) -> None:
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
    require_head(snapshot, assignment.head_sha)
    markers = parse_assignment_markers(snapshot.body)
    if markers != (assignment,):
        raise WorkflowPreconditionError(
            "pull request must contain exactly the expected assignment marker"
        )


def require_assignment_result(
    snapshot: PullRequestSnapshot,
    result: ExperimentResult,
) -> AssignmentRecord:
    try:
        markers = parse_assignment_markers(snapshot.body)
    except ValueError as error:
        raise WorkflowPreconditionError(
            f"pull request contains an invalid assignment marker: {error}"
        ) from error
    if len(markers) != 1:
        raise WorkflowPreconditionError(
            "pull request must contain exactly one assignment marker"
        )
    record = markers[0]
    assignment = result.assignment
    mismatches = [
        name
        for name, current, proposed in (
            ("repo", record.repo, assignment.repo),
            ("assignment_id", record.assignment_id, assignment.assignment_id),
            ("revision_id", record.revision_id, assignment.revision_id),
            ("student", record.student, assignment.student),
            ("head_ref", record.head_ref, snapshot.head_ref),
            ("base_ref", record.base_ref, snapshot.base_ref),
        )
        if current != proposed
    ]
    if mismatches:
        error_type = (
            StaleAssignmentRevisionError
            if record.revision_id != assignment.revision_id
            else WorkflowPreconditionError
        )
        raise error_type(
            "terminal result assignment mismatch "
            f"({', '.join(mismatches)}). Current PR #{snapshot.number} marker: "
            f"revision={record.revision_id!r}, student={record.student!r}, "
            f"head={record.head_sha!r}; result: "
            f"revision={assignment.revision_id!r}, "
            f"student={assignment.student!r}. Refresh PR #{snapshot.number} and "
            "rebuild the result from its current assignment marker before retrying."
        )
    return record


def require_exact_research_base(
    assignment: AssignmentRecord,
    *,
    live_base_sha: str,
    expected_current_base_sha: str,
) -> None:
    if not expected_current_base_sha.strip():
        raise ValueError("expected_current_base_sha must not be empty")
    if live_base_sha != expected_current_base_sha:
        raise StaleResearchBaseError(
            f"expected research base {assignment.base_ref}@"
            f"{expected_current_base_sha}, but live base is {live_base_sha}"
        )


def require_assignment_identity(
    snapshot: PullRequestSnapshot,
    *,
    repo: str,
    assignment_id: str,
) -> AssignmentRecord:
    if not assignment_id.strip():
        raise ValueError("assignment_id must not be empty")
    try:
        markers = parse_assignment_markers(snapshot.body)
    except ValueError as error:
        raise WorkflowPreconditionError(
            f"pull request contains an invalid assignment marker: {error}"
        ) from error
    if len(markers) != 1:
        raise WorkflowPreconditionError(
            "pull request must contain exactly one assignment marker"
        )
    assignment = markers[0]
    if (
        assignment.repo != repo
        or assignment.assignment_id != assignment_id
        or assignment.base_ref != snapshot.base_ref
        or assignment.head_ref != snapshot.head_ref
    ):
        raise WorkflowPreconditionError(
            "pull request assignment identity does not match the requested transition"
        )
    return assignment


def require_open(snapshot: PullRequestSnapshot) -> None:
    if snapshot.state != "open" or snapshot.merged:
        raise WorkflowPreconditionError("pull request must be open and unmerged")


def require_unmerged(snapshot: PullRequestSnapshot) -> None:
    if snapshot.merged:
        raise WorkflowPreconditionError(
            "pull request must be unmerged before it can be closed"
        )


def require_current_revision(
    assignment: AssignmentRecord,
    revision_id: str,
) -> None:
    if assignment.revision_id != revision_id:
        raise StaleAssignmentRevisionError(
            f"assignment revision is {assignment.revision_id!r}, "
            f"expected {revision_id!r}"
        )


def require_active_assignment_routing(
    snapshot: PullRequestSnapshot,
    assignment: AssignmentRecord,
    *,
    allowed_statuses: frozenset[str] = frozenset({"status:wip"}),
) -> None:
    labels = set(snapshot.labels)
    student_labels = {label for label in labels if label.startswith("student:")}
    if student_labels != {f"student:{assignment.student}"}:
        raise WorkflowPreconditionError(
            "pull request must retain exactly its assigned student label"
        )
    active_statuses = labels & {"status:wip", "status:review"}
    if len(active_statuses) != 1 or not active_statuses <= allowed_statuses:
        allowed = " or ".join(sorted(allowed_statuses))
        raise WorkflowPreconditionError(
            f"pull request must have exactly one of {allowed} as its active "
            "assignment status"
        )


def require_labels(
    snapshot: PullRequestSnapshot,
    *,
    required: set[str],
    forbidden: set[str],
) -> None:
    labels = set(snapshot.labels)
    missing = required - labels
    if missing:
        raise WorkflowPreconditionError(
            "pull request is missing required label(s): " + ", ".join(sorted(missing))
        )
    present = forbidden & labels
    if present:
        raise ReconciliationError(
            "pull request retains forbidden label(s): " + ", ".join(sorted(present))
        )


def require_label_update(
    snapshot: PullRequestSnapshot,
    *,
    required: tuple[str, ...],
    forbidden: set[str],
) -> None:
    labels = set(snapshot.labels)
    missing = set(required) - labels
    retained = forbidden & labels
    if missing or retained:
        raise ReconciliationError("GitHub did not reach the requested label state")


def validate_labels(labels: set[str]) -> None:
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be non-empty strings")
