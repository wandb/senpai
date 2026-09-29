"""Student-authored comments on their own and peers' assignments."""

from urllib.parse import urlencode

from senpai_agent.github.pull_authorization import authorized_pulls
from senpai_agent.github.workflow.errors import (
    ReconciliationError,
    WorkflowPreconditionError,
)
from senpai_agent.github.workflow.responses import (
    BroadcastResult,
    MutationResult,
    NumberedResponse,
    validated_response,
)
from senpai_agent.github.workflow.text import marker_body
from senpai_agent.github.workflow.validation import (
    require_active_assignment_routing,
    require_assignment_identity,
    require_assignment_student_label,
    require_open,
)
from senpai_agent.models import (
    MAX_BROADCAST_MESSAGE_CHARS,
    AssignmentCommentRecord,
    StudentPeerCommentRecord,
    authoritative_marker_line,
    parse_assignment_markers,
    parse_student_peer_comment_markers,
    render_assignment_comment_marker,
    render_student_peer_comment_marker,
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
            before, assignment = self._routed_assignment_at_head(
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
            rendered = marker_body(
                marker, f"{content}\n\nWorking PR: [#{number}]({before.url})"
            )
            changed, verified = self._upsert_marker_comment(
                number,
                marker=marker,
                body=rendered,
                conflict_message=(
                    "comment_id already identifies a different message; "
                    "use a new comment_id"
                ),
                exact_conflict=True,
                student=student,
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

    def post_peer_comment(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        student: str,
        advisor_branch: str,
        target_pr_number: int,
        comment_id: str,
        comment: str,
        broadcast_id: str | None = None,
    ) -> MutationResult:
        """Message another assigned student without changing either branch."""

        if self._role != "student":
            raise PermissionError("post_peer_comment requires a student workflow")
        with self._assignment_lifecycle_lock:
            source, assignment = self._routed_assignment_at_head(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                allowed_statuses=_COMMENT_STATUSES,
            )
            if assignment.student != student:
                raise PermissionError("source assignment does not belong to this student")
            if assignment.base_ref != advisor_branch:
                raise WorkflowPreconditionError("source must use the configured advisor base")
            target = self.pull_request(target_pr_number)
            require_open(target)
            try:
                records = parse_assignment_markers(target.body)
            except ValueError as error:
                raise WorkflowPreconditionError("target has an invalid assignment") from error
            if len(records) != 1:
                raise WorkflowPreconditionError("target must have exactly one assignment")
            recipient = require_assignment_identity(
                target, repo=self._repo, assignment_id=records[0].assignment_id
            )
            if broadcast_id is None:
                require_active_assignment_routing(
                    target, recipient, allowed_statuses=_COMMENT_STATUSES
                )
            else:
                require_assignment_student_label(target, recipient)
            if target.number == number or recipient.student == student:
                raise WorkflowPreconditionError("target must belong to another student")
            if recipient.base_ref != assignment.base_ref:
                raise WorkflowPreconditionError("target must use the same advisor base")
            comment_id, content = comment_id.strip(), comment.strip()
            if not comment_id or not content:
                raise ValueError("comment_id and comment must not be empty")
            record = self._peer_comment_binding(
                StudentPeerCommentRecord(
                    repo=self._repo,
                    pr_number=target.number,
                    assignment_id=recipient.assignment_id,
                    revision_id=recipient.revision_id,
                    student=student,
                    source_pr_number=number,
                    source_assignment_id=assignment.assignment_id,
                    source_revision_id=assignment.revision_id,
                    comment_id=comment_id,
                    broadcast_id=broadcast_id,
                )
            )
            marker = render_student_peer_comment_marker(record)
            changed, verified = self._upsert_marker_comment(
                target.number,
                marker=marker,
                body=marker_body(
                    marker, f"{content}\n\nWorking PR: [#{number}]({source.url})"
                ),
                student=student,
                conflict_message=(
                    "comment_id already identifies a different message; use a new comment_id"
                ),
                exact_conflict=True,
            )
            _, current = self._routed_assignment_at_head(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                allowed_statuses=_COMMENT_STATUSES,
            )
            if current.student != student:
                raise ReconciliationError("source student changed while posting comment")
            after = self.pull_request(target.number)
            require_open(after)
            current_recipient = require_assignment_identity(
                after, repo=self._repo, assignment_id=recipient.assignment_id
            )
            if broadcast_id is None:
                require_active_assignment_routing(
                    after, current_recipient, allowed_statuses=_COMMENT_STATUSES
                )
            else:
                require_assignment_student_label(after, current_recipient)
            if current_recipient != recipient:
                raise ReconciliationError("recipient assignment changed while posting comment")
            return MutationResult(
                changed=changed,
                resource_url=verified.url,
                state="peer_comment_posted",
                version=after.head_sha,
            )

    def broadcast_message(
        self,
        number: int,
        *,
        assignment_id: str,
        revision_id: str,
        expected_head_sha: str,
        student: str,
        advisor_branch: str,
        broadcast_id: str,
        message: str,
    ) -> BroadcastResult:
        """Publish one compact discovery and fan it out to current peer PRs."""

        if self._role != "student":
            raise PermissionError("broadcast_message requires a student workflow")
        broadcast_id, message = broadcast_id.strip(), message.strip()
        if not 1 <= len(broadcast_id) <= 256:
            raise ValueError("broadcast_id must contain 1 to 256 characters")
        if not 1 <= len(message) <= MAX_BROADCAST_MESSAGE_CHARS:
            raise ValueError(
                f"message must contain 1 to {MAX_BROADCAST_MESSAGE_CHARS} characters"
            )
        with self._assignment_lifecycle_lock:
            _, assignment = self._routed_assignment_at_head(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                allowed_statuses=_COMMENT_STATUSES,
            )
            if assignment.student != student:
                raise PermissionError("source assignment does not belong to this student")
            if assignment.base_ref != advisor_branch:
                raise WorkflowPreconditionError("source must use the configured advisor base")

            recipients = self._broadcast_recipients(student, advisor_branch)
            comment_id = f"broadcast:{broadcast_id}"
            content = f"**Broadcast FYI**\n\n{message}"
            # The source copy binds the ID to its content even after peers close PRs.
            source = self.post_assignment_comment(
                number,
                assignment_id=assignment_id,
                revision_id=revision_id,
                expected_head_sha=expected_head_sha,
                student=student,
                comment_id=comment_id,
                comment=content,
            )
            changed = source.changed
            for target_pr_number in recipients:
                result = self.post_peer_comment(
                    number,
                    assignment_id=assignment_id,
                    revision_id=revision_id,
                    expected_head_sha=expected_head_sha,
                    student=student,
                    advisor_branch=advisor_branch,
                    target_pr_number=target_pr_number,
                    comment_id=comment_id,
                    comment=(
                        f"{content}\n\n"
                        f"Discovery: [source discussion]({source.resource_url})"
                    ),
                    broadcast_id=broadcast_id,
                )
                changed |= result.changed
            return BroadcastResult(
                changed=changed,
                resource_url=source.resource_url,
                state="broadcast_posted",
                version=source.version,
                delivered_pr_numbers=recipients,
            )

    def _broadcast_recipients(
        self, student: str, advisor_branch: str
    ) -> tuple[int, ...]:
        query = urlencode({"state": "open", "base": advisor_branch, "per_page": 100})
        recipients = set()
        pulls = authorized_pulls(
            self._objects(f"/repos/{self._repo}/pulls?{query}"),
            repo=self._repo,
            get=lambda path: self._request(
                "GET", path, expected_statuses={200}
            ).json_body,
        )
        for item in pulls:
            number = validated_response(NumberedResponse, item, "broadcast PR").number
            target = self.pull_request(number)
            try:
                records = parse_assignment_markers(target.body)
                if len(records) != 1:
                    continue
                recipient = require_assignment_identity(
                    target, repo=self._repo, assignment_id=records[0].assignment_id
                )
                require_open(target)
                require_assignment_student_label(target, recipient)
            except (ValueError, WorkflowPreconditionError):
                # Ordinary PRs and assignments with invalid ownership are not peers.
                continue
            if recipient.student != student and recipient.base_ref == advisor_branch:
                recipients.add(number)
        return tuple(sorted(recipients))

    def _peer_comment_binding(
        self, proposed: StudentPeerCommentRecord
    ) -> StudentPeerCommentRecord:
        """Keep retries bound to the recipient revision first used for this ID."""

        recipient_fields = {"assignment_id", "revision_id"}
        identity = proposed.model_dump(exclude=recipient_fields)
        actor = self._actor().casefold()
        matches = set()
        for comment in self._comments(proposed.pr_number):
            if comment.author.casefold() != actor:
                continue
            marker = authoritative_marker_line(comment.body)
            if not marker.startswith("<!-- senpai-student-peer-comment:"):
                continue
            for record in parse_student_peer_comment_markers(marker):
                if record.model_dump(exclude=recipient_fields) == identity:
                    matches.add(record)
        if len(matches) > 1:
            raise ReconciliationError("peer comment ID has conflicting recipient bindings")
        existing = next(iter(matches), proposed)
        if existing.assignment_id != proposed.assignment_id:
            raise WorkflowPreconditionError(
                "target assignment changed; review it and use a new comment_id"
            )
        return existing
