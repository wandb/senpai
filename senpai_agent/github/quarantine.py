"""Report quarantined student conversations through assignment comments."""

import sys

from senpai_agent.github.workflow import GitHubWorkflow
from senpai_agent.github.workflow.errors import GitHubWorkflowError
from senpai_agent.inbox import STEER_PRIORITY, PersistentInbox
from senpai_agent.models import AssignmentRecord
from senpai_agent.state import AssignmentConversationRegistry


class StudentQuarantineReporter:
    def __init__(
        self,
        inbox: PersistentInbox,
        registry: AssignmentConversationRegistry,
        workflow: GitHubWorkflow,
    ):
        self.inbox = inbox
        self.registry = registry
        self.workflow = workflow
        self._reported: set[str] = set()

    def report(
        self, number: int, assignment: AssignmentRecord, head_sha: str
    ) -> None:
        quarantined = self.inbox.quarantined_turns()
        if not quarantined:
            return
        conversation_id = self.registry.for_assignment(
            assignment.assignment_id, assignment.revision_id
        )
        for turn in quarantined:
            if turn.conversation_id != str(conversation_id):
                continue
            # Human steering can reopen the same turn; each reopening may fail again.
            episode = next(
                (
                    message.delivery_id
                    for message in reversed(turn.messages)
                    if message.priority == STEER_PRIORITY
                ),
                turn.prompt.delivery_id,
            )
            comment_id = f"quarantine:{turn.turn_id}:{episode}"
            if comment_id in self._reported:
                continue
            try:
                self.workflow.post_assignment_comment(
                    number,
                    assignment_id=assignment.assignment_id,
                    revision_id=assignment.revision_id,
                    expected_head_sha=head_sha,
                    student=assignment.student,
                    comment_id=comment_id,
                    comment=(
                        "Controller alert: student conversation quarantined. "
                        "Advisor intervention required.\n\n"
                        f"Assignment: {assignment.assignment_id}\n"
                        f"Revision: {assignment.revision_id}\n"
                        f"Conversation: {turn.conversation_id}\n"
                        f"Turn: {turn.turn_id}\n"
                        f"Reason: {turn.quarantine_reason}\n\n"
                        "Inspect the assignment and diagnose the blocker. "
                        "Ordinary assignment feedback cannot reopen quarantine. "
                        "Escalate through a human Issue if recovery needs operator access."
                    ),
                )
            except GitHubWorkflowError as error:
                # Keep polling and retry this same immutable comment next time.
                print(
                    f"SENPAI_QUARANTINE_REPORT_ERROR pr={number} "
                    f"turn_id={turn.turn_id} {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                self._reported.add(comment_id)
