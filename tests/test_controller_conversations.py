from pathlib import Path
from uuid import UUID

from senpai_agent.controller import Controller, TurnResult
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import ControllerEvent, LocalStudentMailbox
from senpai_agent.state import (
    AssignmentConversationRegistry,
    StudentConversationSelector,
)


class Mailbox:
    def __init__(self, events):
        self.events = tuple(events)
        self.acknowledged = []

    def poll(self):
        events, self.events = self.events, ()
        return events

    def acknowledge(self, dedupe_keys):
        self.acknowledged.append(tuple(dedupe_keys))


class Turns:
    def __init__(self, exit_codes=()):
        self.exit_codes = iter(exit_codes)
        self.calls = []

    def run(
        self,
        prompt,
        *,
        conversation_id,
        event_keys,
        visible_event_keys=frozenset(),
        inbox,
        inbox_turn_id,
    ):
        self.calls.append(
            (prompt, conversation_id, event_keys, visible_event_keys)
        )
        result = TurnResult(exit_code=next(self.exit_codes, 0))
        if result.exit_code == 0:
            turn = inbox.turn(inbox_turn_id)
            for message in turn.messages:
                inbox.record_delivered(message.delivery_id, message.body)
            inbox.record_processed(inbox_turn_id)
        return result


def run_student_controller(tmp_path: Path, mailbox, turns):
    Controller(
        role="student",
        mailbox=mailbox,
        turns=turns,
        conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
        full_prompt="programme",
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(tmp_path / "students.json")
        ),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)


def test_assignment_registry_persists_one_uuid_per_revision(tmp_path: Path):
    path = tmp_path / "students.json"

    first = AssignmentConversationRegistry(path).for_assignment(
        "assignment-1", "revision-2"
    )
    reopened = AssignmentConversationRegistry(path)

    assert reopened.for_assignment("assignment-1", "revision-2") == first
    assert reopened.for_assignment("assignment-1", "revision-3") != first


def test_local_child_events_are_delivered_directly_to_each_parent(tmp_path: Path):
    first_parent = UUID("00000000-0000-0000-0000-000000000011")
    second_parent = UUID("00000000-0000-0000-0000-000000000012")
    store_path = tmp_path / "student-events.sqlite3"
    with LocalEventStore(store_path) as store:
        store.enqueue(
            LocalEvent(
                kind="agent_result",
                dedupe_key="agent_result:first",
                payload={"parent_conversation_id": str(first_parent)},
            )
        )
        store.enqueue(
            LocalEvent(
                kind="agent_result",
                dedupe_key="agent_result:second",
                payload={"parent_conversation_id": str(second_parent)},
            )
        )

    events = LocalStudentMailbox(store_path).poll()
    batches = StudentConversationSelector(
        AssignmentConversationRegistry(tmp_path / "students.json")
    )(events)

    assert [event.dedupe_key for event in events] == [
        "agent_result:first",
        "agent_result:second",
    ]
    assert [batch.conversation_id for batch in batches] == [
        first_parent,
        second_parent,
    ]
    assert "parent_conversation_id" not in events[0].to_prompt()

    mailbox = LocalStudentMailbox(store_path)
    mailbox.acknowledge((events[0].dedupe_key,))
    assert [event.dedupe_key for event in mailbox.poll()] == [
        "agent_result:second"
    ]


def test_assignment_feedback_and_monitor_wake_share_the_assignment_uuid(
    tmp_path: Path,
):
    registry = AssignmentConversationRegistry(tmp_path / "students.json")
    conversation_id = registry.for_assignment("assignment-17", "revision-2")
    events = (
        ControllerEvent(
            kind="student_assignment",
            dedupe_key="student_assignment:assignment-17:revision-2",
            payload={
                "assignment_id": "assignment-17",
                "revision_id": "revision-2",
            },
        ),
        ControllerEvent(
            kind="student_pr_feedback",
            dedupe_key="student_pr_feedback:issue_comment:17:101",
            payload={
                "assignment_id": "assignment-17",
                "revision_id": "revision-2",
            },
        ),
        ControllerEvent(
            kind="training_monitor",
            dedupe_key="training_monitor:run-1:finished",
            payload={"conversation_id": str(conversation_id)},
        ),
    )

    batches = StudentConversationSelector(registry)(events)

    assert len(batches) == 1
    assert batches[0].conversation_id == conversation_id
    assert batches[0].events == events


def test_controller_partitions_and_acknowledges_events_by_conversation(
    tmp_path: Path,
):
    first = UUID("00000000-0000-0000-0000-000000000081")
    second = UUID("00000000-0000-0000-0000-000000000082")
    first_event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="monitor:first",
        payload={"conversation_id": str(first), "summary": "first only"},
    )
    second_event = ControllerEvent(
        kind="agent_result",
        dedupe_key="child:second",
        payload={
            "parent_conversation_id": str(second),
            "summary": "second only",
        },
    )
    mailbox = Mailbox((first_event, second_event))
    turns = Turns()

    run_student_controller(tmp_path, mailbox, turns)

    assert [call[1] for call in turns.calls] == [first, second]
    assert [call[2] for call in turns.calls] == [
        frozenset({first_event.dedupe_key}),
        frozenset({second_event.dedupe_key}),
    ]
    assert mailbox.acknowledged == [
        (first_event.dedupe_key,),
        (second_event.dedupe_key,),
    ]


def test_failed_conversation_does_not_ack_or_starve_another(tmp_path: Path):
    first = UUID("00000000-0000-0000-0000-000000000083")
    second = UUID("00000000-0000-0000-0000-000000000084")
    first_event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="monitor:first",
        payload={"conversation_id": str(first)},
    )
    second_event = ControllerEvent(
        kind="training_monitor",
        dedupe_key="monitor:second",
        payload={"conversation_id": str(second)},
    )
    mailbox = Mailbox((first_event, second_event))
    turns = Turns((1, 0))

    run_student_controller(tmp_path, mailbox, turns)

    assert [call[1] for call in turns.calls] == [first, second]
    assert mailbox.acknowledged == [(second_event.dedupe_key,)]


def test_bounded_backlog_yields_to_another_ready_conversation(tmp_path: Path):
    first = UUID("00000000-0000-0000-0000-000000000085")
    second = UUID("00000000-0000-0000-0000-000000000086")
    backlog = tuple(
        ControllerEvent(
            kind="agent_result",
            dedupe_key=f"first:{index}",
            payload={"parent_conversation_id": str(first), "index": index},
        )
        for index in range(20)
    )
    other = ControllerEvent(
        kind="agent_result",
        dedupe_key="second:0",
        payload={"parent_conversation_id": str(second)},
    )
    turns = Turns()

    run_student_controller(tmp_path, Mailbox((*backlog, other)), turns)

    assert [call[1] for call in turns.calls] == [first, second, first]
    assert [len(call[2]) for call in turns.calls] == [16, 1, 4]


def test_student_human_issue_routes_to_a_quarantined_conversation_first(
    tmp_path: Path,
):
    registry = AssignmentConversationRegistry(tmp_path / "students.json")
    quarantined = registry.for_assignment("assignment-17", "revision-2")
    event = ControllerEvent(
        kind="human_issue",
        dedupe_key="human_issue:23:700",
        payload={"number": 23, "human_message_id": 700},
    )

    routed = StudentConversationSelector(
        registry,
        quarantined=lambda: (str(quarantined),),
    )((event,))
    thread = StudentConversationSelector(registry)((event,))

    assert [batch.conversation_id for batch in routed] == [quarantined]
    assert [batch.conversation_id for batch in thread] == [
        registry.for_assignment("human-issue-23", "thread")
    ]
