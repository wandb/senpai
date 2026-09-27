from copy import deepcopy
from types import SimpleNamespace

import pytest

from senpai_agent.inbox import PersistentInbox
from senpai_agent.models import parse_assignment_markers, render_assignment_marker
from senpai_agent.state import (
    AssignmentConversationRegistry,
    StudentConversationSelector,
    StudentIssueRouter,
)
from test_controller_github_feedback import (
    feedback,
    feedback_responses,
    student_mailbox,
)
from test_controller_github_issues import issue


@pytest.fixture
def student(monkeypatch, tmp_path):
    registry = AssignmentConversationRegistry(tmp_path / "students.json")
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    pulls = mailbox._pulls()
    human_issue = issue(labels=("human", "student:student-1"))
    comments = []
    monkeypatch.setattr(
        mailbox, "student_issue_router", StudentIssueRouter(registry, inbox)
    )
    monkeypatch.setattr(mailbox, "_pulls", lambda: pulls)
    monkeypatch.setattr(mailbox, "_issues", lambda: [human_issue])
    monkeypatch.setattr(mailbox, "_issue_comments", lambda _issue: comments)
    state = SimpleNamespace(
        registry=registry,
        inbox=inbox,
        mailbox=mailbox,
        pulls=pulls,
        issue=human_issue,
        comments=comments,
    )
    yield state
    state.inbox.close()


def quarantine_assignment(student, revision="revision-2"):
    conversation_id = student.registry.for_assignment("assignment-17", revision)
    student.inbox.enqueue(conversation_id, f"assignment:{revision}", "assignment")
    turn = student.inbox.next_turn(conversation_id, "controller prompt")
    assert turn is not None
    student.inbox.quarantine(turn.turn_id, "recovery budget exhausted")
    return turn


def poll_issue(student):
    return next(
        event for event in student.mailbox.poll() if event.kind == "human_issue"
    )


def test_human_issue_poll_reopens_the_original_assignment_only_once(student):
    quarantined = quarantine_assignment(student)

    event = poll_issue(student)

    batches = StudentConversationSelector(student.registry)((event,))
    assert str(batches[0].conversation_id) == quarantined.conversation_id
    assert event.payload["parent_conversation_id"] == quarantined.conversation_id
    reopened = student.inbox.turn(quarantined.turn_id)
    assert reopened.quarantine_reason is None
    assert reopened.event_keys == ("assignment:revision-2", event.dedupe_key)
    assert student.inbox.ready_conversation_ids() == (quarantined.conversation_id,)

    student.inbox.quarantine(quarantined.turn_id, "recovery budget exhausted")
    repeated = poll_issue(student)

    assert repeated == event
    assert student.inbox.turn(quarantined.turn_id).quarantine_reason is not None
    assert student.inbox.ready_conversation_ids() == ()

    student.issue["labels"].append({"name": "team"})
    events = student.mailbox.poll()

    assert not any(event.kind == "human_issue" for event in events)
    owner = student.inbox.event_conversation_id(event.dedupe_key)
    assert owner == quarantined.conversation_id
    assert student.inbox.turn(quarantined.turn_id).quarantine_reason is not None
    assert student.inbox.ready_conversation_ids() == ()

    student.issue["labels"].pop()
    student.issue["body"] = "The provider is fixed; resume this assignment."
    edited = poll_issue(student)

    assert edited.payload["human_message_id"] == event.payload["human_message_id"]
    assert edited.dedupe_key != event.dedupe_key
    assert edited.payload["parent_conversation_id"] == quarantined.conversation_id
    assert student.inbox.turn(quarantined.turn_id).quarantine_reason is None
    assert student.inbox.ready_conversation_ids() == (quarantined.conversation_id,)


def test_issue_binding_survives_restart_and_revision_change(student, monkeypatch):
    old = quarantine_assignment(student)
    first = poll_issue(student)
    student.inbox.quarantine(old.turn_id, "recovery budget exhausted")
    assignment = parse_assignment_markers(student.pulls[0]["body"])[0]
    student.pulls[0]["body"] = render_assignment_marker(
        assignment.model_copy(update={"revision_id": "revision-3"})
    )
    current = quarantine_assignment(student, "revision-3")
    inbox_path = student.inbox.path
    student.inbox.close()
    student.inbox = PersistentInbox(inbox_path)
    student.registry = AssignmentConversationRegistry(student.registry.path)
    monkeypatch.setattr(
        student.mailbox,
        "student_issue_router",
        StudentIssueRouter(student.registry, student.inbox),
    )

    events = student.mailbox.poll()

    assert not any(event.kind == "human_issue" for event in events)
    assert student.inbox.event_conversation_id(first.dedupe_key) == old.conversation_id
    assert student.inbox.turn(old.turn_id).quarantine_reason is not None
    assert student.inbox.turn(current.turn_id).quarantine_reason is not None
    assert student.inbox.ready_conversation_ids() == ()

    student.comments.append(feedback(701, "Resume the current revision."))
    new_message = poll_issue(student)

    assert new_message.dedupe_key != first.dedupe_key
    assert new_message.payload["parent_conversation_id"] == current.conversation_id
    assert student.inbox.turn(current.turn_id).quarantine_reason is None
    assert student.inbox.turn(old.turn_id).quarantine_reason is not None
    assert student.inbox.ready_conversation_ids() == (current.conversation_id,)


@pytest.mark.parametrize(
    ("candidate", "audience", "assignment_target"),
    [
        pytest.param("review", (), True, id="review-assignment"),
        pytest.param("held", (), True, id="held-wip-assignment"),
        pytest.param("absent", (), False, id="no-assignment"),
        pytest.param("duplicate", (), False, id="separate-wip-and-review-assignments"),
        pytest.param("malformed", (), False, id="malformed-assignment"),
        pytest.param("mixed", (), False, id="valid-and-malformed-assignments"),
        pytest.param("contradictory", (), False, id="wip-and-review-assignment"),
        pytest.param(
            "valid",
            ("student:student-1", "bug"),
            True,
            id="student-with-unrelated-label",
        ),
        pytest.param("valid", ("team",), False, id="team-broadcast"),
        pytest.param(
            "valid",
            ("student:student-1", "team"),
            False,
            id="student-and-team",
        ),
        pytest.param(
            "valid",
            ("student:student-1", "student:student-2"),
            False,
            id="multiple-students",
        ),
        pytest.param(
            "valid",
            ("student:student-1", "research"),
            False,
            id="student-and-advisor",
        ),
    ],
)
def test_issue_routing_requires_one_assignment_and_one_student_audience(
    student, candidate, audience, assignment_target
):
    quarantined = quarantine_assignment(student)
    if candidate == "review":
        student.pulls[0]["labels"][-1] = {"name": "status:review"}
    elif candidate == "held":
        student.pulls[0]["labels"].append({"name": "status:hold"})
    elif candidate == "contradictory":
        student.pulls[0]["labels"].append({"name": "status:review"})
    elif candidate == "absent":
        student.pulls.clear()
    elif candidate == "malformed":
        student.pulls[0]["body"] = "<!-- senpai-assignment:v1 not-json -->"
    elif candidate in {"duplicate", "mixed"}:
        other = deepcopy(student.pulls[0])
        other["number"] = 18
        other["head"]["ref"] = "student/other"
        if candidate == "duplicate":
            other["labels"][-1] = {"name": "status:review"}
        assignment = parse_assignment_markers(other["body"])[0]
        other["body"] = (
            render_assignment_marker(
                assignment.model_copy(
                    update={
                        "assignment_id": "assignment-18",
                        "head_ref": "student/other",
                    }
                )
            )
            if candidate == "duplicate"
            else "<!-- senpai-assignment:v1 not-json -->"
        )
        student.pulls.append(other)
    if audience:
        student.issue["labels"] = [{"name": name} for name in ("human", *audience)]

    event = poll_issue(student)

    expected = (
        quarantined.conversation_id
        if assignment_target
        else str(student.registry.for_assignment("human-issue-23", "thread"))
    )
    assert event.payload["parent_conversation_id"] == expected
    batches = StudentConversationSelector(student.registry)((event,))
    assert str(batches[0].conversation_id) == expected
    assert student.inbox.ready_conversation_ids() == (expected,)
    reopened = student.inbox.turn(quarantined.turn_id).quarantine_reason is None
    assert reopened == assignment_target
