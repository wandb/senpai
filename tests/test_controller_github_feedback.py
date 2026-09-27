from pathlib import Path
from uuid import UUID

import pytest
from pydantic import SecretStr

from senpai_agent.controller import Controller, TurnResult
from senpai_agent.github.mailbox import GitHubMailbox
from senpai_agent.inbox import PersistentInbox
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.models import (
    AssignmentCommentRecord,
    AssignmentFeedbackRecord,
    AssignmentRecord,
    render_assignment_feedback_marker,
    render_assignment_comment_marker,
    render_assignment_marker,
)
from senpai_agent.state import (
    AssignmentConversationRegistry,
    StudentConversationSelector,
)


def feedback(
    feedback_id,
    body,
    *,
    author="morganmcg1",
    association="OWNER",
    user_type="User",
    created_at="2026-07-29T18:01:00Z",
    updated_at=None,
    **extra,
):
    return {
        "id": feedback_id,
        "html_url": f"https://github.test/comment/{feedback_id}",
        "body": body,
        "created_at": created_at,
        "updated_at": updated_at or created_at,
        "author_association": association,
        "user": {"login": author, "type": user_type},
        **extra,
    }


def feedback_responses(*, issue_comments=(), reviews=(), inline_comments=()):
    return {
        (
            "https://api.github.test/repos/acme/widgets/issues/17/comments"
            "?per_page=100"
        ): list(issue_comments),
        (
            "https://api.github.test/repos/acme/widgets/pulls/17/reviews"
            "?per_page=100"
        ): list(reviews),
        (
            "https://api.github.test/repos/acme/widgets/pulls/17/comments"
            "?per_page=100"
        ): list(inline_comments),
    }


def student_mailbox(
    monkeypatch,
    responses,
    *,
    status="status:wip",
    revision_id="revision-2",
    feedback_path=None,
    feedback_batch_events=8,
    feedback_batch_bytes=32_000,
    trusted_actor="morganmcg1",
):
    assignment = AssignmentRecord(
        repo="acme/widgets",
        assignment_id="assignment-17",
        revision_id=revision_id,
        student="student-1",
        base_ref="research",
        base_sha="b" * 40,
        head_ref="student/candidate",
        head_sha="a" * 40,
    )
    assigned_pull = {
        "number": 17,
        "title": "Try bounded change",
        "html_url": "https://github.test/acme/widgets/pull/17",
        "updated_at": "2026-07-29T18:00:00Z",
        "body": render_assignment_marker(assignment),
        "user": {"login": "senpai-bot"},
        "head": {
            "ref": "student/candidate",
            "sha": "a" * 40,
            "repo": {"full_name": "acme/widgets"},
        },
        "base": {"ref": "research", "sha": "b" * 40},
        "labels": [
            {"name": "research"},
            {"name": "student:student-1"},
            {"name": status},
        ],
        "url": "https://api.github.test/repos/acme/widgets/pulls/17",
        "comments_url": (
            "https://api.github.test/repos/acme/widgets/issues/17/comments"
        ),
        "review_comments_url": (
            "https://api.github.test/repos/acme/widgets/pulls/17/comments"
        ),
    }
    mailbox = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="student",
        advisor_branch="research",
        student_name="student-1",
        api_url="https://api.github.test",
        trusted_actor=trusted_actor,
        feedback_path=feedback_path,
        feedback_batch_events=feedback_batch_events,
        feedback_batch_bytes=feedback_batch_bytes,
    )
    monkeypatch.setattr(mailbox, "_pulls", lambda: [assigned_pull])
    monkeypatch.setattr(mailbox, "_issues", list)
    monkeypatch.setattr(mailbox, "_has_write_permission", lambda _login: True)
    monkeypatch.setattr(mailbox._github, "objects", lambda url: responses[url])
    return mailbox


@pytest.mark.parametrize(
    ("head_repo", "has_write_permission"),
    [("outsider/widgets", True), ("acme/widgets", False)],
)
def test_student_ignores_assignments_from_unauthorized_pulls(
    monkeypatch,
    head_repo: str,
    has_write_permission: bool,
):
    mailbox = student_mailbox(monkeypatch, {})
    pull = mailbox._pulls()[0]
    pull["head"]["repo"]["full_name"] = head_repo
    monkeypatch.setattr(
        mailbox, "_has_write_permission", lambda _login: has_write_permission
    )

    assert mailbox.poll() == ()


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
        conversation_id=UUID("00000000-0000-0000-0000-000000000017"),
        full_prompt="programme",
        conversation_for_events=StudentConversationSelector(
            AssignmentConversationRegistry(tmp_path / "students.json")
        ),
        sleep=lambda _seconds: None,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)


def test_student_assignment_carries_the_marker_revision_identity(monkeypatch):
    mailbox = student_mailbox(monkeypatch, feedback_responses())

    event = mailbox.poll()[0]

    assert event.kind == "student_assignment"
    assert event.payload["assignment_id"] == "assignment-17"
    assert event.payload["revision_id"] == "revision-2"
    assert event.payload["base_ref"] == "research"
    assert event.payload["base_sha"] == "b" * 40
    assert event.payload["head_ref"] == "student/candidate"
    assert event.payload["head_sha"] == "a" * 40
    assert event.dedupe_key.startswith("student_assignment:v2:17:")
    assert len(event.dedupe_key.rsplit(":", 1)[1]) == 64


def test_repolling_one_assignment_after_mutable_pr_metadata_changes_is_idempotent(
    monkeypatch,
    tmp_path,
):
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    first = mailbox.poll()[0]
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")

    assert inbox.enqueue(UUID(int=17), first.dedupe_key, first.to_prompt()) is True

    pull = mailbox._pulls()[0]
    pull["title"] = "Clarify the bounded change"
    pull["updated_at"] = "2026-07-29T18:05:00Z"
    pull["labels"].append({"name": "operator-note"})
    repeated = mailbox.poll()[0]

    assert repeated.dedupe_key == first.dedupe_key
    assert repeated.to_prompt() == first.to_prompt()
    assert (
        inbox.enqueue(UUID(int=17), repeated.dedupe_key, repeated.to_prompt())
        is False
    )


@pytest.mark.parametrize("blocker", ("blocked", "hold", "needs-rebase"))
def test_actionable_assignment_blockers_create_a_new_event_version(
    monkeypatch,
    blocker,
):
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    first = mailbox.poll()[0]

    mailbox._pulls()[0]["labels"].append({"name": f"status:{blocker}"})
    changed = mailbox.poll()[0]

    assert changed.dedupe_key != first.dedupe_key
    assert changed.payload["blockers"] == [blocker]


def test_v2_assignment_queues_behind_an_unresolved_v1_delivery(
    monkeypatch,
    tmp_path,
):
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    current = mailbox.poll()[0]
    key_parts = current.dedupe_key.split(":")
    legacy_key = "student_assignment:" + ":".join(key_parts[3:-1])
    legacy_payload = {
        **{key: value for key, value in current.payload.items() if key != "blockers"},
        "title": "Try bounded change",
        "labels": ["research", "status:wip", "student:student-1"],
        "updated_at": "2026-07-29T18:00:00Z",
    }
    legacy = ControllerEvent(
        kind="student_assignment",
        dedupe_key=legacy_key,
        payload=legacy_payload,
    )
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    inbox.enqueue(UUID(int=17), legacy.dedupe_key, legacy.to_prompt())
    active = inbox.next_turn(UUID(int=17), "legacy prompt")
    assert active is not None

    assert inbox.enqueue(UUID(int=17), current.dedupe_key, current.to_prompt()) is True
    resumed = inbox.next_turn(UUID(int=17), "new prompt")

    assert resumed is not None
    assert resumed.turn_id == active.turn_id
    assert resumed.events[0].event_key == legacy.dedupe_key
    assert inbox.pending_count(UUID(int=17)) == 1


@pytest.mark.parametrize(
    "assignment",
    [
        AssignmentRecord(
            repo="other/widgets",
            assignment_id="assignment-17",
            revision_id="revision-2",
            student="student-1",
            base_ref="research",
            base_sha="b" * 40,
            head_ref="student/candidate",
            head_sha="a" * 40,
        ),
        AssignmentRecord(
            repo="acme/widgets",
            assignment_id="assignment-17",
            revision_id="revision-2",
            student="student-1",
            base_ref="research",
            base_sha="not-a-sha",
            head_ref="student/candidate",
            head_sha="a" * 40,
        ),
    ],
    ids=["foreign-repo", "invalid-base-sha"],
)
def test_student_rejects_assignment_markers_that_cannot_drive_safe_fetches(
    monkeypatch,
    assignment,
):
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    mailbox._pulls()[0]["body"] = render_assignment_marker(assignment)

    event = mailbox.poll()[0]

    assert event.kind == "malformed_assignment"


def test_trusted_feedback_from_each_github_surface_is_ordered_and_routable(
    monkeypatch,
):
    responses = feedback_responses(
        issue_comments=[feedback(101, "Pause after the current arm.")],
        reviews=[
            feedback(
                201,
                "Please preserve the control.",
                author="ada",
                association="MEMBER",
                submitted_at="2026-07-29T18:02:00Z",
                state="CHANGES_REQUESTED",
            ),
            feedback(
                202,
                "Unsubmitted draft.",
                submitted_at=None,
                state="PENDING",
            ),
        ],
        inline_comments=[
            feedback(
                301,
                "This branch needs the current default.",
                author="grace",
                association="COLLABORATOR",
                created_at="2026-07-29T18:03:00Z",
                pull_request_review_id=201,
                path="train.py",
                line=42,
            ),
            feedback(
                302,
                "Private draft inline comment.",
                created_at="2026-07-29T18:03:30Z",
                pull_request_review_id=202,
                path="train.py",
                line=43,
            ),
        ],
    )
    mailbox = student_mailbox(
        monkeypatch,
        responses,
        trusted_actor="MorganMcG1",
    )

    events = [
        event
        for event in mailbox.poll()
        if event.kind == "student_pr_feedback"
    ]

    assert [":".join(event.dedupe_key.split(":")[:5]) for event in events] == [
        "student_pr_feedback:v2:issue_comment:17:101",
        "student_pr_feedback:v2:review:17:201",
        "student_pr_feedback:v2:inline_comment:17:301",
    ]
    assert {
        (event.payload["assignment_id"], event.payload["revision_id"])
        for event in events
    } == {("assignment-17", "revision-2")}
    assert {
        (
            event.payload["base_sha"],
            event.payload["head_sha"],
        )
        for event in events
    } == {("b" * 40, "a" * 40)}
    assert events[1].payload["state"] == "CHANGES_REQUESTED"
    assert events[2].payload["path"] == "train.py"
    assert events[2].payload["line"] == 42


def test_editing_unacknowledged_feedback_creates_a_new_event_version(monkeypatch):
    comment = feedback(101, "Try the narrow change.")
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(issue_comments=[comment]),
    )
    first = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )

    comment["body"] = "Try the narrow change with a paired baseline."
    comment["updated_at"] = "2026-07-29T18:02:00Z"
    edited = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )

    assert edited.dedupe_key != first.dedupe_key
    assert edited.payload["feedback_id"] == first.payload["feedback_id"]

    comment["body"] = "Try the narrow change."
    comment["updated_at"] = "2026-07-29T18:03:00Z"
    reverted = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )
    assert reverted.dedupe_key != first.dedupe_key
    assert reverted.dedupe_key != edited.dedupe_key
    assert reverted.to_prompt() == first.to_prompt()


def test_reverting_feedback_after_an_acknowledged_version_emits_a_correction(
    monkeypatch,
    tmp_path,
):
    comment = feedback(101, "Use direction A.")
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(issue_comments=[comment]),
        feedback_path=tmp_path / "feedback.json",
    )
    first = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )
    run_student_controller(tmp_path, mailbox, Turns())

    comment["body"] = "Use direction B."
    comment["updated_at"] = "2026-07-29T18:02:00Z"
    edited = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )

    comment["body"] = "Use direction A."
    comment["updated_at"] = "2026-07-29T18:03:00Z"
    reverted = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )

    assert reverted.dedupe_key not in {first.dedupe_key, edited.dedupe_key}
    assert reverted.payload["message"] == "Use direction A."


def test_review_ready_pull_routes_feedback_to_its_assignment_conversation(monkeypatch):
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(
            issue_comments=[feedback(101, "Please revisit this result.")]
        ),
        status="status:review",
    )

    events = mailbox.poll()

    assert [event.kind for event in events] == ["student_pr_feedback"]
    assert events[0].payload["assignment_id"] == "assignment-17"
    assert events[0].payload["revision_id"] == "revision-2"


def test_student_does_not_launch_a_doubly_labeled_assignment(monkeypatch):
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    pull = mailbox._pulls()[0]
    pull["labels"].append({"name": "student:student-2"})

    events = mailbox.poll()

    assert [event.kind for event in events] == ["malformed_assignment"]
    assert "exactly one student label" in events[0].payload["error"]


def test_student_does_not_launch_an_assignment_marked_for_another_student(
    monkeypatch,
):
    mailbox = student_mailbox(monkeypatch, feedback_responses())
    pull = mailbox._pulls()[0]
    pull["body"] = render_assignment_marker(
        AssignmentRecord(
            repo="acme/widgets",
            assignment_id="assignment-17",
            revision_id="revision-2",
            student="student-2",
            base_ref="research",
            base_sha="b" * 40,
            head_ref="student/candidate",
            head_sha="a" * 40,
        )
    )

    events = mailbox.poll()

    assert [event.kind for event in events] == ["malformed_assignment"]
    assert "marker student does not match" in events[0].payload["error"]


def test_inline_comment_without_a_review_id_does_not_poison_feedback_poll(
    monkeypatch,
):
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(
            inline_comments=[
                feedback(
                    301,
                    "Detached inline comment.",
                    pull_request_review_id=None,
                    path="train.py",
                    line=42,
                )
            ]
        ),
    )

    events = mailbox.poll()

    assert [event.kind for event in events] == ["student_assignment"]


def test_untrusted_people_bots_and_automation_comments_are_not_feedback(
    monkeypatch,
):
    responses = feedback_responses(
        issue_comments=[
            feedback(101, "Trusted control."),
            feedback(102, "External advice.", author="mallory", association="NONE"),
            feedback(103, "Automation.", author="ci-bot", user_type="Bot"),
            feedback(104, "<!-- senpai-revision:v1 {} -->\n\nAutomated revision."),
        ]
    )
    mailbox = student_mailbox(monkeypatch, responses)

    events = [
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    ]

    assert [event.payload["feedback_id"] for event in events] == [101]


def test_student_assignment_comment_is_not_replayed_to_its_student(monkeypatch):
    marker = render_assignment_comment_marker(
        AssignmentCommentRecord(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-2",
            student="student-1",
            comment_id="paired-run-started",
        )
    )
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(
            issue_comments=[
                feedback(
                    105,
                    f"{marker}\n\nSTUDENT: The paired run has started.",
                    author="morganmcg1",
                    association="OWNER",
                    user_type="User",
                )
            ]
        ),
    )

    assert not any(event.kind == "student_pr_feedback" for event in mailbox.poll())


def test_typed_actor_feedback_keeps_its_marked_revision_and_hides_other_protocol(
    monkeypatch,
):
    marker = render_assignment_feedback_marker(
        AssignmentFeedbackRecord(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            feedback_id="revision-one-guidance",
        )
    )
    wrong_repo = render_assignment_feedback_marker(
        AssignmentFeedbackRecord(
            repo="other/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            feedback_id="wrong-repository",
        )
    )
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(
            issue_comments=[
                feedback(
                    120,
                    f"{marker}\n\nStop revision one.",
                    association="NONE",
                    user_type="Bot",
                ),
                feedback(121, "<!-- senpai-result:v1 {} -->\n\nAutomation."),
                feedback(122, f"{wrong_repo}\n\nWrong repository."),
            ]
        ),
        revision_id="revision-2",
        trusted_actor="MorganMcG1",
    )

    events = [
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    ]

    assert len(events) == 1
    assert events[0].payload["feedback_id"] == 120
    assert events[0].payload["revision_id"] == "revision-1"
    assert events[0].payload["message"] == "Stop revision one."


def test_feedback_stays_pending_and_bound_until_a_student_turn_succeeds(
    monkeypatch,
    tmp_path: Path,
):
    ledger = tmp_path / "github-feedback.json"
    responses = feedback_responses(
        issue_comments=[feedback(131, "Durable feedback.")]
    )
    failed = student_mailbox(
        monkeypatch,
        responses,
        feedback_path=ledger,
    )
    original = next(
        event for event in failed.poll() if event.kind == "student_pr_feedback"
    )
    run_student_controller(tmp_path, failed, Turns((1,)))

    revised = student_mailbox(
        monkeypatch,
        responses,
        revision_id="revision-3",
        feedback_path=ledger,
    )
    revised._pulls()[0]["head"]["sha"] = "c" * 40
    pending = revised.poll()
    assert [event.kind for event in pending] == ["student_pr_feedback"]
    assert pending[0].payload["revision_id"] == "revision-2"
    assert pending[0].dedupe_key == original.dedupe_key
    assert pending[0].to_prompt() == original.to_prompt()

    turns = Turns()
    run_student_controller(tmp_path, revised, turns)
    assert [
        next(iter(event_keys)).split(":", 1)[0]
        for _, _, event_keys, _ in turns.calls
    ] == ["student_pr_feedback", "student_assignment"]

    restarted = student_mailbox(
        monkeypatch,
        responses,
        revision_id="revision-4",
        feedback_path=ledger,
    )
    assert [event.kind for event in restarted.poll()] == ["student_assignment"]


def test_controller_drains_feedback_batches_oldest_first_after_success(
    monkeypatch,
    tmp_path: Path,
):
    comments = [
        feedback(
            140 + index,
            f"feedback-{index}",
            created_at=f"2026-07-29T18:01:{index:02d}Z",
        )
        for index in range(5)
    ]
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(issue_comments=comments),
        feedback_path=tmp_path / "feedback.json",
        feedback_batch_events=2,
        feedback_batch_bytes=100_000,
    )
    turns = Turns()

    run_student_controller(tmp_path, mailbox, turns)

    feedback_batches = [
        sorted(
            int(key.split(":")[4])
            for key in event_keys
            if key.startswith("student_pr_feedback:")
        )
        for _, _, event_keys, _ in turns.calls
    ]
    assert feedback_batches == [[140, 141], [142, 143], [144]]
    assert not any(event.kind == "student_pr_feedback" for event in mailbox.poll())


def test_feedback_batch_limit_counts_the_rendered_prompt_bytes(monkeypatch):
    comments = [
        feedback(
            150 + index,
            "x" * 1_000,
            created_at=f"2026-07-29T18:02:{index:02d}Z",
        )
        for index in range(2)
    ]
    responses = feedback_responses(issue_comments=comments)
    probe = student_mailbox(monkeypatch, responses)
    probe_events = [
        event for event in probe.poll() if event.kind == "student_pr_feedback"
    ]
    byte_limit = len(probe_events[0].to_prompt().encode())
    bounded = student_mailbox(
        monkeypatch,
        responses,
        feedback_batch_bytes=byte_limit,
    )

    batch = [
        event for event in bounded.poll() if event.kind == "student_pr_feedback"
    ]

    assert len(batch) == 1
    assert len(batch[0].to_prompt().encode()) <= byte_limit


def test_long_feedback_keeps_actionable_head_tail_and_source_link(monkeypatch):
    message = "ACTION: stop after this run.\n" + "x" * 5_000 + "\nTAIL: thanks"
    mailbox = student_mailbox(
        monkeypatch,
        feedback_responses(issue_comments=[feedback(160, message)]),
    )

    event = next(
        event for event in mailbox.poll() if event.kind == "student_pr_feedback"
    )
    excerpt = str(event.payload["message"])

    assert excerpt.startswith("ACTION: stop after this run.")
    assert excerpt.endswith("TAIL: thanks")
    assert len(excerpt.encode()) <= 4_000
    assert event.payload["message_truncated"] is True
    assert event.payload["full_message_instruction"] == (
        "Open feedback_url to read the omitted text."
    )
