"""Quarantine alerts cross the real student workflow and advisor mailbox."""

from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import SecretStr

from github_workflow_support import (
    API_URL,
    ASSIGNMENT_ID,
    REPO,
    FakeGitHub,
    assignment_record,
    pull_request,
    workflow,
)
from senpai_agent.github.http import GitHubReader
from senpai_agent.github.mailbox import GitHubMailbox
from senpai_agent.github.quarantine import StudentQuarantineReporter
from senpai_agent.github.workflow import GitHubTransportError, HttpResponse
from senpai_agent.inbox import STEER_PRIORITY, PersistentInbox
from senpai_agent.models import (
    parse_assignment_comment_markers,
    render_assignment_marker,
)
from senpai_agent.state import AssignmentConversationRegistry


class PollingGitHub(FakeGitHub):
    """Add mailbox REST endpoints to the existing workflow transport fixture."""

    fail_next_post = False
    before_pull_read = None

    def _pull_payload(self):
        payload = super()._pull_payload()
        payload.update(
            user={"login": self.actor_login},
            updated_at="2099-01-01T00:00:00Z",
            comments_url=f"{API_URL}/repos/{REPO}/issues/7/comments",
        )
        payload["head"]["repo"] = {"full_name": REPO}
        return payload

    def request(self, method, url, *, headers, json_body=None):
        parsed = urlsplit(url)
        if method == "GET" and parsed.path.endswith("/permission"):
            return HttpResponse(200, {"permission": "write"})
        if method == "GET" and parsed.path == f"/repos/{REPO}/pulls":
            if "head" not in parse_qs(parsed.query):
                return HttpResponse(200, [self._pull_payload()])
        if method == "GET" and parsed.path.endswith("/pulls/7"):
            if change := self.before_pull_read:
                self.before_pull_read = None
                change(self.pr)
        if method == "POST" and parsed.path.endswith("/issues/7/comments"):
            if self.fail_next_post:
                self.fail_next_post = False
                raise GitHubTransportError(method, url)
        response = super().request(method, url, headers=headers, json_body=json_body)
        for comment in self.comments:
            timestamp = f"2026-10-05T10:00:{int(comment['id']):02d}Z"
            comment.setdefault("created_at", timestamp)
            comment.setdefault("updated_at", timestamp)
        return response


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    fake = PollingGitHub(
        pull_request(labels={"schmidhuber", "student:student-one", "status:wip"})
    )

    def read(_reader, path, *, json_body=None):
        url = path if path.startswith("https://") else f"{API_URL}{path}"
        response = fake.request("GET", url, headers={}, json_body=json_body)
        return response.json_body, None

    monkeypatch.setattr(GitHubReader, "_request", read)
    inbox = PersistentInbox(tmp_path / "inbox.sqlite3")
    registry = AssignmentConversationRegistry(tmp_path / "conversations.json")
    yield SimpleNamespace(path=tmp_path, fake=fake, inbox=inbox, registry=registry)
    inbox.close()


def quarantine(runtime, assignment_id=ASSIGNMENT_ID, revision_id="revision-1"):
    conversation = runtime.registry.for_assignment(assignment_id, revision_id)
    runtime.inbox.enqueue(
        conversation, f"assignment:{assignment_id}:{revision_id}", "assignment"
    )
    turn = runtime.inbox.next_turn(conversation, "Research the assigned experiment.")
    assert turn is not None
    runtime.inbox.quarantine(turn.turn_id, "context recovery exhausted")
    return conversation, turn


def mailbox(runtime, role, reporter=None):
    return GitHubMailbox(
        repo=REPO,
        token=SecretStr("fixture-token"),
        role=role,
        advisor_branch="schmidhuber",
        students=("student-one",),
        student_name="student-one" if role == "student" else None,
        human_issues_enabled=False,
        api_url=API_URL,
        trusted_actor=runtime.fake.actor_login,
        quarantine_reporter=reporter,
    )


def reporter(runtime, inbox=None):
    return StudentQuarantineReporter(
        inbox or runtime.inbox,
        AssignmentConversationRegistry(runtime.registry.path),
        workflow(runtime.fake, role="student"),
    )


def alerts(advisor):
    return [
        event for event in advisor.poll()
        if event.kind == "student_assignment_comment"
    ]


@pytest.mark.parametrize("status", ["status:wip", "status:review"])
def test_existing_quarantine_reaches_advisor_once_across_restarts(runtime, status):
    runtime.fake.pr["labels"] = {"schmidhuber", "student:student-one", status}
    conversation, turn = quarantine(runtime)
    original_pr = deepcopy(runtime.fake.pr)

    with PersistentInbox(runtime.path / "inbox.sqlite3") as restored:
        student = mailbox(runtime, "student", reporter(runtime, restored))
        student.poll()
        student.poll()
    assert len(runtime.fake.comments) == 1
    advisor = mailbox(runtime, "advisor")
    (first,) = alerts(advisor)
    assert first.payload["assignment_id"] == ASSIGNMENT_ID
    assert first.payload["revision_id"] == "revision-1"
    for evidence in (str(conversation), turn.turn_id, "context recovery exhausted"):
        assert evidence in first.payload["message"]
    assert "Advisor intervention required" in first.payload["message"]

    with PersistentInbox(runtime.path / "inbox.sqlite3") as restored:
        mailbox(runtime, "student", reporter(runtime, restored)).poll()
    (repeated,) = alerts(mailbox(runtime, "advisor"))
    assert repeated.dedupe_key == first.dedupe_key
    assert repeated.payload == first.payload
    assert len(runtime.fake.comments) == 1
    assert runtime.fake.pr == original_pr
    assert runtime.inbox.turn(turn.turn_id).quarantine_reason is not None


def test_failed_alert_retries_without_blocking_student_poll(runtime, capsys):
    quarantine(runtime)
    runtime.fake.fail_next_post = True
    student = mailbox(runtime, "student", reporter(runtime))

    assert [event.kind for event in student.poll()] == ["student_assignment"]
    assert runtime.fake.comments == []
    assert "SENPAI_QUARANTINE_REPORT_ERROR pr=7" in capsys.readouterr().err

    assert [event.kind for event in student.poll()] == ["student_assignment"]
    assert len(alerts(mailbox(runtime, "advisor"))) == 1
    assert len(runtime.fake.comments) == 1


@pytest.mark.parametrize(
    "assignment_id,revision_id",
    [
        (ASSIGNMENT_ID, "obsolete-revision"),
        ("foreign-assignment", "revision-1"),
        ("human-issue-23", "thread"),
        ("student-control", "current"),
    ],
)
def test_only_current_assignment_quarantine_is_reported(
    runtime, assignment_id, revision_id
):
    quarantine(runtime, assignment_id, revision_id)
    student = mailbox(runtime, "student", reporter(runtime))

    assert [event.kind for event in student.poll()] == ["student_assignment"]
    assert runtime.fake.comments == []

    quarantine(runtime)
    student.poll()
    assert len(runtime.fake.comments) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"head_sha": "c" * 40},
        {"body": render_assignment_marker(assignment_record(revision_id="revision-2"))},
        {"labels": {"student:student-two", "status:wip"}},
    ],
    ids=["head", "revision", "student-routing"],
)
def test_alert_revalidates_a_changed_assignment_before_post(runtime, capsys, change):
    quarantine(runtime)
    runtime.fake.before_pull_read = lambda pr: pr.update(change)

    events = mailbox(runtime, "student", reporter(runtime)).poll()

    assert [event.kind for event in events] == ["student_assignment"]
    assert runtime.fake.comments == []
    assert "SENPAI_QUARANTINE_REPORT_ERROR pr=7" in capsys.readouterr().err


def test_human_reopening_then_quarantine_reports_a_new_episode(runtime):
    conversation, turn = quarantine(runtime)
    student = mailbox(runtime, "student", reporter(runtime))
    student.poll()
    runtime.inbox.steer(
        conversation, "human:retry", "Retry after the operator repair.",
        priority=STEER_PRIORITY, once=True,
    )
    assert runtime.inbox.turn(turn.turn_id).quarantine_reason is None
    student.poll()
    assert len(runtime.fake.comments) == 1

    runtime.inbox.quarantine(turn.turn_id, "context recovery exhausted")
    student.poll()
    records = [
        parse_assignment_comment_markers(comment["body"])[0]
        for comment in runtime.fake.comments
    ]
    assert len(records) == 2
    assert records[0].comment_id != records[1].comment_id
    assert all(record.revision_id == "revision-1" for record in records)
    assert all(turn.turn_id in comment["body"] for comment in runtime.fake.comments)
    assert len(alerts(mailbox(runtime, "advisor"))) == 2

    mailbox(runtime, "student", reporter(runtime)).poll()
    assert len(runtime.fake.comments) == 2
