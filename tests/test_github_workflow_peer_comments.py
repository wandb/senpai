import json
from copy import deepcopy
from urllib.parse import urlsplit

import pytest
from github_workflow_support import (
    ASSIGNMENT_ID,
    HEAD_SHA,
    FakeGitHub,
    assignment_record,
    pull_request,
    workflow,
)

from senpai_agent.github.workflow import (
    GitHubTransportError,
    PullHeadMismatchError,
    ReconciliationError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker


class PeerGitHub:
    """Route two PR resources through the existing stateful GitHub transport."""

    actor_login = "senpai-bot"

    def __init__(self):
        self.sender = FakeGitHub(
            pull_request(labels={"student:student-one", "status:wip"}, draft=True)
        )
        recipient = pull_request(
            labels={"student:student-two", "status:wip"},
            draft=True,
            head_ref="student-two/handoff",
            body=render_assignment_marker(
                assignment_record(
                    assignment_id="assignment-8",
                    student="student-two",
                    head_ref="student-two/handoff",
                )
            ),
        )
        recipient.update(number=8, html_url="https://github.com/acme/widgets/pull/8")
        self.recipient = FakeGitHub(recipient)
        self.requests = []
        self.after_post = None

    @property
    def mutations(self):
        return [item for item in self.requests if item[0] != "GET"]

    def request(self, method, url, *, headers, json_body=None):
        path = urlsplit(url).path
        self.requests.append((method, path, json_body))
        target = "/pulls/8" in path or "/issues/8/" in path
        fake = self.recipient if target else self.sender
        routed = url.replace("/pulls/8", "/pulls/7").replace(
            "/issues/8/", "/issues/7/"
        )
        response = fake.request(method, routed, headers=headers, json_body=json_body)
        if target:
            for comment in fake.comments:
                comment["html_url"] = str(comment["html_url"]).replace(
                    "/pull/7#", "/pull/8#"
                )
        if method == "POST" and self.after_post is not None:
            self.after_post(method, url)
        return response


def post_peer(client, **overrides):
    fields = {
        "assignment_id": ASSIGNMENT_ID,
        "revision_id": "revision-1",
        "expected_head_sha": HEAD_SHA,
        "student": "student-one",
        "advisor_branch": "schmidhuber",
        "target_pr_number": 8,
        "comment_id": "handoff-question",
        "comment": "Does the handoff preserve optimizer state?",
    }
    fields.update(overrides)
    return client.post_peer_comment(7, **fields)


def test_peer_message_identifies_sender_and_working_pr_without_changing_assignments():
    fake = PeerGitHub()
    original = deepcopy((fake.sender.pr, fake.recipient.pr))
    client = workflow(fake, role="student")

    first = post_peer(client)
    replay = post_peer(client)

    assert first.changed is True
    assert replay.changed is False
    assert first.state == "peer_comment_posted"
    assert first.resource_url == "https://github.com/acme/widgets/pull/8#issuecomment-1"
    body = fake.recipient.comments[0]["body"]
    marker, visible = body.split("\n\n", 1)
    assert json.loads(marker[marker.index("{") : marker.rindex("}") + 1]) == {
        "schema_version": 1,
        "repo": "acme/widgets",
        "pr_number": 8,
        "assignment_id": "assignment-8",
        "revision_id": "revision-1",
        "student": "student-one",
        "source_pr_number": 7,
        "source_assignment_id": "assignment-7",
        "source_revision_id": "revision-1",
        "comment_id": "handoff-question",
    }
    assert visible == (
        "**STUDENT: student-one**\n\n"
        "Does the handoff preserve optimizer state?\n\n"
        "Working PR: [#7](https://github.com/acme/widgets/pull/7)"
    )
    assert fake.sender.comments == []

    reply = client.post_peer_comment(
        8,
        assignment_id="assignment-8",
        revision_id="revision-1",
        expected_head_sha=HEAD_SHA,
        student="student-two",
        advisor_branch="schmidhuber",
        target_pr_number=7,
        comment_id="handoff-answer",
        comment="The producer saves optimizer state.",
    )

    assert reply.changed is True
    assert fake.sender.comments[0]["body"].split("\n\n", 1)[1] == (
        "**STUDENT: student-two**\n\n"
        "The producer saves optimizer state.\n\n"
        "Working PR: [#8](https://github.com/acme/widgets/pull/8)"
    )
    assert (fake.sender.pr, fake.recipient.pr) == original
    assert [(method, path) for method, path, _ in fake.mutations] == [
        ("POST", "/repos/acme/widgets/issues/8/comments"),
        ("POST", "/repos/acme/widgets/issues/7/comments"),
    ]


def test_peer_comment_ids_cannot_rewrite_messages_and_new_ids_append():
    fake = PeerGitHub()
    client = workflow(fake, role="student")
    post_peer(client)

    fake.recipient.pr["body"] = render_assignment_marker(
        assignment_record(
            assignment_id="assignment-8",
            revision_id="revision-2",
            student="student-two",
            head_ref="student-two/handoff",
        )
    )
    replay = post_peer(client)

    assert replay.changed is False
    assert len(fake.recipient.comments) == 1

    with pytest.raises(WorkflowPreconditionError, match="new comment_id"):
        post_peer(client, comment="A different request.")

    assert len(fake.mutations) == 1
    post_peer(client, comment_id="handoff-follow-up", comment="I can test the interface.")
    assert len(fake.recipient.comments) == 2
    marker = fake.recipient.comments[1]["body"].split("\n\n", 1)[0]
    record = json.loads(marker[marker.index("{") : marker.rindex("}") + 1])
    assert record["revision_id"] == "revision-2"
    assert all(method == "POST" for method, _, _ in fake.mutations)


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"assignment_id": "other-assignment"}, WorkflowPreconditionError),
        ({"revision_id": "revision-2"}, StaleAssignmentRevisionError),
        ({"expected_head_sha": "c" * 40}, PullHeadMismatchError),
        ({"student": "student-two"}, PermissionError),
        ({"advisor_branch": "other-program"}, WorkflowPreconditionError),
    ],
    ids=("assignment", "revision", "head", "student", "other-program"),
)
def test_peer_message_requires_the_senders_current_assignment(fields, error):
    fake = PeerGitHub()

    with pytest.raises(error):
        post_peer(workflow(fake, role="student"), **fields)

    assert fake.mutations == []


@pytest.mark.parametrize(
    "invalid_target",
    [
        "self",
        "same-student",
        "other-base",
        "other-repo",
        "closed",
        "unassigned",
        "ambiguous",
    ],
)
def test_peer_message_rejects_targets_outside_active_peer_assignments(invalid_target):
    fake = PeerGitHub()
    target_pr = 8
    if invalid_target == "self":
        target_pr = 7
    elif invalid_target in {"same-student", "other-base", "other-repo"}:
        changes = {
            "same-student": {"student": "student-one", "head_ref": "student-one/other"},
            "other-base": {"base_ref": "other-launch"},
            "other-repo": {"repo": "other/widgets"},
        }[invalid_target]
        record = assignment_record(
            assignment_id="assignment-8",
            student="student-two",
            head_ref="student-two/handoff",
        ).model_copy(update=changes)
        fake.recipient.pr.update(
            body=render_assignment_marker(record),
            base_ref=record.base_ref,
            head_ref=record.head_ref,
            labels={f"student:{record.student}", "status:wip"},
        )
    elif invalid_target == "closed":
        fake.recipient.pr["state"] = "closed"
    elif invalid_target == "unassigned":
        fake.recipient.pr["body"] = "An ordinary unrelated pull request."
    else:
        fake.recipient.pr["labels"].add("status:review")

    with pytest.raises(WorkflowPreconditionError):
        post_peer(workflow(fake, role="student"), target_pr_number=target_pr)

    assert fake.mutations == []


def test_advisor_cannot_send_a_student_peer_message():
    fake = PeerGitHub()

    with pytest.raises(PermissionError, match="student workflow"):
        post_peer(workflow(fake, role="advisor"))

    assert fake.requests == []


def test_peer_comment_recovers_a_lost_post_response_without_duplicating_delivery():
    fake = PeerGitHub()

    def lose_response(method, url):
        fake.after_post = None
        raise GitHubTransportError(method, url)

    fake.after_post = lose_response
    client = workflow(fake, role="student")
    recovered = post_peer(client)
    replay = post_peer(client)

    assert recovered.changed is True
    assert replay.changed is False
    assert len(fake.recipient.comments) == 1
    assert len(fake.mutations) == 1


@pytest.mark.parametrize("changed_pr", ["sender", "recipient"])
def test_peer_comment_rechecks_both_assignments_after_delivery(changed_pr):
    fake = PeerGitHub()

    def revise_assignment(_method, _url):
        owner = getattr(fake, changed_pr)
        owner.pr["body"] = render_assignment_marker(
            assignment_record(
                assignment_id="assignment-7" if changed_pr == "sender" else "assignment-8",
                student="student-one" if changed_pr == "sender" else "student-two",
                head_ref=str(owner.pr["head_ref"]),
                revision_id="revision-2",
            )
        )

    fake.after_post = revise_assignment
    error_type = (
        StaleAssignmentRevisionError if changed_pr == "sender" else ReconciliationError
    )

    with pytest.raises(error_type) as error:
        post_peer(workflow(fake, role="student"))

    if changed_pr == "recipient":
        assert not isinstance(error.value, StaleAssignmentRevisionError)
    assert len(fake.mutations) == 1
    assert len(fake.recipient.comments) == 1
