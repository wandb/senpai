import json
import re
from copy import deepcopy
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

import pytest
from github_workflow_support import (
    ASSIGNMENT_ID,
    HEAD_SHA,
    FakeGitHub,
    assignment_record,
    pull_request,
    workflow,
)

from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.workflow import (
    GitHubAPIError,
    GitHubTransportError,
    HttpResponse,
    PullHeadMismatchError,
    ReconciliationError,
    StaleAssignmentRevisionError,
    WorkflowPreconditionError,
)
from senpai_agent.models import render_assignment_marker


class PeerGitHub:
    """Route student PRs through the existing stateful GitHub transport."""

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
        self.pulls = {7: self.sender, 8: self.recipient}
        self.requests = []
        self.after_post = None
        self.fail_posts = set()
        self.page_size = 2
        self.permissions = {
            "senpai-bot": HttpResponse(200, {"permission": "write"}),
        }

    @property
    def mutations(self):
        return [item for item in self.requests if item[0] != "GET"]

    def add_peer(
        self,
        number,
        *,
        student=None,
        author="senpai-bot",
        head_repo="acme/widgets",
        **overrides,
    ):
        student = student or f"student-{number}"
        fields = {
            "labels": {f"student:{student}", "status:wip"},
            "head_ref": f"{student}/experiment",
            **overrides,
        }
        fields.setdefault(
            "body",
            render_assignment_marker(
                assignment_record(
                    assignment_id=f"assignment-{number}",
                    student=student,
                    head_ref=fields["head_ref"],
                    base_ref=fields.get("base_ref", "schmidhuber"),
                )
            ),
        )
        pr = pull_request(**fields)
        pr.update(
            number=number,
            html_url=f"https://github.com/acme/widgets/pull/{number}",
            author=author,
            head_repo=head_repo,
        )
        fake = self.pulls[number] = FakeGitHub(pr)
        return fake

    def _pull_payload(self, number):
        fake = self.pulls[number]
        payload = fake._pull_payload()
        payload["head"]["repo"] = {"full_name": fake.pr.get("head_repo", "acme/widgets")}
        payload["base"]["repo"] = {"full_name": "acme/widgets"}
        payload["user"] = {"login": fake.pr.get("author", "senpai-bot")}
        return payload

    def request(self, method, url, *, headers, json_body=None):
        parsed = urlsplit(url)
        path = parsed.path
        self.requests.append((method, url, json_body))
        if method == "GET" and "/collaborators/" in path:
            login = unquote(
                path.split("/collaborators/", 1)[1].removesuffix("/permission")
            )
            return self.permissions[login.casefold()]
        if method == "GET" and path == "/repos/acme/widgets/pulls":
            query = parse_qs(parsed.query)
            candidates = [
                self._pull_payload(number)
                for number, fake in self.pulls.items()
                if fake.pr["state"] == query.get("state", ["open"])[0]
                and fake.pr["base_ref"] == query.get("base", ["schmidhuber"])[0]
            ]
            page = int(query.get("page", ["1"])[0])
            start = (page - 1) * self.page_size
            end = start + self.page_size
            response_headers = ()
            if end < len(candidates):
                query["page"] = [str(page + 1)]
                next_url = parsed._replace(query=urlencode(query, doseq=True)).geturl()
                response_headers = (("Link", f'<{next_url}>; rel="next"'),)
            return HttpResponse(200, candidates[start:end], response_headers)
        match = re.search(r"/(?:pulls|issues)/(\d+)(?:/|$)", path)
        number = int(match[1]) if match else 7
        if method == "GET" and path == f"/repos/acme/widgets/pulls/{number}":
            return HttpResponse(200, self._pull_payload(number))
        if method == "POST" and number in self.fail_posts:
            return HttpResponse(503, {"message": "Service unavailable"})
        fake = self.pulls[number]
        routed = url.replace(f"/pulls/{number}", "/pulls/7").replace(
            f"/issues/{number}/", "/issues/7/"
        )
        response = fake.request(method, routed, headers=headers, json_body=json_body)
        for comment in fake.comments:
            comment["html_url"] = str(comment["html_url"]).replace(
                "/pull/7#", f"/pull/{number}#"
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
    assert [(method, urlsplit(url).path) for method, url, _ in fake.mutations] == [
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


def broadcast(client, **overrides):
    fields = {
        "assignment_id": ASSIGNMENT_ID,
        "revision_id": "revision-1",
        "expected_head_sha": HEAD_SHA,
        "student": "student-one",
        "advisor_branch": "schmidhuber",
        "broadcast_id": "optimizer-state-discovery",
        "message": (
            "The checkpoint omits optimizer state, so resumed runs reset momentum. "
            "Relevant if your assignment resumes training; keep your current focus. "
            "Evidence: https://github.com/acme/widgets/pull/7/files"
        ),
    }
    fields.update(overrides)
    return client.broadcast_message(7, **fields)


def test_broadcast_reaches_paginated_peer_assignments_regardless_of_workflow_status():
    fake = PeerGitHub()
    fake.add_peer(9, labels={"student:student-9", "status:review"})
    fake.add_peer(10, labels={"student:student-10", "status:hold"})
    fake.add_peer(11, labels={"student:student-11", "status:done"}, draft=True)
    fake.add_peer(12, state="closed")
    fake.add_peer(13, base_ref="another-advisor")
    fake.add_peer(14, body="An ordinary PR without an assignment.")
    fake.add_peer(15, student="student-one")
    fake.add_peer(16, labels={"student:someone-else", "status:wip"})
    fake.add_peer(17, head_repo="external/widgets")
    fake.add_peer(18, author="untrusted-contributor")
    fake.permissions["untrusted-contributor"] = HttpResponse(200, {"permission": "read"})
    before = {number: deepcopy(pr.pr) for number, pr in fake.pulls.items()}
    client = workflow(fake, role="student")

    first = broadcast(client)
    replay = broadcast(client)

    assert first.changed is True
    assert replay.changed is False
    assert first.delivered_pr_numbers == replay.delivered_pr_numbers == (8, 9, 10, 11)
    assert first.resource_url == "https://github.com/acme/widgets/pull/7#issuecomment-1"
    assert {number: pr.pr for number, pr in fake.pulls.items()} == before
    assert len(fake.sender.comments) == 1
    for number in (8, 9, 10, 11):
        comments = fake.pulls[number].comments
        assert len(comments) == 1
        marker, visible = comments[0]["body"].split("\n\n", 1)
        record = json.loads(marker[marker.index("{") : marker.rindex("}") + 1])
        assert record["broadcast_id"] == "optimizer-state-discovery"
        assert record["pr_number"] == number
        assert record["assignment_id"] == f"assignment-{number}"
        assert visible.startswith("**STUDENT: student-one**\n\n")
        assert f"Discovery: [source discussion]({first.resource_url})" in visible
        assert "Evidence: https://github.com/acme/widgets/pull/7/files" in visible
        assert visible.endswith("Working PR: [#7](https://github.com/acme/widgets/pull/7)")
    assert all(not fake.pulls[number].comments for number in range(12, 19))
    assert len(fake.mutations) == 5
    assert all(
        method == "POST" and urlsplit(url).path.endswith("/comments")
        for method, url, _ in fake.mutations
    )
    assert any(
        parse_qs(urlsplit(url).query).get("page") == ["2"]
        for _, url, _ in fake.requests
    )


def test_broadcast_retry_recovers_partial_delivery_and_keeps_original_recipient_revision():
    fake = PeerGitHub()
    fake.add_peer(9)
    fake.fail_posts.add(9)
    client = workflow(fake, role="student")

    with pytest.raises(GitHubAPIError):
        broadcast(client)
    assert len(fake.sender.comments) == len(fake.recipient.comments) == 1
    assert fake.pulls[9].comments == []
    fake.fail_posts.clear()
    fake.recipient.pr["body"] = render_assignment_marker(
        assignment_record(
            assignment_id="assignment-8",
            revision_id="revision-2",
            student="student-two",
            head_ref="student-two/handoff",
        )
    )
    fake.add_peer(10)

    recovered = broadcast(client)
    replay = broadcast(client)

    assert recovered.changed is True
    assert replay.changed is False
    assert recovered.delivered_pr_numbers == (8, 9, 10)
    assert all(len(fake.pulls[number].comments) == 1 for number in (7, 8, 9, 10))
    marker = fake.recipient.comments[0]["body"].split("\n\n", 1)[0]
    record = json.loads(marker[marker.index("{") : marker.rindex("}") + 1])
    assert record["revision_id"] == "revision-1"


def test_broadcast_source_record_prevents_changed_text_after_a_recipient_closes():
    fake = PeerGitHub()
    client = workflow(fake, role="student")
    broadcast(client)
    fake.recipient.pr["state"] = "closed"
    newcomer = fake.add_peer(9)
    mutation_count = len(fake.mutations)

    with pytest.raises(WorkflowPreconditionError):
        broadcast(client, message="A changed conclusion under the same ID.")

    assert len(fake.mutations) == mutation_count
    assert newcomer.comments == []
    broadcast(client, broadcast_id="revised-conclusion", message="A revised conclusion.")
    assert len(newcomer.comments) == 1


@pytest.mark.parametrize(
    "message", ["", "   ", "x" * 1501], ids=("empty", "whitespace", "oversized")
)
def test_broadcast_rejects_empty_or_oversized_messages_before_writing(message):
    fake = PeerGitHub()

    with pytest.raises(ValueError):
        broadcast(workflow(fake, role="student"), message=message)

    assert fake.mutations == []


@pytest.mark.parametrize(
    ("role", "overrides", "error"),
    [
        ("advisor", {}, PermissionError),
        ("student", {"student": "student-two"}, PermissionError),
        ("student", {"advisor_branch": "other-advisor"}, WorkflowPreconditionError),
        ("student", {"revision_id": "revision-2"}, StaleAssignmentRevisionError),
    ],
)
def test_broadcast_validates_sender_before_any_publication(role, overrides, error):
    fake = PeerGitHub()

    with pytest.raises(error):
        broadcast(workflow(fake, role=role), **overrides)

    assert fake.mutations == []


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (HttpResponse(200, {"permission": "unexpected"}), GitHubReadError),
        (HttpResponse(503, {"message": "Unavailable"}), GitHubAPIError),
    ],
    ids=("invalid-permission", "failed-permission-read"),
)
def test_broadcast_aborts_before_any_writes_when_recipient_permissions_are_unknown(
    response, error
):
    fake = PeerGitHub()
    fake.add_peer(9, author="another-author")
    fake.permissions["another-author"] = response

    with pytest.raises(error):
        broadcast(workflow(fake, role="student"))

    assert fake.mutations == []
