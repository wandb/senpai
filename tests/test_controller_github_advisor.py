import pytest
from pydantic import SecretStr

from senpai_agent.github.http import GitHubReadError
from senpai_agent.github.mailbox import GitHubMailbox
from senpai_agent.github.mailbox.values import payload_digest
from senpai_agent.models import (
    AssignmentCommentRecord,
    AssignmentKey,
    AssignmentRecord,
    ExperimentResult,
    ResearchBaseAcceptanceRecord,
    ResultStatus,
    experiment_result_digest,
    render_assignment_marker,
    render_assignment_comment_marker,
    render_research_base_acceptance_marker,
    render_result_comment,
)


def pull(
    *,
    labels,
    number=17,
    body="",
    head_sha=None,
    head_repo="acme/widgets",
    author="maintainer",
    comments_url=None,
    updated_at="2099-07-29T18:00:00Z",
):
    value = {
        "number": number,
        "title": "Try bounded change",
        "html_url": f"https://github.test/acme/widgets/pull/{number}",
        "updated_at": updated_at,
        "body": body,
        "user": {"login": author},
        "head": {
            "ref": f"student/candidate-{number}",
            "sha": head_sha or str(number % 10) * 40,
            "repo": {"full_name": head_repo},
        },
        "labels": [{"name": label} for label in labels],
    }
    if comments_url is not None:
        value["comments_url"] = comments_url
    return value


def mailbox(monkeypatch, pulls, *, students=()):
    value = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
        students=students,
        trusted_actor="senpai-bot",
    )
    monkeypatch.setattr(value, "_pulls", lambda: list(pulls))
    monkeypatch.setattr(value, "_issues", list)
    monkeypatch.setattr(value, "_has_write_permission", lambda _login: True)
    monkeypatch.setattr(value._github, "objects", lambda _url: [])
    return value


@pytest.mark.parametrize(
    ("permission", "expected"),
    [
        ("admin", True),
        ("maintain", True),
        ("write", True),
        ("triage", False),
        ("read", False),
        ("none", False),
    ],
)
def test_advisor_write_authorization_uses_current_collaborator_permission(
    monkeypatch,
    permission,
    expected,
):
    advisor = GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role="advisor",
        advisor_branch="research",
    )
    requests = []

    def get(path):
        requests.append(path)
        return {"permission": permission}

    monkeypatch.setattr(advisor._github, "get", get)

    assert advisor._has_write_permission("maintainer") is expected
    assert requests == [
        "/repos/acme/widgets/collaborators/maintainer/permission"
    ]


def test_advisor_rejects_a_fork_head_before_checking_author_permission(monkeypatch):
    fork = pull(
        labels=("research", "student:student-1", "status:review"),
        head_repo="outsider/widgets",
    )
    advisor = mailbox(monkeypatch, [fork])
    permission_checks = []
    monkeypatch.setattr(
        advisor,
        "_has_write_permission",
        lambda login: permission_checks.append(login) or True,
    )

    assert advisor.poll() == ()
    assert permission_checks == []


def test_advisor_rejects_a_same_repo_pull_without_current_write_permission(
    monkeypatch,
):
    untrusted = pull(
        labels=("research", "student:student-1", "status:review"),
        author="former-maintainer",
    )
    advisor = mailbox(monkeypatch, [untrusted])
    permission_checks = []
    monkeypatch.setattr(
        advisor,
        "_has_write_permission",
        lambda login: permission_checks.append(login) or False,
    )

    assert advisor.poll() == ()
    assert permission_checks == ["former-maintainer"]


def test_advisor_permission_read_failure_rejects_the_pull(monkeypatch, capsys):
    candidate = pull(
        labels=("research", "student:student-1", "status:review"),
    )
    advisor = mailbox(monkeypatch, [candidate])

    def fail(_login):
        raise GitHubReadError("permission unavailable")

    monkeypatch.setattr(advisor, "_has_write_permission", fail)

    assert advisor.poll() == ()
    assert "SENPAI_ADVISOR_PULL_AUTHORIZATION_ERROR pr=17" in capsys.readouterr().err


def assignment(
    *,
    base_sha="b" * 40,
    base_ref="research",
    number=17,
    revision_id="revision-2",
):
    return AssignmentRecord(
        repo="acme/widgets",
        assignment_id=f"assignment-{number}",
        revision_id=revision_id,
        student="student-1",
        base_ref=base_ref,
        base_sha=base_sha,
        head_ref=f"student/candidate-{number}",
        head_sha=str(number % 10) * 40,
    )


def assignment_comment(
    *,
    github_id: int = 501,
    comment_id: str = "paired-run-started",
    student: str = "student-1",
    revision_id: str = "revision-2",
    author: str = "senpai-bot",
    author_type: str = "Bot",
    message: str = "The paired run has started.",
    updated_at: str = "2026-08-11T07:20:00Z",
):
    marker = render_assignment_comment_marker(
        AssignmentCommentRecord(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id=revision_id,
            student=student,
            comment_id=comment_id,
        )
    )
    return {
        "id": github_id,
        "body": f"{marker}\n\nSTUDENT: {message}",
        "html_url": (
            "https://github.test/acme/widgets/pull/17"
            f"#issuecomment-{github_id}"
        ),
        "created_at": "2026-08-11T07:20:00Z",
        "updated_at": updated_at,
        "user": {"login": author, "type": author_type},
        "author_association": "MEMBER",
    }


def human_pr_comment(
    *,
    github_id: int,
    body: str,
    author: str = "maintainer",
    author_type: str = "User",
    association: str = "OWNER",
    created_at: str = "2026-08-11T07:20:00Z",
    updated_at: str | None = None,
):
    return {
        "id": github_id,
        "body": body,
        "html_url": (
            "https://github.test/acme/widgets/pull/17"
            f"#issuecomment-{github_id}"
        ),
        "created_at": created_at,
        "updated_at": updated_at or created_at,
        "user": {"login": author, "type": author_type},
        "author_association": association,
    }


def test_student_assignment_comment_wakes_advisor_once_per_semantic_message(
    monkeypatch,
):
    comments_url = "https://api.github.test/repos/acme/widgets/issues/17/comments"
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
        head_sha="7" * 40,
        comments_url=comments_url,
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            assignment_comment(github_id=502),
            assignment_comment(github_id=501),
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    comments = [
        event for event in advisor.poll() if event.kind == "student_assignment_comment"
    ]

    assert len(comments) == 1
    expected_comment = assignment_comment(github_id=501)
    assert comments[0].payload == {
        "number": 17,
        "pr_url": "https://github.test/acme/widgets/pull/17",
        "comment_id": "paired-run-started",
        "assignment_id": "assignment-17",
        "revision_id": "revision-2",
        "student": "student-1",
        "message": "STUDENT: The paired run has started.",
        "content_digest": payload_digest({"body": expected_comment["body"]}),
    }


def test_duplicate_comment_retry_metadata_does_not_change_the_event(monkeypatch):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    visible_comments = [
        assignment_comment(github_id=501),
        assignment_comment(github_id=502),
    ]
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: list(visible_comments),
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    first = next(
        event for event in advisor.poll() if event.kind == "student_assignment_comment"
    )
    visible_comments.pop(0)
    second = next(
        event for event in advisor.poll() if event.kind == "student_assignment_comment"
    )

    assert second == first


def test_advisor_delivers_a_comment_from_an_earlier_revision(monkeypatch):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
        head_sha="7" * 40,
        comments_url="https://api.github.test/repos/acme/widgets/issues/17/comments",
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [assignment_comment(github_id=502, revision_id="revision-old")],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    event = next(
        event for event in advisor.poll() if event.kind == "student_assignment_comment"
    )

    assert event.payload["revision_id"] == "revision-old"


def test_advisor_ignores_forged_assignment_comments(monkeypatch):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            assignment_comment(github_id=501, student="student-2"),
            assignment_comment(github_id=503, author="mallory"),
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    assert not any(
        event.kind == "student_assignment_comment" for event in advisor.poll()
    )


def test_conflicting_student_comment_bodies_fail_closed(monkeypatch, capsys):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            assignment_comment(github_id=501),
            assignment_comment(github_id=502, message="A conflicting update."),
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    assert not any(
        event.kind == "student_assignment_comment" for event in advisor.poll()
    )
    assert "conflicting bodies for assignment comment identity" in capsys.readouterr().err


def test_edited_student_comment_fails_closed(monkeypatch, capsys):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            assignment_comment(
                updated_at="2026-08-11T07:21:00Z",
                message="Edited after publication.",
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    assert not any(
        event.kind == "student_assignment_comment" for event in advisor.poll()
    )
    assert "edited assignment comment rejected" in capsys.readouterr().err


def test_advisor_receives_every_trusted_human_pr_comment_and_student_message(
    monkeypatch,
):
    assigned = pull(
        labels=("research", "student:student-1", "status:review"),
        body=render_assignment_marker(assignment()),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [assigned], students=("student-1",))
    visible_comments = [
        human_pr_comment(
            github_id=601,
            body="Owner direction.",
            association="OWNER",
        ),
        human_pr_comment(
            github_id=602,
            body="Member direction.",
            author="member",
            association="MEMBER",
        ),
        human_pr_comment(
            github_id=603,
            body="Collaborator direction.",
            author="collaborator",
            association="COLLABORATOR",
        ),
        assignment_comment(github_id=604, author_type="User"),
        human_pr_comment(
            github_id=605,
            body="Untrusted suggestion.",
            author="outsider",
            association="NONE",
        ),
        human_pr_comment(
            github_id=606,
            body="Bot suggestion.",
            author="automation",
            author_type="Bot",
            association="OWNER",
        ),
        human_pr_comment(
            github_id=607,
            body="Contributor suggestion.",
            author="contributor",
            association="CONTRIBUTOR",
        ),
        human_pr_comment(
            github_id=608,
            body="   ",
            author="maintainer",
            association="OWNER",
        ),
    ]
    reads = []

    def objects(url):
        reads.append(url)
        return list(visible_comments)

    monkeypatch.setattr(advisor._github, "objects", objects)
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    events = advisor.poll()
    human_events = [event for event in events if event.kind == "human_pr_comment"]
    student_events = [
        event for event in events if event.kind == "student_assignment_comment"
    ]

    assert [event.payload["feedback_id"] for event in human_events] == [
        601,
        602,
        603,
    ]
    assert [event.payload["message"] for event in human_events] == [
        "Owner direction.",
        "Member direction.",
        "Collaborator direction.",
    ]
    assert len(student_events) == 1
    assert reads == [
        "/repos/acme/widgets/issues/17/comments?per_page=100"
    ]


def test_student_and_human_parsers_share_a_failed_comment_read(monkeypatch):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    advisor = mailbox(monkeypatch, [assigned])
    reads = []

    def objects(url):
        reads.append(url)
        raise GitHubReadError("temporary issue-comment failure")

    monkeypatch.setattr(advisor._github, "objects", objects)
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    events = advisor.poll()

    assert not any(
        event.kind in {"human_pr_comment", "student_assignment_comment"}
        for event in events
    )
    assert reads.count(
        "/repos/acme/widgets/issues/17/comments?per_page=100"
    ) == 1


def test_advisor_receives_trusted_human_comment_without_a_valid_assignment(
    monkeypatch,
):
    malformed = pull(
        labels=("research",),
        body="This PR has no assignment marker.",
    )
    advisor = mailbox(monkeypatch, [malformed])
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            human_pr_comment(
                github_id=611,
                body="Repair the assignment metadata before continuing.",
            )
        ],
    )

    event = next(
        event for event in advisor.poll() if event.kind == "human_pr_comment"
    )

    assert event.payload["number"] == 17
    assert event.payload["message"] == (
        "Repair the assignment metadata before continuing."
    )


def test_malformed_human_pr_comment_is_reported(monkeypatch, capsys):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    malformed = human_pr_comment(github_id=620, body="Malformed direction.")
    malformed["created_at"] = "not-a-timestamp"
    advisor = mailbox(monkeypatch, [assigned])
    monkeypatch.setattr(advisor._github, "objects", lambda _url: [malformed])
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    assert not any(
        event.kind == "human_pr_comment" for event in advisor.poll()
    )
    error = capsys.readouterr().err
    assert "SENPAI_HUMAN_PR_COMMENT_READ_ERROR" in error
    assert "pr=17 comment_id=620 ValueError" in error


def test_human_pr_comment_versions_edits_but_not_pull_metadata(monkeypatch):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    comment = human_pr_comment(github_id=621, body="Use the narrow control.")
    advisor = mailbox(monkeypatch, [assigned])
    monkeypatch.setattr(advisor._github, "objects", lambda _url: [comment])
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    first = next(
        event for event in advisor.poll() if event.kind == "human_pr_comment"
    )
    assigned["head"]["sha"] = "8" * 40
    assigned["labels"] = [
        {"name": "research"},
        {"name": "student:student-1"},
        {"name": "status:review"},
    ]
    unchanged = next(
        event for event in advisor.poll() if event.kind == "human_pr_comment"
    )
    comment["author_association"] = "MEMBER"
    metadata_changed = next(
        event for event in advisor.poll() if event.kind == "human_pr_comment"
    )
    comment["body"] = "Use the wide control."
    comment["updated_at"] = "2026-08-11T07:25:00Z"
    edited = next(
        event for event in advisor.poll() if event.kind == "human_pr_comment"
    )
    comment["body"] = "Use the narrow control."
    comment["updated_at"] = "2026-08-11T07:30:00Z"
    reverted = next(
        event for event in advisor.poll() if event.kind == "human_pr_comment"
    )

    assert unchanged == first
    assert metadata_changed == first
    assert len({first.dedupe_key, edited.dedupe_key, reverted.dedupe_key}) == 3


def test_shared_actor_plain_human_comment_is_visible_but_protocol_output_is_not(
    monkeypatch,
):
    assigned = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment()),
    )
    advisor = mailbox(monkeypatch, [assigned])
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            human_pr_comment(
                github_id=631,
                body="Human direction from the shared account.",
                author="senpai-bot",
            ),
            human_pr_comment(
                github_id=632,
                body=(
                    "<!-- senpai-assignment-feedback:v1:{} -->\n\n"
                    "Advisor protocol output."
                ),
                author="senpai-bot",
            ),
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "b" * 40}},
    )

    events = [event for event in advisor.poll() if event.kind == "human_pr_comment"]

    assert len(events) == 1
    assert events[0].payload["feedback_id"] == 631


def test_review_label_wakes_the_advisor_and_keeps_the_student_assigned(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
            )
        ],
        students=("student-1", "student-2"),
    )

    events = advisor.poll()

    assert [event.kind for event in events] == [
        "review_ready",
        "student_available_for_assignment",
    ]
    assert events[0].payload["number"] == 17
    assert (
        events[1].dedupe_key
        == "student_available_for_assignment:student-2"
    )
    assert events[1].payload == {"student": "student-2"}
    assert events[1].to_prompt().startswith(
        "## Student available for assignment: `student-2`"
    )


@pytest.mark.parametrize("status", ["status:wip", "status:review"])
@pytest.mark.parametrize(
    "blocker",
    ["status:hold", "status:blocked", "status:needs-rebase"],
)
def test_assignment_action_labels_do_not_make_the_student_available(
    monkeypatch,
    status,
    blocker,
):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=(
                    "research",
                    "student:student-1",
                    status,
                    blocker,
                )
            )
        ],
        students=("student-1",),
    )

    assert "student_available_for_assignment" not in {
        event.kind for event in advisor.poll()
    }


def test_new_review_revision_at_the_same_head_wakes_the_advisor(monkeypatch):
    reviewed_pull = pull(
        labels=("research", "student:student-1", "status:review"),
        body=render_assignment_marker(assignment(revision_id="revision-1")),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [reviewed_pull])

    first = next(event for event in advisor.poll() if event.kind == "review_ready")
    reviewed_pull["body"] = render_assignment_marker(
        assignment(revision_id="revision-2")
    )
    second = next(event for event in advisor.poll() if event.kind == "review_ready")

    assert first.dedupe_key.startswith(
        f"review_ready:v2:17:assignment-17:revision-1:{'7' * 40}:"
    )
    assert second.dedupe_key.startswith(
        f"review_ready:v2:17:assignment-17:revision-2:{'7' * 40}:"
    )
    assert first.payload["revision_id"] == "revision-1"
    assert second.payload["revision_id"] == "revision-2"


def test_review_event_ignores_mutable_pull_presentation(monkeypatch):
    reviewed_pull = pull(
        labels=("research", "student:student-1", "status:review"),
        body=render_assignment_marker(assignment()),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [reviewed_pull])
    first = next(event for event in advisor.poll() if event.kind == "review_ready")

    reviewed_pull["title"] = "Clarify the review"
    reviewed_pull["updated_at"] = "2099-07-29T19:00:00Z"
    reviewed_pull["labels"].append({"name": "operator-note"})
    repeated = next(
        event for event in advisor.poll() if event.kind == "review_ready"
    )

    assert repeated.dedupe_key == first.dedupe_key
    assert repeated.to_prompt() == first.to_prompt()


def test_review_event_versions_a_branch_rename_at_the_same_head(monkeypatch):
    reviewed_pull = pull(
        labels=("research", "student:student-1", "status:review"),
        body=render_assignment_marker(assignment()),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [reviewed_pull])
    first = next(event for event in advisor.poll() if event.kind == "review_ready")

    reviewed_pull["head"]["ref"] = "student/renamed-candidate"
    renamed = next(event for event in advisor.poll() if event.kind == "review_ready")

    assert renamed.dedupe_key != first.dedupe_key
    assert renamed.payload["head_ref"] == "student/renamed-candidate"


@pytest.mark.parametrize(
    ("labels", "updated_at", "reason"),
    [
        (("student:student-1", "status:blocked"), "2099-01-01T00:00:00Z", "blocked"),
        (
            ("student:student-1", "status:needs-rebase"),
            "2099-01-01T00:00:00Z",
            "needs_rebase",
        ),
        (("status:review",), "2099-01-01T00:00:00Z", "missing_student"),
        (
            ("student:student-1", "student:student-2", "status:review"),
            "2099-01-01T00:00:00Z",
            "multiple_students",
        ),
        (
            ("student:student-1", "status:wip"),
            "2020-01-01T00:00:00Z",
            "stale_wip",
        ),
    ],
)
def test_advisor_action_reports_each_unsafe_assignment_state(
    monkeypatch,
    labels,
    updated_at,
    reason,
):
    advisor = mailbox(
        monkeypatch,
        [pull(labels=("research", *labels), updated_at=updated_at)],
    )

    actions = [event for event in advisor.poll() if event.kind == "advisor_action"]

    assert len(actions) == 1
    assert actions[0].payload["reasons"] == [reason]


def test_advisor_action_ignores_metadata_that_does_not_change_its_reasons(
    monkeypatch,
):
    assigned_pull = pull(
        labels=("research", "student:student-1", "status:wip", "status:blocked"),
    )
    advisor = mailbox(monkeypatch, [assigned_pull])
    first = next(event for event in advisor.poll() if event.kind == "advisor_action")

    assigned_pull["title"] = "Clarify the blocked work"
    assigned_pull["updated_at"] = "2099-07-29T19:00:00Z"
    assigned_pull["labels"].append({"name": "operator-note"})
    repeated = next(
        event for event in advisor.poll() if event.kind == "advisor_action"
    )

    assert repeated.dedupe_key == first.dedupe_key
    assert repeated.to_prompt() == first.to_prompt()


def test_advisor_action_versions_a_branch_rename_at_the_same_head(monkeypatch):
    assigned_pull = pull(
        labels=("research", "student:student-1", "status:wip", "status:blocked"),
    )
    advisor = mailbox(monkeypatch, [assigned_pull])
    first = next(event for event in advisor.poll() if event.kind == "advisor_action")

    assigned_pull["head"]["ref"] = "student/renamed-candidate"
    renamed = next(event for event in advisor.poll() if event.kind == "advisor_action")

    assert renamed.dedupe_key != first.dedupe_key
    assert renamed.payload["head_ref"] == "student/renamed-candidate"


def test_advisor_action_reports_hold_as_a_blocker(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=(
                    "research",
                    "student:student-1",
                    "status:wip",
                    "status:hold",
                )
            )
        ],
    )

    action = next(event for event in advisor.poll() if event.kind == "advisor_action")

    assert action.payload["reasons"] == ["hold"]


def test_duplicate_assignments_report_every_pr_for_the_student(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(labels=("student:student-1", "status:wip"), number=17),
            pull(labels=("student:student-1", "status:review"), number=18),
        ],
        students=("student-1",),
    )

    duplicate = next(
        event for event in advisor.poll() if event.kind == "duplicate_assignment"
    )

    assert duplicate.dedupe_key == "duplicate_assignment:student-1:17,18"
    assert duplicate.payload["pull_requests"] == [17, 18]


def test_research_base_change_uses_the_fresh_live_branch_head_on_each_poll(
    monkeypatch,
):
    assigned_sha = "b" * 40
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:wip"),
                body=render_assignment_marker(assignment(base_sha=assigned_sha)),
                head_sha="7" * 40,
            )
        ],
        students=("student-1",),
    )
    current_sha = ["c" * 40]
    ref_reads = []

    def get_ref(path):
        ref_reads.append(path)
        return {"object": {"sha": current_sha[0]}}

    monkeypatch.setattr(advisor._github, "get", get_ref)

    first = next(
        event for event in advisor.poll() if event.kind == "research_base_changed"
    )
    current_sha[0] = "d" * 40
    second = next(
        event for event in advisor.poll() if event.kind == "research_base_changed"
    )

    assert first.dedupe_key.startswith(
        f"research_base_changed:v2:17:assignment-17:revision-2:"
        f"{'7' * 40}:research:{assigned_sha}:{'c' * 40}:"
    )
    assert first.payload["required_base_sha"] == assigned_sha
    assert first.payload["current_base_sha"] == "c" * 40
    assert first.payload["compare_url"] == (
        f"https://github.test/acme/widgets/compare/{assigned_sha}...{'c' * 40}"
    )
    assert second.payload["current_base_sha"] == "d" * 40
    assert ref_reads == [
        "/repos/acme/widgets/git/ref/heads/research",
        "/repos/acme/widgets/git/ref/heads/research",
    ]


def test_research_base_event_ignores_mutable_pull_presentation(monkeypatch):
    assigned_pull = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment(base_sha="b" * 40)),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [assigned_pull])
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    first = next(
        event for event in advisor.poll() if event.kind == "research_base_changed"
    )

    assigned_pull["title"] = "Clarify the base change"
    assigned_pull["updated_at"] = "2099-07-29T19:00:00Z"
    assigned_pull["labels"].append({"name": "operator-note"})
    repeated = next(
        event for event in advisor.poll() if event.kind == "research_base_changed"
    )

    assert repeated.dedupe_key == first.dedupe_key
    assert repeated.to_prompt() == first.to_prompt()


def test_research_base_event_versions_a_branch_rename_at_the_same_head(monkeypatch):
    assigned_pull = pull(
        labels=("research", "student:student-1", "status:wip"),
        body=render_assignment_marker(assignment(base_sha="b" * 40)),
        head_sha="7" * 40,
    )
    advisor = mailbox(monkeypatch, [assigned_pull])
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    first = next(
        event for event in advisor.poll() if event.kind == "research_base_changed"
    )

    assigned_pull["head"]["ref"] = "student/renamed-candidate"
    renamed = next(
        event for event in advisor.poll() if event.kind == "research_base_changed"
    )

    assert renamed.dedupe_key != first.dedupe_key
    assert renamed.payload["head_ref"] == "student/renamed-candidate"


def terminal_result(*, summary="The candidate remains valid."):
    return ExperimentResult(
        assignment=AssignmentKey(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-2",
            expected_head_sha="7" * 40,
            student="student-1",
        ),
        status=ResultStatus.SUCCEEDED,
        hypothesis="The candidate improves the primary metric.",
        summary=summary,
        runs=(),
        commit_sha="7" * 40,
    )


def result_comment(*, result=None, author="senpai-bot"):
    current = terminal_result() if result is None else result
    return {
        "body": render_result_comment(current),
        "user": {"login": author},
    }


def acceptance_comment(
    *,
    result=None,
    accepted_base_sha="c" * 40,
    author="senpai-bot",
):
    current = terminal_result() if result is None else result
    marker = render_research_base_acceptance_marker(
        ResearchBaseAcceptanceRecord(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-2",
            result_head_sha="7" * 40,
            result_digest=experiment_result_digest(current),
            evaluated_base_sha="b" * 40,
            base_ref="research",
            accepted_base_sha=accepted_base_sha,
        )
    )
    return {
        "body": f"{marker}\n\nThe result remains valid.",
        "user": {"login": author},
    }


def test_exact_trusted_acceptance_suppresses_review_restart_redelivery(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [result_comment(), acceptance_comment()],
    )

    assert "research_base_changed" not in {
        event.kind for event in advisor.poll()
    }


@pytest.mark.parametrize(
    "extra",
    ["duplicate-acceptance", "duplicate-result", "malformed-result"],
)
def test_acceptance_suppression_tolerates_idempotent_or_malformed_noise(
    monkeypatch,
    extra,
):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    acceptance = acceptance_comment()
    if extra == "duplicate-acceptance":
        noise = acceptance_comment()
    elif extra == "duplicate-result":
        noise = result_comment()
    else:
        noise = {
            "body": '<!-- senpai-result:v2 {"assignment_id":"assignment-17"} -->',
            "user": {"login": "senpai-bot"},
        }
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [result_comment(), acceptance, noise],
    )

    assert "research_base_changed" not in {
        event.kind for event in advisor.poll()
    }


@pytest.mark.parametrize(
    "comment",
    [
        acceptance_comment(author="untrusted-user"),
        acceptance_comment(accepted_base_sha="d" * 40),
    ],
    ids=("untrusted", "stale"),
)
def test_untrusted_or_stale_acceptance_does_not_suppress_change(
    monkeypatch,
    comment,
):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [result_comment(), comment],
    )

    assert "research_base_changed" in {event.kind for event in advisor.poll()}


def test_acceptance_for_different_result_at_same_head_does_not_suppress_change(
    monkeypatch,
):
    accepted = terminal_result(summary="Result A remains valid.")
    current = terminal_result(summary="Result B supersedes it.")
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            result_comment(result=current),
            acceptance_comment(result=accepted),
        ],
    )

    assert "research_base_changed" in {event.kind for event in advisor.poll()}


def test_malformed_trusted_acceptance_does_not_suppress_change(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [
            result_comment(),
            {
                "body": "<!-- senpai-research-base-acceptance:v1 malformed -->",
                "user": {"login": "senpai-bot"},
            },
        ],
    )

    assert "research_base_changed" in {event.kind for event in advisor.poll()}


@pytest.mark.parametrize("separator", ["\n\n", "\r", "\x85", "\u2028"])
def test_embedded_acceptance_marker_is_not_trusted_protocol_evidence(
    monkeypatch,
    separator,
):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    monkeypatch.setattr(advisor._github, "actor", lambda: "senpai-bot")
    embedded = acceptance_comment()
    embedded["body"] = (
        f"<!-- senpai-assignment-feedback:v1 {{}} -->{separator}"
        + str(embedded["body"])
    )
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda _url: [result_comment(), embedded],
    )

    assert "research_base_changed" in {event.kind for event in advisor.poll()}


def test_wip_base_change_shares_the_single_student_comment_read(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:wip"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
                comments_url=(
                    "https://api.github.test/repos/acme/widgets/"
                    "issues/17/comments"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": "c" * 40}},
    )
    comment_reads = []
    monkeypatch.setattr(
        advisor._github,
        "objects",
        lambda url: comment_reads.append(url) or [],
    )

    assert "research_base_changed" in {event.kind for event in advisor.poll()}
    assert comment_reads == [
        "/repos/acme/widgets/issues/17/comments?per_page=100"
    ]


def test_current_assignment_base_does_not_emit_a_false_change(monkeypatch):
    current_sha = "b" * 40
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:wip"),
                body=render_assignment_marker(assignment(base_sha=current_sha)),
                head_sha="7" * 40,
            )
        ],
        students=("student-1",),
    )
    monkeypatch.setattr(
        advisor._github,
        "get",
        lambda _path: {"object": {"sha": current_sha}},
    )

    assert "research_base_changed" not in {
        event.kind for event in advisor.poll()
    }


def test_each_assignment_watches_its_own_research_base_ref(monkeypatch):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                number=17,
                labels=("research-a", "student:student-1", "status:wip"),
                body=render_assignment_marker(
                    assignment(base_ref="research-a", number=17)
                ),
            ),
            pull(
                number=18,
                labels=("research-b", "student:student-2", "status:wip"),
                body=render_assignment_marker(
                    assignment(base_ref="research-b", number=18)
                ),
            ),
        ],
    )
    reads = []

    def get_ref(path):
        reads.append(path)
        sha = "c" * 40 if path.endswith("research-a") else "d" * 40
        return {"object": {"sha": sha}}

    monkeypatch.setattr(advisor._github, "get", get_ref)

    changed = [
        event
        for event in advisor.poll()
        if event.kind == "research_base_changed"
    ]

    assert {event.payload["base_ref"] for event in changed} == {
        "research-a",
        "research-b",
    }
    assert reads == [
        "/repos/acme/widgets/git/ref/heads/research-a",
        "/repos/acme/widgets/git/ref/heads/research-b",
    ]


def test_research_base_ref_failure_does_not_suppress_other_advisor_events(
    monkeypatch,
    capsys,
):
    advisor = mailbox(
        monkeypatch,
        [
            pull(
                labels=("research", "student:student-1", "status:review"),
                body=render_assignment_marker(assignment()),
                head_sha="7" * 40,
            )
        ],
        students=("student-1", "student-2"),
    )

    def invalid_ref(_path):
        raise TypeError("invalid ref response")

    monkeypatch.setattr(advisor._github, "get", invalid_ref)

    events = advisor.poll()

    assert {event.kind for event in events} == {
        "review_ready",
        "student_available_for_assignment",
    }
    available = next(
        event
        for event in events
        if event.kind == "student_available_for_assignment"
    )
    assert available.payload == {"student": "student-2"}
    assert "SENPAI_RESEARCH_BASE_WATCH_ERROR" in capsys.readouterr().err
