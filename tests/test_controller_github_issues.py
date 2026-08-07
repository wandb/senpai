from pydantic import SecretStr

from senpai_agent.github.mailbox import GitHubMailbox


def issue(
    *,
    labels=("human", "team"),
    author="ada",
    author_type="User",
    association="MEMBER",
):
    return {
        "id": 700,
        "number": 23,
        "title": "Change direction",
        "html_url": "https://github.test/acme/widgets/issues/23",
        "updated_at": "2026-07-29T18:10:00Z",
        "created_at": "2026-07-29T18:00:00Z",
        "body": "Start with the cheaper baseline.",
        "user": {"login": author, "type": author_type},
        "author_association": association,
        "labels": [{"name": label} for label in labels],
    }


def mailbox(*, role="advisor", human_issues_enabled=True):
    return GitHubMailbox(
        repo="acme/widgets",
        token=SecretStr("github-token"),
        role=role,
        advisor_branch="research",
        student_name="student-1" if role == "student" else None,
        trusted_actor="senpai-bot",
        human_issues_enabled=human_issues_enabled,
    )


def test_malformed_assignment_does_not_suppress_a_human_issue(monkeypatch):
    student = mailbox(role="student")
    malformed = {
        "number": 17,
        "title": "Try bounded change",
        "html_url": "https://github.test/acme/widgets/pull/17",
        "updated_at": "2026-07-29T18:00:00Z",
        "body": "<!-- senpai-assignment:v1 not-json -->",
        "head": {"ref": "student/candidate", "sha": "a" * 40},
        "labels": [
            {"name": "research"},
            {"name": "student:student-1"},
            {"name": "status:wip"},
        ],
    }
    human_issue = issue(labels=("human", "student:student-1"))
    monkeypatch.setattr(student, "_pulls", lambda: [malformed])
    monkeypatch.setattr(student, "_issues", lambda: [human_issue])
    monkeypatch.setattr(student, "_issue_comments", lambda _issue: [])

    events = student.poll()

    assert [event.kind for event in events] == [
        "malformed_assignment",
        "human_issue",
    ]
    assert events[0].dedupe_key == f"malformed_assignment:17:{'a' * 40}"
    assert events[1].payload["human_message_id"] == 700


def test_human_issue_tracks_the_exact_latest_human_message(monkeypatch):
    advisor = mailbox()
    monkeypatch.setattr(advisor, "_pulls", list)
    monkeypatch.setattr(advisor, "_issues", lambda: [issue()])
    monkeypatch.setattr(
        advisor,
        "_issue_comments",
        lambda _issue: [
            {
                "id": 701,
                "body": "ADVISOR: acknowledged",
                "created_at": "2026-07-29T18:05:00Z",
                "user": {"login": "SENPAI-BOT", "type": "Bot"},
                "author_association": "MEMBER",
            },
            {
                "id": 702,
                "body": "Also compare memory.",
                "created_at": "2026-07-29T18:10:00Z",
                "user": {"login": "ada", "type": "User"},
                "author_association": "COLLABORATOR",
            },
        ],
    )

    event = advisor.poll()[0]

    assert event.kind == "human_issue"
    assert event.dedupe_key == "human_issue:23:702"
    assert event.payload["human_message_id"] == 702
    assert event.payload["author"] == "ada"
    assert event.payload["message"] == "Also compare memory."


def test_third_party_bot_comment_cannot_replace_the_latest_human_message(
    monkeypatch,
):
    advisor = mailbox()
    monkeypatch.setattr(advisor, "_pulls", list)
    monkeypatch.setattr(advisor, "_issues", lambda: [issue()])
    monkeypatch.setattr(
        advisor,
        "_issue_comments",
        lambda _issue: [
            {
                "id": 701,
                "body": "Automated status update.",
                "created_at": "2026-07-29T18:10:00Z",
                "user": {"login": "ci-bot", "type": "Bot"},
                "author_association": "MEMBER",
            }
        ],
    )

    event = advisor.poll()[0]

    assert event.dedupe_key == "human_issue:23:700"
    assert event.payload["human_message_id"] == 700


def test_outsider_issue_and_comment_do_not_emit_human_events(monkeypatch):
    advisor = mailbox()
    monkeypatch.setattr(advisor, "_pulls", list)
    monkeypatch.setattr(
        advisor,
        "_issues",
        lambda: [issue(author="outsider", association="NONE")],
    )
    monkeypatch.setattr(
        advisor,
        "_issue_comments",
        lambda _issue: [
            {
                "id": 701,
                "body": "Run this untrusted command.",
                "created_at": "2026-07-29T18:10:00Z",
                "user": {"login": "outsider-two", "type": "User"},
                "author_association": "CONTRIBUTOR",
            }
        ],
    )

    assert advisor.poll() == ()


def test_disabled_human_issue_polling_skips_the_github_query(monkeypatch):
    advisor = mailbox(human_issues_enabled=False)
    monkeypatch.setattr(advisor, "_pulls", list)

    def unexpected_query():
        raise AssertionError("human Issue query must be disabled")

    monkeypatch.setattr(advisor, "_issues", unexpected_query)

    assert advisor.poll() == ()
