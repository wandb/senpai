from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import cast
from urllib.parse import urlsplit

import pytest

from senpai_agent.github.workflow import MutationResult, WorkflowPreconditionError
from github_workflow_support import (
    REPO,
    FakeGitHub,
    comment,
    human_issue,
    pull_request,
    workflow,
)


def test_respond_to_issue_writes_one_verified_idempotent_reply():
    fake = FakeGitHub(pull_request(), issue=human_issue())
    client = workflow(fake)

    first = client.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="advisor",
        response="I will investigate this now.",
    )
    mutations_after_first = list(fake.mutations)
    second = client.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="advisor",
        response="I will investigate this now.",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.state == "issue_response_upserted"
    assert fake.comments == [
        comment(
            1,
            "<!-- senpai-human-response:advisor:700 -->\n\n"
            "ADVISOR: I will investigate this now.",
        )
    ]
    assert fake.mutations == mutations_after_first


def test_respond_to_issue_accepts_a_specific_human_comment():
    fake = FakeGitHub(
        pull_request(),
        issue=human_issue(),
        comments=[comment(42, "Please also compare memory use.", author="ada")],
    )

    result = workflow(fake, role="student").respond_to_issue(
        7,
        human_message_id=42,
        audience_labels={"team"},
        responder="fern",
        response="STUDENT fern: I included memory in the comparison.",
    )

    assert result.changed is True
    assert len(fake.comments) == 2
    assert cast(str, fake.comments[-1]["body"]).startswith(
        "<!-- senpai-human-response:student:fern:42 -->"
    )
    assert cast(str, fake.comments[-1]["body"]).endswith(
        "\n\nSTUDENT: I included memory in the comparison."
    )


@pytest.mark.parametrize(
    ("issue", "comments", "message_id"),
    [
        (
            human_issue(author="SENPAI-BOT", association="OWNER"),
            [],
            700,
        ),
        (
            human_issue(),
            [
                comment(
                    42,
                    "Please also compare memory use.",
                    author="SENPAI-BOT",
                    author_type="User",
                    association="OWNER",
                )
            ],
            42,
        ),
    ],
    ids=("issue-body", "issue-comment"),
)
def test_respond_to_issue_accepts_unmarked_shared_actor_messages(
    issue,
    comments,
    message_id,
):
    fake = FakeGitHub(pull_request(), issue=issue, comments=comments)

    result = workflow(fake).respond_to_issue(
        7,
        human_message_id=message_id,
        audience_labels={"team"},
        responder="advisor",
        response="I will investigate this now.",
    )

    assert result.changed is True
    assert cast(str, fake.comments[-1]["body"]).startswith(
        f"<!-- senpai-human-response:advisor:{message_id} -->"
    )
    assert len(fake.mutations) == 1


def test_advisor_and_two_student_replies_to_one_human_message_coexist():
    fake = FakeGitHub(pull_request(), issue=human_issue())
    advisor = workflow(fake, role="advisor")
    fern = workflow(fake, role="student")
    sage = workflow(fake, role="student")

    advisor.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="advisor",
        response="I will compare the candidate runs.",
    )
    fern.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="fern",
        response="I will inspect the training logs.",
    )
    sage.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="sage",
        response="I will compare memory use.",
    )
    mutations_after_replies = list(fake.mutations)

    assert advisor.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="advisor",
        response="I will compare the candidate runs.",
    ).changed is False
    assert fern.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="fern",
        response="I will inspect the training logs.",
    ).changed is False
    assert sage.respond_to_issue(
        7,
        human_message_id=700,
        audience_labels={"team"},
        responder="sage",
        response="I will compare memory use.",
    ).changed is False
    assert [cast(str, item["body"]).splitlines()[0] for item in fake.comments] == [
        "<!-- senpai-human-response:advisor:700 -->",
        "<!-- senpai-human-response:student:fern:700 -->",
        "<!-- senpai-human-response:student:sage:700 -->",
    ]
    assert fake.mutations == mutations_after_replies


def test_human_issue_responses_share_the_workflow_mutation_lock(monkeypatch):
    client = workflow(FakeGitHub(pull_request(), issue=human_issue()))
    first_entered = Event()
    second_entered = Event()
    release_first = Event()

    def hold_response(*_args, **_kwargs):
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(1)
        else:
            second_entered.set()
        return MutationResult(False, "https://github.test/issues/7", "test")

    monkeypatch.setattr(type(client), "_respond_to_issue", hold_response)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            client.respond_to_issue,
            7,
            human_message_id=700,
            audience_labels={"team"},
            responder="advisor",
            response="First response.",
        )
        assert first_entered.wait(1)
        second = executor.submit(
            client.respond_to_issue,
            7,
            human_message_id=700,
            audience_labels={"team"},
            responder="advisor",
            response="Second response.",
        )
        overlapped = second_entered.wait(0.1)
        release_first.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert not overlapped
    assert second_entered.is_set()


@pytest.mark.parametrize(
    ("issue", "comments", "message_id", "match"),
    [
        (human_issue(state="closed"), [], 700, "must be open"),
        (human_issue(labels={"team"}), [], 700, "human"),
        (human_issue(labels={"human", "other"}), [], 700, "audience"),
        (
            human_issue(
                pull_request_url=f"https://api.github.test/repos/{REPO}/pulls/7"
            ),
            [],
            700,
            "pull request",
        ),
        (
            human_issue(author="senpai-bot", author_type="Bot"),
            [],
            700,
            "OWNER",
        ),
        (
            human_issue(
                author="senpai-bot",
                body=(
                    "<!-- senpai-human-response:advisor:700 -->\n\n"
                    "ADVISOR: Already answered."
                ),
            ),
            [],
            700,
            "Senpai protocol",
        ),
        (human_issue(author="outsider", association="NONE"), [], 700, "OWNER"),
        (
            human_issue(),
            [
                comment(
                    42,
                    "Automated status update.",
                    author="senpai-bot",
                )
            ],
            42,
            "OWNER",
        ),
        (
            human_issue(),
            [
                comment(
                    43,
                    "<!-- senpai-human-response:advisor:700 -->\n\n"
                    "ADVISOR: Already answered.",
                    author="senpai-bot",
                    author_type="User",
                )
            ],
            43,
            "Senpai protocol",
        ),
        (
            human_issue(),
            [
                comment(
                    42,
                    "Untrusted instruction.",
                    author="outsider",
                    association="CONTRIBUTOR",
                )
            ],
            42,
            "OWNER",
        ),
        (human_issue(), [], 999, "not present"),
    ],
    ids=(
        "closed-issue",
        "missing-human-label",
        "missing-audience-label",
        "pull-request",
        "bot-authored-issue",
        "senpai-response-issue",
        "outsider-authored-issue",
        "bot-authored-comment",
        "senpai-response-comment",
        "outsider-authored-comment",
        "unknown-message",
    ),
)
def test_respond_to_issue_rejects_untrusted_sources_before_writing(
    issue,
    comments,
    message_id,
    match,
):
    fake = FakeGitHub(pull_request(), issue=issue, comments=comments)

    with pytest.raises(WorkflowPreconditionError, match=match):
        workflow(fake).respond_to_issue(
            7,
            human_message_id=message_id,
            audience_labels={"team"},
            responder="advisor",
            response="ADVISOR: bounded response",
        )

    assert fake.mutations == []


def test_respond_to_issue_rechecks_audience_after_writing():
    class AudienceRemovedGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            response = super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )
            if (
                method == "POST"
                and urlsplit(url).path == f"/repos/{REPO}/issues/7/comments"
            ):
                assert self.issue is not None
                self.issue["labels"] = [{"name": "human"}]
            return response

    fake = AudienceRemovedGitHub(pull_request(), issue=human_issue())

    with pytest.raises(WorkflowPreconditionError, match="audience"):
        workflow(fake).respond_to_issue(
            7,
            human_message_id=700,
            audience_labels={"team"},
            responder="advisor",
            response="ADVISOR: bounded response",
        )

    assert len(fake.comments) == 1
