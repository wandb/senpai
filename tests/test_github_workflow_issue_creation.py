from typing import cast
from urllib.parse import urlsplit

import pytest
from github_workflow_support import (
    REPO,
    AmbiguousMutationGitHub,
    FakeGitHub,
    comment,
    human_issue,
    pull_request,
    workflow,
)

from senpai_agent.github.workflow import (
    GitHubTransportError,
    HttpResponse,
    ReconciliationError,
    WorkflowPreconditionError,
)

QUESTION = {
    "issue_id": "validation-budget",
    "title": "Increase the validation budget?",
    "body": "May we evaluate two more seeds?",
}


@pytest.mark.parametrize("operation", ["create", "reply"])
@pytest.mark.parametrize(
    "owner_response",
    [
        HttpResponse(200, {"login": "senpai[bot]", "type": "Bot"}),
        HttpResponse(200, {"login": "acme", "type": "Organization"}),
        HttpResponse(200, {"login": "operator"}),
        HttpResponse(200, {"login": "operator @someone-else", "type": "User"}),
        HttpResponse(403),
        HttpResponse(503),
        GitHubTransportError("GET", "https://api.github.test/user"),
    ],
    ids=["bot", "organization", "missing-type", "invalid-login", "no-user", "unavailable", "transport"],
)
def test_issue_writes_succeed_without_an_optional_notification_owner(
    operation, owner_response
):
    class OwnerLookupGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            if method == "GET" and urlsplit(url).path == "/user":
                if isinstance(owner_response, Exception):
                    raise owner_response
                return owner_response
            return super().request(method, url, headers=headers, json_body=json_body)

    fake = OwnerLookupGitHub(
        pull_request(), issue=human_issue() if operation == "reply" else None
    )
    client = workflow(fake)
    if operation == "create":
        result = client.create_human_issue(
            **QUESTION, audience_label="schmidhuber", creator="advisor"
        )
        body = fake.issue["body"]
    else:
        result = client.respond_to_issue(
            7,
            human_message_id=700,
            audience_labels={"team"},
            responder="advisor",
            response="I will investigate.",
        )
        body = fake.comments[-1]["body"]

    assert result.changed is True
    assert "@" not in body
    assert len(fake.mutations) == 1


@pytest.mark.parametrize("foreign_origin", [True, False], ids=["foreign-page", "cycle"])
def test_issue_creation_rejects_unsafe_pagination_before_writing(foreign_origin):
    class InvalidPaginationGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            if method == "GET" and urlsplit(url).path == f"/repos/{REPO}/issues":
                next_page = "https://other.example/issues" if foreign_origin else url
                return HttpResponse(200, [], (("Link", f'<{next_page}>; rel="next"'),))
            return super().request(method, url, headers=headers, json_body=json_body)

    fake = InvalidPaginationGitHub(pull_request())
    with pytest.raises(ReconciliationError, match="origin|cycle"):
        workflow(fake).create_human_issue(
            **QUESTION, audience_label="schmidhuber", creator="advisor"
        )
    assert fake.mutations == []


def test_closed_issue_retry_keeps_its_original_mentions_and_does_not_reopen():
    fake = FakeGitHub(pull_request(), actor_login="operator", actor_type="User")
    workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )
    assert fake.issue is not None
    original_body = fake.issue["body"]
    fake.issue["state"] = "closed"
    fake.actor_type = "Bot"
    mutations = list(fake.mutations)

    result = workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )

    assert result.changed is False
    assert result.resource_url == fake.issue["html_url"]
    assert fake.issue["state"] == "closed"
    assert fake.issue["body"] == original_body
    assert fake.mutations == mutations


@pytest.mark.parametrize("field", ["title", "body"])
def test_issue_id_rejects_a_changed_question_without_writing(field):
    fake = FakeGitHub(pull_request())
    workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )
    mutations = list(fake.mutations)

    with pytest.raises(WorkflowPreconditionError, match="different question"):
        workflow(fake).create_human_issue(
            **(QUESTION | {field: "A different research question."}),
            audience_label="schmidhuber",
            creator="advisor",
        )

    assert fake.mutations == mutations


def test_reply_to_a_senpai_opened_issue_does_not_repeat_owner_mention():
    fake = FakeGitHub(pull_request(), actor_login="operator", actor_type="User")
    workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )
    fake.comments.append(comment(42, "Yes, evaluate two more seeds.", author="ada"))
    reply = {
        "human_message_id": 42,
        "audience_labels": {"schmidhuber"},
        "responder": "advisor",
        "response": "I will evaluate two more seeds.",
    }

    first = workflow(fake).respond_to_issue(7, **reply)
    mutations = list(fake.mutations)
    retry = workflow(fake).respond_to_issue(7, **reply)

    assert first.changed is True
    assert retry.changed is False
    assert cast(str, fake.comments[-1]["body"]).endswith(
        "ADVISOR: I will evaluate two more seeds."
    )
    assert "@operator" not in fake.comments[-1]["body"]
    assert fake.mutations == mutations


def test_issue_creation_recovers_when_github_applies_post_but_loses_response():
    fake = AmbiguousMutationGitHub(
        pull_request(), fail_method="POST", fail_path=f"/repos/{REPO}/issues"
    )

    created = workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )
    retry = workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )

    assert fake.failed is True
    assert created.changed is True
    assert retry.changed is False
    assert created.resource_url == retry.resource_url
    assert len(fake.mutations) == 1
