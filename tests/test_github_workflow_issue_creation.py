from typing import cast

import pytest
from github_workflow_support import (
    REPO,
    AmbiguousMutationGitHub,
    FakeGitHub,
    comment,
    pull_request,
    workflow,
)

from senpai_agent.github.workflow import WorkflowPreconditionError

QUESTION = {
    "issue_id": "validation-budget",
    "title": "Increase the validation budget?",
    "body": "May we evaluate two more seeds?",
}


def test_closed_issue_retry_keeps_its_original_mentions_and_does_not_reopen():
    fake = FakeGitHub(pull_request())
    fake.collaborators = [
        {"login": "ada", "type": "User", "permissions": {"push": True}}
    ]
    workflow(fake).create_human_issue(
        **QUESTION, audience_label="schmidhuber", creator="advisor"
    )
    assert fake.issue is not None
    original_body = fake.issue["body"]
    fake.issue["state"] = "closed"
    fake.collaborators = [
        {"login": "grace", "type": "User", "permissions": {"push": True}}
    ]
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


def test_reply_to_a_senpai_opened_issue_does_not_repeat_maintainer_mentions():
    fake = FakeGitHub(pull_request())
    fake.collaborators = [
        {"login": "ada", "type": "User", "permissions": {"push": True}}
    ]
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
    assert "@ada" not in fake.comments[-1]["body"]
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
