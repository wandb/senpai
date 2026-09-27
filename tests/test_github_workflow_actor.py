from urllib.parse import urlsplit

import pytest
from github_workflow_support import (
    API_URL,
    HEAD_SHA,
    REPO,
    FakeGitHub,
    experiment_result,
    human_issue,
    pull_request,
)
from pydantic import SecretStr

from senpai_agent.github.workflow import (
    GitHubAPIError,
    GitHubWorkflow,
    HttpResponse,
    ReconciliationError,
)


class InstallationGitHub(FakeGitHub):
    def __init__(self, *, user_status=403, viewer_response=None):
        super().__init__(pull_request(draft=True), actor_login="research[bot]")
        self.user_status = user_status
        self.viewer_response = viewer_response or HttpResponse(
            200, {"data": {"viewer": {"login": self.actor_login}}}
        )
        self.identity_requests = []

    def request(self, method, url, *, headers, json_body=None):
        path = urlsplit(url).path
        if method == "GET" and path == "/user":
            self.identity_requests.append(path)
            return HttpResponse(self.user_status)
        if method == "POST" and path == "/graphql" and json_body == {
            "query": "query { viewer { login } }"
        }:
            self.identity_requests.append(path)
            return self.viewer_response
        return super().request(method, url, headers=headers, json_body=json_body)


def installation_workflow(fake):
    return GitHubWorkflow(
        REPO,
        SecretStr("github-secret"),
        role="student",
        transport=fake,
        api_url=API_URL,
    )


def test_installation_token_can_publish_and_replay_without_configured_actor():
    fake = InstallationGitHub()
    client = installation_workflow(fake)
    result = experiment_result()

    first = client.submit_result(7, expected_head_sha=HEAD_SHA, result=result)
    replay = client.submit_result(7, expected_head_sha=HEAD_SHA, result=result)

    assert first.changed is True
    assert replay.changed is False
    assert len(fake.comments) == 1
    assert fake.comments[0]["user"]["login"] == "research[bot]"
    assert fake.identity_requests == ["/user", "/graphql"]


@pytest.mark.parametrize("operation", ["create", "reply"])
def test_installation_token_can_write_issues_without_an_actor_override_or_tag(operation):
    fake = InstallationGitHub()
    client = installation_workflow(fake)

    if operation == "create":
        result = client.create_human_issue(
            issue_id="budget",
            title="Confirm the experiment budget",
            body="Can we run another seed?",
            audience_label="student:fern",
            creator="fern",
        )
        body = fake.issue["body"]
    else:
        fake.issue = human_issue()
        result = client.respond_to_issue(
            7,
            human_message_id=700,
            audience_labels={"team"},
            responder="fern",
            response="I will investigate.",
        )
        body = fake.comments[-1]["body"]

    assert result.changed is True
    assert "@" not in body
    assert fake.identity_requests == ["/user", "/graphql", "/user"]
    assert len(fake.mutations) == 1


@pytest.mark.parametrize(
    "viewer_response",
    [
        HttpResponse(200, {"data": {"viewer": None}}),
        HttpResponse(
            200,
            {
                "data": {"viewer": {"login": "research[bot]"}},
                "errors": [{"message": "Forbidden"}],
            },
        ),
        HttpResponse(403),
    ],
)
def test_unverified_installation_actor_cannot_publish(viewer_response):
    fake = InstallationGitHub(viewer_response=viewer_response)

    with pytest.raises((GitHubAPIError, ReconciliationError)):
        installation_workflow(fake).submit_result(
            7, expected_head_sha=HEAD_SHA, result=experiment_result()
        )

    assert fake.identity_requests == ["/user", "/graphql"]
    assert fake.mutations == []


def test_invalid_credentials_do_not_attempt_installation_identity():
    fake = InstallationGitHub(user_status=401)

    with pytest.raises(GitHubAPIError, match="HTTP 401"):
        installation_workflow(fake).submit_result(
            7, expected_head_sha=HEAD_SHA, result=experiment_result()
        )

    assert fake.identity_requests == ["/user"]
    assert fake.mutations == []
