from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast
from urllib.error import URLError

import pytest
from pydantic import SecretStr

from senpai_agent.github.workflow import (
    GitHubAPIError,
    GitHubTransportError,
    GitHubWorkflow,
    HttpResponse,
    PullRequestSnapshot,
    ReconciliationError,
)
from github_workflow_support import (
    API_URL,
    HEAD_SHA,
    REPO,
    AmbiguousMutationGitHub,
    FakeGitHub,
    pull_request,
    workflow,
)


def test_github_code_uses_one_bounded_module_tree():
    agent_package = Path(__file__).parents[1] / "senpai_agent"
    modules = [
        *(agent_package / "github").rglob("*.py"),
        agent_package / "git_workflow.py",
    ]
    oversized = {
        str(path.relative_to(agent_package)): lines
        for path in modules
        if (lines := len(path.read_text().splitlines())) > 400
    }
    stray_github_paths = sorted(
        path.name for path in agent_package.glob("github_*")
    )

    assert oversized == {}
    assert stray_github_paths == []


def test_pull_request_returns_an_immutable_typed_snapshot():
    fake = FakeGitHub(pull_request(labels={"status:wip", "student:one"}))

    snapshot = workflow(fake).pull_request(7)

    assert snapshot == PullRequestSnapshot(
        number=7,
        node_id="PR_node_7",
        url=f"https://github.com/{REPO}/pull/7",
        head_sha=HEAD_SHA,
        base_ref="schmidhuber",
        head_ref="student-one/lower-lr",
        title="Try lower learning rate",
        body=cast(str, fake.pr["body"]),
        labels=("status:wip", "student:one"),
        draft=False,
        state="open",
        merged=False,
        mergeable=True,
        merge_commit_sha=None,
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.state = "closed"  # type: ignore[misc]


def test_pull_request_rejects_values_that_violate_the_response_contract():
    fake = FakeGitHub(pull_request())
    fake.pr["draft"] = 0

    with pytest.raises(ReconciliationError, match="invalid pull request"):
        workflow(fake).pull_request(7)


def test_workflow_authenticates_with_the_secret_but_does_not_render_it():
    fake = FakeGitHub(pull_request())
    client = workflow(fake)

    client.pull_request(7)

    assert fake.requests[0][3]["Authorization"] == "Bearer github-secret"
    assert "github-secret" not in repr(client)


def test_api_errors_do_not_expose_the_token():
    class FailingTransport:
        def request(self, method, url, *, headers, json_body=None):
            return HttpResponse(503, {"message": "unavailable"})

    client = GitHubWorkflow(
        REPO,
        SecretStr("never-show-this"),
        role="advisor",
        transport=FailingTransport(),
        api_url=API_URL,
    )

    with pytest.raises(GitHubAPIError) as raised:
        client.pull_request(7)

    assert "never-show-this" not in str(raised.value)
    assert "never-show-this" not in repr(raised.value)


def test_network_failure_raises_a_token_safe_transport_error(monkeypatch):
    def offline(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    client = GitHubWorkflow(
        REPO,
        SecretStr("never-show-this"),
        role="advisor",
        api_url=API_URL,
    )

    with pytest.raises(GitHubTransportError) as raised:
        client.pull_request(7)

    assert "never-show-this" not in str(raised.value)
    assert "never-show-this" not in repr(raised.value)


def test_draft_mutation_recovers_an_ambiguous_response_after_application():
    fake = AmbiguousMutationGitHub(
        pull_request(draft=False),
        fail_method="POST",
        fail_path="/graphql",
    )
    client = workflow(fake)

    changed = client._set_draft(client.pull_request(7), draft=True)

    assert changed is True
    assert fake.pr["draft"] is True


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"errors": [{"message": "denied"}]}, "returned errors"),
        ({"data": {"convertPullRequestToDraft": {}}}, "invalid GraphQL"),
        (
            {
                "data": {
                    "convertPullRequestToDraft": {
                        "pullRequest": {"id": "other", "isDraft": True}
                    }
                }
            },
            "wrong pull request",
        ),
        (
            {
                "data": {
                    "convertPullRequestToDraft": {
                        "pullRequest": {"id": "PR_node_7", "isDraft": False}
                    }
                }
            },
            "wrong draft state",
        ),
    ],
    ids=("errors", "malformed", "wrong-pull", "wrong-state"),
)
def test_draft_mutation_rejects_invalid_graphql_results(payload, message):
    class GraphQLResultGitHub(FakeGitHub):
        def request(self, method, url, *, headers, json_body=None):
            if method == "POST" and url.endswith("/graphql"):
                return HttpResponse(200, payload)
            return super().request(
                method,
                url,
                headers=headers,
                json_body=json_body,
            )

    fake = GraphQLResultGitHub(pull_request(draft=False))
    client = workflow(fake)

    with pytest.raises(ReconciliationError, match=message):
        client._set_draft(client.pull_request(7), draft=True)
