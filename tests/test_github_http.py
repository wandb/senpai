import json
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from pydantic import SecretStr

from senpai_agent.github import http as github_http
from senpai_agent.github.http import GitHubReader, GitHubReadError, next_link


class Response:
    def __init__(self, payload, *, link=None):
        self.payload = payload
        self.headers = {"Link": link} if link else {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


def test_reader_follows_pagination_with_typed_auth(monkeypatch):
    responses = {
        "/items?per_page=1": Response(
            [{"id": 1}],
            link=(
                '<https://api.github.test/items?page=2>; rel="next", '
                '<https://api.github.test/items?page=2>; rel="last"'
            ),
        ),
        "/items?page=2": Response([{"id": 2}]),
    }

    def urlopen(github_request, timeout):
        assert github_request.headers["Authorization"] == "Bearer github-secret"
        assert timeout == 30
        parsed = urlsplit(github_request.full_url)
        key = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        return responses[key]

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(
        SecretStr("github-secret"),
        api_url="https://api.github.test",
    )

    assert reader.objects("/items?per_page=1") == [{"id": 1}, {"id": 2}]


def test_reader_rejects_foreign_pagination_origin(monkeypatch):
    monkeypatch.setattr(
        github_http.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(
            [],
            link='<https://attacker.example/items?page=2>; rel="next"',
        ),
    )

    with pytest.raises(GitHubReadError):
        GitHubReader(
            SecretStr("github-secret"),
            api_url="https://api.github.test",
        ).objects("/items")


def test_reader_rejects_pagination_cycles(monkeypatch):
    page = "https://api.github.test/items?page=1"
    monkeypatch.setattr(
        github_http.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(
            [],
            link=f'<{page}>; rel="next"',
        ),
    )

    with pytest.raises(GitHubReadError):
        GitHubReader(
            SecretStr("github-secret"),
            api_url="https://api.github.test",
        ).objects(page)


def test_reader_rejects_non_list_paginated_responses(monkeypatch):
    monkeypatch.setattr(
        github_http.request,
        "urlopen",
        lambda *_args, **_kwargs: Response({"items": []}),
    )

    with pytest.raises(GitHubReadError):
        GitHubReader(SecretStr("github-secret")).objects("/items")


def test_reader_errors_do_not_expose_token(monkeypatch):
    def fail(github_request, timeout):
        raise HTTPError(github_request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubReadError) as raised:
        GitHubReader(SecretStr("github-secret")).get("/user")

    assert "github-secret" not in str(raised.value)
    assert "/user" in str(raised.value)


@pytest.mark.parametrize("trusted_actor", [None, "configured[bot]"])
def test_reader_resolves_installation_actor_and_caches_it(monkeypatch, trusted_actor):
    calls = []

    def urlopen(github_request, timeout):
        calls.append(github_request)
        assert github_request.headers["Authorization"] == "Bearer github-secret"
        if github_request.full_url.endswith("/user"):
            raise HTTPError(github_request.full_url, 403, "forbidden", {}, None)
        assert github_request.full_url == "https://api.github.test/graphql"
        assert github_request.get_method() == "POST"
        assert json.loads(github_request.data) == {"query": "query { viewer { login } }"}
        return Response({"data": {"viewer": {"login": "research[bot]"}}})

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(
        SecretStr("github-secret"),
        api_url="https://api.github.test",
        trusted_actor=trusted_actor,
    )

    assert reader.actor() == (trusted_actor or "research[bot]")
    assert reader.actor() == (trusted_actor or "research[bot]")
    assert len(calls) == (0 if trusted_actor else 2)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"data": None},
        {"data": {"viewer": None}},
        {"data": {"viewer": {"login": ""}}},
        {"data": {"viewer": {"login": " "}}},
        {"data": {"viewer": {"login": 12}}},
        {
            "data": {"viewer": {"login": "unverified[bot]"}},
            "errors": [{"message": "Resource not accessible by integration"}],
        },
    ],
)
def test_reader_rejects_unverified_installation_actor(monkeypatch, payload):
    def urlopen(github_request, timeout):
        if github_request.full_url.endswith("/user"):
            raise HTTPError(github_request.full_url, 403, "forbidden", {}, None)
        return Response(payload)

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)

    with pytest.raises(GitHubReadError, match="GraphQL"):
        GitHubReader(SecretStr("github-secret")).actor()


@pytest.mark.parametrize("status", [401, 429, 500])
def test_reader_actor_preserves_other_http_failures(monkeypatch, status):
    calls = []

    def urlopen(github_request, timeout):
        calls.append(github_request.full_url)
        raise HTTPError(github_request.full_url, status, "failed", {}, None)

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)

    with pytest.raises(GitHubReadError, match=f"HTTP {status}"):
        GitHubReader(SecretStr("github-secret")).actor()
    assert calls == ["https://api.github.com/user"]


def test_next_link_extracts_only_the_next_relation():
    assert (
        next_link(
            '<https://api.github.test/items?page=1>; rel="prev", '
            '<https://api.github.test/items?page=3>; rel="next"'
        )
        == "https://api.github.test/items?page=3"
    )
    assert next_link(None) is None
