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


def test_reader_preserves_github_rate_limit_reset(monkeypatch):
    monkeypatch.setattr(github_http.time, "time", lambda: 1_000)

    def fail(github_request, timeout):
        raise HTTPError(
            github_request.full_url,
            403,
            "rate limited",
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1120"},
            None,
        )

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubReadError) as raised:
        GitHubReader(SecretStr("github-secret")).get("/user")

    assert raised.value.retry_after_seconds == 121
    assert "github-secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (None, None),
        ({"Retry-After": "inf"}, None),
        ({"Retry-After": "999999999"}, 3_600),
    ],
)
def test_reader_bounds_invalid_or_unsafe_retry_headers(
    monkeypatch,
    headers,
    expected,
):
    def fail(github_request, timeout):
        raise HTTPError(
            github_request.full_url,
            403,
            "rate limited",
            headers,
            None,
        )

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubReadError) as raised:
        GitHubReader(SecretStr("github-secret")).get("/user")

    assert raised.value.retry_after_seconds == expected


def test_next_link_extracts_only_the_next_relation():
    assert (
        next_link(
            '<https://api.github.test/items?page=1>; rel="prev", '
            '<https://api.github.test/items?page=3>; rel="next"'
        )
        == "https://api.github.test/items?page=3"
    )
    assert next_link(None) is None
