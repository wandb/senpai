import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from pydantic import SecretStr

from senpai_agent.github import http as github_http
from senpai_agent.github.http import (
    GitHubRateLimitError,
    GitHubReader,
    GitHubReadError,
    next_link,
)


class Response:
    def __init__(self, payload, *, link=None, headers=None):
        self.payload = payload
        self.headers = dict(headers or {})
        if link:
            self.headers["Link"] = link

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return (
            self.payload
            if isinstance(self.payload, bytes)
            else json.dumps(self.payload).encode()
        )


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


def test_cancelled_reader_stops_before_the_next_request(monkeypatch):
    calls = []
    monkeypatch.setattr(
        github_http.request,
        "urlopen",
        lambda *_args, **_kwargs: calls.append(True),
    )
    reader = GitHubReader(SecretStr("token"), api_url="https://api.github.test")

    reader.cancel()

    with pytest.raises(GitHubReadError, match="cancelled"):
        reader.get("/items")
    assert calls == []


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


def test_reader_reuses_an_etag_response_without_exposing_mutable_cache(
    monkeypatch,
):
    calls = 0

    def urlopen(github_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        if calls == 1:
            assert github_request.get_header("If-none-match") is None
            return Response([{"id": 1}], headers={"ETag": '"version-1"'})
        assert github_request.get_header("If-none-match") == '"version-1"'
        raise HTTPError(
            github_request.full_url,
            304,
            "not modified",
            {},
            None,
        )

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(None, api_url="https://api.github.test")

    first = reader.objects("/items")
    first[0]["id"] = 99

    assert reader.objects("/items") == [{"id": 1}]
    assert calls == 2


def test_reader_uses_last_modified_when_etag_is_absent(monkeypatch):
    calls = 0

    def urlopen(github_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        if calls == 1:
            assert github_request.get_header("If-modified-since") is None
            return Response(
                {"id": 1},
                headers={"Last-Modified": "Wed, 19 Aug 2026 12:00:00 GMT"},
            )
        assert (
            github_request.get_header("If-modified-since")
            == "Wed, 19 Aug 2026 12:00:00 GMT"
        )
        raise HTTPError(github_request.full_url, 304, "not modified", {}, None)

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(None, api_url="https://api.github.test")

    assert reader.get("/item") == {"id": 1}
    assert reader.get("/item") == {"id": 1}


def test_invalid_json_is_not_cached_with_its_validator(monkeypatch):
    calls = 0

    def urlopen(github_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 30
        assert github_request.get_header("If-none-match") is None
        if calls == 1:
            return Response(b"not-json", headers={"ETag": '"broken"'})
        return Response({"id": 1}, headers={"ETag": '"valid"'})

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(None, api_url="https://api.github.test")

    with pytest.raises(GitHubReadError, match="invalid JSON"):
        reader.get("/item")

    assert reader.get("/item") == {"id": 1}


def test_paginated_cache_preserves_the_next_link_after_304(monkeypatch):
    calls = []

    def urlopen(github_request, timeout):
        assert timeout == 30
        parsed = urlsplit(github_request.full_url)
        key = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        calls.append(key)
        if calls.count(key) == 1:
            if key == "/items":
                return Response(
                    [{"id": 1}],
                    link='<https://api.github.test/items?page=2>; rel="next"',
                    headers={"ETag": '"page-1"'},
                )
            return Response([{"id": 2}], headers={"ETag": '"page-2"'})
        raise HTTPError(github_request.full_url, 304, "not modified", {}, None)

    monkeypatch.setattr(github_http.request, "urlopen", urlopen)
    reader = GitHubReader(None, api_url="https://api.github.test")

    assert reader.objects("/items") == [{"id": 1}, {"id": 2}]
    assert reader.objects("/items") == [{"id": 1}, {"id": 2}]
    assert calls == [
        "/items",
        "/items?page=2",
        "/items",
        "/items?page=2",
    ]


def test_reader_surfaces_rate_limit_retry_after_without_secrets(monkeypatch):
    def fail(github_request, timeout):
        assert timeout == 30
        raise HTTPError(
            github_request.full_url,
            429,
            "too many requests",
            {"Retry-After": "120"},
            BytesIO(b'{"message":"secondary rate limit"}'),
        )

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubRateLimitError) as raised:
        GitHubReader(SecretStr("github-secret")).get("/user")

    assert raised.value.retry_after_seconds == 120
    assert "github-secret" not in str(raised.value)


def test_reader_uses_rate_limit_reset_when_retry_after_is_absent(monkeypatch):
    monkeypatch.setattr(github_http.time, "time", lambda: 1_000)

    def fail(github_request, timeout):
        assert timeout == 30
        raise HTTPError(
            github_request.full_url,
            403,
            "rate limited",
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1120"},
            BytesIO(b'{"message":"API rate limit exceeded"}'),
        )

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubRateLimitError) as raised:
        GitHubReader(None).get("/user")

    assert raised.value.retry_after_seconds == 120


def test_reader_parses_http_date_retry_after(monkeypatch):
    monkeypatch.setattr(github_http.time, "time", lambda: 1_777_344_400)

    def fail(github_request, timeout):
        assert timeout == 30
        raise HTTPError(
            github_request.full_url,
            429,
            "rate limited",
            {"Retry-After": "Tue, 28 Apr 2026 02:48:40 GMT"},
            BytesIO(b'{"message":"secondary rate limit"}'),
        )

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubRateLimitError) as raised:
        GitHubReader(None).get("/user")

    assert raised.value.retry_after_seconds == 120


def test_ordinary_permission_403_is_not_misclassified_as_rate_limit(monkeypatch):
    def fail(github_request, timeout):
        assert timeout == 30
        raise HTTPError(
            github_request.full_url,
            403,
            "forbidden",
            {},
            BytesIO(b'{"message":"resource not accessible"}'),
        )

    monkeypatch.setattr(github_http.request, "urlopen", fail)

    with pytest.raises(GitHubReadError) as raised:
        GitHubReader(None).get("/private")

    assert not isinstance(raised.value, GitHubRateLimitError)


def test_next_link_extracts_only_the_next_relation():
    assert (
        next_link(
            '<https://api.github.test/items?page=1>; rel="prev", '
            '<https://api.github.test/items?page=3>; rel="next"'
        )
        == "https://api.github.test/items?page=3"
    )
    assert next_link(None) is None
