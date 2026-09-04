"""Small authenticated GitHub reader shared by Senpai control-plane code."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from pydantic import SecretStr


class GitHubReadError(RuntimeError):
    """A GitHub read failed or returned an invalid response."""


class GitHubRateLimitError(GitHubReadError):
    """GitHub asked the caller to defer further requests."""

    def __init__(self, message: str, *, retry_after_seconds: float):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True, slots=True)
class _CachedResponse:
    body: bytes
    next_url: str | None
    etag: str | None
    last_modified: str | None


class GitHubReader:
    """Read JSON objects and paginated lists from one GitHub API origin."""

    def __init__(
        self,
        token: SecretStr | None,
        *,
        api_url: str = "https://api.github.com",
        trusted_actor: str | None = None,
        timeout: int = 30,
    ):
        if token is not None and not isinstance(token, SecretStr):
            raise TypeError("token must be a SecretStr")
        if token is not None and not token.get_secret_value().strip():
            raise ValueError("token must not be empty")
        if trusted_actor is not None and not trusted_actor.strip():
            raise ValueError("trusted actor must not be empty")
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._origin = urlsplit(self._api_url)[:2]
        self._actor = trusted_actor
        self._timeout = timeout
        self._cache: dict[str, _CachedResponse] = {}
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Stop this reader before its next network request."""

        self._cancelled.set()

    def get(self, path: str) -> object:
        """Return one decoded GitHub JSON response."""

        payload, _ = self._request(path)
        return payload

    def _request(self, path: str) -> tuple[object, str | None]:
        if self._cancelled.is_set():
            raise GitHubReadError("GitHub read was cancelled")
        url = self._url(path)
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token.get_secret_value()}"
        cached = self._cache.get(url)
        if cached is not None:
            if cached.etag:
                headers["If-None-Match"] = cached.etag
            elif cached.last_modified:
                headers["If-Modified-Since"] = cached.last_modified
        github_request = request.Request(url, headers=headers)
        try:
            with request.urlopen(github_request, timeout=self._timeout) as response:
                body = response.read()
                payload = self._decode(body, url)
                next_url = next_link(response.headers.get("Link"))
                self._cache[url] = _CachedResponse(
                    body=body,
                    next_url=next_url,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                )
                return payload, next_url
        except HTTPError as error:
            if error.code == 304:
                if cached is None:
                    raise GitHubReadError(
                        "GitHub returned HTTP 304 without a cached response"
                    ) from error
                return self._decode(cached.body, url), cached.next_url
            rate_delay = _rate_limit_delay(error)
            if rate_delay is not None:
                raise GitHubRateLimitError(
                    f"GitHub GET {self._safe_path(url)} was rate limited",
                    retry_after_seconds=rate_delay,
                ) from error
            raise GitHubReadError(
                f"GitHub GET {self._safe_path(url)} returned HTTP {error.code}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise GitHubReadError(
                f"GitHub GET {self._safe_path(url)} failed before an HTTP response"
            ) from error

    def pages(self, path: str) -> tuple[object, ...]:
        """Return every page while rejecting cycles and foreign origins."""

        pages: list[object] = []
        url: str | None = self._url(path)
        visited: set[str] = set()
        while url is not None:
            if url in visited:
                raise GitHubReadError("GitHub pagination contains a cycle")
            visited.add(url)
            page, url = self._request(url)
            pages.append(page)
        return tuple(pages)

    def objects(self, path: str) -> list[dict[str, object]]:
        """Return all objects from a paginated list response."""

        objects: list[dict[str, object]] = []
        for page in self.pages(path):
            if not isinstance(page, list) or any(
                not isinstance(item, dict) for item in page
            ):
                raise GitHubReadError("GitHub returned an invalid paginated list")
            objects.extend(page)
        return objects

    def actor(self) -> str:
        """Return and cache the authenticated GitHub login."""

        if self._actor is None:
            user = self.get("/user")
            if not isinstance(user, dict) or not isinstance(user.get("login"), str):
                raise GitHubReadError("GitHub returned an invalid authenticated user")
            self._actor = user["login"]
        return self._actor

    def _url(self, path: str) -> str:
        url = (
            path
            if path.startswith(("https://", "http://"))
            else f"{self._api_url}/{path.lstrip('/')}"
        )
        if urlsplit(url)[:2] != self._origin:
            raise GitHubReadError("GitHub pagination returned an unexpected origin")
        return url

    @staticmethod
    def _safe_path(url: str) -> str:
        parsed = urlsplit(url)
        return f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path

    def _decode(self, body: bytes, url: str) -> object:
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitHubReadError(
                f"GitHub GET {self._safe_path(url)} returned invalid JSON"
            ) from error


def _rate_limit_delay(error: HTTPError) -> float | None:
    if error.code not in (403, 429):
        return None
    headers = error.headers or {}
    retry_after = _retry_after_seconds(headers.get("Retry-After"))
    remaining = headers.get("X-RateLimit-Remaining")
    reset = headers.get("X-RateLimit-Reset")
    body = error.read(4096) if error.fp is not None else b""
    message = body.decode(errors="ignore").casefold()
    rate_limited = (
        error.code == 429
        or retry_after is not None
        or remaining == "0"
        or "rate limit" in message
        or "abuse detection" in message
    )
    if not rate_limited:
        return None
    if retry_after is not None:
        return retry_after
    if reset is not None:
        try:
            return max(0, float(reset) - time.time())
        except ValueError:
            pass
    return 60


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def next_link(value: str | None) -> str | None:
    """Extract the RFC 8288 ``rel=next`` target from a GitHub Link header."""

    if value is None:
        return None
    for part in value.split(","):
        sections = [section.strip() for section in part.split(";")]
        if 'rel="next"' in sections[1:]:
            target = sections[0]
            if target.startswith("<") and target.endswith(">"):
                return target[1:-1]
    return None
