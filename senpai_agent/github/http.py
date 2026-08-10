"""Small authenticated GitHub reader shared by Senpai control-plane code."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from pydantic import SecretStr

MAX_GITHUB_RETRY_SECONDS = 3_600.0


class GitHubReadError(RuntimeError):
    """A GitHub read failed or returned an invalid response."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


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

    def get(self, path: str) -> object:
        """Return one decoded GitHub JSON response."""

        payload, _ = self._request(path)
        return payload

    def _request(self, path: str) -> tuple[object, str | None]:
        url = self._url(path)
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token.get_secret_value()}"
        github_request = request.Request(url, headers=headers)
        try:
            with request.urlopen(github_request, timeout=self._timeout) as response:
                return json.loads(response.read()), next_link(
                    response.headers.get("Link")
                )
        except HTTPError as error:
            raise GitHubReadError(
                f"GitHub GET {self._safe_path(url)} returned HTTP {error.code}",
                retry_after_seconds=_retry_after_seconds(error),
            ) from error
        except (URLError, TimeoutError) as error:
            raise GitHubReadError(
                f"GitHub GET {self._safe_path(url)} failed before an HTTP response"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitHubReadError(
                f"GitHub GET {self._safe_path(url)} returned invalid JSON"
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

    def objects_bounded(
        self,
        path: str,
        *,
        limit: int,
        stop: Callable[[dict[str, object]], bool] | None = None,
    ) -> tuple[list[dict[str, object]], bool]:
        """Read a sorted list only until its useful window or hard bound ends.

        The boolean is true when the remote sequence ended or ``stop`` ended the
        requested window. It is false when ``limit`` truncated matching data.
        """

        if limit <= 0:
            raise ValueError("GitHub object limit must be positive")
        objects: list[dict[str, object]] = []
        url: str | None = self._url(path)
        visited: set[str] = set()
        while url is not None:
            if url in visited:
                raise GitHubReadError("GitHub pagination contains a cycle")
            visited.add(url)
            page, url = self._request(url)
            if not isinstance(page, list) or any(
                not isinstance(item, dict) for item in page
            ):
                raise GitHubReadError("GitHub returned an invalid paginated list")
            for item in page:
                if stop is not None and stop(item):
                    return objects, True
                if len(objects) == limit:
                    return objects, False
                objects.append(item)
        return objects, True

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


def _retry_after_seconds(error: HTTPError) -> float | None:
    """Return GitHub's requested retry delay without retaining response headers."""

    headers = error.headers
    if headers is None:
        return None

    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return _bounded_retry_delay(float(retry_after))
        except ValueError:
            pass

    reset = headers.get("X-RateLimit-Reset")
    if reset is None or headers.get("X-RateLimit-Remaining") != "0":
        return None
    try:
        return _bounded_retry_delay(max(0.0, float(reset) - time.time()) + 1.0)
    except ValueError:
        return None


def _bounded_retry_delay(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return min(max(0.0, value), MAX_GITHUB_RETRY_SECONDS)
