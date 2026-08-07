"""Complete, context-bounded GitHub pull-request retrieval."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from pydantic import SecretStr

from senpai_agent.github.artifacts import store_pull_requests
from senpai_agent.github.http import GitHubReader, GitHubReadError
from senpai_agent.github.rendering import render_pull_requests


@dataclass(frozen=True)
class PRManifestEntry:
    """Compact identity for one rendered pull request."""

    number: int
    title: str
    head_sha: str
    url: str


@dataclass(frozen=True)
class PRRetrievalResult:
    """Markdown returned inline or written to one external artifact."""

    manifest: tuple[PRManifestEntry, ...]
    markdown: str | None
    path: Path | None


@dataclass(frozen=True)
class PullRequest:
    """Complete GitHub record used to render one pull request."""

    details: dict[str, Any]
    issue_comments: tuple[dict[str, Any], ...]
    reviews: tuple[dict[str, Any], ...]
    inline_comments: tuple[dict[str, Any], ...]

    @property
    def number(self) -> int:
        return int(self.details["number"])

    @property
    def manifest_entry(self) -> PRManifestEntry:
        return PRManifestEntry(
            number=self.number,
            title=str(self.details.get("title") or ""),
            head_sha=str((self.details.get("head") or {}).get("sha") or ""),
            url=str(self.details.get("html_url") or ""),
        )


def get_prs(
    repo: str,
    *,
    numbers: Sequence[int] = (),
    date_range: tuple[str | date, str | date] | None = None,
    search: str | None = None,
    max_inline_prs: int = 5,
    artifact_dir: str | Path | None = None,
    target_workspace: str | Path | None = None,
    token: SecretStr | None = None,
) -> PRRetrievalResult:
    """Retrieve selected pull requests as complete Markdown.

    Explicit numbers are combined with results from ``search`` and the inclusive
    PR creation ``date_range``. Each selected PR includes its full body, issue
    comments, review submissions, and inline review comments from every API page.

    At most ``max_inline_prs`` PRs are returned inline. Larger selections are
    written to one deterministic artifact outside ``target_workspace``. Raising
    the default limit above five warns because it can pollute agent context.

    ``token`` must be a typed credential. Ambient GitHub token variables are
    deliberately ignored.
    """
    _validate_repo(repo)
    explicit_numbers = _normalize_numbers(numbers)
    normalized_range = _normalize_date_range(date_range)
    normalized_search = search.strip() if search and search.strip() else None
    if not explicit_numbers and normalized_range is None and normalized_search is None:
        raise ValueError("get_prs requires at least one selector")
    if isinstance(max_inline_prs, bool) or max_inline_prs < 0:
        raise ValueError("max_inline_prs must be a non-negative integer")
    if max_inline_prs > 5:
        warnings.warn(
            "Raising max_inline_prs above 5 risks polluting agent context.",
            UserWarning,
            stacklevel=2,
        )
    if token is not None and not isinstance(token, SecretStr):
        raise TypeError("token must be a SecretStr")

    reader = GitHubReader(token)
    selected_numbers = set(explicit_numbers)
    if normalized_range is not None or normalized_search is not None:
        selected_numbers.update(
            _search_pr_numbers(reader, repo, normalized_range, normalized_search)
        )

    pull_requests = tuple(
        _fetch_pull_request(reader, repo, number)
        for number in sorted(selected_numbers)
    )
    markdown = render_pull_requests(repo, pull_requests)
    manifest = tuple(pr.manifest_entry for pr in pull_requests)
    if len(pull_requests) <= max_inline_prs:
        return PRRetrievalResult(manifest=manifest, markdown=markdown, path=None)

    path = store_pull_requests(
        repo=repo,
        numbers=explicit_numbers,
        date_range=normalized_range,
        search=normalized_search,
        manifest=manifest,
        markdown=markdown,
        artifact_dir=artifact_dir,
        target_workspace=target_workspace,
    )
    return PRRetrievalResult(manifest=manifest, markdown=None, path=path)


def _validate_repo(repo: str) -> None:
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repo must use owner/name form")


def _normalize_numbers(numbers: Sequence[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError("PR numbers must be positive integers")
        normalized.add(number)
    return tuple(sorted(normalized))


def _normalize_date_range(
    value: tuple[str | date, str | date] | None,
) -> tuple[str, str] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("date_range must contain exactly a start and end date")
    start, end = (_iso_date(item) for item in value)
    if start > end:
        raise ValueError("date_range start must not be after its end")
    return start, end


def _iso_date(value: str | date) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid ISO date: {value!r}") from error


def _search_pr_numbers(
    reader: GitHubReader,
    repo: str,
    date_range: tuple[str, str] | None,
    search: str | None,
) -> tuple[int, ...]:
    query = [f"repo:{repo}", "is:pr"]
    if search is not None:
        query.append(search)
    if date_range is not None:
        query.append(f"created:{date_range[0]}..{date_range[1]}")
    endpoint = "/search/issues?" + urlencode({"q": " ".join(query), "per_page": 100})
    numbers: set[int] = set()
    for page in reader.pages(endpoint):
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise GitHubReadError("GitHub returned invalid issue search results")
        numbers.update(
            int(item["number"])
            for item in page["items"]
            if isinstance(item, dict)
        )
    return tuple(sorted(numbers))


def _fetch_pull_request(
    reader: GitHubReader,
    repo: str,
    number: int,
) -> PullRequest:
    root = f"/repos/{repo}"
    details = reader.get(f"{root}/pulls/{number}")
    if not isinstance(details, dict):
        raise TypeError(f"GitHub returned an invalid PR #{number} response")
    return PullRequest(
        details=details,
        issue_comments=tuple(
            reader.objects(f"{root}/issues/{number}/comments?per_page=100")
        ),
        reviews=tuple(reader.objects(f"{root}/pulls/{number}/reviews?per_page=100")),
        inline_comments=tuple(
            reader.objects(f"{root}/pulls/{number}/comments?per_page=100")
        ),
    )
