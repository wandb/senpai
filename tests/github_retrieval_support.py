from urllib.parse import parse_qs, urlsplit

from senpai_agent.github import pull_requests

REPO = "acme/widgets"


def pull_request(
    number: int,
    *,
    head: str | None = None,
    body: str | None = None,
) -> dict:
    return {
        "number": number,
        "title": f"Experiment {number}",
        "body": body if body is not None else f"Complete PR body {number}",
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "head": {"sha": head or f"head-{number}"},
    }


def issue_comment(comment_id: int, body: str) -> dict:
    return {
        "id": comment_id,
        "body": body,
        "created_at": f"2026-06-01T10:00:{comment_id:02d}Z",
    }


def review(review_id: int, body: str) -> dict:
    return {
        "id": review_id,
        "body": body,
        "submitted_at": f"2026-06-02T10:00:{review_id:02d}Z",
    }


def inline_comment(comment_id: int, body: str) -> dict:
    return {
        "id": comment_id,
        "body": body,
        "created_at": f"2026-06-03T10:00:{comment_id:02d}Z",
    }


class FakeGitHubReader:
    def __init__(
        self,
        pulls: dict[int, dict],
        *,
        comments: dict[int, list[dict]] | None = None,
        reviews: dict[int, list[dict]] | None = None,
        inline_comments: dict[int, list[dict]] | None = None,
        search_pages: list[object] | None = None,
    ):
        self.pulls = pulls
        self.comments = comments or {}
        self.reviews = reviews or {}
        self.inline_comments = inline_comments or {}
        self.search_pages = search_pages or []
        self.search_query: str | None = None

    def factory(self, _token):
        return self

    def get(self, endpoint: str):
        path = urlsplit(endpoint).path
        prefix = f"/repos/{REPO}/pulls/"
        if path.startswith(prefix) and path.count("/") == 5:
            return self.pulls[int(path.removeprefix(prefix))]
        raise AssertionError(f"Unexpected GitHub endpoint: {endpoint}")

    def pages(self, endpoint: str):
        parsed = urlsplit(endpoint)
        if parsed.path != "/search/issues":
            raise AssertionError(f"Unexpected GitHub search endpoint: {endpoint}")
        self.search_query = parse_qs(parsed.query)["q"][0]
        return tuple(self.search_pages)

    def objects(self, endpoint: str):
        path = urlsplit(endpoint).path
        pull_prefix = f"/repos/{REPO}/pulls/"
        issue_prefix = f"/repos/{REPO}/issues/"
        if path.startswith(issue_prefix) and path.endswith("/comments"):
            number = int(path.removeprefix(issue_prefix).split("/", 1)[0])
            return self.comments.get(number, [])
        if path.startswith(pull_prefix) and path.endswith("/reviews"):
            number = int(path.removeprefix(pull_prefix).split("/", 1)[0])
            return self.reviews.get(number, [])
        if path.startswith(pull_prefix) and path.endswith("/comments"):
            number = int(path.removeprefix(pull_prefix).split("/", 1)[0])
            return self.inline_comments.get(number, [])
        raise AssertionError(f"Unexpected paginated GitHub endpoint: {endpoint}")


def install_fake_github(monkeypatch, fake: FakeGitHubReader) -> None:
    monkeypatch.setattr(pull_requests, "GitHubReader", fake.factory)
