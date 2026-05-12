"""Read-oriented GitHub helpers for the workshop."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from .config import WorkshopConfig


def repo_slug(repo_url: str) -> str:
    parsed = urllib.parse.urlparse(repo_url)
    if parsed.netloc == "github.com":
        slug = parsed.path.strip("/")
    else:
        slug = repo_url.split("github.com", 1)[-1].lstrip(":/")
    return slug.removesuffix(".git")


def github_api(config: WorkshopConfig, path: str, *, method: str = "GET", data: dict[str, Any] | None = None) -> Any:
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=body,
        method=method,
        headers={
            "authorization": f"Bearer {config.github_token}",
            "accept": "application/vnd.github+json",
            "content-type": "application/json",
            "user-agent": "senpai-workshop",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
    return json.loads(raw or b"{}")


def get_repo(config: WorkshopConfig) -> dict[str, Any]:
    return github_api(config, f"/repos/{repo_slug(config.target_repo_url)}")


def get_branch(config: WorkshopConfig, branch: str | None = None) -> dict[str, Any]:
    branch_name = branch or config.target_repo_branch
    quoted = urllib.parse.quote(branch_name, safe="")
    return github_api(config, f"/repos/{repo_slug(config.target_repo_url)}/branches/{quoted}")


def list_open_prs(config: WorkshopConfig, *, labels: list[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
    slug = repo_slug(config.target_repo_url)
    query = urllib.parse.urlencode({"state": "open", "per_page": str(limit)})
    prs = github_api(config, f"/repos/{slug}/pulls?{query}")
    if not labels:
        return prs
    wanted = set(labels)
    filtered = []
    for pr in prs:
        issue = github_api(config, f"/repos/{slug}/issues/{pr['number']}")
        names = {label["name"] for label in issue.get("labels", [])}
        if wanted.issubset(names):
            pr["label_names"] = sorted(names)
            filtered.append(pr)
    return filtered


def get_pr_comments(config: WorkshopConfig, number: int) -> list[dict[str, Any]]:
    slug = repo_slug(config.target_repo_url)
    issue_comments = github_api(config, f"/repos/{slug}/issues/{number}/comments")
    review_comments = github_api(config, f"/repos/{slug}/pulls/{number}/comments")
    return [
        {"kind": "issue", **comment}
        for comment in issue_comments
    ] + [
        {"kind": "review", **comment}
        for comment in review_comments
    ]


def require_mutations_enabled(allow_mutations: bool) -> None:
    if not allow_mutations:
        raise RuntimeError(
            "This workshop notebook disables GitHub mutations by default. "
            "Set ALLOW_MUTATIONS = True in the visible notebook cell if you really intend to mutate state."
        )


def routing_labels(advisor_branch: str, student_names: list[str]) -> dict[str, str]:
    return {
        advisor_branch: f"Advisor branch: {advisor_branch}",
        "status:wip": "Work in progress",
        "status:review": "Ready for advisor review",
        **{f"student:{name}": f"Assigned to student {name}" for name in student_names},
    }
