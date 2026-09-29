"""Shared trust boundary for student PR inventories."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from urllib.parse import quote

from senpai_agent.github.http import GitHubReadError


def authorized_pulls(
    pulls: Iterable[dict[str, object]],
    *,
    repo: str,
    get: Callable[[str], object],
) -> tuple[dict[str, object], ...]:
    """Keep same-repository PRs whose authors currently have write access."""

    permissions: dict[str, bool] = {}
    authorized: list[dict[str, object]] = []
    for pull in pulls:
        try:
            head = pull["head"]
            if not isinstance(head, dict):
                raise TypeError("GitHub mailbox returned an invalid object")
            head_repo = head["repo"]
            if not isinstance(head_repo, dict):
                raise TypeError("GitHub mailbox returned an invalid object")
            head_repo = head_repo["full_name"]
            author = pull["user"]
            if not isinstance(author, dict):
                raise TypeError("GitHub mailbox returned an invalid object")
            author = author["login"]
            if not isinstance(head_repo, str) or not isinstance(author, str):
                raise TypeError("GitHub pull request has invalid trust metadata")
        except (KeyError, TypeError) as error:
            print(
                "SENPAI_PULL_AUTHORIZATION_ERROR "
                f"pr={pull.get('number')!r} {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if head_repo.casefold() != repo.casefold():
            continue
        login = author.casefold()
        if login not in permissions:
            permission = get(
                f"/repos/{repo}/collaborators/{quote(author, safe='')}/permission"
            )
            if not isinstance(permission, dict) or permission.get("permission") not in (
                "admin", "write", "read", "none"
            ):
                raise GitHubReadError("GitHub returned an invalid collaborator permission")
            # GitHub maps maintain to write and triage to read in this field.
            permissions[login] = permission["permission"] in {"admin", "write"}
        if permissions[login]:
            authorized.append(pull)
    return tuple(authorized)
