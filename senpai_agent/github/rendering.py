"""Stable Markdown rendering for complete pull-request records."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from senpai_agent.github.pull_requests import PullRequest


CLIPPED_NOTICE = "\n\n[Truncated to fit the review budget.]"


def render_pull_requests(repo: str, pull_requests: Sequence[PullRequest]) -> str:
    parts = [
        "# GitHub pull requests",
        "",
        f"Repository: `{repo}`",
        "",
        f"Selected pull requests: {len(pull_requests)}",
    ]
    if not pull_requests:
        parts.extend(("", "_No pull requests selected._"))
    for pull_request in pull_requests:
        parts.extend(("", "---", "", _render_pull_request(pull_request)))
    return "\n".join(parts) + "\n"


def _render_pull_request(pull_request: PullRequest) -> str:
    details = pull_request.details
    base = details.get("base") or {}
    head = details.get("head") or {}
    author = details.get("user") or {}
    sections = [
        f"## PR #{pull_request.number} — {details.get('title') or ''}",
        "",
        f"- URL: {details.get('html_url') or ''}",
        f"- Author: @{author.get('login') or 'unknown'}",
        f"- State: {details.get('state') or 'unknown'}",
        f"- Draft: {'yes' if details.get('draft') else 'no'}",
        f"- Created: {details.get('created_at') or 'unknown'}",
        f"- Updated: {details.get('updated_at') or 'unknown'}",
        f"- Base: `{base.get('ref') or ''}` (`{base.get('sha') or ''}`)",
        f"- Head: `{head.get('ref') or ''}` (`{head.get('sha') or ''}`)",
        "",
        "### PR body",
        "",
        _body(details.get("body")),
        "",
        f"### Issue comments ({len(pull_request.issue_comments)})",
        "",
        _render_entries(
            sorted(pull_request.issue_comments, key=_created_key),
            _render_issue_comment,
            "issue comments",
        ),
        "",
        f"### Review submissions ({len(pull_request.reviews)})",
        "",
        _render_entries(
            sorted(pull_request.reviews, key=_review_key),
            _render_review,
            "review submissions",
        ),
        "",
        f"### Inline review comments ({len(pull_request.inline_comments)})",
        "",
        _render_entries(
            sorted(pull_request.inline_comments, key=_created_key),
            _render_inline_comment,
            "inline review comments",
        ),
    ]
    return "\n".join(sections)


def render_review_context(pull_request: PullRequest, max_chars: int) -> str:
    """Render one PR's body and newest discussion in at most ``max_chars``.

    The body and each entry keep at most a quarter of the budget. The newest
    entries that fit follow the body in chronological order.
    """
    part_chars = max_chars // 4
    details = pull_request.details
    head = "\n".join((
        f"## PR #{pull_request.number} — {details.get('title') or ''}",
        "",
        f"- URL: {details.get('html_url') or ''}",
        "",
        "### PR body",
        "",
        clip_text(_body(details.get("body")), part_chars, CLIPPED_NOTICE),
    ))
    entries = sorted(
        (key(entry), render(entry, index))
        for records, key, render in (
            (pull_request.issue_comments, _created_key, _render_issue_comment),
            (pull_request.reviews, _review_key, _render_review),
            (pull_request.inline_comments, _created_key, _render_inline_comment),
        )
        for index, entry in enumerate(sorted(records, key=key), 1)
    )
    total = len(entries)
    heading = "### Recent discussion ({} of {} entries; {} older omitted)"
    remaining = max_chars - len(head) - len(heading.format(total, total, total)) - 2
    recent: list[str] = []
    for _key, text in reversed(entries):
        text = clip_text(text, part_chars, CLIPPED_NOTICE)
        remaining -= len(text) + 2
        if remaining < 0:
            break
        recent.insert(0, text)
    heading = heading.format(len(recent), total, total - len(recent))
    return "\n\n".join((head, heading, *recent))


def clip_text(text: str, limit: int, notice: str) -> str:
    """Return ``text``, or its start and ``notice`` in at most ``limit`` characters."""

    if len(text) <= limit:
        return text
    return text[:max(0, limit - len(notice))] + notice


def _render_entries(
    entries: list[dict[str, Any]],
    render: Callable[[dict[str, Any], int], str],
    label: str,
) -> str:
    if not entries:
        return f"_No {label}._"
    return "\n\n".join(render(entry, index) for index, entry in enumerate(entries, 1))


def _render_issue_comment(comment: dict[str, Any], index: int) -> str:
    user = comment.get("user") or {}
    return "\n".join(
        (
            f"#### Issue comment {index} — @{user.get('login') or 'unknown'}",
            "",
            f"- Created: {comment.get('created_at') or 'unknown'}",
            f"- Updated: {comment.get('updated_at') or 'unknown'}",
            f"- URL: {comment.get('html_url') or ''}",
            "",
            _body(comment.get("body")),
        )
    )


def _render_review(review: dict[str, Any], index: int) -> str:
    user = review.get("user") or {}
    return "\n".join(
        (
            f"#### Review {index} — @{user.get('login') or 'unknown'}",
            "",
            f"- State: {review.get('state') or 'unknown'}",
            f"- Submitted: {review.get('submitted_at') or 'unknown'}",
            f"- Commit: `{review.get('commit_id') or ''}`",
            f"- URL: {review.get('html_url') or ''}",
            "",
            _body(review.get("body")),
        )
    )


def _render_inline_comment(comment: dict[str, Any], index: int) -> str:
    user = comment.get("user") or {}
    line = comment.get("line")
    if line is None:
        line = comment.get("original_line")
    location = str(comment.get("path") or "")
    if line is not None:
        location += f":{line}"
    reply_to = comment.get("in_reply_to_id")
    return "\n".join(
        (
            f"#### Inline comment {index} — @{user.get('login') or 'unknown'}",
            "",
            f"- Location: `{location}`",
            f"- Side: {comment.get('side') or 'unknown'}",
            f"- Created: {comment.get('created_at') or 'unknown'}",
            f"- Updated: {comment.get('updated_at') or 'unknown'}",
            f"- Commit: `{comment.get('commit_id') or ''}`",
            f"- Reply to: {reply_to if reply_to is not None else 'none'}",
            f"- URL: {comment.get('html_url') or ''}",
            "",
            _body(comment.get("body")),
        )
    )


def _body(value: Any) -> str:
    return str(value) if value else "_No body._"


def _created_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item.get("created_at") or ""), int(item.get("id") or 0)


def _review_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item.get("submitted_at") or ""), int(item.get("id") or 0)
