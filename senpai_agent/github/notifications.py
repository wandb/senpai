"""Explicit researcher recipients for GitHub issue notifications."""

import re
from collections.abc import Sequence


def normalize_researcher_handles(handles: Sequence[str]) -> tuple[str, ...]:
    """Accept GitHub user handles, with optional @ prefixes, once per user."""

    recipients = []
    for value in handles:
        if not value.strip():
            continue
        handle = value.strip().removeprefix("@").lower()
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){0,38}", handle):
            raise ValueError(
                f"invalid researcher GitHub handle {value!r}; use individual "
                "usernames, with an optional @ prefix"
            )
        recipients.append(handle)
    return tuple(dict.fromkeys(recipients))
