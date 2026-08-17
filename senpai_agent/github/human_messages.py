# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Trust classification for human-authored GitHub messages."""

from senpai_agent.models import authoritative_marker_line

TRUSTED_HUMAN_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


def is_trusted_human_author(*, author_type: str, association: str) -> bool:
    return (
        author_type == "User" and association in TRUSTED_HUMAN_ASSOCIATIONS
    )


def is_trusted_human_message(
    *,
    author: str,
    author_type: str,
    association: str,
    body: str,
    actor: str,
) -> bool:
    """Return whether a message is trusted human input, not Senpai output."""

    if not is_trusted_human_author(
        author_type=author_type,
        association=association,
    ):
        return False
    return not (
        author.casefold() == actor.casefold()
        and authoritative_marker_line(body).startswith("<!-- senpai-")
    )
