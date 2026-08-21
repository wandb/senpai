# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Safe text conversion at model and persistence boundaries."""

import re


_SURROGATE = re.compile(r"[\ud800-\udfff]")


def utf8_text(value: object) -> str:
    """Return text that always has a valid UTF-8 encoding."""

    return _SURROGATE.sub(
        lambda match: f"\\u{ord(match.group()):04x}",
        str(value),
    )
