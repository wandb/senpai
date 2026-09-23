# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Validation for optional OpenHands model transport overrides."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

MODEL_API_MODES = frozenset({"auto", "chat", "responses"})


def model_base_url(value: str | None) -> str | None:
    """Return a validated HTTP(S) API base URL, or ``None`` when unset."""

    if not value or not value.strip():
        return None
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("model_base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("model_base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("model_base_url must not contain a query or fragment")
    return value.rstrip("/")


def model_api_mode(value: str | None) -> str | None:
    """Return a validated OpenHands API mode, or ``None`` when unset."""

    if not value or not value.strip():
        return None
    value = value.strip().lower()
    if value not in MODEL_API_MODES:
        choices = ", ".join(sorted(MODEL_API_MODES))
        raise ValueError(f"model_api_mode must be one of: {choices}")
    return value


def model_extra_headers(value: str, *, source: str) -> dict[str, str]:
    """Parse and validate a secret JSON object of HTTP request headers."""

    try:
        headers = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} must contain a JSON object") from error
    if not isinstance(headers, dict) or any(
        not isinstance(name, str) or not isinstance(header_value, str)
        for name, header_value in headers.items()
    ):
        raise ValueError(f"{source} must contain a JSON object of string headers")

    normalized_names: set[str] = set()
    for name, header_value in headers.items():
        if not name or name != name.strip():
            raise ValueError(f"{source} contains a blank or padded header name")
        normalized = name.casefold()
        if normalized in normalized_names:
            raise ValueError(f"{source} contains duplicate header names")
        normalized_names.add(normalized)
        if "\r" in name or "\n" in name or "\r" in header_value or "\n" in header_value:
            raise ValueError(f"{source} header names and values must not contain CR/LF")
    return headers
