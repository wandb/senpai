"""Senpai-specific OpenHands security boundaries."""

from __future__ import annotations

from typing import Any


def _no_ambient_plugins(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {}


def disable_ambient_plugin_discovery() -> None:
    """Allow only the explicit, runner-owned plugin for Senpai conversations."""

    from openhands.sdk.conversation.impl import local_conversation

    # OpenHands 1.40.0 has no public switch for ambient plugin discovery. The
    # container asserts that exact SDK version, so replacing its imported
    # discovery function is a fail-closed boundary until the SDK exposes one.
    local_conversation.load_available_plugins = _no_ambient_plugins
