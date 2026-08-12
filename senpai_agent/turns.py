# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Shared outcomes for bounded OpenHands turns."""

from enum import IntEnum


class TurnOutcome(IntEnum):
    PROCESSED = 0
    FAILED = 1
    PAUSED_TIMEOUT = 75

    @classmethod
    def from_exit_code(cls, exit_code: int) -> "TurnOutcome":
        if exit_code == cls.PROCESSED:
            return cls.PROCESSED
        if exit_code == cls.PAUSED_TIMEOUT:
            return cls.PAUSED_TIMEOUT
        return cls.FAILED
