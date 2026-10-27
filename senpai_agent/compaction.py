# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Compaction settings shared by the launcher and the OpenHands runner."""

# Provider-native compaction (OpenAI stored Responses, Anthropic) and the
# OpenHands summarizing condenser both start at this many input tokens.
DEFAULT_COMPACTION_TRIGGER_TOKENS = 200_000
# The OpenHands condenser also compacts when the conversation view holds more
# than this many events the model sees: the system prompt, messages, actions
# and observations. Hook executions and other bookkeeping events are not in
# the view. Token pressure normally fires first; this cap bounds the history
# when token counting is unavailable.
DEFAULT_COMPACTION_TRIGGER_EVENTS = 400
# The condenser never summarizes the leading events: the system prompt and the
# task that opened the conversation.
CONDENSER_KEEP_FIRST = 4
# A condensation keeps half the window, which must hold the kept events and
# the summary.
MIN_COMPACTION_TRIGGER_EVENTS = 2 * (CONDENSER_KEEP_FIRST + 2)
