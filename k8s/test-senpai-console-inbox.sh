#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# Unit test for senpai_drain_inbox (console -> agent near-live steer).
# Run: bash k8s/test-senpai-console-inbox.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/senpai-console-inbox.sh"

fail() { echo "FAIL: $1"; exit 1; }

# 1. Inert when the env var is unset.
unset SENPAI_CONSOLE_INBOX_DIR || true
out="$(senpai_drain_inbox)"
[ -z "$out" ] || fail "expected empty output when inbox dir unset"

# 2. Inert when the dir doesn't exist.
export SENPAI_CONSOLE_INBOX_DIR="/tmp/senpai-inbox-missing-$$"
out="$(senpai_drain_inbox)"
[ -z "$out" ] || fail "expected empty output when inbox dir absent"

# 3. Drains directives and archives them.
tmp="$(mktemp -d)"
export SENPAI_CONSOLE_INBOX_DIR="$tmp"
printf 'Focus on the n=16 arm.' > "$tmp/steer-1.md"
printf 'Researcher updated DATASET_ANALYSIS.md.' > "$tmp/ping-2.md"
out="$(senpai_drain_inbox)"
echo "$out" | grep -q "Console directives" || fail "missing header"
echo "$out" | grep -q "n=16 arm" || fail "missing directive 1"
echo "$out" | grep -q "DATASET_ANALYSIS" || fail "missing directive 2"
# .md files are consumed (archived), so a second drain is empty.
ls "$tmp"/*.md >/dev/null 2>&1 && fail "directives not archived"
[ -d "$tmp/.archived" ] || fail "archive dir not created"
out2="$(senpai_drain_inbox)"
[ -z "$out2" ] || fail "second drain should be empty"
rm -rf "$tmp"

echo "PASS: senpai_drain_inbox"
