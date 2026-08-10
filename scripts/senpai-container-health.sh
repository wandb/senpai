#!/bin/sh

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

set -eu

lease_path="$1"
started_path="${SENPAI_BOOTSTRAP_STARTED_PATH:-/var/lib/senpai/.bootstrap-started}"
grace_seconds="${SENPAI_BOOTSTRAP_GRACE_SECONDS:-600}"
failure_path="${SENPAI_HEALTH_FAILURES_PATH:-/var/lib/senpai/.health-failures}"
failure_threshold="${SENPAI_HEALTH_FAILURE_THRESHOLD:-5}"
started="$(cat "$started_path" 2>/dev/null || true)"
now="$(date +%s)"

case "$started:$grace_seconds" in
    *[!0-9:]*|:*|*:) ;;
    *)
        if [ $((now - started)) -lt "$grace_seconds" ]; then
            rm -f "$failure_path"
            exit 0
        fi
        ;;
esac

if /usr/local/bin/senpai-run-controller health "$lease_path"; then
    rm -f "$failure_path"
    exit 0
fi

failures="$(cat "$failure_path" 2>/dev/null || true)"
case "$failures:$failure_threshold" in
    *[!0-9:]*|:*|*:) failures=0 ;;
esac
failures=$((failures + 1))
printf '%s\n' "$failures" > "$failure_path"
if [ "$failures" -lt "$failure_threshold" ]; then
    echo "Senpai supervisor health failure $failures/$failure_threshold" >&2
    exit 1
fi

echo "Senpai supervisor is unhealthy; terminating the container" >&2
kill -TERM 1
exit 1
