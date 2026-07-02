# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
#
# Console -> agent inbox (SENPAI Console Phase 2, near-live steer).
#
# The console writes directive files (a Supervisor steer, or a "researcher edited
# <doc>" ping from the file-edit flow) into $SENPAI_CONSOLE_INBOX_DIR. Each loop
# iteration the entrypoint drains any pending directives into that iteration's
# prompt and archives them, so a console message reaches the agent on its next
# heartbeat without scraping GitHub. Inert when the env var is unset or the dir
# is empty, so it never changes current behaviour unless the console is wired up.

senpai_drain_inbox() {
    local dir="${SENPAI_CONSOLE_INBOX_DIR:-}"
    [ -n "$dir" ] && [ -d "$dir" ] || return 0
    local archive="$dir/.archived"
    local out="" f
    for f in "$dir"/*.md; do
        [ -e "$f" ] || continue   # no matches -> the glob stays literal; skip it
        out+=$'\n\n'"$(cat "$f")"
        mkdir -p "$archive"
        mv "$f" "$archive/$(basename "$f").$(date +%s)" 2>/dev/null || rm -f "$f"
    done
    [ -n "$out" ] && printf '# Console directives (address these first)%s' "$out"
    return 0
}
