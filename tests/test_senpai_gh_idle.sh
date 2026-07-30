#!/bin/bash

set -e
set -o pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

source "$WORKDIR/plugins/senpai/scripts/senpai-gh.sh"

assert_json_eq() {
    local expected="$1" actual="$2"
    python3 - "$expected" "$actual" <<'PY'
import json
import sys

expected = json.loads(sys.argv[1])
actual = json.loads(sys.argv[2])
if actual != expected:
    raise SystemExit(f"expected {expected!r}, got {actual!r}")
PY
}

rest_labeled_pull_details() {
    printf '%s\n' '[{"number":1,"labels":[{"name":"status:wip"},{"name":"student:busy"}]}]'
}

mock_bin=$(mktemp -d "${TMPDIR:-/tmp}/senpai-idle-test.XXXXXX")
trap 'rm -rf "$mock_bin"' EXIT

cat > "$mock_bin/kubectl" <<'SH'
#!/bin/sh

case "$*" in
    *student=busy*) printf 'busy-pod\tTrue\n' ;;
    *student=ready*) printf 'ready-pod\tTrue\n' ;;
    *student=notready*) printf 'notready-pod\tFalse\n' ;;
    *student=dead*) : ;;
    *) exit 1 ;;
esac
SH
chmod +x "$mock_bin/kubectl"

export PATH="$mock_bin:$PATH"
export RESEARCH_TAG="test"

actual=$(list_idle_students "busy,ready,notready,dead" "advisor")
assert_json_eq '["ready"]' "$actual"

unset RESEARCH_TAG
actual=$(list_idle_students "busy,ready,notready,dead" "advisor")
assert_json_eq '["ready","notready","dead"]' "$actual"
