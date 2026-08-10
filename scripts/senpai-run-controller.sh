#!/bin/sh

# Run control-plane Python from immutable image paths, independent of the
# agent-controlled PATH, PYTHONPATH, user site, and current working directory.

set -eu

case "${1:-}" in
    advisor|student)
        module=senpai_agent.supervisor
        role="$1"
        shift
        set -- "$role" "$@"
        ;;
    operational-supervisor)
        module=senpai_agent.operational_supervisor
        shift
        set -- run "$@"
        ;;
    health)
        [ "$#" -eq 2 ] || exit 2
        module=senpai_agent.supervisor
        shift
        set -- health "$1"
        ;;
    *)
        echo "unsupported Senpai controller mode: ${1:-<missing>}" >&2
        exit 2
        ;;
esac

exec /opt/senpai-venv/bin/python -I -m "$module" "$@"
