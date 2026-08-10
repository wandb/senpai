#!/usr/bin/env python3
"""Stdlib-only command executor copied into immutable role images."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


# JSON may expand one decoded character to six bytes. Two streams at this
# bound remain comfortably below the broker's 2 MiB response frame.
REPAIR_STREAM_LIMIT_CHARS = 128 * 1024


def execute_local_repair(
    command: str,
    cwd: Path,
    timeout_seconds: float,
) -> dict[str, int | str]:
    """Run one shell in a reaped process group and return bounded output."""

    try:
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as error:
        return {
            "exit_code": 126,
            "stdout": "",
            "stderr": f"repair command could not start ({type(error).__name__})",
        }
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {
            "exit_code": process.returncode,
            "stdout": stdout[-REPAIR_STREAM_LIMIT_CHARS:],
            "stderr": stderr[-REPAIR_STREAM_LIMIT_CHARS:],
        }
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {
            "exit_code": 124,
            "stdout": stdout[-REPAIR_STREAM_LIMIT_CHARS:],
            "stderr": (stderr + "\nrepair command timed out")[
                -REPAIR_STREAM_LIMIT_CHARS:
            ],
        }


def repair_executor_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a bounded repair command.")
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = execute_local_repair(sys.stdin.read(), args.cwd, args.timeout)
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(repair_executor_main())
