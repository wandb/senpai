#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Inspect, follow, or terminate a standalone Docker Senpai run."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai.launch.docker_backend import (  # noqa: E402
    logs_docker,
    status_docker,
    terminate_docker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("status", "logs", "terminate"):
        command = actions.add_parser(name)
        command.add_argument("tag")
        command.add_argument("--run-root", default="~/.senpai/runs")
        if name == "logs":
            command.add_argument("--role", default="")
            command.add_argument("--tail", type=int, default=200)
            command.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    try:
        if args.action == "status":
            status_docker(args.tag, args.run_root)
        elif args.action == "logs":
            logs_docker(
                args.tag,
                args.run_root,
                role_key=args.role,
                follow=args.follow,
                tail=args.tail,
            )
        else:
            terminate_docker(args.tag, args.run_root)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
