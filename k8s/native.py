#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch or operate persistent native Senpai roles on macOS."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai.launch.native_backend import (  # noqa: E402
    DEFAULT_NATIVE_RUN_ROOT,
    logs_native,
    run_from_payload,
    run_role,
    status_native,
    terminate_native,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("preflight-payload", "launch-payload", "run-role"):
        command = actions.add_parser(name)
        command.add_argument("path", type=Path)
    for name in ("status", "logs", "terminate"):
        command = actions.add_parser(name)
        command.add_argument("tag")
        command.add_argument("--run-root", default=DEFAULT_NATIVE_RUN_ROOT)
        if name == "logs":
            command.add_argument("--role", default="")
            command.add_argument("--tail", type=int, default=200)
            command.add_argument("--follow", action="store_true")
    args = parser.parse_args()

    try:
        if args.action in {"preflight-payload", "launch-payload"}:
            run_from_payload(args.action.removesuffix("-payload"), args.path)
        elif args.action == "run-role":
            run_role(args.path)
        elif args.action == "status":
            status_native(args.tag, args.run_root)
        elif args.action == "logs":
            logs_native(
                args.tag,
                args.run_root,
                role_key=args.role,
                follow=args.follow,
                tail=args.tail,
            )
        else:
            terminate_native(args.tag, args.run_root)
    except (
        ValueError,
        RuntimeError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
