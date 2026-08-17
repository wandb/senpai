#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Inspect or terminate a recorded native EC2 Mac Senpai fleet."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai.launch.aws_mac_backend import (  # noqa: E402
    AwsCommandError,
    logs_aws_mac,
    status_aws_mac,
    terminate_aws_mac,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("status", "logs", "terminate"):
        command = actions.add_parser(name)
        command.add_argument("tag")
        command.add_argument("--state-root", default="~/.senpai/aws")
        command.add_argument("--profile", default="")
        if name == "logs":
            command.add_argument("--role", default="advisor")
            command.add_argument("--tail", type=int, default=200)
    args = parser.parse_args()

    try:
        if args.action == "status":
            status_aws_mac(
                args.tag,
                args.state_root,
                profile=args.profile,
            )
        elif args.action == "logs":
            logs_aws_mac(
                args.tag,
                args.state_root,
                profile=args.profile,
                role_key=args.role,
                tail=args.tail,
            )
        else:
            terminate_aws_mac(
                args.tag,
                args.state_root,
                profile=args.profile,
            )
    except (AwsCommandError, OSError, ValueError, RuntimeError) as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
