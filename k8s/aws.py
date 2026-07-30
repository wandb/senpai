#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Inspect or terminate an AWS Senpai run."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai.launch.aws_backend import status_aws, terminate_aws  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "terminate"))
    parser.add_argument("tag")
    parser.add_argument("--state-root", default="~/.senpai/aws")
    parser.add_argument("--profile", default="")
    args = parser.parse_args()

    command = status_aws if args.action == "status" else terminate_aws
    try:
        command(args.tag, args.state_root, profile=args.profile)
    except (ValueError, RuntimeError) as error:
        sys.exit(f"ERROR: {error}")


if __name__ == "__main__":
    main()
