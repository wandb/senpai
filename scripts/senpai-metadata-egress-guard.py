#!/usr/bin/env python3
"""Fail startup if a cloud metadata endpoint accepts a TCP connection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import socket
import sys


DEFAULT_ENDPOINTS = (
    "169.254.169.254:80",
    "169.254.170.2:80",
    "[fd00:ec2::254]:80",
)


def parse_endpoint(value: str) -> tuple[str, int]:
    if value.startswith("["):
        host, separator, port = value[1:].partition("]:")
    else:
        host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError(f"invalid host:port endpoint: {value}")
    try:
        number = int(port)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid endpoint port: {value}") from error
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError(f"endpoint port is out of range: {value}")
    return host, number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    endpoints = args.endpoint or DEFAULT_ENDPOINTS
    reachable = []
    for value in endpoints:
        host, port = parse_endpoint(value)
        try:
            with socket.create_connection((host, port), timeout=args.timeout):
                reachable.append(value)
        except OSError:
            pass
    if reachable:
        print(
            "ERROR: cloud metadata endpoint is reachable despite the required "
            f"egress boundary: {', '.join(reachable)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
