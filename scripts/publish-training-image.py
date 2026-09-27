#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Publish a local training image and print its immutable registry reference."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def docker(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["docker", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else sys.stderr,
    )
    return result.stdout or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dockerfile", type=Path, help="Dockerfile to build")
    source.add_argument("--archive", type=Path, help="Archive from docker image save")
    parser.add_argument("--tag", required=True, help="Destination registry/repository:tag")
    parser.add_argument(
        "--context", type=Path,
        help="Build context; defaults to Dockerfile directory",
    )
    parser.add_argument("--archive-image", help="Exact image tag saved in the archive")
    parser.add_argument(
        "--platform", default="linux/amd64",
        help="Training platform (default: linux/amd64)",
    )
    args = parser.parse_args()

    repository, separator, tag = args.tag.rpartition(":")
    if not separator or not tag or "/" in tag or "@" in args.tag:
        parser.error(
            "--tag must include an explicit destination tag, "
            "such as ghcr.io/team/train:v1"
        )
    path = args.dockerfile or args.archive
    if not path.is_file():
        parser.error(f"source file does not exist: {path}")
    if args.dockerfile:
        if args.archive_image:
            parser.error("--archive-image requires --archive")
        context = args.context or args.dockerfile.parent
        if not context.is_dir():
            parser.error(f"build context does not exist: {context}")
        with tempfile.TemporaryDirectory(prefix="senpai-training-image-") as directory:
            metadata = Path(directory) / "metadata.json"
            docker(
                "buildx", "build", "--platform", args.platform,
                "--file", str(args.dockerfile.resolve()), "--tag", args.tag,
                "--push", "--metadata-file", str(metadata), str(context.resolve()),
            )
            digest = json.loads(metadata.read_text())["containerimage.digest"]
    else:
        if args.context:
            parser.error("--context requires --dockerfile")
        if not args.archive_image:
            parser.error("--archive requires --archive-image")
        requested_platform = args.platform.split("/")
        if len(requested_platform) not in (2, 3) or not all(requested_platform):
            parser.error("archive --platform must use os/architecture[/variant]")
        # A platform-filtered load can skip tags already present in the local daemon.
        with tarfile.open(args.archive) as archive:
            try:
                manifest_file = archive.extractfile("manifest.json")
            except KeyError:
                parser.error("archive must contain manifest.json from docker image save")
            manifest = json.load(manifest_file)
            selected = [
                image for image in manifest
                if args.archive_image in (image.get("RepoTags") or [])
            ]
            if not selected:
                parser.error(
                    f"--archive-image {args.archive_image!r} is not saved in {args.archive}"
                )
            matching_configs = []
            for image in selected:
                config_bytes = archive.extractfile(image["Config"]).read()
                config = json.loads(config_bytes)
                platform = [
                    config.get(key, "") for key in ("os", "architecture", "variant")
                ]
                if platform[:len(requested_platform)] == requested_platform:
                    matching_configs.append(config_bytes)
            if len(matching_configs) != 1:
                parser.error(
                    f"archive must contain exactly one {args.platform} image "
                    f"tagged {args.archive_image!r}; found {len(matching_configs)}"
                )
            config_digest = "sha256:" + hashlib.sha256(matching_configs[0]).hexdigest()
        docker("load", "--input", str(args.archive.resolve()), "--platform", args.platform)
        docker("tag", args.archive_image, args.tag)
        docker("push", "--platform", args.platform, args.tag)
        manifest = json.loads(docker(
            "buildx", "imagetools", "inspect", args.tag,
            "--format", "{{json .Manifest}}", capture=True,
        ))
        digest = manifest["digest"]
        published = json.loads(docker(
            "buildx", "imagetools", "inspect", f"{repository}@{digest}",
            "--raw", capture=True,
        ))
        if published.get("config", {}).get("digest") != config_digest:
            parser.error("published image does not match the selected archive image")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"Docker returned an invalid image digest: {digest!r}")
    print(f"{repository}@{digest}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
