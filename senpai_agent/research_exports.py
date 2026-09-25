"""Generated research exports with descriptor-bound, symlink-safe writes."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def export_directory(path: Path) -> Iterator[int]:
    """Pin every directory component; never follow a symlink during a write."""
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


class ResearchExports:
    def __init__(self, root: Path, descriptor: int, *, secrets: Sequence[str] = ()):
        self.root, self.descriptor = root, descriptor
        self.secrets = tuple(
            json.dumps(secret, ensure_ascii=False)[1:-1] for secret in secrets if secret
        )

    @contextmanager
    def create(self, suffix: str):
        name = f"{uuid4().hex}.{suffix}"
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                yield self.root / name, stream
        except BaseException:
            os.unlink(name, dir_fd=self.descriptor)
            raise

    def jsonl(self, rows: Iterable[object]) -> tuple[str, int]:
        count = 0
        with self.create("jsonl") as (path, stream):
            for row in rows:
                encoded = json.dumps(row, ensure_ascii=False)
                for secret in self.secrets:
                    encoded = encoded.replace(secret, "[REDACTED]")
                stream.write((encoded + "\n").encode())
                count += 1
        return str(path), count
