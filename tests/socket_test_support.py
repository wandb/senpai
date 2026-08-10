from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


def short_socket_path(scope: Path, suffix: str, *, prefix: str = "senpai") -> Path:
    """Return a unique Unix-socket path below a short platform directory."""

    root = Path("/private/tmp") if sys.platform == "darwin" else Path("/tmp")
    digest = hashlib.sha256(str(scope).encode()).hexdigest()[:10]
    return root / f"{prefix}-{os.getpid()}-{digest}-{suffix}.sock"
