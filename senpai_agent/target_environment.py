"""Run shared dependency commands with the writable target interpreter."""

import shlex
import sys
import sysconfig
from importlib.metadata import entry_points
from pathlib import Path


def install_shared_console_scripts(target_env: Path) -> None:
    runtime_bin = Path(sysconfig.get_path("scripts"))
    target_bin = target_env / "bin"
    target_python = shlex.quote(str(target_bin / "python"))
    for entry_point in entry_points(group="console_scripts"):
        target_script = target_bin / entry_point.name
        runtime_script = shlex.quote(str(runtime_bin / entry_point.name))
        try:
            launcher = target_script.open("x", encoding="utf-8")
        except FileExistsError:
            # Preserve target installs, custom commands, and dangling symlinks.
            continue
        with launcher:
            launcher.write(
                f'#!/bin/sh\nexec {target_python} {runtime_script} "$@"\n'
            )
        target_script.chmod(0o755)


if __name__ == "__main__":
    install_shared_console_scripts(Path(sys.argv[1]))
