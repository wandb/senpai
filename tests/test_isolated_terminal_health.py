import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def test_health_client_imports_without_openhands():
    script = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(ROOT)!r})

class RejectOpenHands(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openhands" or fullname.startswith("openhands."):
            raise RuntimeError("unexpected heavyweight import: " + fullname)
        return None

sys.meta_path.insert(0, RejectOpenHands())
import senpai_agent.isolated_terminal_health
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_lightweight_health_client_round_trips_to_the_real_server(tmp_path):
    from senpai_agent.isolated_terminal import IsolatedTerminalServer
    from senpai_agent.isolated_terminal_health import check_isolated_terminal_health

    socket_path = Path(tempfile.gettempdir()) / (
        f"senpai-{uuid4().hex[:12]}-terminal-health.sock"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with IsolatedTerminalServer(
        socket_path=socket_path,
        working_dir=workspace,
    ):
        check_isolated_terminal_health(socket_path)
