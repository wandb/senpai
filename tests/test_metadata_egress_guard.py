from contextlib import nullcontext
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "scripts" / "senpai-metadata-egress-guard.py"
SPEC = importlib.util.spec_from_file_location("senpai_metadata_egress_guard", GUARD_PATH)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_guard_fails_closed_when_a_metadata_endpoint_accepts_tcp(
    monkeypatch, capsys
):
    monkeypatch.setattr(guard.socket, "create_connection", lambda *_a, **_k: nullcontext())

    result = guard.main(["--endpoint", "169.254.169.254:80", "--timeout", "0.1"])

    assert result == 1
    error = capsys.readouterr().err
    assert "169.254.169.254:80" in error
    assert "reachable" in error


def test_guard_allows_startup_when_the_endpoint_cannot_accept_tcp(
    monkeypatch, capsys
):
    def unavailable(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(guard.socket, "create_connection", unavailable)

    result = guard.main(["--endpoint", "169.254.169.254:80", "--timeout", "0.1"])

    assert result == 0
    assert capsys.readouterr().err == ""
