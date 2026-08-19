from pathlib import Path


def test_openhands_runtime_uses_bounded_modules():
    agent_package = Path(__file__).parents[1] / "senpai_agent"
    modules = [
        *(agent_package / "openhands").rglob("*.py"),
        agent_package / "openhands_runner.py",
    ]
    oversized = {
        str(path.relative_to(agent_package)): lines
        for path in modules
        if (lines := len(path.read_text().splitlines())) > 400
    }
    stray_openhands_paths = sorted(
        path.name
        for path in agent_package.glob("openhands_*")
        if path.name != "openhands_runner.py"
    )

    assert oversized == {}
    assert stray_openhands_paths == []
