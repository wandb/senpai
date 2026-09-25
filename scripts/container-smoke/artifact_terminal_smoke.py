"""Offline smoke: full Exa evidence remains readable through the real final terminal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

root = Path(sys.argv[1]).resolve()
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
sys.path.insert(0, str(root))

from pydantic import SecretStr
from openhands.tools.terminal import TerminalAction
from senpai_agent import exa_tool
from senpai_agent.tools import SenpaiTerminalTool
from senpai_agent.research_exports import ResearchExports, export_directory

with tempfile.TemporaryDirectory(prefix="pr3515-cross-artifact-") as tmp:
    directory = Path(tmp).resolve()
    workspace = directory / "target"
    workspace.mkdir()
    output = directory / "outputs"
    target_env = directory / "target-env"
    subprocess.run(
        [sys.executable, "-P", "-m", "venv", "--without-pip", str(target_env)],
        check=True,
    )
    os.environ["SENPAI_TARGET_PYTHON_ENV"] = str(target_env)
    os.environ.pop("PYTHONSAFEPATH", None)
    state = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(workspace)),
        env_observation_persistence_dir=str(output),
    )
    conversation = SimpleNamespace(state=state)
    original_request = exa_tool.Exa.request

    def request(client, path, options):
        assert path == "/search"
        assert client.headers["x-api-key"] == "cross-slice-exa-fixture"
        assert options["numResults"] == 100
        return {"results": [
            {"id": str(index), "title": f"Result {index}",
             "url": f"https://example.test/{index}",
             "highlights": ["Evidence " * 100]}
            for index in range(1, 101)
        ]}

    exa_tool.Exa.request = request
    exa_tool.configure_exa_credentials(SecretStr("cross-slice-exa-fixture"))
    tool = None
    try:
        observation = exa_tool.ExaSearchExecutor()(
            exa_tool.ExaSearchAction(query="artifact retrieval", num_results=100),
            conversation,
        )
        first = observation.markdown.splitlines()[0]
        assert first.startswith("Complete results saved to: "), first
        exa_path = Path(first.removeprefix("Complete results saved to: "))
        assert "## 100. Result 100" not in observation.markdown
        assert "## 100. Result 100" in exa_path.read_text()
        assert "cross-slice-exa-fixture" not in observation.markdown

        export_root = directory / "research"
        with export_directory(export_root) as descriptor:
            exports = ResearchExports(export_root, descriptor)
            wandb_path, _ = exports.jsonl([
                {"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25},
            ])
        result_path = directory / "readback.json"
        command = "python - <<'PYCODE'\n" + (
            "import json,sys\nfrom pathlib import Path\n"
            f"exa = Path({str(exa_path)!r}).read_text()\n"
            f"rows = [json.loads(line) for line in Path({wandb_path!r}).read_text().splitlines()]\n"
            f"Path({str(result_path)!r}).write_text(json.dumps({{"
            "'prefix':sys.prefix,'tail':exa[-2000:],'loss':rows[-1]['loss']}))\n"
        ) + "PYCODE"
        tool = SenpaiTerminalTool.create(state, role="advisor")[0]
        result = tool.executor(TerminalAction(command=command, timeout=30))
        assert not result.is_error and result.exit_code == 0, result.text
        readback = json.loads(result_path.read_text())
        assert Path(readback["prefix"]).resolve() == target_env
        assert "## 100. Result 100" in readback["tail"]
        assert readback["loss"] == 0.25
        print(json.dumps({
            "result": "PASS",
            "exa_requested": 100,
            "complete_evidence_retrieved_after_preview": True,
            "wandb_export_readable": True,
            "uses_target_python": True,
            "pooled_terminal": tool.executor.is_pooled,
        }))
    finally:
        if tool is not None:
            tool.executor.close()
        exa_tool.configure_exa_credentials(None)
        exa_tool.Exa.request = original_request
