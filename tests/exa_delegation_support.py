"""Exercise real child exec and runner startup without making model requests."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

# Child commands use -P, so select this checkout explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openhands.sdk import Tool
from openhands.sdk.tool import resolve_tool

from senpai_agent import exa_tool, openhands_runner
from senpai_agent.delegation import DelegationRequest, OpenHandsChildProcess
from senpai_agent.secrets import MODEL_CREDENTIALS_FD_ENV

_native_command = OpenHandsChildProcess.command.fget


def child_command(child):
    command = _native_command(child)
    return (*command[:2], str(Path(__file__).resolve()), *command[4:])


def run_without_model(_prompt, config):
    assert "EXA_API_KEY" not in os.environ
    assert MODEL_CREDENTIALS_FD_ENV not in os.environ
    shell_environment = subprocess.check_output(["/bin/sh", "-c", "env"], text=True)
    assert "nested-exa-key" not in shell_environment
    assert "EXA_API_KEY=" not in shell_environment
    assert "EXA_API_KEY" not in config.conversation_secrets

    hop = {"agent": config.agent_name, "pid": os.getpid()}
    if config.agent_name == "general-purpose":
        child = OpenHandsChildProcess(
            openhands_runner.delegation_config(config),
            DelegationRequest(
                task_id=str(uuid.uuid4()),
                parent_conversation_id=str(config.conversation_id),
                parent_context=(),
                agent="search",
                model="fast",
                search_mode="general-web",
                depth=config.delegation_depth + 1,
            ),
        )
        result = json.loads(child.run("Find the requested evidence.", 30))
        result["hops"].insert(0, hop)
    else:
        assert config.agent_name == "search"
        exa_tool.configure_exa_credentials(config.exa_api_key)
        openhands_runner.register_senpai_tools()
        tool = resolve_tool(Tool(name="senpai_exa"), SimpleNamespace())[0]
        assert "nested-exa-key" not in tool.model_dump_json()

        def request(client, path, options):
            assert client.headers["x-api-key"] == "nested-exa-key"
            assert path == "/search"
            assert options["query"] == "nested delegation evidence"
            return {
                "results": [
                    {
                        "title": "Nested search result",
                        "url": "https://example.test/nested",
                        "highlights": [
                            "Evidence returned through both child processes."
                        ],
                    }
                ],
            }

        exa_tool.Exa.request = request
        observation = tool.executor(
            exa_tool.ExaSearchAction(query="nested delegation evidence")
        )
        result = {
            "hops": [hop],
            "evidence": "\n".join(item.text for item in observation.to_llm_content),
        }
    print(
        "OPENHANDS_RESULT "
        + json.dumps({"status": "finished", "result": json.dumps(result)}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    OpenHandsChildProcess.command = property(child_command)
    openhands_runner.run_openhands = run_without_model
    raise SystemExit(openhands_runner.main())
