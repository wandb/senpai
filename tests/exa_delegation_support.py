"""Exercise real child exec and runner startup without making model requests."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

# Child commands use -P, so select this checkout explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openhands.sdk import Agent, LLM, LocalConversation
from openhands.sdk.subagent import agent_definition_to_factory
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

    assert config.agent_name == "general-purpose"
    hop = {
        "agent": config.agent_name,
        "depth": config.delegation_depth,
        "pid": os.getpid(),
    }
    if config.delegation_depth == 1:
        child = OpenHandsChildProcess(
            openhands_runner.delegation_config(config),
            DelegationRequest(
                task_id=str(uuid.uuid4()),
                parent_conversation_id=str(config.conversation_id),
                parent_context=(),
                agent="general-purpose",
                model="fast",
                depth=config.delegation_depth + 1,
            ),
        )
        result = json.loads(child.run("Find the requested evidence.", 30))
        result["hops"].insert(0, hop)
    else:
        assert config.delegation_depth == 2
        openhands_runner.register_senpai_tools()
        definition = openhands_runner.depth_aware_child_definition(
            openhands_runner.find_named_agent(
                config.agent_name,
                openhands_runner.sanitized_agent_definitions(config.workspace),
            ),
            child=config.child,
            depth=config.delegation_depth,
        )
        agent = agent_definition_to_factory(definition, work_dir=config.workspace)(
            LLM(model=config.model, api_key=config.api_key)
        )
        spec = next(tool for tool in agent.tools if tool.name == "senpai_exa")
        conversation = LocalConversation(
            agent=Agent(llm=agent.llm, tools=[spec]),
            workspace=config.workspace,
            persistence_dir=config.state_dir,
            conversation_id=config.conversation_id,
            visualizer=None,
            delete_on_close=True,
        )

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

        try:
            exa_tool.configure_exa_credentials(config.exa_api_key)
            exa_tool.Exa.request = request
            tool = resolve_tool(spec, conversation.state)[0]
            assert "nested-exa-key" not in tool.model_dump_json()
            observation = tool.executor(
                exa_tool.ExaSearchAction(query="nested delegation evidence"),
                conversation,
            )
            result = {
                "hops": [hop],
                "evidence": "\n".join(item.text for item in observation.to_llm_content),
            }
        finally:
            exa_tool.configure_exa_credentials(None)
            conversation.close()
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
