import json
import os
import uuid
from io import StringIO
from types import SimpleNamespace

import pytest
from openhands.sdk.conversation import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.tool import resolve_tool
from openhands_support import runtime_env

import senpai_agent.openhands_runner as runner
import senpai_agent.supervisor_worker as supervisor


@pytest.mark.parametrize("outcome", ["repaired", "unfinished", "failed"])
def test_supervisor_repairs_local_workspace_and_publishes_only_after_cleanup(
    tmp_path,
    monkeypatch,
    outcome,
):
    environment = runtime_env(tmp_path)
    workspace = tmp_path / "target"
    broken = workspace / "config.py"
    broken.write_text("workers = 0\n")
    prompt = (
        "Repair config.py so workers is 1. The current runtime cannot start workers."
    )
    credentials = {
        name: environment.pop(name)
        for name in (
            "GITHUB_TOKEN",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
        )
    }
    credentials.pop("GITHUB_TOKEN")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w") as stream:
        json.dump(credentials, stream)
    monkeypatch.setenv("SENPAI_MODEL_CREDENTIALS_FD", str(read_fd))
    monkeypatch.setenv("SENPAI_DELEGATION_TASK_ID", "repair-task")
    monkeypatch.setenv(
        "SENPAI_PARENT_CONVERSATION_HISTORY_DIR", "/private/advisor/history"
    )
    monkeypatch.setattr("sys.stdin", StringIO(prompt))
    closed = False
    published = []

    def record(task_id, **values):
        assert closed, "parent resumed while Supervisor tools were still open"
        published.append((task_id, values))

    class RepairConversation:
        def __init__(self, **kwargs):
            self.id = kwargs["conversation_id"]
            agent = kwargs["agent"]
            assert agent.llm.model == "anthropic/claude-opus-5-5"
            assert agent.llm.reasoning_effort == "xhigh"
            suffix = agent.agent_context.system_message_suffix
            assert "independent Supervisor" in suffix
            assert "advisor role" not in suffix
            assert "program.md" in suffix
            assert "SENPAI_PARENT_CONVERSATION_HISTORY_DIR" not in os.environ
            assert "GITHUB_TOKEN" not in kwargs["secrets"]
            assert "GITHUB_TOKEN" not in os.environ
            assert "ANTHROPIC_API_KEY" not in os.environ
            tool_names = {tool.name for tool in agent.tools}
            assert {
                "file_editor",
                "senpai_terminal",
                "task_tracker",
                "FinishTool",
            } <= tool_names
            assert not (
                {"senpai_github", "senpai_training", "spawn_agents"} & tool_names
            )
            self.state = SimpleNamespace(
                execution_status=ConversationExecutionStatus.IDLE,
                workspace=SimpleNamespace(working_dir=str(workspace)),
                agent=agent,
                view=SimpleNamespace(events=[]),
            )
            definitions = [
                resolve_tool(spec, self.state)[0]
                for spec in agent.tools
                if spec.name in {"file_editor", "FinishTool"}
            ]
            self.agent = SimpleNamespace(
                tools_map={tool.name: tool for tool in definitions}
            )
            self.editor = next(tool for tool in definitions if tool.name != "finish")
            self.finish = self.agent.tools_map["finish"]
            schema = self.finish.to_openai_tool()["function"]["parameters"]
            assert "actionable" in schema["properties"]["repair_summary"]["description"]

        def send_message(self, supplied_prompt):
            assert supplied_prompt == prompt

        async def arun(self):
            if outcome == "failed":
                raise RuntimeError("local repair failed")
            if outcome == "unfinished":
                self.state.execution_status = ConversationExecutionStatus.PAUSED
                return
            edit = self.editor.action_from_arguments(
                {
                    "command": "str_replace",
                    "path": str(broken),
                    "old_str": "workers = 0",
                    "new_str": "workers = 1",
                }
            )
            observation = self.editor.executor(edit)
            assert not observation.is_error
            assert broken.read_text() == "workers = 1\n"
            arguments = {
                "message": "Repair complete.",
                "resolved": True,
                "repair_summary": "Changed workers to 1 and verified the file.",
            }
            action = self.finish.action_from_arguments(arguments)
            call = MessageToolCall(
                id="repair-finish",
                name="finish",
                arguments=json.dumps(arguments),
                origin="completion",
            )
            self.state.view.events.append(
                ActionEvent(
                    thought=[],
                    action=action,
                    tool_name="finish",
                    tool_call_id=call.id,
                    tool_call=call,
                    llm_response_id="repair-response",
                )
            )
            self.state.execution_status = ConversationExecutionStatus.FINISHED

        def close(self):
            nonlocal closed
            closed = True
            self.editor.executor.close()

    monkeypatch.setattr(runner, "LocalConversation", RepairConversation)
    monkeypatch.setattr(runner, "record_delegated_task_result", record)
    monkeypatch.setattr(supervisor, "record_delegated_task_result", record)
    monkeypatch.setattr(supervisor, "finish_weave_monitoring", lambda: None)
    args = [
        "--child",
        "--agent",
        "supervisor",
        "--max-turns",
        "1",
        "--model",
        "anthropic/claude-haiku-4-5",
        "--reasoning-effort",
        "low",
        "--workspace",
        str(workspace),
        "--state-dir",
        str(tmp_path / "worker-state"),
        "--conversation-id",
        str(uuid.uuid4()),
    ]

    if outcome == "repaired":
        assert supervisor.main(args) == 0
        assert json.loads(published[0][1]["result"]) == {
            "resolved": True,
            "repair_summary": "Changed workers to 1 and verified the file.",
        }
    else:
        reason = (
            "structured result" if outcome == "unfinished" else "local repair failed"
        )
        with pytest.raises(RuntimeError, match=reason):
            supervisor.main(args)
        assert reason in published[0][1]["error"]
        assert broken.read_text() == "workers = 0\n"
    assert len(published) == 1
    assert published[0][0] == "repair-task"
