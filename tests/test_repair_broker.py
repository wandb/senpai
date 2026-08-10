import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from senpai_agent.kubernetes_operations import KubectlCampaignBackend
from senpai_agent.operations import CampaignInventory, RoleTarget
from senpai_agent.repair_broker import (
    RepairBrokerClient,
    RepairBrokerServer,
    RepairRequest,
    RepairResult,
)
from senpai_agent.repair_executor import (
    REPAIR_STREAM_LIMIT_CHARS,
    execute_local_repair,
)


class RecordingRepairBackend:
    def __init__(self):
        self.calls = []

    def run_repair(self, target, *, command, cwd, timeout_seconds):
        self.calls.append((target, command, cwd, timeout_seconds))
        return RepairResult(exit_code=0, stdout="fixed\n", stderr="")


def inventory():
    return CampaignInventory(
        research_tag="maple",
        repo="acme/widgets",
        advisor_branch="maple-advisor",
        students=("fern", "frieren"),
    )


def test_repair_broker_binds_arbitrary_commands_to_exact_role_target(tmp_path):
    backend = RecordingRepairBackend()
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-repair.sock"
    with RepairBrokerServer(socket_path, inventory(), backend):
        result = RepairBrokerClient(socket_path).execute(
            RepairRequest(
                target=RoleTarget(research_tag="maple", role="student", student="fern"),
                command="git reset --soft HEAD^ && git status --short",
                cwd="workspace",
                timeout_seconds=123,
            )
        )

    assert result == RepairResult(exit_code=0, stdout="fixed\n", stderr="")
    assert backend.calls == [
        (
            RoleTarget(research_tag="maple", role="student", student="fern"),
            "git reset --soft HEAD^ && git status --short",
            "workspace",
            123,
        )
    ]


@pytest.mark.parametrize(
    "target",
    [
        RoleTarget(research_tag="cedar", role="advisor"),
        RoleTarget(research_tag="maple", role="student", student="other"),
    ],
)
def test_repair_broker_rejects_cross_campaign_and_unknown_roles(
    tmp_path,
    target,
    capsys,
):
    socket_path = Path("/private/tmp") / f"{tmp_path.name}-{target.role}-repair.sock"
    with RepairBrokerServer(socket_path, inventory(), RecordingRepairBackend()):
        with pytest.raises(PermissionError):
            RepairBrokerClient(socket_path).execute(
                RepairRequest(target=target, command="true")
            )
    audit = capsys.readouterr().err
    assert "SENPAI_REPAIR_COMMAND" in audit
    assert '"outcome": "denied"' in audit
    assert target.key in audit


def test_repair_request_has_no_namespace_pod_container_or_host_escape_hatch():
    payload = {
        "target": {"research_tag": "maple", "role": "advisor"},
        "command": "true",
        "container": "advisor",
    }

    with pytest.raises(ValueError):
        RepairRequest.model_validate(payload)


def test_kubectl_repair_transport_executes_only_the_repair_sidecar(monkeypatch):
    backend = KubectlCampaignBackend(inventory(), namespace="research")
    calls = []
    monkeypatch.setattr(backend, "_pod", lambda target: "senpai-maple-fern-abc")

    def record(command, *, input_text=None, timeout_seconds=None):
        calls.append((command, input_text, timeout_seconds))
        return json.dumps({"exit_code": 0, "stdout": "ok", "stderr": ""})

    monkeypatch.setattr(backend, "_run", record)

    result = backend.run_repair(
        RoleTarget(research_tag="maple", role="student", student="fern"),
        command="env && git status",
        cwd="state",
        timeout_seconds=90,
    )

    command, input_text, timeout = calls[0]
    assert result.exit_code == 0
    assert command[command.index("-c") + 1] == "repair"
    assert "student" not in command
    assert "advisor" not in command
    assert "env && git status" not in command
    assert "/repair/workspace/senpai" not in command
    assert command[command.index("--") + 1] == (
        "/usr/local/bin/senpai-repair-executor"
    )
    assert input_text == "env && git status"
    assert timeout == 105


def test_repair_timeout_reaps_the_descendant_process_group(tmp_path):
    marker = tmp_path / "survived"

    result = execute_local_repair(
        f"sleep 0.4; printf survived > {marker}",
        tmp_path,
        0.05,
    )
    time.sleep(0.5)

    assert result["exit_code"] == 124
    assert not marker.exists()


def test_repair_executor_bounds_combined_json_response_below_socket_frame(tmp_path):
    producer = (
        "import sys; "
        "sys.stdout.write('x' * 1200000); "
        "sys.stderr.write('y' * 1200000)"
    )

    result = execute_local_repair(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(producer)}",
        tmp_path,
        10,
    )
    response = json.dumps({"result": result}).encode() + b"\n"

    assert len(result["stdout"]) == REPAIR_STREAM_LIMIT_CHARS
    assert len(result["stderr"]) == REPAIR_STREAM_LIMIT_CHARS
    assert len(response) < 2 * 1024 * 1024


def test_repair_executor_replaces_undecodable_command_output(tmp_path):
    producer = "import sys; sys.stdout.buffer.write(b'\\xff')"

    result = execute_local_repair(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(producer)}",
        tmp_path,
        10,
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "\N{REPLACEMENT CHARACTER}"


def test_repair_broker_audits_backend_errors_without_logging_command(
    tmp_path,
    capsys,
):
    class FailingBackend:
        def run_repair(self, target, *, command, cwd, timeout_seconds):
            raise RuntimeError("transport failed")

    socket_path = Path("/private/tmp") / f"{tmp_path.name}-error-repair.sock"
    command = "sensitive diagnostic text"
    with RepairBrokerServer(socket_path, inventory(), FailingBackend()):
        with pytest.raises(RuntimeError, match="transport failed"):
            RepairBrokerClient(socket_path).execute(
                RepairRequest(
                    target=RoleTarget(research_tag="maple", role="advisor"),
                    command=command,
                )
            )

    audit = capsys.readouterr().err
    assert '"outcome": "error"' in audit
    assert '"error_type": "RuntimeError"' in audit
    assert command not in audit
