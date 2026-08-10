from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from senpai_agent.kubernetes_operations import KubectlCampaignBackend
from senpai_agent.operations import CampaignInventory, RoleObservation, RoleTarget
from senpai_agent.protocols import MANAGEMENT_PROTOCOL_VERSION
from senpai_agent.role_control import (
    RoleControlRequest,
    RoleResearchTail,
    RoleResearchTailItem,
    RoleRuntimeState,
)


CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000301")


def inventory() -> CampaignInventory:
    return CampaignInventory(
        research_tag="maple",
        repo="example/research",
        advisor_branch="maple-advisor",
        students=("fern",),
    )


def runtime_json(target: RoleTarget) -> str:
    state = RoleRuntimeState(
        target=target,
        observation=RoleObservation(
            target=target,
            observed_at=datetime.now(UTC),
            control_token="token-1",
            restart_control_token="restart-token-1",
            controller_alive=True,
            controller_phase="sleep",
            conversation_id=CONVERSATION_ID,
            active_turn=False,
            unmatched_actions=0,
            raw_history_event_count=4,
            raw_history_digest="digest-1",
            active_delegation_count=2,
        ),
        lease_deadline_seconds=600,
        completed_turns=2,
        running_training_count=0,
        active_delegation_count=2,
        wandb_run_inventory_complete=True,
        cpu_percent=10,
        memory_percent=20,
        disk_percent=30,
        gpu_percent=40,
    )
    return json.dumps(
        {
            "protocol_version": MANAGEMENT_PROTOCOL_VERSION,
            "result": state.model_dump(mode="json"),
        }
    )


def test_backend_selects_exact_campaign_role_labels_and_parses_role_state(
    monkeypatch,
):
    calls = []
    target = RoleTarget(research_tag="maple", role="student", student="fern")

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs.get("input")))
        if "get" in command:
            output = json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "senpai-maple-fern"},
                            "status": {"phase": "Running"},
                        }
                    ]
                }
            )
        else:
            output = runtime_json(target)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("senpai_agent.kubernetes_operations.subprocess.run", run)
    backend = KubectlCampaignBackend(inventory(), namespace="research")

    observation = backend.collect_role(target)

    selector = calls[0][0][calls[0][0].index("-l") + 1]
    assert calls[0][0][0] == "/usr/local/bin/kubectl"
    assert selector == "app=senpai,role=student,research-tag=maple,student=fern"
    assert calls[1][0][-4:] == (
        "/opt/senpai-venv/bin/python",
        "-I",
        "-m",
        "senpai_agent.role_control",
    )
    assert json.loads(calls[1][1])["command"] == "observe"
    assert (
        json.loads(calls[1][1])["protocol_version"]
        == MANAGEMENT_PROTOCOL_VERSION
    )
    assert observation.conversation_id == CONVERSATION_ID
    assert observation.active_delegation_count == 2


def test_backend_rejects_roles_outside_inventory_before_kubectl(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "senpai_agent.kubernetes_operations.subprocess.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    backend = KubectlCampaignBackend(inventory(), namespace="research")

    with pytest.raises(PermissionError, match="campaign inventory"):
        backend.collect_role(
            RoleTarget(research_tag="maple", role="student", student="other")
        )

    assert calls == []


def test_log_window_covers_cadence_and_both_supervisor_turns():
    backend = KubectlCampaignBackend(
        inventory(),
        namespace="research",
        environment={
            "SENPAI_SUPERVISOR_INTERVAL_SECONDS": "120",
            "SENPAI_OPENHANDS_TIMEOUT_SECONDS": "45",
        },
    )

    assert backend.log_since_seconds == 120 + (2 * 45) + 120


def test_runtime_collection_preserves_errors_and_repeated_deferred_markers(
    monkeypatch,
):
    advisor = RoleTarget(research_tag="maple", role="advisor")
    student = RoleTarget(research_tag="maple", role="student", student="fern")

    log_commands = []

    def run(command, **kwargs):
        if "get" in command:
            role = command[command.index("-l") + 1].split("role=", 1)[1].split(",", 1)[0]
            name = "senpai-advisor-maple" if role == "advisor" else "senpai-maple-fern"
            output = json.dumps(
                {"items": [{"metadata": {"name": name}, "status": {"phase": "Running"}}]}
            )
        elif "logs" in command:
            log_commands.append(tuple(command))
            output = (
                "ordinary line\n"
                "candidate failed to improve after an error analysis\n"
                "2026-08-06T12:00:01Z SENPAI_TURN_DEFERRED "
                "student-only-secret conversation_id=x\n"
                "2026-08-06T12:00:02Z SENPAI_TURN_DEFERRED "
                "conversation_id=x\n"
            )
        else:
            target = advisor if "advisor" in command else student
            output = runtime_json(target)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("senpai_agent.kubernetes_operations.subprocess.run", run)
    backend = KubectlCampaignBackend(
        inventory(),
        namespace="research",
        environment={
            "SENPAI_SUPERVISOR_INTERVAL_SECONDS": "900",
            "EXA_API_KEY": "student-only-secret",
        },
    )

    observations, gaps = backend.collect_runtimes()

    assert gaps == ()
    assert len(observations) == 2
    assert all(runtime.controller_healthy is True for runtime in observations)
    assert all(runtime.active_delegation_count == 2 for runtime in observations)
    assert all(len(runtime.recent_errors) == 2 for runtime in observations)
    assert all(
        "SENPAI_TURN_DEFERRED" in marker
        for runtime in observations
        for marker in runtime.recent_errors
    )
    assert "student-only-secret" not in json.dumps(
        [runtime.model_dump(mode="json") for runtime in observations]
    )
    assert all(
        "fingerprint=" in marker
        for runtime in observations
        for marker in runtime.recent_errors
    )
    assert all("--timestamps=true" in command for command in log_commands)
    assert all("--since=2820s" in command for command in log_commands)


def test_runtime_collection_reports_a_truncated_log_window(monkeypatch):
    advisor = RoleTarget(research_tag="maple", role="advisor")
    student = RoleTarget(research_tag="maple", role="student", student="fern")

    def run(command, **kwargs):
        if "get" in command:
            selector = command[command.index("-l") + 1]
            name = (
                "senpai-advisor-maple"
                if "role=advisor" in selector
                else "senpai-maple-fern"
            )
            output = json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": name},
                            "status": {"phase": "Running"},
                        }
                    ]
                }
            )
        elif "logs" in command:
            output = "\n".join(f"ordinary line {index}" for index in range(400))
        else:
            target = advisor if "advisor" in command else student
            output = runtime_json(target)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("senpai_agent.kubernetes_operations.subprocess.run", run)

    observations, gaps = KubectlCampaignBackend(
        inventory(), namespace="research"
    ).collect_runtimes()

    assert len(observations) == 2
    assert {gap.subject for gap in gaps} == {"maple:advisor", "maple:student:fern"}
    assert all("bounded 400-line tail" in gap.detail for gap in gaps)


def test_runtime_collection_rejects_a_mismatched_role_payload(monkeypatch):
    advisor = RoleTarget(research_tag="maple", role="advisor")

    def run(command, **kwargs):
        if "get" in command:
            selector = command[command.index("-l") + 1]
            name = (
                "senpai-advisor-maple"
                if "role=advisor" in selector
                else "senpai-maple-fern"
            )
            output = json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": name},
                            "status": {"phase": "Running"},
                        }
                    ]
                }
            )
        elif "logs" in command:
            output = ""
        else:
            output = runtime_json(advisor)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("senpai_agent.kubernetes_operations.subprocess.run", run)

    observations, gaps = KubectlCampaignBackend(
        inventory(), namespace="research"
    ).collect_runtimes()

    assert observations[0].machine == "senpai-advisor-maple"
    assert observations[1].machine == "unavailable"
    assert gaps[0].subject == "maple:student:fern"


def test_backend_collects_only_the_exact_advisor_research_tail(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs.get("input")))
        if "get" in command:
            output = json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "senpai-advisor-maple"},
                            "status": {"phase": "Running"},
                        }
                    ]
                }
            )
        else:
            tail = RoleResearchTail(
                conversation_id=CONVERSATION_ID,
                observed_at=datetime.now(UTC),
                advisor_guidance="Prefer causal, mechanism-led research.",
                messages=(
                    RoleResearchTailItem(
                        index=7,
                        kind="MessageEvent",
                        source="agent",
                        summary="Compare mechanisms before another sweep.",
                    ),
                ),
            )
            output = json.dumps(
                {
                    "protocol_version": MANAGEMENT_PROTOCOL_VERSION,
                    "result": tail.model_dump(mode="json"),
                }
            )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr("senpai_agent.kubernetes_operations.subprocess.run", run)
    backend = KubectlCampaignBackend(inventory(), namespace="research")

    tail = backend.collect_advisor_research_tail()

    assert tail.conversation_id == CONVERSATION_ID
    assert tail.messages[0].source == "agent"
    assert tail.advisor_guidance == "Prefer causal, mechanism-led research."
    assert "role=advisor" in calls[0][0][calls[0][0].index("-l") + 1]
    assert json.loads(calls[1][1])["command"] == "research_tail"


@pytest.mark.parametrize(
    "response",
    [
        {"result": {}},
        {"protocol_version": "senpai-management/v0", "result": {}},
    ],
)
def test_backend_rejects_missing_or_stale_role_management_protocol(
    monkeypatch,
    response,
):
    backend = KubectlCampaignBackend(inventory(), namespace="research")
    monkeypatch.setattr(backend, "_pod", lambda _target: "senpai-advisor-maple")
    monkeypatch.setattr(backend, "_run", lambda *_args, **_kwargs: json.dumps(response))

    with pytest.raises(RuntimeError, match="management protocol"):
        backend._role_control(
            RoleTarget(research_tag="maple", role="advisor"),
            RoleControlRequest(
                protocol_version=MANAGEMENT_PROTOCOL_VERSION,
                command="observe",
            ),
        )
