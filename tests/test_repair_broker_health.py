import os
import threading
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from senpai_agent.operations import CampaignInventory, RoleTarget
from senpai_agent.repair_broker import (
    RepairBrokerClient,
    RepairBrokerServer,
    RepairRequest,
    RepairResult,
)
from senpai_agent.repair_broker_health import (
    RepairBrokerHealthError,
    check_repair_broker_health,
)


class BlockingBackend:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def run_repair(self, target, *, command, cwd, timeout_seconds):
        del target, command, cwd, timeout_seconds
        self.started.set()
        assert self.release.wait(5)
        return RepairResult(exit_code=0, stdout="repaired", stderr="")


def inventory() -> CampaignInventory:
    return CampaignInventory(
        research_tag="maple",
        repo="acme/widgets",
        advisor_branch="maple-advisor",
        students=("fern",),
    )


def short_socket_path() -> Path:
    return Path(tempfile.gettempdir()) / (
        f"senpai-{os.getpid()}-{uuid4().hex[:10]}-broker.sock"
    )


def test_broker_health_is_authoritative_while_idle_and_during_bounded_work(tmp_path):
    backend = BlockingBackend()
    socket_path = short_socket_path()
    request = RepairRequest.create(
        operation_id="health-active",
        target=RoleTarget(research_tag="maple", role="advisor"),
        command="true",
        timeout_seconds=5,
    )
    outcome = []

    with RepairBrokerServer(
        socket_path,
        inventory(),
        backend,
        ledger_path=tmp_path / "repair.sqlite3",
    ):
        check_repair_broker_health(socket_path, expected_pid=os.getpid())
        worker = threading.Thread(
            target=lambda: outcome.append(RepairBrokerClient(socket_path).execute(request))
        )
        worker.start()
        assert backend.started.wait(2)

        check_repair_broker_health(socket_path, expected_pid=os.getpid())
        backend.release.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        check_repair_broker_health(socket_path, expected_pid=os.getpid())

    assert outcome == [RepairResult(exit_code=0, stdout="repaired", stderr="")]
    with pytest.raises(RepairBrokerHealthError):
        check_repair_broker_health(socket_path, expected_pid=os.getpid())


def test_broker_health_rejects_a_stale_or_overdue_heartbeat():
    from senpai_agent.repair_broker_health import broker_heartbeat_is_healthy

    idle = {
        "protocol": "senpai-repair-broker-health/v1",
        "server_pid": 12,
        "state": "idle",
        "heartbeat_monotonic": 10.0,
        "operation_deadline_monotonic": None,
    }
    active = {
        **idle,
        "state": "active",
        "operation_deadline_monotonic": 20.0,
    }

    assert broker_heartbeat_is_healthy(idle, expected_pid=12, now=11.0)
    assert not broker_heartbeat_is_healthy(idle, expected_pid=12, now=13.0)
    assert broker_heartbeat_is_healthy(active, expected_pid=12, now=19.0)
    assert not broker_heartbeat_is_healthy(active, expected_pid=12, now=26.0)
