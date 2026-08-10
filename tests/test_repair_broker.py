import hashlib
import json
import os
import shlex
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import pytest
from pydantic import ValidationError

from senpai_agent.kubernetes_operations import KubectlCampaignBackend
from senpai_agent.operations import CampaignInventory, RoleTarget
from senpai_agent.protocols import REPAIR_PROTOCOL_VERSION
from senpai_agent.repair_broker import (
    RepairBrokerClient,
    RepairBrokerServer,
    RepairIdempotencyConflict,
    RepairLedger,
    RepairOutcomeUnknown,
    RepairReceiptExpired,
    RepairRequest,
    RepairResult,
    repair_broker_main,
)
from senpai_agent.repair_executor import (
    DEFAULT_EXECUTOR_SOCKET,
    REPAIR_STREAM_LIMIT_BYTES,
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


def repair_request(
    operation_id="repair-1",
    *,
    target=None,
    command="true",
    cwd="workspace",
    timeout_seconds=300,
):
    return RepairRequest.create(
        operation_id=operation_id,
        target=target
        or RoleTarget(research_tag="maple", role="student", student="fern"),
        command=command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )


def test_repair_broker_binds_arbitrary_commands_to_exact_role_target(tmp_path):
    backend = RecordingRepairBackend()
    socket_path = _test_socket(tmp_path, "repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        backend,
        ledger_path=tmp_path / "repair.sqlite3",
    ):
        result = RepairBrokerClient(socket_path).execute(
            repair_request(
                "repair-bind",
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


def test_repair_wire_preserves_exact_command_and_output_whitespace(tmp_path):
    backend = RecordingRepairBackend()
    socket_path = _test_socket(tmp_path, "fidelity-repair")
    command = "  printf 'first\\nsecond ' \\\n"
    request = RepairRequest.create(
        operation_id="repair-fidelity:1",
        target=RoleTarget(research_tag="maple", role="advisor"),
        command=command,
    )

    with RepairBrokerServer(
        socket_path,
        inventory(),
        backend,
        ledger_path=tmp_path / "repair.sqlite3",
    ):
        RepairBrokerClient(socket_path).execute(request)

    assert request.command == command
    assert backend.calls[0][1] == command
    result = RepairResult(exit_code=0, stdout=" leading\ntrailing \n", stderr="\t")
    assert RepairResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("command", ["", " ", "\n\t"])
def test_repair_request_rejects_empty_commands_without_normalizing(command):
    with pytest.raises(ValidationError):
        RepairRequest.create(
            operation_id="repair-empty",
            target=RoleTarget(research_tag="maple", role="advisor"),
            command=command,
        )


@pytest.mark.parametrize(
    "operation_id",
    [" leading", "trailing ", "contains/control\n", ""],
)
def test_repair_operation_id_is_an_audit_safe_canonical_token(operation_id):
    with pytest.raises(ValidationError):
        RepairRequest.create(
            operation_id=operation_id,
            target=RoleTarget(research_tag="maple", role="advisor"),
            command="true",
        )


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
    socket_path = _test_socket(tmp_path, f"{target.role}-repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        RecordingRepairBackend(),
        ledger_path=tmp_path / f"{target.role}.sqlite3",
    ):
        with pytest.raises(PermissionError):
            RepairBrokerClient(socket_path).execute(
                repair_request("repair-denied", target=target)
            )
    audit = capsys.readouterr().err
    assert "SENPAI_REPAIR_COMMAND" in audit
    assert '"outcome": "denied"' in audit
    assert target.key in audit


def test_repair_request_has_no_namespace_pod_container_or_host_escape_hatch():
    payload = {
        "operation_id": "repair-escape",
        "target": {"research_tag": "maple", "role": "advisor"},
        "command": "true",
        "command_fingerprint": "0" * 64,
        "container": "advisor",
    }

    with pytest.raises(ValueError):
        RepairRequest.model_validate(payload)


def test_role_shell_requires_a_stable_operation_id_before_execution(capsys):
    with pytest.raises(SystemExit):
        repair_broker_main(
            [
                "repair",
                "--research-tag",
                "maple",
                "--role",
                "advisor",
                "--command",
                "true",
            ]
        )

    assert "--operation-id is required" in capsys.readouterr().err


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
    executor = command.index("--") + 1
    assert command[executor : executor + 4] == (
        "/opt/senpai-venv/bin/python",
        "-I",
        "/usr/local/bin/senpai-repair-executor",
        "client",
    )
    assert DEFAULT_EXECUTOR_SOCKET in command
    assert DEFAULT_EXECUTOR_SOCKET.startswith("/run/senpai-repair-executor/")
    assert input_text == "env && git status"
    assert timeout == 105


def test_kubectl_transport_uses_an_explicit_secret_free_environment(monkeypatch):
    captured = {}
    environment = {
        "PATH": "/usr/bin",
        "KUBECONFIG": "/var/run/senpai/kubeconfig",
        "LANG": "C.UTF-8",
        "DATABASE_URL": "postgres://credential@database/research",
        "SENTRY_DSN": "https://credential@sentry.example/1",
        "CUSTOM_AUTH": "credential",
        "OPENAI_API_KEY": "credential",
        "PYTHONPATH": "/workspaces/poisoned",
        "SENPAI_SUPERVISOR_INTERVAL_SECONDS": "900",
    }

    def record(command, **kwargs):
        captured["command"] = tuple(command)
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("senpai_agent.kubernetes_operations.subprocess.run", record)
    backend = KubectlCampaignBackend(
        inventory(),
        namespace="research",
        environment=environment,
    )
    assert backend._run((backend.kubectl, "version")) == "ok"
    assert captured["command"][0] == "/usr/local/bin/kubectl"

    assert captured["env"] == {
        "PATH": "/usr/bin",
        "KUBECONFIG": "/var/run/senpai/kubeconfig",
        "LANG": "C.UTF-8",
    }


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

    assert len(result["stdout"].encode()) <= REPAIR_STREAM_LIMIT_BYTES
    assert len(result["stderr"].encode()) <= REPAIR_STREAM_LIMIT_BYTES
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert len(response) < 2 * 1024 * 1024


def test_repair_truncation_is_explicit_at_a_multibyte_utf8_boundary(tmp_path):
    producer = (
        "import sys; "
        f"sys.stdout.buffer.write('€'.encode() * {REPAIR_STREAM_LIMIT_BYTES})"
    )

    result = execute_local_repair(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(producer)}",
        tmp_path,
        10,
    )

    assert result["exit_code"] == 0
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is False
    assert len(result["stdout"].encode()) <= REPAIR_STREAM_LIMIT_BYTES


def test_repair_executor_drains_high_volume_output_with_bounded_caller_memory(tmp_path):
    producer = (
        "import sys; chunk=b'x'*65536; "
        "[(sys.stdout.buffer.write(chunk), sys.stderr.buffer.write(chunk)) "
        "for _ in range(256)]"
    )
    tracemalloc.start()
    try:
        result = execute_local_repair(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(producer)}",
            tmp_path,
            20,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result["exit_code"] == 0
    assert peak < 8 * 1024 * 1024


def test_repair_broker_returns_worst_case_unicode_within_one_valid_frame(tmp_path):
    class UnicodeBackend:
        def run_repair(self, target, *, command, cwd, timeout_seconds):
            return RepairResult(
                exit_code=0,
                stdout="\N{COLLISION SYMBOL}" * REPAIR_STREAM_LIMIT_BYTES,
                stderr="\N{COLLISION SYMBOL}" * REPAIR_STREAM_LIMIT_BYTES,
            )

    socket_path = _test_socket(tmp_path, "unicode-repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        UnicodeBackend(),
        ledger_path=tmp_path / "unicode.sqlite3",
    ):
        result = RepairBrokerClient(socket_path).execute(
            repair_request("repair-unicode")
        )

    assert len(result.stdout.encode()) <= REPAIR_STREAM_LIMIT_BYTES
    assert len(result.stderr.encode()) <= REPAIR_STREAM_LIMIT_BYTES
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


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

    socket_path = _test_socket(tmp_path, "error-repair")
    command = "sensitive diagnostic text"
    with RepairBrokerServer(
        socket_path,
        inventory(),
        FailingBackend(),
        ledger_path=tmp_path / "error.sqlite3",
    ):
        with pytest.raises(RepairOutcomeUnknown, match="outcome is unknown"):
            RepairBrokerClient(socket_path).execute(
                repair_request(
                    "repair-error",
                    target=RoleTarget(research_tag="maple", role="advisor"),
                    command=command,
                )
            )

    audit = capsys.readouterr().err
    assert '"outcome": "unknown"' in audit
    assert '"error_type": "RuntimeError"' in audit
    assert command not in audit


def test_repair_receipt_replays_after_lost_reply_and_broker_restart(tmp_path):
    backend = RecordingRepairBackend()
    socket_path = _test_socket(tmp_path, "lost-repair")
    ledger_path = tmp_path / "repair.sqlite3"
    request = repair_request("repair-lost", command="printf fixed")
    envelope = json.dumps(
        {
            "protocol": REPAIR_PROTOCOL_VERSION,
            "operation": "execute",
            "request": request.model_dump(mode="json"),
        },
        separators=(",", ":"),
    ).encode() + b"\n"

    with RepairBrokerServer(
        socket_path,
        inventory(),
        backend,
        ledger_path=ledger_path,
    ):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(socket_path))
        connection.sendall(envelope)
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = RepairBrokerClient(socket_path).status("repair-lost")
            if status.status == "completed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("repair did not reach a durable completed state")

    with RepairBrokerServer(
        socket_path,
        inventory(),
        backend,
        ledger_path=ledger_path,
    ):
        replay = RepairBrokerClient(socket_path).execute(request)
        status = RepairBrokerClient(socket_path).status("repair-lost")

    assert replay == RepairResult(exit_code=0, stdout="fixed\n", stderr="")
    assert status.status == "completed"
    assert status.result == replay
    assert len(backend.calls) == 1


def test_repair_ledger_retains_only_the_newest_128_full_receipts(tmp_path):
    ledger_path = tmp_path / "repair.sqlite3"
    requests = [repair_request(f"repair-retention-{index:03d}") for index in range(129)]

    with RepairLedger(ledger_path) as ledger:
        for index, request in enumerate(requests):
            assert ledger.reserve(request).execute is True
            ledger.complete(
                request.operation_id,
                RepairResult(exit_code=index, stdout=f"receipt-{index}", stderr=""),
            )

        oldest = ledger.status(requests[0].operation_id)
        newest = ledger.status(requests[-1].operation_id)
        retained = ledger._connection.execute(
            "SELECT COUNT(*) FROM repair_operations WHERE result_json IS NOT NULL"
        ).fetchone()

    assert retained is not None
    assert retained[0] == 128
    assert oldest.status == "completed"
    assert oldest.receipt_retained is False
    assert oldest.result is None
    assert oldest.exit_code == 0
    assert oldest.payload_pruned_at is not None
    assert oldest.command_fingerprint == requests[0].command_fingerprint
    assert newest.receipt_retained is True
    assert newest.result == RepairResult(exit_code=128, stdout="receipt-128", stderr="")
    assert newest.exit_code == 128
    assert newest.payload_pruned_at is None


def test_pruned_repair_receipt_replay_is_typed_and_never_reexecutes_after_restart(
    tmp_path,
):
    ledger_path = tmp_path / "repair.sqlite3"
    requests = [repair_request(f"repair-expired-{index:03d}") for index in range(129)]
    with RepairLedger(ledger_path) as ledger:
        for index, request in enumerate(requests):
            assert ledger.reserve(request).execute is True
            ledger.complete(
                request.operation_id,
                RepairResult(exit_code=index, stdout=f"receipt-{index}", stderr=""),
            )

    backend = RecordingRepairBackend()
    socket_path = _test_socket(tmp_path, "expired-repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        backend,
        ledger_path=ledger_path,
    ):
        client = RepairBrokerClient(socket_path)
        with pytest.raises(RepairReceiptExpired) as raised:
            client.execute(requests[0])
        with pytest.raises(RepairIdempotencyConflict):
            client.execute(
                repair_request(requests[0].operation_id, command="changed command")
            )
        status = client.status(requests[0].operation_id)

    assert backend.calls == []
    assert raised.value.status == status
    assert status.status == "completed"
    assert status.receipt_retained is False
    assert status.exit_code == 0
    assert status.payload_pruned_at is not None


def test_repair_completion_and_payload_pruning_are_one_transaction(
    tmp_path,
    monkeypatch,
):
    ledger_path = tmp_path / "repair.sqlite3"
    request = repair_request("repair-atomic-retention")
    result = RepairResult(exit_code=0, stdout="receipt", stderr="")

    with RepairLedger(ledger_path) as ledger:
        assert ledger.reserve(request).execute is True

        def fail_pruning(_now):
            raise RuntimeError("injected pruning failure")

        original_pruning = ledger._prune_receipt_payloads
        monkeypatch.setattr(ledger, "_prune_receipt_payloads", fail_pruning)
        with pytest.raises(RuntimeError, match="injected pruning failure"):
            ledger.complete(request.operation_id, result)
        after_failure = ledger.status(request.operation_id)

        monkeypatch.setattr(ledger, "_prune_receipt_payloads", original_pruning)
        completed = ledger.complete(request.operation_id, result)

    assert after_failure.status == "running"
    assert after_failure.result is None
    assert after_failure.receipt_retained is False
    assert completed.status == "completed"
    assert completed.result == result


def test_repair_ledger_migrates_legacy_receipts_and_prunes_them_on_recovery(
    tmp_path,
    monkeypatch,
):
    from senpai_agent import repair_broker

    ledger_path = tmp_path / "repair.sqlite3"
    request = repair_request("repair-legacy")
    result = RepairResult(exit_code=17, stdout="legacy", stderr="")
    with sqlite3.connect(ledger_path) as connection:
        connection.execute(
            """
            CREATE TABLE repair_operations (
                operation_id TEXT PRIMARY KEY,
                command_fingerprint TEXT NOT NULL,
                research_tag TEXT NOT NULL,
                role TEXT NOT NULL,
                student TEXT,
                cwd TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                requested_at REAL NOT NULL,
                completed_at REAL,
                status TEXT NOT NULL,
                result_json TEXT,
                error_type TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO repair_operations (
                operation_id, command_fingerprint, research_tag, role, student,
                cwd, timeout_seconds, requested_at, completed_at, status,
                result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 2, 'completed', ?)
            """,
            (
                request.operation_id,
                request.command_fingerprint,
                request.target.research_tag,
                request.target.role,
                request.target.student,
                request.cwd,
                request.timeout_seconds,
                result.model_dump_json(),
            ),
        )

    with RepairLedger(ledger_path) as ledger:
        migrated = ledger.status(request.operation_id)
        columns = {
            row[1]
            for row in ledger._connection.execute(
                "PRAGMA table_info(repair_operations)"
            ).fetchall()
        }
        monkeypatch.setattr(repair_broker, "_RETAINED_REPAIR_PAYLOADS", 0)
        ledger.recover_interrupted()
        pruned = ledger.status(request.operation_id)
        audit = ledger.recent(limit=1)[0]

    assert {"receipt_retained", "exit_code", "payload_pruned_at"} <= columns
    assert migrated.receipt_retained is True
    assert migrated.result == result
    assert migrated.exit_code == 17
    assert migrated.payload_pruned_at is None
    assert pruned.receipt_retained is False
    assert pruned.result is None
    assert pruned.exit_code == 17
    assert pruned.payload_pruned_at is not None
    assert audit.receipt_retained is False
    assert audit.exit_code == 17
    assert audit.payload_pruned_at == pruned.payload_pruned_at


def test_repair_payload_pages_are_reused_after_the_retention_window(
    tmp_path,
    monkeypatch,
):
    from senpai_agent import repair_broker

    monkeypatch.setattr(repair_broker, "_RETAINED_REPAIR_PAYLOADS", 4)
    ledger_path = tmp_path / "repair.sqlite3"
    payload = "x" * (32 * 1024)
    with RepairLedger(ledger_path) as ledger:
        for index in range(8):
            _complete_repair(ledger, index, payload)
        first_page_count = ledger._connection.execute("PRAGMA page_count").fetchone()[0]
        first_free_pages = ledger._connection.execute(
            "PRAGMA freelist_count"
        ).fetchone()[0]

        for index in range(8, 40):
            _complete_repair(ledger, index, payload)
        second_page_count = ledger._connection.execute("PRAGMA page_count").fetchone()[
            0
        ]
        second_free_pages = ledger._connection.execute(
            "PRAGMA freelist_count"
        ).fetchone()[0]
        retained = ledger._connection.execute(
            "SELECT COUNT(*) FROM repair_operations WHERE result_json IS NOT NULL"
        ).fetchone()[0]

    first_live_pages = first_page_count - first_free_pages
    second_live_pages = second_page_count - second_free_pages
    assert retained == 4
    assert second_live_pages <= first_live_pages + 4
    assert second_page_count <= first_page_count + 8


def test_repair_operation_id_rejects_a_changed_command(tmp_path):
    socket_path = _test_socket(tmp_path, "conflict-repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        RecordingRepairBackend(),
        ledger_path=tmp_path / "repair.sqlite3",
    ):
        client = RepairBrokerClient(socket_path)
        client.execute(repair_request("repair-stable", command="true"))
        with pytest.raises(RepairIdempotencyConflict, match="different command"):
            client.execute(repair_request("repair-stable", command="false"))


def test_running_repair_becomes_queryable_unknown_after_broker_restart(tmp_path):
    ledger_path = tmp_path / "repair.sqlite3"
    request = repair_request("repair-interrupted", command="dangerous mutation")
    with RepairLedger(ledger_path) as ledger:
        assert ledger.reserve(request).status.status == "running"

    socket_path = _test_socket(tmp_path, "interrupted-repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        RecordingRepairBackend(),
        ledger_path=ledger_path,
    ) as restarted:
        status = restarted.ledger.status("repair-interrupted")

    assert status.status == "unknown"
    assert status.command_fingerprint == request.command_fingerprint


def test_repair_ledger_uses_race_safe_rollback_journal(tmp_path):
    from sqlite_test_support import assert_concurrent_first_open

    path = tmp_path / "repair.sqlite3"
    assert_concurrent_first_open(lambda: RepairLedger(path), workers=8)
    with RepairLedger(path) as ledger:
        mode = ledger._connection.execute("PRAGMA journal_mode").fetchone()

    assert mode is not None
    assert mode[0] == "delete"


@pytest.mark.parametrize("protocol", [None, "senpai-repair-broker/v0"])
def test_repair_broker_rejects_missing_or_stale_protocol(tmp_path, protocol):
    socket_path = _test_socket(tmp_path, "protocol-repair")
    envelope = {"operation": "status", "operation_id": "missing"}
    if protocol is not None:
        envelope["protocol"] = protocol

    with RepairBrokerServer(
        socket_path,
        inventory(),
        RecordingRepairBackend(),
        ledger_path=tmp_path / "repair.sqlite3",
    ):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
            connection.sendall(json.dumps(envelope).encode() + b"\n")
            response = json.loads(connection.makefile("rb").readline())

    assert response["protocol"] == REPAIR_PROTOCOL_VERSION
    assert response["error_type"] == "ValueError"
    assert "protocol" in response["error"]


def test_repair_broker_recovers_after_a_slow_drip_frame(tmp_path, monkeypatch):
    from senpai_agent import repair_broker

    monkeypatch.setattr(repair_broker, "_REQUEST_READ_TIMEOUT_SECONDS", 0.12)
    socket_path = _test_socket(tmp_path, "slow-drip-repair")
    with RepairBrokerServer(
        socket_path,
        inventory(),
        RecordingRepairBackend(),
        ledger_path=tmp_path / "repair.sqlite3",
    ):
        attacker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        attacker.connect(str(socket_path))

        def drip() -> None:
            try:
                for byte in b'{"protocol":':
                    attacker.send(bytes([byte]))
                    time.sleep(0.04)
            except OSError:
                pass
            finally:
                attacker.close()

        thread = threading.Thread(target=drip)
        thread.start()
        thread.join(timeout=2)
        result = RepairBrokerClient(socket_path).execute(
            repair_request("repair-after-slow-drip")
        )

    assert result.exit_code == 0

def _test_socket(tmp_path: Path, suffix: str) -> Path:
    digest = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:10]
    return Path("/private/tmp") / f"senpai-{os.getpid()}-{digest}-{suffix}.sock"


def _complete_repair(ledger: RepairLedger, index: int, payload: str) -> None:
    request = repair_request(f"repair-pages-{index:03d}")
    assert ledger.reserve(request).execute is True
    ledger.complete(
        request.operation_id,
        RepairResult(exit_code=index, stdout=payload, stderr=""),
    )
