from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from openhands.sdk.event import MessageEvent
from openhands.sdk.event.types import ROOT_PARENT_ID
from openhands.sdk.llm import Message, TextContent

from senpai_agent.advisor import AdvisorEventStore
from senpai_agent.operations import (
    ContextResetRequest,
    ContextResetRequestStore,
    RestartRequest,
    RestartRequestStore,
    RoleObservation,
    RoleTarget,
)
from senpai_agent.role_control import (
    RoleRuntimeState,
    _active_delegation_count,
    _control_token,
    _pending_event_keys,
    _require_control_token,
    _require_restart_control_token,
    _restart_control_token,
    _training_state,
    advisor_research_tail,
    nudge,
    observe_role,
    request_context_reset,
    restart_controller,
)
from senpai_agent.supervisor import WorkerLease
from senpai_agent.training import (
    TRAINING_INVENTORY_FILENAME,
    TrainingResult,
    TrainingState,
    TrainingSupervisor,
    read_training_inventory,
)


CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000201")


def student_env(tmp_path: Path) -> dict[str, str]:
    state = tmp_path / "state"
    state.mkdir()
    return {
        "SENPAI_ROLE": "student",
        "RESEARCH_TAG": "maple",
        "STUDENT_NAME": "fern",
        "SENPAI_OPENHANDS_STATE_DIR": str(state),
        "SENPAI_OPENHANDS_WORKSPACE": str(tmp_path),
    }


def runtime_state(
    target: RoleTarget,
    *,
    running: int = 0,
    active_delegations: int | None = 0,
) -> RoleRuntimeState:
    return RoleRuntimeState(
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
            raw_history_event_count=2,
            raw_history_digest="digest-1",
            pending_event_keys=("existing",),
            active_delegation_count=active_delegations,
        ),
        lease_deadline_seconds=60,
        completed_turns=3,
        running_training_count=running,
        active_delegation_count=active_delegations,
        wandb_run_inventory_complete=True,
        cpu_percent=10,
        memory_percent=20,
        disk_percent=30,
        gpu_percent=40,
    )


def test_control_token_ignores_routine_lease_phase_and_deadline_changes():
    first = WorkerLease(pid=123, phase="sleep", deadline=100)
    second = WorkerLease(pid=123, phase="poll", deadline=200)

    assert _control_token(first, CONVERSATION_ID, "digest", ("event",)) == (
        _control_token(second, CONVERSATION_ID, "digest", ("event",))
    )


def test_restart_control_token_binds_phase_and_completed_turns():
    stable = "stable-token"
    sleeping = WorkerLease(pid=123, phase="sleep", deadline=100, completed_turns=4)
    reconciling = WorkerLease(
        pid=123,
        phase="reconcile",
        deadline=200,
        completed_turns=4,
    )
    advanced = WorkerLease(pid=123, phase="sleep", deadline=300, completed_turns=5)

    assert _restart_control_token(stable, sleeping) != _restart_control_token(
        stable,
        reconciling,
    )
    assert _restart_control_token(stable, sleeping) != _restart_control_token(
        stable,
        advanced,
    )


def test_control_token_changes_with_active_delegation_state():
    lease = WorkerLease(pid=123, phase="sleep", deadline=100)

    assert _control_token(
        lease,
        CONVERSATION_ID,
        "digest",
        ("event",),
        0,
    ) != _control_token(
        lease,
        CONVERSATION_ID,
        "digest",
        ("event",),
        1,
    )


def test_role_observation_counts_all_active_delegation_states(tmp_path: Path):
    env = student_env(tmp_path)
    database_path = (
        Path(env["SENPAI_OPENHANDS_STATE_DIR"])
        / "delegation"
        / "tasks.sqlite3"
    )
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE tasks (status TEXT NOT NULL)")
        database.executemany(
            "INSERT INTO tasks VALUES (?)",
            [
                ("queued",),
                ("starting",),
                ("running",),
                ("finished",),
                ("failed",),
                ("cancelled",),
            ],
        )

    runtime = observe_role(env)

    assert runtime.active_delegation_count == 3
    assert runtime.observation.active_delegation_count == 3
    assert runtime.observation.control_token == _control_token(
        None,
        None,
        None,
        (),
        3,
    )
    assert runtime.observation.worker_generation is None
    assert runtime.observation.restart_control_token is None


def test_new_delegation_invalidates_a_previously_observed_control_token(
    tmp_path: Path,
):
    env = student_env(tmp_path)
    before = observe_role(env)
    database_path = (
        Path(env["SENPAI_OPENHANDS_STATE_DIR"])
        / "delegation"
        / "tasks.sqlite3"
    )
    database_path.parent.mkdir()
    with sqlite3.connect(database_path) as database:
        database.execute("CREATE TABLE tasks (status TEXT NOT NULL)")
        database.execute("INSERT INTO tasks VALUES ('queued')")

    with pytest.raises(RuntimeError, match="role state changed"):
        _require_control_token(before.observation.control_token, env)


def test_unreadable_delegation_inventory_is_reported_unknown(tmp_path: Path):
    env = student_env(tmp_path)
    database_path = (
        Path(env["SENPAI_OPENHANDS_STATE_DIR"])
        / "delegation"
        / "tasks.sqlite3"
    )
    database_path.parent.mkdir()
    database_path.write_text("not a sqlite database")

    assert _active_delegation_count(database_path.parents[1]) is None
    runtime = observe_role(env)
    assert runtime.active_delegation_count is None
    assert runtime.observation.active_delegation_count is None


def test_pending_event_observation_fails_closed_when_the_queue_is_unreadable(
    tmp_path: Path,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "student-events.sqlite3").write_text("not a sqlite database")
    target = RoleTarget(research_tag="maple", role="student", student="fern")

    with pytest.raises(RuntimeError, match="pending role events are unreadable"):
        _pending_event_keys(state_dir, target, CONVERSATION_ID)


def test_research_tail_contains_only_active_branch_agent_messages(
    tmp_path: Path,
):
    env = {
        **student_env(tmp_path),
        "SENPAI_ROLE": "advisor",
        "EXA_API_KEY": "role-only-exa-secret",
    }
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"])
    (state_dir / "advisor-conversation-id").write_text(f"{CONVERSATION_ID}\n")
    events_dir = state_dir / CONVERSATION_ID.hex / "events"
    events_dir.mkdir(parents=True)
    user = MessageEvent(
        source="user",
        parent_id=ROOT_PARENT_ID,
        llm_message=Message(
            role="user",
            content=[TextContent(text="secret user instruction")],
        ),
    )
    first = MessageEvent(
        source="agent",
        parent_id=user.id,
        llm_message=Message(
            role="assistant",
            content=[TextContent(text="I am comparing two mechanisms.")],
        ),
    )
    abandoned = MessageEvent(
        source="agent",
        parent_id=user.id,
        llm_message=Message(
            role="assistant",
            content=[TextContent(text="abandoned branch noise")],
        ),
    )
    current = MessageEvent(
        source="agent",
        parent_id=first.id,
        llm_message=Message(
            role="assistant",
            content=[
                TextContent(
                    text=(
                        "The mechanism evidence favors fusion. "
                        "role-only-exa-secret"
                    )
                )
            ],
        ),
    )
    for index, event in enumerate((user, first, abandoned, current)):
        (events_dir / f"event-{index:05d}-{event.id}.json").write_text(
            event.model_dump_json(exclude_none=True)
        )
    (state_dir / CONVERSATION_ID.hex / "base_state.json").write_text(
        json.dumps({"leaf_event_id": current.id})
    )

    tail = advisor_research_tail(env)

    summaries = [message.summary for message in tail.messages]
    assert summaries == [
        "I am comparing two mechanisms.",
        "The mechanism evidence favors fusion. <redacted>",
    ]
    assert "role-only-exa-secret" not in " ".join(summaries)
    assert "secret user instruction" not in " ".join(summaries)
    assert "abandoned branch noise" not in " ".join(summaries)


def test_research_tail_fails_closed_if_history_changes_during_read(
    tmp_path: Path,
    monkeypatch,
):
    env = {
        **student_env(tmp_path),
        "SENPAI_ROLE": "advisor",
    }
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"])
    (state_dir / "advisor-conversation-id").write_text(f"{CONVERSATION_ID}\n")
    checkpoints = iter(((1, "first"), (2, "second")))
    monkeypatch.setattr(
        "senpai_agent.role_control._raw_history_checkpoint",
        lambda *_args, **_kwargs: next(checkpoints),
    )
    monkeypatch.setattr("senpai_agent.role_control._active_branch", lambda *_args: [])

    with pytest.raises(RuntimeError, match="changed during inspection"):
        advisor_research_tail(env)


def test_student_nudge_is_bound_to_the_existing_conversation(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    monkeypatch.setattr(
        "senpai_agent.role_control._require_control_token",
        lambda _token, _env: runtime_state(target),
    )

    receipt = nudge(
        CONVERSATION_ID,
        "token-1",
        "Resume the existing assignment after checking the stale worker.",
        "nudge-fern-1",
        env,
    )

    assert receipt.conversation_id == CONVERSATION_ID
    with AdvisorEventStore(
        Path(env["SENPAI_OPENHANDS_STATE_DIR"]) / "student-events.sqlite3"
    ) as store:
        event = store.pending()[0]
    assert event.dedupe_key == "supervisor-nudge:nudge-fern-1"
    assert event.payload["parent_conversation_id"] == str(CONVERSATION_ID)


def test_same_nudge_text_with_a_new_operation_key_is_delivered_again(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    monkeypatch.setattr(
        "senpai_agent.role_control._require_control_token",
        lambda _token, _env: runtime_state(target),
    )
    message = "Resume after checking the same recurring worker fault."
    database = Path(env["SENPAI_OPENHANDS_STATE_DIR"]) / "student-events.sqlite3"

    nudge(CONVERSATION_ID, "token-1", message, "first-operation", env)
    with AdvisorEventStore(database) as store:
        store.acknowledge("supervisor-nudge:first-operation")
    nudge(CONVERSATION_ID, "token-1", message, "second-operation", env)
    nudge(CONVERSATION_ID, "token-1", message, "second-operation", env)

    with AdvisorEventStore(database) as store:
        assert [event.dedupe_key for event in store.pending()] == [
            "supervisor-nudge:second-operation"
        ]


def test_context_reset_is_only_queued_and_does_not_touch_raw_history(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"])
    history = state_dir / CONVERSATION_ID.hex / "events" / "event-0.json"
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps({"raw": "preserve-me"}))
    monkeypatch.setattr(
        "senpai_agent.role_control._require_control_token",
        lambda _token, _env: runtime_state(target),
    )
    request = ContextResetRequest(
        request_id="reset-fern-1",
        target=target,
        expected_conversation_id=CONVERSATION_ID,
        expected_control_token="token-1",
        expected_raw_history_event_count=2,
        expected_raw_history_digest="digest-1",
        expected_pending_event_keys=("existing",),
        recovery_prompt="Resume from the current assignment and ignore old noise.",
    )

    receipt = request_context_reset(request, env)

    assert receipt.status == "queued"
    assert json.loads(history.read_text()) == {"raw": "preserve-me"}
    with ContextResetRequestStore(state_dir / "context-resets.sqlite3") as store:
        assert store.pending(target) == (request,)


def test_role_inspection_exposes_a_sanitized_processing_reset_after_a_crash(
    tmp_path: Path,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    state_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"])
    request = ContextResetRequest(
        request_id="reset-processing-fern",
        target=target,
        expected_conversation_id=CONVERSATION_ID,
        expected_control_token="private-control-token",
        expected_raw_history_event_count=2,
        expected_raw_history_digest="private-history-digest",
        expected_pending_event_keys=("private-event-key",),
        recovery_prompt="private recovery instructions that must not be inspected",
    )
    with ContextResetRequestStore(state_dir / "context-resets.sqlite3") as store:
        store.enqueue(request)
        assert store.claim_next(target) == request

    runtime = observe_role(env)

    assert len(runtime.context_resets) == 1
    assert runtime.context_resets[0].request_id == request.request_id
    assert runtime.context_resets[0].status == "processing"
    assert runtime.context_resets[0].conversation_id == CONVERSATION_ID
    encoded = runtime.model_dump_json()
    assert "private recovery instructions" not in encoded
    assert "private-control-token" not in encoded
    assert "private-history-digest" not in encoded
    assert "private-event-key" not in encoded


def test_stale_running_training_state_does_not_block_recovery(tmp_path: Path):
    env = student_env(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    training_dir = Path(env["SENPAI_OPENHANDS_STATE_DIR"]) / "training"
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=training_dir)
    result = TrainingResult(
        training_id="stale-training",
        state=TrainingState.RUNNING,
        pid=999_999_999,
        process_group_id=999_999_999,
        process_start_time=1.0,
        exit_code=None,
        elapsed_seconds=60,
        log_path="/private/stale.log",
        wandb_run_ids=("stale-run",),
        error_tail="role-only-secret-must-not-leak",
    )
    supervisor._write_result(result)

    runtime = observe_role(env)

    assert runtime.running_training_count == 0
    assert runtime.running_wandb_run_ids == ()
    assert runtime.wandb_run_inventory_complete is True
    assert any("SENPAI_TRAINING_STALE" in item for item in runtime.recent_errors)
    assert "role-only-secret-must-not-leak" not in runtime.model_dump_json()


@pytest.mark.parametrize(
    ("run_count", "inventory_complete"),
    ((50, True), (51, False)),
)
def test_running_wandb_inventory_reports_cap_truncation(
    tmp_path: Path,
    monkeypatch,
    run_count: int,
    inventory_complete: bool,
):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    training_dir = state_dir / "training"
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=training_dir)
    result = TrainingResult(
        training_id="running-training",
        state=TrainingState.RUNNING,
        pid=123,
        process_group_id=123,
        process_start_time=1.0,
        exit_code=None,
        elapsed_seconds=60,
        log_path="/private/running.log",
        wandb_run_ids=tuple(f"running-{index}" for index in range(run_count)),
    )
    supervisor._write_result(result)
    monkeypatch.setattr(
        "senpai_agent.role_control.training_process_is_live",
        lambda _result: True,
    )

    running, _, running_ids, recent_ids, complete = _training_state(
        state_dir,
        "student",
    )

    assert running == 1
    assert len(running_ids) == min(run_count, 50)
    assert len(recent_ids) == run_count
    assert complete is inventory_complete


def test_advisor_job_inventory_is_not_reported_as_idle(
    tmp_path: Path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    supervisor = TrainingSupervisor(
        workspace=workspace,
        state_dir=state_dir / "training",
    )
    supervisor._write_result(
        TrainingResult(
            training_id="advisor-job",
            state=TrainingState.RUNNING,
            pid=123,
            process_group_id=123,
            process_start_time=1.0,
            exit_code=None,
            elapsed_seconds=60,
            log_path="/private/advisor-job.log",
            wandb_run_ids=("advisor-run",),
        )
    )
    monkeypatch.setattr(
        "senpai_agent.role_control.training_process_is_live",
        lambda _result: True,
    )

    running, _, running_ids, _, complete = _training_state(state_dir, "advisor")

    assert running == 1
    assert running_ids == ("advisor-run",)
    assert complete is True


@pytest.mark.parametrize(
    ("run_count", "inventory_complete"),
    ((200, True), (201, False)),
)
def test_recent_wandb_inventory_reports_cap_truncation(
    tmp_path: Path,
    run_count: int,
    inventory_complete: bool,
):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    training_dir = state_dir / "training"
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=training_dir)
    result = TrainingResult(
        training_id="finished-training",
        state=TrainingState.FINISHED,
        exit_code=0,
        elapsed_seconds=60,
        log_path="/private/finished.log",
        wandb_run_ids=tuple(f"finished-{index}" for index in range(run_count)),
    )
    supervisor._write_result(result)

    running, _, running_ids, recent_ids, complete = _training_state(
        state_dir,
        "student",
    )

    assert running == 0
    assert running_ids == ()
    assert len(recent_ids) == min(run_count, 200)
    assert complete is inventory_complete


def test_training_observation_cost_is_bounded_by_inventory_not_history(
    tmp_path: Path,
):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    training_dir = state_dir / "training"
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=training_dir)
    first_path = None
    for index in range(300):
        result = TrainingResult(
            training_id=f"finished-{index}",
            state=TrainingState.FINISHED,
            exit_code=0,
            elapsed_seconds=60,
            log_path=f"/private/finished-{index}.log",
            wandb_run_ids=(f"run-{index}",),
        )
        supervisor._write_result(result)
        first_path = first_path or training_dir / f"{result.training_id}.json"

    assert first_path is not None
    first_path.write_text("corrupt historical result")
    inventory = read_training_inventory(training_dir)
    running, errors, running_ids, recent_ids, complete = _training_state(
        state_dir,
        "student",
    )

    assert len(inventory.recent_terminal) == 64
    assert len(inventory.recent_wandb_run_ids) == 200
    assert inventory.wandb_run_inventory_overflow is True
    assert running == 0
    assert errors == ()
    assert running_ids == ()
    assert len(recent_ids) == 200
    assert complete is False


def test_corrupt_training_inventory_fails_closed(tmp_path: Path):
    state_dir = tmp_path / "state"
    training_dir = state_dir / "training"
    training_dir.mkdir(parents=True)
    (training_dir / TRAINING_INVENTORY_FILENAME).write_text("not valid json")

    running, errors, running_ids, recent_ids, complete = _training_state(
        state_dir,
        "student",
    )

    assert running is None
    assert errors == ("SENPAI_TRAINING_INVENTORY_UNREADABLE",)
    assert running_ids == ()
    assert recent_ids == ()
    assert complete is None


def test_active_result_disagreement_fails_closed(tmp_path: Path, monkeypatch):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    training_dir = state_dir / "training"
    supervisor = TrainingSupervisor(workspace=workspace, state_dir=training_dir)
    result = TrainingResult(
        training_id="running-training",
        state=TrainingState.RUNNING,
        pid=123,
        process_group_id=123,
        process_start_time=1.0,
        exit_code=None,
        elapsed_seconds=60,
        log_path="/private/running.log",
        wandb_run_ids=("run-a",),
    )
    supervisor._write_result(result)
    (training_dir / "running-training.json").write_text(
        result.model_copy(update={"wandb_run_ids": ("unexpected",)}).model_dump_json()
    )
    monkeypatch.setattr(
        "senpai_agent.role_control.training_process_is_live",
        lambda _result: True,
    )

    running, errors, running_ids, recent_ids, complete = _training_state(
        state_dir,
        "student",
    )

    assert running is None
    assert errors == ("SENPAI_TRAINING_INVENTORY_UNREADABLE",)
    assert running_ids == ()
    assert recent_ids == ()
    assert complete is None


def test_controller_restart_refuses_to_interrupt_a_running_experiment(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: runtime_state(target, running=1),
    )

    with pytest.raises(RuntimeError, match="supervised job is running"):
        restart_controller(CONVERSATION_ID, "token-1", env)


def test_controller_restart_is_queued_for_the_worker_owner_without_signalling(
    tmp_path: Path,
    monkeypatch,
):
    """
    Requirement: the role-control endpoint may request a restart but cannot signal
    the controller process owned by WorkerSupervisor.
    Interface: restart_controller's receipt and the role's durable state directory.
    """

    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    state = runtime_state(target)
    lease = WorkerLease(
        pid=123,
        phase="sleep",
        deadline=100,
        completed_turns=state.completed_turns or 0,
        conversation_id=str(CONVERSATION_ID),
        generation=7,
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: state,
    )
    monkeypatch.setattr("senpai_agent.role_control._read_lease", lambda _path: lease)
    monkeypatch.setattr(
        "senpai_agent.role_control._controller_alive",
        lambda _lease, _role: True,
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._training_state",
        lambda _state_dir, _role: (0, (), (), (), True),
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._restart_control_token",
        lambda *_args: "restart-token-1",
    )
    killed = []
    monkeypatch.setattr(
        "senpai_agent.role_control.os.kill",
        lambda *args: killed.append(args),
    )

    receipt = restart_controller(CONVERSATION_ID, "restart-token-1", env)
    replay = restart_controller(CONVERSATION_ID, "restart-token-1", env)

    assert killed == []
    assert replay == receipt
    assert receipt.model_dump()["status"] == "queued"
    assert receipt.model_dump()["request_id"]
    assert receipt.model_dump()["expected_worker_generation"] == 7
    assert receipt.state_preserved is None
    assert receipt.compute_preserved is None
    assert (
        Path(env["SENPAI_OPENHANDS_STATE_DIR"]) / "controller-restarts.sqlite3"
    ).is_file()


def test_role_observation_exposes_sanitized_durable_restart_status(
    tmp_path: Path,
):
    """
    Requirement: operational snapshots distinguish queued/processing/completed/
    rejected restarts without exposing restart authorization tokens.
    Interface: observe_role's bounded RoleRuntimeState payload.
    """

    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    request = RestartRequest(
        request_id="restart-status-1",
        target=target,
        expected_conversation_id=CONVERSATION_ID,
        expected_restart_control_token="must-not-appear-in-observation",
        expected_worker_generation=3,
        expected_completed_turns=2,
    )
    with RestartRequestStore(
        Path(env["SENPAI_OPENHANDS_STATE_DIR"]) / "controller-restarts.sqlite3"
    ) as store:
        store.enqueue(request)

    runtime = observe_role(env)

    assert len(runtime.controller_restarts) == 1
    status = runtime.controller_restarts[0]
    assert status.request_id == request.request_id
    assert status.status == "queued"
    assert status.source_generation == 3
    assert "must-not-appear" not in runtime.model_dump_json()


def test_controller_restart_fails_closed_for_a_legacy_unowned_worker_lease(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    state = runtime_state(target)
    lease = WorkerLease(
        pid=123,
        phase="sleep",
        deadline=100,
        completed_turns=state.completed_turns or 0,
        conversation_id=str(CONVERSATION_ID),
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: state,
    )
    monkeypatch.setattr("senpai_agent.role_control._read_lease", lambda _path: lease)
    monkeypatch.setattr(
        "senpai_agent.role_control._controller_alive",
        lambda _lease, _role: True,
    )

    with pytest.raises(RuntimeError, match="worker generation"):
        restart_controller(CONVERSATION_ID, "restart-token-1", env)

    assert not (
        Path(env["SENPAI_OPENHANDS_STATE_DIR"]) / "controller-restarts.sqlite3"
    ).exists()


def test_controller_restart_refuses_an_advisor_with_a_running_job(
    tmp_path: Path,
    monkeypatch,
):
    env = {**student_env(tmp_path), "SENPAI_ROLE": "advisor"}
    target = RoleTarget(research_tag="maple", role="advisor")
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: runtime_state(target, running=1),
    )

    with pytest.raises(RuntimeError, match="supervised job is running"):
        restart_controller(CONVERSATION_ID, "token-1", env)


def test_controller_restart_rechecks_jobs_immediately_before_queueing(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    state = runtime_state(target)
    lease = WorkerLease(
        pid=123,
        phase="sleep",
        deadline=100,
        completed_turns=state.completed_turns or 0,
        generation=1,
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: state,
    )
    monkeypatch.setattr("senpai_agent.role_control._read_lease", lambda _path: lease)
    monkeypatch.setattr(
        "senpai_agent.role_control._controller_alive",
        lambda _lease, _role: True,
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._training_state",
        lambda _state_dir, _role: (1, (), (), (), True),
    )
    killed = []
    monkeypatch.setattr(
        "senpai_agent.role_control.os.kill",
        lambda *args: killed.append(args),
    )

    with pytest.raises(RuntimeError, match="started before controller restart"):
        restart_controller(CONVERSATION_ID, "token-1", env)

    assert killed == []


def test_controller_restart_rejects_a_missing_conversation_uuid(tmp_path: Path):
    env = student_env(tmp_path)

    with pytest.raises(ValueError, match="expected conversation UUID"):
        restart_controller(None, "token-1", env)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("active_delegations", "message"),
    [
        (1, "delegated task is active"),
        (None, "delegation activity is unknown"),
    ],
)
def test_controller_restart_refuses_active_or_unknown_delegation_inventory(
    tmp_path: Path,
    monkeypatch,
    active_delegations: int | None,
    message: str,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: runtime_state(
            target,
            active_delegations=active_delegations,
        ),
    )
    killed = []
    monkeypatch.setattr("senpai_agent.role_control.os.kill", lambda *args: killed.append(args))

    with pytest.raises(RuntimeError, match=message):
        restart_controller(CONVERSATION_ID, "token-1", env)

    assert killed == []


def test_controller_restart_refuses_a_non_quiescent_phase(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    reconciling = runtime_state(target).model_copy(
        update={
            "observation": runtime_state(target).observation.model_copy(
                update={"controller_phase": "reconcile"}
            )
        }
    )
    monkeypatch.setattr(
        "senpai_agent.role_control._require_restart_control_token",
        lambda _token, _env: reconciling,
    )
    killed = []
    monkeypatch.setattr(
        "senpai_agent.role_control.os.kill",
        lambda *args: killed.append(args),
    )

    with pytest.raises(RuntimeError, match="quiescent sleep phase"):
        restart_controller(CONVERSATION_ID, "restart-token-1", env)

    assert killed == []


def test_sleep_observation_cannot_authorize_a_reconcile_restart(
    tmp_path: Path,
    monkeypatch,
):
    env = student_env(tmp_path)
    target = RoleTarget(research_tag="maple", role="student", student="fern")
    before = runtime_state(target)
    after = before.model_copy(
        update={
            "observation": before.observation.model_copy(
                update={
                    "controller_phase": "reconcile",
                    "restart_control_token": "restart-token-2",
                }
            )
        }
    )
    monkeypatch.setattr(
        "senpai_agent.role_control.observe_role",
        lambda _env: after,
    )
    killed = []
    monkeypatch.setattr(
        "senpai_agent.role_control.os.kill",
        lambda *args: killed.append(args),
    )

    with pytest.raises(RuntimeError, match="restart state changed"):
        _require_restart_control_token(
            before.observation.restart_control_token,
            env,
        )

    assert killed == []
