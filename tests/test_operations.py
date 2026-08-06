from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from senpai_agent.operations import (
    CampaignInventory,
    CollectRole,
    CollectRoleReceipt,
    ContextReset,
    ContextResetCompletion,
    ContextResetRequest,
    ContextResetRequestStore,
    ContextResetReceipt,
    IdempotencyConflict,
    Nudge,
    NudgeReceipt,
    OperationBackend,
    OperationInvariantError,
    OperationLedger,
    OperationPolicy,
    OperationService,
    RecordedOperationError,
    Restart,
    RestartReceipt,
    RoleObservation,
    RoleTarget,
    UnsafeContextReset,
)


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
ADVISOR_ID = UUID("00000000-0000-0000-0000-000000000101")
STUDENT_ID = UUID("00000000-0000-0000-0000-000000000102")


def target(role: str = "advisor", student: str | None = None) -> RoleTarget:
    return RoleTarget(
        research_tag="maple",
        role=role,
        student=student,
    )


def observation(
    role_target: RoleTarget,
    *,
    conversation_id: UUID | None = None,
    active_turn: bool | None = False,
    unmatched_actions: int | None = 0,
    history: tuple[str, ...] = ("old error", "useful result"),
    pending_event_keys: tuple[str, ...] = ("review:17",),
    control_token: str = "lease-1",
    restart_control_token: str | None = "restart-1",
) -> RoleObservation:
    if conversation_id is None:
        conversation_id = ADVISOR_ID if role_target.role == "advisor" else STUDENT_ID
    digest = hashlib.sha256(json.dumps(history).encode()).hexdigest()
    return RoleObservation(
        target=role_target,
        observed_at=NOW,
        control_token=control_token,
        restart_control_token=restart_control_token,
        controller_alive=True,
        controller_phase="sleep",
        conversation_id=conversation_id,
        active_turn=active_turn,
        unmatched_actions=unmatched_actions,
        raw_history_event_count=len(history),
        raw_history_digest=digest,
        pending_event_keys=pending_event_keys,
    )


class FakeBackend(OperationBackend):
    def __init__(self):
        self.roles = {
            target(): observation(target()),
            target("student", "fern"): observation(target("student", "fern")),
            target("student", "frieren"): observation(
                target("student", "frieren")
            ),
        }
        self.calls: list[tuple[object, ...]] = []
        self.raw_history = {
            role_target.key: ["old error", "useful result"]
            for role_target in self.roles
        }
        self.active_history = {
            role_target.key: list(history)
            for role_target, history in (
                (role_target, self.raw_history[role_target.key])
                for role_target in self.roles
            )
        }
        self.failure: BaseException | None = None
        self.damage_reset_request = False
        self.reset_requests: list[ContextResetRequest] = []
        self.restart_preserves_state = True
        self.restart_preserves_compute = True

    def _fail(self):
        if self.failure is not None:
            raise self.failure

    def collect_role(self, role_target: RoleTarget) -> RoleObservation:
        self.calls.append(("collect_role", role_target))
        self._fail()
        return self.roles[role_target]

    def nudge(
        self,
        role_target: RoleTarget,
        *,
        operation_key: str,
        expected_conversation_id: UUID,
        message: str,
        control_token: str,
    ) -> NudgeReceipt:
        self.calls.append(
            (
                "nudge",
                role_target,
                operation_key,
                expected_conversation_id,
                message,
                control_token,
            )
        )
        self._fail()
        current = self.roles[role_target]
        assert current.conversation_id == expected_conversation_id
        assert current.control_token == control_token
        delivery_key = f"supervisor-nudge:{len(self.calls)}"
        self.roles[role_target] = current.model_copy(
            update={
                "pending_event_keys": (*current.pending_event_keys, delivery_key),
                "control_token": f"lease-{len(self.calls)}",
            }
        )
        return NudgeReceipt(
            target=role_target,
            conversation_id=expected_conversation_id,
            delivery_key=delivery_key,
        )

    def restart_controller(
        self,
        role_target: RoleTarget,
        *,
        expected_conversation_id: UUID,
        restart_control_token: str,
    ) -> RestartReceipt:
        self.calls.append(
            (
                "restart_controller",
                role_target,
                expected_conversation_id,
                restart_control_token,
            )
        )
        self._fail()
        current = self.roles[role_target]
        assert current.restart_control_token == restart_control_token
        assert current.conversation_id == expected_conversation_id
        self.roles[role_target] = current.model_copy(
            update={
                "control_token": f"lease-{len(self.calls)}",
                "restart_control_token": f"restart-{len(self.calls)}",
                "controller_phase": "startup",
            }
        )
        return RestartReceipt(
            target=role_target,
            conversation_id=expected_conversation_id,
            state_preserved=self.restart_preserves_state,
            compute_preserved=self.restart_preserves_compute,
        )

    def request_context_reset(
        self,
        role_target: RoleTarget,
        *,
        request: ContextResetRequest,
    ) -> ContextResetReceipt:
        self.calls.append(
            (
                "request_context_reset",
                role_target,
                request,
            )
        )
        self._fail()
        current = self.roles[role_target]
        assert current.conversation_id == request.expected_conversation_id
        assert current.control_token == request.expected_control_token
        self.reset_requests.append(request)
        return ContextResetReceipt(
            target=role_target,
            request_id=request.request_id,
            expected_conversation_id=request.expected_conversation_id,
            expected_raw_history_event_count=(
                request.expected_raw_history_event_count
            ),
            expected_raw_history_digest=(
                "damaged"
                if self.damage_reset_request
                else request.expected_raw_history_digest
            ),
            expected_pending_event_keys=request.expected_pending_event_keys,
        )

@pytest.fixture
def inventory() -> CampaignInventory:
    return CampaignInventory(
        research_tag="maple",
        repo="acme/widgets",
        advisor_branch="maple-advisor",
        students=("fern", "frieren"),
    )


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def service(
    tmp_path: Path,
    inventory: CampaignInventory,
    backend: FakeBackend,
) -> OperationService:
    return OperationService(
        inventory,
        backend,
        OperationLedger(tmp_path / "operations.sqlite3"),
        policy=OperationPolicy(
            mutation_cooldown_seconds=60,
        ),
    )


def test_inventory_accepts_only_the_exact_campaign_roles(
    service: OperationService,
    backend: FakeBackend,
):
    allowed = [target(), target("student", "fern"), target("student", "frieren")]

    for index, role_target in enumerate(allowed):
        result = service.execute(
            CollectRole(
                operation_key=f"collect-{index}",
                target=role_target,
            ),
            now=NOW,
        )
        assert isinstance(result.receipt, CollectRoleReceipt)
        assert result.receipt.observation.target == role_target

    before = list(backend.calls)
    for forbidden in (
        RoleTarget(research_tag="cedar", role="advisor"),
        RoleTarget(research_tag="maple", role="student", student="other"),
    ):
        with pytest.raises(PermissionError, match="campaign inventory"):
            service.execute(
                CollectRole(operation_key=f"bad-{forbidden.key}", target=forbidden),
                now=NOW,
            )
    assert backend.calls == before


def test_collect_role_executes_fresh_even_when_the_operation_key_is_reused(
    service: OperationService,
    backend: FakeBackend,
):
    first = CollectRole(operation_key="collect-1", target=target())

    executed = service.execute(first, now=NOW)
    backend.roles[target()] = backend.roles[target()].model_copy(
        update={
            "observed_at": NOW + timedelta(seconds=1),
            "control_token": "lease-2",
        }
    )
    refreshed_same_key = service.execute(first, now=NOW + timedelta(seconds=1))
    refreshed = service.execute(
        CollectRole(operation_key="collect-2", target=target()),
        now=NOW + timedelta(seconds=2),
    )

    assert executed.disposition == "executed"
    assert refreshed_same_key.disposition == "executed"
    assert isinstance(refreshed_same_key.receipt, CollectRoleReceipt)
    assert refreshed_same_key.receipt.observation.control_token == "lease-2"
    assert refreshed.disposition == "executed"
    assert [call[0] for call in backend.calls] == [
        "collect_role",
        "collect_role",
        "collect_role",
    ]
    assert service.ledger.records() == []


def test_operation_timestamp_must_be_timezone_aware(
    service: OperationService,
    backend: FakeBackend,
):
    with pytest.raises(ValueError, match="timezone-aware"):
        service.execute(
            CollectRole(operation_key="naive-time", target=target()),
            now=datetime(2026, 8, 6, 12, 0),
        )

    assert backend.calls == []


def test_mutation_idempotency_and_cooldown_survive_a_service_restart(
    tmp_path: Path,
    inventory: CampaignInventory,
    backend: FakeBackend,
):
    database = tmp_path / "operations.sqlite3"
    policy = OperationPolicy(mutation_cooldown_seconds=60)
    action = Nudge(
        operation_key="nudge-1",
        incident_key="idle-fern",
        target=target("student", "fern"),
        expected_conversation_id=STUDENT_ID,
        message="Resume the existing assignment and report the blocker.",
        reason="The student has been idle for three snapshots.",
    )
    first_service = OperationService(
        inventory,
        backend,
        OperationLedger(database),
        policy=policy,
    )

    first = first_service.execute(action, now=NOW)
    replay = first_service.execute(action, now=NOW + timedelta(seconds=1))
    first_service.close()
    restarted = OperationService(
        inventory,
        backend,
        OperationLedger(database),
        policy=policy,
    )
    suppressed = restarted.execute(
        action.model_copy(
            update={
                "operation_key": "nudge-2",
                "incident_key": "the-model-renamed-this-incident",
            }
        ),
        now=NOW + timedelta(seconds=30),
    )
    after_cooldown = restarted.execute(
        action.model_copy(update={"operation_key": "nudge-3"}),
        now=NOW + timedelta(seconds=61),
    )

    assert first.disposition == "executed"
    assert replay.disposition == "replayed"
    assert suppressed.disposition == "suppressed"
    assert suppressed.source_operation_key == "nudge-1"
    assert after_cooldown.disposition == "executed"
    assert [call[0] for call in backend.calls].count("nudge") == 2


def test_cooldown_is_per_typed_action_and_target_not_model_incident_name(
    service: OperationService,
    backend: FakeBackend,
):
    base = Nudge(
        operation_key="base",
        incident_key="stale-wip",
        anomaly_category="stale_wip",
        target=target("student", "fern"),
        expected_conversation_id=STUDENT_ID,
        message="Please inspect the stale WIP.",
        reason="No progress was observed.",
    )
    service.execute(base, now=NOW)

    renamed_incident = service.execute(
        base.model_copy(
            update={"operation_key": "other-incident", "incident_key": "deferred"}
        ),
        now=NOW + timedelta(seconds=1),
    )
    different_target = service.execute(
        base.model_copy(
            update={
                "operation_key": "other-target",
                "target": target("student", "frieren"),
            }
        ),
        now=NOW + timedelta(seconds=2),
    )
    different_category = service.execute(
        base.model_copy(
            update={
                "operation_key": "other-category",
                "incident_key": "renamed-again",
                "anomaly_category": "recovery_deferral",
            }
        ),
        now=NOW + timedelta(seconds=3),
    )
    restart_same_incident = service.execute(
        Restart(
            operation_key="restart",
            incident_key="stale-wip",
            target=target("student", "fern"),
            expected_conversation_id=STUDENT_ID,
            reason="A controller restart is now required.",
        ),
        now=NOW + timedelta(seconds=4),
    )

    assert renamed_incident.disposition == "suppressed"
    assert renamed_incident.source_operation_key == "base"
    assert different_target.disposition == "executed"
    assert different_category.disposition == "executed"
    assert restart_same_incident.disposition == "executed"
    assert [call[0] for call in backend.calls].count("nudge") == 3


def test_operation_key_cannot_be_reused_for_different_semantics(
    service: OperationService,
):
    action = Nudge(
        operation_key="stable-key",
        incident_key="idle",
        target=target(),
        expected_conversation_id=ADVISOR_ID,
        message="Review the current queue.",
        reason="The queue is stalled.",
    )
    service.execute(action, now=NOW)

    with pytest.raises(IdempotencyConflict):
        service.execute(
            action.model_copy(update={"message": "Do something unrelated."}),
            now=NOW + timedelta(seconds=1),
        )


def test_nudge_is_bound_to_the_existing_conversation(
    service: OperationService,
    backend: FakeBackend,
):
    existing_pending = backend.roles[target()].pending_event_keys
    action = Nudge(
        operation_key="nudge-advisor",
        incident_key="deferred-turns",
        target=target(),
        expected_conversation_id=ADVISOR_ID,
        message="Reconcile the repeated deferred events.",
        reason="Three consecutive snapshots contain deferred turns.",
    )

    outcome = service.execute(action, now=NOW)

    assert isinstance(outcome.receipt, NudgeReceipt)
    assert outcome.receipt.conversation_id == ADVISOR_ID
    assert backend.roles[target()].conversation_id == ADVISOR_ID
    assert set(existing_pending) < set(backend.roles[target()].pending_event_keys)


def test_stale_conversation_identity_rejects_a_nudge_before_delivery(
    service: OperationService,
    backend: FakeBackend,
):
    with pytest.raises(OperationInvariantError, match="conversation"):
        service.execute(
            Nudge(
                operation_key="stale-nudge",
                incident_key="idle",
                target=target(),
                expected_conversation_id=UUID(
                    "00000000-0000-0000-0000-000000000999"
                ),
                message="This belongs to an obsolete conversation.",
                reason="Stale supervisor snapshot.",
            ),
            now=NOW,
        )

    assert [call[0] for call in backend.calls] == ["collect_role"]


def test_restart_is_controller_only_and_must_preserve_state_and_compute(
    service: OperationService,
    backend: FakeBackend,
):
    action = Restart(
        operation_key="restart-fern",
        incident_key="restart-churn",
        target=target("student", "fern"),
        expected_conversation_id=STUDENT_ID,
        reason="The controller is wedged.",
    )

    outcome = service.execute(action, now=NOW)

    assert isinstance(outcome.receipt, RestartReceipt)
    assert outcome.receipt.state_preserved is True
    assert outcome.receipt.compute_preserved is True
    assert backend.roles[action.target].conversation_id == STUDENT_ID
    assert "restart_controller" in [call[0] for call in backend.calls]


def test_restart_rejects_a_changed_conversation_before_delivery(
    service: OperationService,
    backend: FakeBackend,
):
    with pytest.raises(OperationInvariantError, match="expected conversation"):
        service.execute(
            Restart(
                operation_key="restart-stale-conversation",
                incident_key="restart-churn",
                target=target("student", "fern"),
                expected_conversation_id=UUID(
                    "00000000-0000-0000-0000-000000000999"
                ),
                reason="A stale snapshot suggested a restart.",
            ),
            now=NOW,
        )

    assert [call[0] for call in backend.calls] == ["collect_role"]


def test_restart_fails_closed_when_the_role_protocol_omits_restart_authorization(
    service: OperationService,
    backend: FakeBackend,
):
    role_target = target("student", "fern")
    backend.roles[role_target] = backend.roles[role_target].model_copy(
        update={"restart_control_token": None}
    )

    with pytest.raises(OperationInvariantError, match="restart authorization"):
        service.execute(
            Restart(
                operation_key="restart-old-protocol",
                incident_key="controller-failure",
                target=role_target,
                expected_conversation_id=STUDENT_ID,
                reason="The controller is unresponsive.",
            ),
            now=NOW,
        )

    assert [call[0] for call in backend.calls] == ["collect_role"]


@pytest.mark.parametrize("lost", ["state", "compute"])
def test_restart_fails_closed_if_the_backend_cannot_prove_preservation(
    service: OperationService,
    backend: FakeBackend,
    lost: str,
):
    if lost == "state":
        backend.restart_preserves_state = False
    else:
        backend.restart_preserves_compute = False

    with pytest.raises(OperationInvariantError, match="preserve"):
        service.execute(
            Restart(
                operation_key=f"unsafe-restart-{lost}",
                incident_key=f"unsafe-{lost}",
                target=target(),
                expected_conversation_id=ADVISOR_ID,
                reason="Try a safe controller restart.",
            ),
            now=NOW,
        )


def test_context_reset_queues_an_owner_consumed_compare_and_reset_request(
    service: OperationService,
    backend: FakeBackend,
):
    role_target = target()
    raw_before = list(backend.raw_history[role_target.key])
    pending_before = backend.roles[role_target].pending_event_keys
    action = ContextReset(
        operation_key="reset-advisor-context",
        incident_key="malformed-history",
        target=role_target,
        expected_conversation_id=ADVISOR_ID,
        recovery_prompt=(
            "Resume from the current operational summary and pending events; "
            "the raw trace remains available for bounded searches."
        ),
        reason="The model-visible branch is dominated by old transport errors.",
    )

    outcome = service.execute(action, now=NOW)

    assert isinstance(outcome.receipt, ContextResetReceipt)
    assert outcome.receipt.status == "queued"
    assert outcome.receipt.expected_conversation_id == ADVISOR_ID
    assert backend.roles[role_target].conversation_id == ADVISOR_ID
    assert backend.raw_history[role_target.key] == raw_before
    assert backend.active_history[role_target.key] == raw_before
    assert backend.roles[role_target].pending_event_keys == pending_before
    assert backend.reset_requests == [
        ContextResetRequest(
            request_id=action.operation_key,
            target=role_target,
            expected_conversation_id=ADVISOR_ID,
            expected_control_token="lease-1",
            expected_raw_history_event_count=backend.roles[
                role_target
            ].raw_history_event_count,
            expected_raw_history_digest=backend.roles[
                role_target
            ].raw_history_digest,
            expected_pending_event_keys=pending_before,
            recovery_prompt=action.recovery_prompt,
        )
    ]


@pytest.mark.parametrize(
    ("active_turn", "unmatched_actions", "digest", "message"),
    [
        (True, 0, "keep", "active turn"),
        (False, 1, "keep", "unmatched tool"),
        (None, 0, "keep", "activity is unknown"),
        (False, None, "keep", "tool-action state is unknown"),
        (False, 0, None, "history digest"),
    ],
)
def test_context_reset_requires_a_quiescent_fully_observed_conversation(
    service: OperationService,
    backend: FakeBackend,
    active_turn: bool | None,
    unmatched_actions: int | None,
    digest: str | None,
    message: str,
):
    current = backend.roles[target()]
    backend.roles[target()] = current.model_copy(
        update={
            "active_turn": active_turn,
            "unmatched_actions": unmatched_actions,
            "raw_history_digest": (
                current.raw_history_digest if digest == "keep" else digest
            ),
        }
    )

    with pytest.raises(UnsafeContextReset, match=message):
        service.execute(
            ContextReset(
                operation_key=f"unsafe-reset-{message}",
                incident_key=f"unsafe-{message}",
                target=target(),
                expected_conversation_id=ADVISOR_ID,
                recovery_prompt="Recover safely.",
                reason="Test unsafe context reset preconditions.",
            ),
            now=NOW,
        )

    assert "request_context_reset" not in [call[0] for call in backend.calls]


def test_context_reset_requires_a_raw_history_event_count(
    service: OperationService,
    backend: FakeBackend,
):
    backend.roles[target()] = backend.roles[target()].model_copy(
        update={"raw_history_event_count": None}
    )

    with pytest.raises(UnsafeContextReset, match="history count"):
        service.execute(
            ContextReset(
                operation_key="missing-history-count",
                incident_key="missing-history-count",
                target=target(),
                expected_conversation_id=ADVISOR_ID,
                recovery_prompt="Recover safely.",
                reason="Raw history preservation must be verifiable.",
            ),
            now=NOW,
        )


def test_context_reset_rejects_a_transport_that_changes_the_durable_request(
    service: OperationService,
    backend: FakeBackend,
):
    backend.damage_reset_request = True

    with pytest.raises(OperationInvariantError, match="compare-and-reset"):
        service.execute(
            ContextReset(
                operation_key="damaging-reset",
                incident_key="damage",
                target=target(),
                expected_conversation_id=ADVISOR_ID,
                recovery_prompt="Recover safely.",
                reason="The transport must preserve durable request state.",
            ),
            now=NOW,
        )


def reset_request(
    *,
    request_id: str = "reset-1",
    role_target: RoleTarget | None = None,
) -> ContextResetRequest:
    role_target = role_target or target()
    current = observation(role_target)
    assert current.conversation_id is not None
    assert current.raw_history_digest is not None
    return ContextResetRequest(
        request_id=request_id,
        target=role_target,
        expected_conversation_id=current.conversation_id,
        expected_control_token=current.control_token,
        expected_raw_history_event_count=current.raw_history_event_count,
        expected_raw_history_digest=current.raw_history_digest,
        expected_pending_event_keys=current.pending_event_keys,
        recovery_prompt="Continue from a clean model-visible branch.",
    )


def test_context_reset_store_is_durable_idempotent_and_claimed_by_exact_role(
    tmp_path: Path,
):
    database = tmp_path / "context-resets.sqlite3"
    advisor = reset_request()
    student = reset_request(
        request_id="reset-student",
        role_target=target("student", "fern"),
    )
    store = ContextResetRequestStore(database)

    assert store.enqueue(advisor) is True
    assert store.enqueue(advisor) is False
    assert store.enqueue(student) is True
    assert store.pending(target()) == (advisor,)
    store.close()

    reopened = ContextResetRequestStore(database)
    assert reopened.claim_next(target()) == advisor
    assert reopened.claim_next(target()) is None
    assert reopened.pending() == (student,)
    assert reopened.result(advisor.request_id).status == "processing"
    reopened.close()


def test_context_reset_store_claims_only_the_requested_student_conversation(
    tmp_path: Path,
):
    database = tmp_path / "context-resets.sqlite3"
    student = target("student", "fern")
    first = reset_request(request_id="reset-first", role_target=student)
    second_id = UUID("00000000-0000-0000-0000-000000000199")
    second = first.model_copy(
        update={
            "request_id": "reset-second",
            "expected_conversation_id": second_id,
        }
    )

    with ContextResetRequestStore(database) as store:
        store.enqueue(first)
        store.enqueue(second)

        assert store.claim_next(student, conversation_id=second_id) == second
        assert store.pending(student) == (first,)


def test_context_reset_store_rejects_request_id_reuse_with_different_content(
    tmp_path: Path,
):
    store = ContextResetRequestStore(tmp_path / "context-resets.sqlite3")
    original = reset_request()
    store.enqueue(original)

    with pytest.raises(IdempotencyConflict):
        store.enqueue(
            original.model_copy(update={"recovery_prompt": "Different semantics."})
        )


def test_context_reset_completion_proves_owner_preserved_all_durable_state(
    tmp_path: Path,
):
    request = reset_request()
    store = ContextResetRequestStore(tmp_path / "context-resets.sqlite3")
    store.enqueue(request)
    store.claim_next(request.target)
    completion = ContextResetCompletion(
        request_id=request.request_id,
        target=request.target,
        conversation_id=request.expected_conversation_id,
        raw_history_event_count_after=(
            request.expected_raw_history_event_count + 1
        ),
        raw_history_digest=request.expected_raw_history_digest,
        pending_event_keys=(*request.expected_pending_event_keys, "arrived-during-reset"),
    )

    with pytest.raises(OperationInvariantError, match="raw conversation history"):
        store.complete(
            completion.model_copy(update={"raw_history_digest": "changed"})
        )
    with pytest.raises(OperationInvariantError, match="removed raw"):
        store.complete(
            completion.model_copy(
                update={
                    "raw_history_event_count_after": (
                        request.expected_raw_history_event_count - 1
                    )
                }
            )
        )
    assert store.complete(completion) is True
    assert store.complete(completion) is False
    result = store.result(request.request_id)

    assert result.status == "completed"
    assert result.completion == completion
    assert result.rejection_code is None


def test_context_reset_completion_refuses_to_lose_a_preexisting_role_event(
    tmp_path: Path,
):
    request = reset_request()
    store = ContextResetRequestStore(tmp_path / "context-resets.sqlite3")
    store.enqueue(request)
    store.claim_next(request.target)
    completion = ContextResetCompletion(
        request_id=request.request_id,
        target=request.target,
        conversation_id=request.expected_conversation_id,
        raw_history_event_count_after=request.expected_raw_history_event_count,
        raw_history_digest=request.expected_raw_history_digest,
        pending_event_keys=(),
    )

    with pytest.raises(OperationInvariantError, match="lost pending role events"):
        store.complete(completion)


def test_context_reset_rejection_records_only_a_bounded_reason_code(
    tmp_path: Path,
):
    secret = "do-not-persist-reset-error"
    request = reset_request()
    store = ContextResetRequestStore(tmp_path / "context-resets.sqlite3")
    store.enqueue(request)

    assert store.reject(request.request_id, "stale_control_token") is True
    assert store.reject(request.request_id, "stale_control_token") is False
    result = store.result(request.request_id)

    assert result.status == "rejected"
    assert result.rejection_code == "stale_control_token"
    assert secret.encode() not in store.path.read_bytes()
    with pytest.raises(ValueError, match="bounded"):
        store.reject(request.request_id, secret * 20)


def test_context_reset_statuses_are_sanitized_and_bounded(tmp_path: Path):
    store = ContextResetRequestStore(tmp_path / "context-resets.sqlite3")
    template = reset_request()
    for index in range(25):
        store.enqueue(
            template.model_copy(update={"request_id": f"reset-status-{index:02d}"})
        )
    processing = template.model_copy(update={"request_id": "reset-processing"})
    completed = template.model_copy(update={"request_id": "reset-completed"})
    rejected = template.model_copy(update={"request_id": "reset-rejected"})
    for request in (processing, completed, rejected):
        store.enqueue(request)
    for index in range(25):
        store.reject(f"reset-status-{index:02d}", "superseded")
    assert store.claim_next(template.target) == processing
    assert store.claim_next(template.target) == completed
    store.complete(
        ContextResetCompletion(
            request_id=completed.request_id,
            target=completed.target,
            conversation_id=completed.expected_conversation_id,
            raw_history_event_count_after=completed.expected_raw_history_event_count,
            raw_history_digest=completed.expected_raw_history_digest,
            pending_event_keys=completed.expected_pending_event_keys,
        )
    )
    store.reject(rejected.request_id, "stale-control-token")
    queued = template.model_copy(update={"request_id": "reset-queued"})
    store.enqueue(queued)

    statuses = store.statuses(template.target)

    assert len(statuses) == 20
    by_id = {status.request_id: status for status in statuses}
    assert by_id[queued.request_id].status == "queued"
    assert by_id[processing.request_id].status == "processing"
    assert by_id[completed.request_id].status == "completed"
    assert by_id[rejected.request_id].status == "rejected"
    assert by_id[rejected.request_id].rejection_code == "stale-control-token"
    encoded = "".join(status.model_dump_json() for status in statuses)
    assert template.recovery_prompt not in encoded
    assert template.expected_control_token not in encoded


def test_failed_mutation_is_durable_and_does_not_storm_the_backend(
    service: OperationService,
    backend: FakeBackend,
):
    action = Nudge(
        operation_key="failed-nudge",
        incident_key="provider-error",
        target=target(),
        expected_conversation_id=ADVISOR_ID,
        message="Retry only after the action cooldown.",
        reason="The delivery transport is failing.",
    )
    backend.failure = RuntimeError("transport failed with secret-value")

    with pytest.raises(RuntimeError, match="transport failed"):
        service.execute(action, now=NOW)
    first_calls = len(backend.calls)
    with pytest.raises(RecordedOperationError, match="RuntimeError") as recorded:
        service.execute(action, now=NOW + timedelta(seconds=1))
    suppressed = service.execute(
        action.model_copy(update={"operation_key": "failed-nudge-2"}),
        now=NOW + timedelta(seconds=2),
    )

    assert len(backend.calls) == first_calls
    assert "secret-value" not in str(recorded.value)
    assert suppressed.disposition == "suppressed"
    assert suppressed.prior_status == "failed"


def test_audit_is_timestamped_but_never_persists_messages_commands_or_errors(
    service: OperationService,
    backend: FakeBackend,
):
    secret = "do-not-persist-this-sentinel"
    action = Nudge(
        operation_key="audited-nudge",
        incident_key="idle",
        target=target(),
        expected_conversation_id=ADVISOR_ID,
        message=f"Investigate {secret}.",
        reason=f"The log contained {secret}.",
    )

    service.execute(action, now=NOW)
    records = service.ledger.records()
    database_bytes = service.ledger.path.read_bytes()

    assert len(records) == 1
    assert records[0].operation_key == "audited-nudge"
    assert records[0].target == target()
    assert records[0].action_kind == "nudge"
    assert records[0].incident_key == "idle"
    assert records[0].anomaly_category == "other_operational"
    assert records[0].stable_incident_key.startswith("incident-")
    assert records[0].stable_incident_key != action.incident_key
    assert records[0].requested_at == NOW
    assert records[0].completed_at == NOW
    assert records[0].status == "succeeded"
    assert secret.encode() not in database_bytes


def test_recent_mutation_audit_is_bounded_newest_first_and_excludes_inspection(
    service: OperationService,
):
    first = Nudge(
        operation_key="first-nudge",
        incident_key="first-model-label",
        target=target("student", "fern"),
        expected_conversation_id=STUDENT_ID,
        message="Resume the current assignment.",
        reason="The role is idle.",
    )
    service.execute(first, now=NOW)
    suppressed = first.model_copy(
        update={
            "operation_key": "renamed-nudge",
            "incident_key": "unrelated-looking-model-label",
        }
    )
    service.execute(suppressed, now=NOW + timedelta(seconds=1))
    service.execute(
        Restart(
            operation_key="restart-student",
            incident_key="restart-model-label",
            target=target("student", "fern"),
            expected_conversation_id=STUDENT_ID,
            reason="The controller requires a safe restart.",
        ),
        now=NOW + timedelta(seconds=2),
    )
    service.execute(
        CollectRole(operation_key="latest-inspection", target=target()),
        now=NOW + timedelta(seconds=3),
    )

    recent = service.ledger.recent_mutations(limit=2)

    assert [record.operation_key for record in recent] == [
        "restart-student",
        "renamed-nudge",
    ]
    assert [record.status for record in recent] == ["succeeded", "suppressed"]
    assert recent[1].stable_incident_key == service.ledger.records()[0].stable_incident_key
    assert all(record.action_kind != "collect_role" for record in recent)

    with pytest.raises(ValueError, match="between 1 and 50"):
        service.ledger.recent_mutations(limit=0)


def test_legacy_ledger_rows_gain_a_canonical_default_category(
    tmp_path: Path,
    inventory: CampaignInventory,
    backend: FakeBackend,
):
    database = tmp_path / "legacy-operations.sqlite3"
    action = Nudge(
        operation_key="legacy-nudge",
        incident_key="original-readable-key",
        target=target(),
        expected_conversation_id=ADVISOR_ID,
        message="Resume the current work.",
        reason="The role is idle.",
    )
    service = OperationService(
        inventory,
        backend,
        OperationLedger(database),
        policy=OperationPolicy(mutation_cooldown_seconds=60),
    )
    service.execute(action, now=NOW)
    service.close()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE operation_audit
            SET anomaly_category = NULL, cooldown_key = 'legacy-model-derived-key'
            WHERE operation_key = 'legacy-nudge'
            """
        )

    reopened = OperationService(
        inventory,
        backend,
        OperationLedger(database),
        policy=OperationPolicy(mutation_cooldown_seconds=60),
    )
    record = reopened.ledger.records()[0]
    suppressed = reopened.execute(
        action.model_copy(
            update={
                "operation_key": "renamed-legacy-nudge",
                "incident_key": "renamed-readable-key",
            }
        ),
        now=NOW + timedelta(seconds=1),
    )

    assert record.incident_key == "original-readable-key"
    assert record.anomaly_category == "other_operational"
    assert record.stable_incident_key.startswith("incident-")
    assert suppressed.disposition == "suppressed"
