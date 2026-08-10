"""Portable Senpai control loop with GitHub as its only remote mailbox."""

from __future__ import annotations

import math
import os
import random
import signal
import sys
import time
from base64 import b64decode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from string import Template
from typing import Literal, Protocol
from uuid import UUID

from senpai_agent.agent_markdown import read_agent_markdown, strip_spdx_header
from senpai_agent.advisor import (
    AdvisorEvent,
    AdvisorEventStore,
    compose_system_instructions,
)
from senpai_agent.github.mailbox import ActiveGitHubWatcher, GitHubMailbox
from senpai_agent.inbox import (
    DeliveryState,
    InboxTurn,
    InboxTurnQuarantined,
    PersistentInbox,
)
from senpai_agent.mailbox import (
    CompositeMailbox,
    ContextResetMailbox,
    ControllerEvent,
    LocalAdvisorMailbox,
    LocalStudentMailbox,
    Mailbox,
)
from senpai_agent.monitor import (
    MonitorMailbox,
    TrainingMonitorEngine,
    WandbMetricSource,
)
from senpai_agent.state import (
    AssignmentConversationRegistry,
    ConversationBatch,
    ConversationStateLedger,
    StudentConversationSelector,
    WorkspaceDivergenceLedger,
)
from senpai_agent.supervisor import LEASE_ENV, ProgressLease
from senpai_agent.workspace import (
    StudentWorkspaceReconciler,
    WorkspaceDivergence,
    WorkspaceJobRunning,
)


_EDGE_TRIGGERED_EVENT_KINDS = frozenset({"research_base_changed"})
_PROMPT_TEMPLATE_VARIABLES = frozenset(
    {
        "ADVISOR_BRANCH",
        "GH_REPO",
        "GPUS_PER_STUDENT",
        "PROBLEM_DIR",
        "RESEARCH_TAG",
        "STUDENT_NAME",
        "STUDENT_NAMES",
        "TARGET_REPO_URL",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    }
)


@dataclass(frozen=True, slots=True)
class TurnResult:
    exit_code: int
    delivered_event_keys: frozenset[str] = frozenset()


class ConversationRecoveryExhausted(RuntimeError):
    def __init__(self, conversation_id: UUID, error: Exception):
        self.conversation_id = conversation_id
        self.error = error
        super().__init__(
            f"context recovery failed for conversation {conversation_id}: {error}"
        )


class TurnRunner(Protocol):
    def run(
        self,
        prompt: str,
        *,
        conversation_id: UUID,
        event_keys: frozenset[str],
        visible_event_keys: frozenset[str] = frozenset(),
        inbox: PersistentInbox,
        inbox_turn_id: str,
    ) -> TurnResult: ...


def _is_context_history_failure(error: Exception) -> bool:
    from openhands.sdk.conversation.exceptions import ConversationRunError
    from openhands.sdk.llm.exceptions import (
        LLMContextWindowExceedError,
        LLMMalformedConversationHistoryError,
    )

    cause = (
        error.original_exception
        if isinstance(error, ConversationRunError)
        else error
    )
    return isinstance(
        cause,
        (LLMContextWindowExceedError, LLMMalformedConversationHistoryError),
    )


def _context_recovery_prompt(full_prompt: str, current_prompt: str) -> str:
    research_brief = "" if full_prompt in current_prompt else f"{full_prompt}\n\n"
    return research_brief + (
        "# Conversation context recovery\n\n"
        "The previous model-visible conversation branch exhausted or corrupted "
        "its context. Its complete raw trace and workspace are preserved, but "
        "the active model context was reset. Inspect preserved state as needed, "
        "and verify any interrupted action before relying on it."
        f"\n\n# Current actionable state\n\n{current_prompt}"
    )


class OpenHandsTurnRunner:
    def __init__(
        self,
        config: object,
        *,
        full_prompt: str,
        github_mailbox: GitHubMailbox | None = None,
        active_poll_interval_seconds: float = 120,
    ):
        self.config = config
        self.full_prompt = full_prompt.strip()
        if not self.full_prompt:
            raise ValueError("full prompt must not be empty")
        self.github_mailbox = github_mailbox
        self.active_poll_interval_seconds = active_poll_interval_seconds

    def run(
        self,
        prompt: str,
        *,
        conversation_id: UUID,
        event_keys: frozenset[str],
        visible_event_keys: frozenset[str] = frozenset(),
        inbox: PersistentInbox | None = None,
        inbox_turn_id: str | None = None,
    ) -> TurnResult:
        from senpai_agent.openhands_runner import local_event_db_path, run_openhands

        config = replace(
            self.config,
            conversation_id=conversation_id,
        )
        reset_request = _claim_context_reset(config.state_dir, conversation_id)
        turn_deadline = time.monotonic() + config.timeout_seconds
        if (inbox is None) != (inbox_turn_id is None):
            raise ValueError("inbox and inbox turn ID must be provided together")

        def inbox_options() -> dict[str, object]:
            if inbox is None or inbox_turn_id is None:
                return {}
            return {
                "inbox": inbox,
                "inbox_turn_id": inbox_turn_id,
                "recovery_prompt": _context_recovery_prompt(
                    self.full_prompt,
                    prompt,
                ),
            }

        def run_turn(
            reset_delivered_event_keys: frozenset[str] = frozenset(),
        ) -> int:
            current_prompt = prompt
            reset_callback = None
            if reset_request is not None:
                current_prompt = _context_recovery_prompt(
                    self.full_prompt,
                    f"{reset_request.recovery_prompt}\n\n{prompt}",
                )

                def reset_callback() -> None:
                    if reset_delivered_event_keys:
                        with AdvisorEventStore(local_event_db_path(config)) as store:
                            for event_key in reset_delivered_event_keys:
                                store.acknowledge(event_key)
                    _complete_context_reset(
                        config.state_dir,
                        reset_request,
                        delivered_event_keys=tuple(
                            sorted(
                                reset_delivered_event_keys.intersection(
                                    reset_request.expected_pending_event_keys
                                )
                            )
                        ),
                    )
            try:
                if reset_callback is not None:
                    return run_openhands(
                        current_prompt,
                        config,
                        reset_context=True,
                        context_reset_applied=reset_callback,
                        **inbox_options(),
                    )
                return run_openhands(current_prompt, config, **inbox_options())
            except Exception as error:
                if reset_request is not None or not _is_context_history_failure(error):
                    raise
                print(
                    "SENPAI_CONTEXT_RECOVERY "
                    f"conversation_id={conversation_id} "
                    f"error={type(error).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                remaining = turn_deadline - time.monotonic()
                if remaining <= 0:
                    timeout = TimeoutError(
                        "no turn time remains for context recovery"
                    )
                    raise ConversationRecoveryExhausted(
                        conversation_id,
                        timeout,
                    ) from error
                recovery_config = replace(
                    config,
                    timeout_seconds=remaining,
                )
                try:
                    return run_openhands(
                        _context_recovery_prompt(self.full_prompt, prompt),
                        recovery_config,
                        reset_context=True,
                        **inbox_options(),
                    )
                except Exception as recovery_error:
                    if _is_context_history_failure(recovery_error):
                        raise ConversationRecoveryExhausted(
                            conversation_id,
                            recovery_error,
                        ) from recovery_error
                    raise

        if self.github_mailbox is None:
            return TurnResult(exit_code=run_turn())

        store_path = local_event_db_path(config)
        map_event = None
        if config.role == "student":
            registry = AssignmentConversationRegistry(
                config.state_dir / "student-conversations.json"
            )
            map_event = partial(
                _student_feedback_event,
                conversation_id=conversation_id,
                registry=registry,
            )

        # A late watcher event may still be pending locally when the next
        # controller prompt carries that same GitHub event. The prompt is the
        # delivery path for this turn, so keep the event pump from repeating it.
        if reset_request is None:
            with AdvisorEventStore(store_path) as store:
                for event_key in event_keys:
                    store.acknowledge(event_key)
        with ActiveGitHubWatcher(
            self.github_mailbox,
            store_path,
            known_keys=visible_event_keys | event_keys,
            poll_interval_seconds=self.active_poll_interval_seconds,
            map_event=map_event,
        ) as watcher:
            exit_code = run_turn(
                event_keys if reset_request is not None else frozenset()
            )
        with AdvisorEventStore(store_path) as store:
            delivered = store.acknowledged(tuple(watcher.enqueued_keys))
        return TurnResult(
            exit_code=exit_code,
            delivered_event_keys=frozenset(delivered),
        )


def _claim_context_reset(state_dir: Path, conversation_id: UUID):
    from senpai_agent.operations import ContextResetRequestStore, RoleTarget
    from senpai_agent.role_control import (
        _active_delegation_count,
        _control_token,
        _pending_event_keys,
        _raw_history_checkpoint,
        _read_lease,
    )

    queue_path = state_dir / "context-resets.sqlite3"
    if not queue_path.is_file():
        return None
    role = os.environ.get("SENPAI_ROLE")
    research_tag = os.environ.get("RESEARCH_TAG")
    if role not in {"advisor", "student"} or not research_tag:
        return None
    target = RoleTarget(
        research_tag=research_tag,
        role=role,
        student=os.environ.get("STUDENT_NAME") if role == "student" else None,
    )
    with ContextResetRequestStore(queue_path) as store:
        request = store.claim_next(target, conversation_id=conversation_id)
        if request is None:
            return None
        count, digest = _raw_history_checkpoint(state_dir, conversation_id)
        pending = _pending_event_keys(state_dir, target, conversation_id)
        token = _control_token(
            _read_lease(state_dir / "controller-lease.json"),
            conversation_id,
            digest,
            request.expected_pending_event_keys,
            _active_delegation_count(state_dir),
        )
        if count != request.expected_raw_history_event_count:
            store.reject(request.request_id, "raw-history-count-changed")
            return None
        if digest != request.expected_raw_history_digest:
            store.reject(request.request_id, "raw-history-prefix-changed")
            return None
        if not set(request.expected_pending_event_keys).issubset(pending):
            store.reject(request.request_id, "pending-events-lost")
            return None
        if token != request.expected_control_token:
            store.reject(request.request_id, "controller-identity-changed")
            return None
        return request


def _complete_context_reset(
    state_dir: Path,
    request: object,
    *,
    delivered_event_keys: tuple[str, ...] = (),
) -> None:
    from senpai_agent.operations import (
        ContextResetCompletion,
        ContextResetRequest,
        ContextResetRequestStore,
    )
    from senpai_agent.role_control import (
        _pending_event_keys,
        _raw_history_checkpoint,
    )

    reset = ContextResetRequest.model_validate(request)
    total_count, _ = _raw_history_checkpoint(
        state_dir,
        reset.expected_conversation_id,
    )
    _, prefix_digest = _raw_history_checkpoint(
        state_dir,
        reset.expected_conversation_id,
        event_count=reset.expected_raw_history_event_count,
    )
    if total_count is None or prefix_digest is None:
        raise RuntimeError("context reset history checkpoint is unreadable")
    pending = _pending_event_keys(
        state_dir,
        reset.target,
        reset.expected_conversation_id,
    )
    with ContextResetRequestStore(state_dir / "context-resets.sqlite3") as store:
        store.complete(
            ContextResetCompletion(
                request_id=reset.request_id,
                target=reset.target,
                conversation_id=reset.expected_conversation_id,
                raw_history_event_count_after=total_count,
                raw_history_digest=prefix_digest,
                pending_event_keys=pending,
                delivered_event_keys=delivered_event_keys,
            )
        )


def _student_feedback_event(
    event: ControllerEvent,
    *,
    conversation_id: UUID,
    registry: AssignmentConversationRegistry,
) -> AdvisorEvent | None:
    if event.kind != "student_pr_feedback":
        return None
    target = registry.for_assignment(
        str(event.payload["assignment_id"]),
        str(event.payload["revision_id"]),
    )
    if target != conversation_id:
        return None
    return AdvisorEvent(
        kind=event.kind,
        dedupe_key=event.dedupe_key,
        payload={
            **event.payload,
            "parent_conversation_id": str(conversation_id),
        },
    )


class Controller:
    """Poll, reconcile, run one turn, and immediately verify GitHub again."""

    def __init__(
        self,
        *,
        role: Literal["advisor", "student"],
        mailbox: Mailbox,
        turns: TurnRunner,
        conversation_id: UUID,
        full_prompt: str,
        system_context: str = "",
        conversation_state: ConversationStateLedger | None = None,
        inbox: PersistentInbox | None = None,
        workspace_divergence_state: WorkspaceDivergenceLedger | None = None,
        conversation_for_events: (
            Callable[[Sequence[ControllerEvent]], Sequence[ConversationBatch]] | None
        ) = None,
        reconcile: Callable[[Sequence[ControllerEvent]], None] | None = None,
        progress: ProgressLease | None = None,
        operation_timeout_seconds: float = 300,
        turn_timeout_seconds: float = 3660,
        max_consecutive_turn_failures: int = 2,
        event_reminder_seconds: float | None = None,
        start_gate_path: Path | None = None,
        launch_gate_path: Path | None = None,
        start_gate_poll_seconds: float = 30,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 600,
        jitter_seconds: float = 120,
        next_monitor_poll_seconds: Callable[[], float | None] | None = None,
    ):
        if (
            not all(
                math.isfinite(value)
                for value in (poll_interval_seconds, jitter_seconds)
            )
            or min(poll_interval_seconds, jitter_seconds) < 0
        ):
            raise ValueError("poll and jitter intervals must not be negative")
        if start_gate_poll_seconds <= 0:
            raise ValueError("start-gate polling interval must be positive")
        if operation_timeout_seconds <= 0 or turn_timeout_seconds <= 0:
            raise ValueError("controller phase timeouts must be positive")
        if max_consecutive_turn_failures <= 0:
            raise ValueError("maximum consecutive turn failures must be positive")
        self.role = role
        self.mailbox = mailbox
        self.turns = turns
        self.conversation_id = conversation_id
        self.conversation_for_events = conversation_for_events
        self.reconcile = reconcile
        self.progress = progress
        self.operation_timeout_seconds = operation_timeout_seconds
        self.turn_timeout_seconds = turn_timeout_seconds
        self.max_consecutive_turn_failures = max_consecutive_turn_failures
        self.start_gate_paths = tuple(
            path for path in (start_gate_path, launch_gate_path) if path is not None
        )
        self.start_gate_poll_seconds = start_gate_poll_seconds
        self.full_prompt = full_prompt.strip()
        self.system_context = system_context.strip()
        self.conversation_state = conversation_state
        self.inbox = inbox or PersistentInbox()
        self.workspace_divergence_state = workspace_divergence_state
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.jitter_seconds = jitter_seconds
        self.next_monitor_poll_seconds = next_monitor_poll_seconds
        self.event_reminder_seconds = (
            max(poll_interval_seconds, 600)
            if event_reminder_seconds is None
            else event_reminder_seconds
        )
        if self.event_reminder_seconds <= 0:
            raise ValueError("event reminder interval must be positive")
        self._started: set[UUID] = set()
        resumed_at = time.monotonic()
        self._visible: dict[str, float] = {
            key: resumed_at for key in self.inbox.acknowledged_event_keys()
        }
        self._deferred_until: dict[str, float] = {}
        self._deferred_conversations: dict[UUID, float] = {}
        self._workspace_divergence: dict[UUID, str] = {}

    def run(self, *, max_cycles: int | None = None) -> None:
        self._wait_for_start_gates()
        for turn in self.inbox.quarantined_turns():
            print(
                "SENPAI_TURN_QUARANTINED "
                f"conversation_id={turn.conversation_id} "
                f"turn_id={turn.turn_id} reason={turn.quarantine_reason}",
                file=sys.stderr,
                flush=True,
            )
        cycles = 0
        turn_failures: dict[UUID, int] = {}
        while max_cycles is None or cycles < max_cycles:
            self._acknowledge_processed_turns()
            self._poll_into_inbox()
            cycle_had_failure = False
            failed_conversations: set[UUID] = set()
            served_conversations: set[UUID] = set()
            while True:
                turn = self._next_ready_turn(
                    failed_conversations | served_conversations
                )
                if turn is None and served_conversations:
                    served_conversations.clear()
                    turn = self._next_ready_turn(failed_conversations)
                if turn is None:
                    break
                conversation_id = UUID(turn.conversation_id)
                try:
                    self._publish_progress(
                        "openhands-turn",
                        self.turn_timeout_seconds,
                        conversation_id=conversation_id,
                    )
                    result = self.turns.run(
                        turn.prompt.body,
                        conversation_id=conversation_id,
                        event_keys=frozenset(turn.event_keys),
                        visible_event_keys=(
                            frozenset(self._visible) | frozenset(turn.event_keys)
                        ),
                        inbox=self.inbox,
                        inbox_turn_id=turn.turn_id,
                    )
                except ConversationRecoveryExhausted as error:
                    retry_delay = max(self.poll_interval_seconds, 600)
                    retry_at = time.monotonic() + retry_delay
                    self._deferred_conversations[conversation_id] = retry_at
                    print(
                        "SENPAI_TURN_DEFERRED "
                        f"conversation_id={conversation_id} "
                        f"event_keys={','.join(turn.event_keys)} "
                        f"retry_after_seconds={retry_delay:g} "
                        f"error={error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                except InboxTurnQuarantined as error:
                    turn_failures.pop(conversation_id, None)
                    served_conversations.add(conversation_id)
                    print(
                        "SENPAI_TURN_QUARANTINED "
                        f"conversation_id={conversation_id} "
                        f"turn_id={error.turn_id} reason={error.reason}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                except Exception as error:  # noqa: BLE001
                    failures = turn_failures.get(conversation_id, 0) + 1
                    turn_failures[conversation_id] = failures
                    print(
                        f"SENPAI_TURN_EXCEPTION {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if failures >= self.max_consecutive_turn_failures:
                        raise RuntimeError(
                            "controller exceeded its consecutive turn-failure "
                            f"limit ({self.max_consecutive_turn_failures})"
                        ) from error
                    failed_conversations.add(conversation_id)
                    cycle_had_failure = True
                    continue
                if result.exit_code != 0:
                    failures = turn_failures.get(conversation_id, 0) + 1
                    turn_failures[conversation_id] = failures
                    print(
                        "SENPAI_TURN_ERROR "
                        f"exit_code={result.exit_code} "
                        f"conversation_id={conversation_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if failures >= self.max_consecutive_turn_failures:
                        raise RuntimeError(
                            "controller exceeded its consecutive turn-failure "
                            f"limit ({self.max_consecutive_turn_failures})"
                        )
                    failed_conversations.add(conversation_id)
                    cycle_had_failure = True
                    continue

                self._assert_processed(turn.turn_id)
                self._acknowledge_processed_turns()
                turn_failures.pop(conversation_id, None)
                served_conversations.add(conversation_id)
                self._poll_into_inbox(allow_reminders=False)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return
            if cycle_had_failure:
                longest_streak = max(
                    (turn_failures[value] for value in failed_conversations),
                    default=1,
                )
                delay = min(
                    self.poll_interval_seconds,
                    2 ** min(longest_streak, 8),
                )
                self._sleep("turn-backoff", delay)
                continue
            phase, delay = self._idle_delay()
            self._sleep(phase, delay)

    def _poll_into_inbox(self, *, allow_reminders: bool = True) -> None:
        self._publish_progress("poll")
        try:
            polled = self.mailbox.poll()
        except Exception as error:  # noqa: BLE001
            if allow_reminders:
                raise
            print(
                f"SENPAI_POST_TURN_POLL_ERROR {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return
        events = self._new_events(
            polled,
            allow_reminders=allow_reminders,
        )
        self._enqueue_events(events)

    def _enqueue_events(self, events: Sequence[ControllerEvent]) -> None:
        for batch in self._event_batches(events):
            batch_events = batch.events
            conversation_id = batch.conversation_id
            if self.reconcile is not None:
                self._publish_progress("reconcile")
                try:
                    self.reconcile(batch_events)
                except WorkspaceJobRunning as busy:
                    retry_delay = 30.0
                    retry_at = time.monotonic() + retry_delay
                    checkout_kinds = {
                        "student_assignment",
                        "student_pr_feedback",
                    }
                    deferred_events = tuple(
                        event
                        for event in batch_events
                        if event.kind in checkout_kinds
                    )
                    batch_events = tuple(
                        event
                        for event in batch_events
                        if event.kind not in checkout_kinds
                    )
                    for event in deferred_events:
                        self._visible.pop(event.dedupe_key, None)
                        self._deferred_until[event.dedupe_key] = retry_at
                    print(
                        "SENPAI_WORKSPACE_JOB_DEFERRED "
                        f"conversation_id={conversation_id} "
                        f"retry_after_seconds={retry_delay:g} "
                        f"jobs={','.join(busy.job_ids)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not batch_events:
                        continue
                except WorkspaceDivergence as conflict:
                    print(
                        f"SENPAI_WORKSPACE_DIVERGENCE {conflict}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if (
                        self._workspace_divergence_key(conversation_id)
                        == conflict.event.dedupe_key
                    ):
                        batch_events = tuple(
                            event
                            for event in batch_events
                            if event.kind != "student_assignment"
                        )
                        if not batch_events:
                            print(
                                "SENPAI_WORKSPACE_DIVERGENCE_SUPPRESSED "
                                f"conversation_id={conversation_id} "
                                f"event_key={conflict.event.dedupe_key}",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                    batch_events = (*batch_events, conflict.event)
                else:
                    if any(
                        event.kind == "student_assignment" for event in batch_events
                    ):
                        self._clear_workspace_divergence(conversation_id)
            for event in batch_events:
                self.inbox.enqueue(
                    conversation_id,
                    event.dedupe_key,
                    event.to_prompt(),
                )

    def _next_ready_turn(
        self,
        excluded: set[UUID] | frozenset[UUID] = frozenset(),
    ) -> InboxTurn | None:
        now = time.monotonic()
        self._deferred_conversations = {
            conversation_id: retry_at
            for conversation_id, retry_at in self._deferred_conversations.items()
            if now < retry_at
        }
        for value in self.inbox.ready_conversation_ids():
            conversation_id = UUID(value)
            if (
                conversation_id in excluded
                or conversation_id in self._deferred_conversations
            ):
                continue
            continuing = self._has_started(conversation_id)
            refresh_system_context = (
                continuing
                and self.conversation_state is not None
                and not self.conversation_state.is_context_current(
                    conversation_id,
                    self.system_context,
                )
            )
            turn = self.inbox.next_turn(
                conversation_id,
                self._prompt(
                    (),
                    continuing=continuing,
                    refresh_system_context=refresh_system_context,
                ),
                legacy_prompt_identity=(
                    f"initial:{self.full_prompt}"
                    if not continuing
                    else (
                        f"system-context:{self.system_context}"
                        if refresh_system_context
                        else None
                    )
                ),
            )
            if turn is not None:
                return turn
        return None

    def _assert_processed(self, turn_id: str) -> None:
        turn = self.inbox.latest_turn(turn_id)
        if turn.state is not DeliveryState.PROCESSED:
            raise RuntimeError(
                "turn runner returned success without a processed inbox receipt: "
                f"{turn.turn_id}"
            )

    def _acknowledge_processed_turns(self) -> None:
        for turn in self.inbox.processed_turns():
            keys = tuple(dict.fromkeys(turn.acknowledgement_keys))
            if keys:
                self._publish_progress("acknowledge")
                self.mailbox.acknowledge(keys)
            conversation_id = UUID(turn.conversation_id)
            self._mark_success(conversation_id)
            divergence = next(
                (
                    key
                    for key in turn.event_keys
                    if key.startswith("workspace_diverged:")
                ),
                None,
            )
            if divergence is not None:
                self._record_workspace_divergence(conversation_id, divergence)
            self.inbox.acknowledge(turn.turn_id)
            for key in turn.event_keys:
                self._visible[key] = time.monotonic()
            self._publish_progress("turn-complete", completed_turn=True)

    def _event_batches(
        self,
        events: Sequence[ControllerEvent],
    ) -> tuple[ConversationBatch, ...]:
        if self.conversation_for_events is not None:
            return tuple(self.conversation_for_events(events))
        return (ConversationBatch(self.conversation_id, tuple(events)),)

    def _wait_for_start_gates(self) -> None:
        while any(not path.is_file() for path in self.start_gate_paths):
            self._publish_progress(
                "start-gate",
                self.start_gate_poll_seconds + self.operation_timeout_seconds,
            )
            self.sleep(self.start_gate_poll_seconds)

    def _publish_progress(
        self,
        phase: str,
        timeout_seconds: float | None = None,
        *,
        completed_turn: bool = False,
        conversation_id: UUID | None = None,
    ) -> None:
        if self.progress is not None:
            self.progress.update(
                phase,
                timeout_seconds or self.operation_timeout_seconds,
                completed_turn=completed_turn,
                conversation_id=(
                    str(conversation_id) if conversation_id is not None else None
                ),
            )

    def _sleep(self, phase: str, seconds: float) -> None:
        self._publish_progress(
            phase,
            max(seconds + self.operation_timeout_seconds, 1),
        )
        self.sleep(seconds)

    def _idle_delay(self) -> tuple[str, float]:
        heartbeat = max(
            1.0,
            self.poll_interval_seconds + random.uniform(0, self.jitter_seconds),
        )
        if self.next_monitor_poll_seconds is None:
            return "sleep", heartbeat
        try:
            monitor_delay = self.next_monitor_poll_seconds()
        except Exception as error:  # noqa: BLE001
            print(
                f"SENPAI_MONITOR_SCHEDULE_ERROR {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return "monitor-backoff", min(heartbeat, 5.0)
        if monitor_delay is None:
            return "sleep", heartbeat
        if not math.isfinite(monitor_delay) or monitor_delay < 0:
            print(
                f"SENPAI_MONITOR_SCHEDULE_INVALID delay={monitor_delay!r}",
                file=sys.stderr,
                flush=True,
            )
            return "monitor-backoff", min(heartbeat, 5.0)
        return "monitor-sleep", max(1.0, min(heartbeat, monitor_delay))

    def _has_started(self, conversation_id: UUID) -> bool:
        return conversation_id in self._started or (
            self.conversation_state is not None
            and self.conversation_state.has_started(conversation_id)
        )

    def _mark_success(self, conversation_id: UUID) -> None:
        self._started.add(conversation_id)
        if self.conversation_state is not None:
            self.conversation_state.mark_success(
                conversation_id,
                self.system_context,
            )

    def _workspace_divergence_key(self, conversation_id: UUID) -> str | None:
        if self.workspace_divergence_state is not None:
            return self.workspace_divergence_state.current(conversation_id)
        return self._workspace_divergence.get(conversation_id)

    def _record_workspace_divergence(
        self,
        conversation_id: UUID,
        event_key: str,
    ) -> None:
        if self.workspace_divergence_state is not None:
            self.workspace_divergence_state.record(conversation_id, event_key)
        else:
            self._workspace_divergence[conversation_id] = event_key

    def _clear_workspace_divergence(self, conversation_id: UUID) -> None:
        if self.workspace_divergence_state is not None:
            self.workspace_divergence_state.clear(conversation_id)
        else:
            self._workspace_divergence.pop(conversation_id, None)

    def _new_events(
        self,
        events: Sequence[ControllerEvent],
        *,
        allow_reminders: bool = True,
    ) -> tuple[ControllerEvent, ...]:
        now = time.monotonic()
        current: dict[str, float] = {}
        new: list[ControllerEvent] = []
        self._deferred_until = {
            key: deadline
            for key, deadline in self._deferred_until.items()
            if now < deadline
        }
        for event in events:
            retry_at = self._deferred_until.get(event.dedupe_key)
            if retry_at is not None:
                continue
            delivered_at = self._visible.get(event.dedupe_key)
            reminder_due = (
                allow_reminders
                and event.kind not in _EDGE_TRIGGERED_EVENT_KINDS
                and delivered_at is not None
                and now - delivered_at >= self.event_reminder_seconds
            )
            if delivered_at is None or reminder_due:
                new.append(event)
                delivered_at = now
            current[event.dedupe_key] = delivered_at
        self._visible = current
        return tuple(new)

    def _prompt(
        self,
        _events: Sequence[ControllerEvent],
        *,
        continuing: bool,
        refresh_system_context: bool = False,
    ) -> str:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not continuing:
            return (
                f"{self.full_prompt}\n\nCurrent time (UTC): {now}\n\n"
                "# Current GitHub state\n\n"
                "Actionable events follow as separately tracked messages."
            )
        prompt = (
            f"Continue the {self.role} loop. Current time (UTC): {now}. "
            "Actionable GitHub events follow as separately tracked messages."
        )
        if refresh_system_context:
            prompt += (
                "\n\n# Updated Senpai system context\n\n"
                "The Senpai operating context or research brief changed since "
                "this conversation last ran. Treat the following as current:\n\n"
                f"{self.system_context}"
            )
        return prompt


def _full_prompt(role: Literal["advisor", "student"], env: Mapping[str, str]) -> str:
    workspace = Path(env["SENPAI_OPENHANDS_WORKSPACE"]).resolve()
    instructions = workspace / "instructions" / f"prompt-{role}.md"
    program = workspace / "program.md"
    if not program.is_file() and (workspace / "senpai" / "program.md").is_file():
        program = workspace / "senpai" / "program.md"
    template_env = {
        key: env[key] for key in _PROMPT_TEMPLATE_VARIABLES if key in env
    }
    role_prompt = (
        Template(read_agent_markdown(instructions))
        .safe_substitute(template_env)
        .strip()
        if instructions.is_file()
        else (
            "Follow the repository AGENTS.md, the assigned GitHub work, "
            "and the Senpai role charter."
        )
    )
    prompt = (
        "# Research programme\n\n"
        f"{read_agent_markdown(program).strip()}\n\n"
        f"# {role.title()} task\n\n"
        f"{role_prompt}"
    )
    encoded_extra = env.get("EXTRA_INSTRUCTIONS_B64")
    if encoded_extra:
        extra = b64decode(encoded_extra, validate=True).decode()
        prompt += (
            "\n\n# Additional launch instructions\n\n"
            f"{strip_spdx_header(extra).strip()}"
        )
    identity = (
        f"Role: {role}; repository: {env['GH_REPO']}; "
        f"advisor branch: {env['ADVISOR_BRANCH']}; "
        f"W&B: {env['WANDB_ENTITY']}/{env['WANDB_PROJECT']}."
    )
    if role == "advisor":
        identity += (
            f" Advisor: {env.get('ADVISOR_NAME', 'advisor')}."
            f" Students: {env.get('STUDENT_NAMES', '')}."
        )
    else:
        identity += f" Student: {env['STUDENT_NAME']}."
    return f"{prompt}\n\n# Runtime identity\n\n{identity}"


def _role_interval(
    env: Mapping[str, str],
    role: Literal["advisor", "student"],
    suffix: str,
    default: float,
) -> float:
    role_key = f"SENPAI_{role.upper()}_{suffix}"
    shared_key = f"SENPAI_{suffix}"
    return float(env.get(role_key, env.get(shared_key, str(default))))


def controller_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    import argparse

    progress = ProgressLease(Path(env[LEASE_ENV])) if env.get(LEASE_ENV) else None
    if progress is not None:
        progress.update("startup", 300)

    from senpai_agent.delegation import reconcile_delegated_tasks
    from senpai_agent.openhands_runner import (
        parse_runner_args,
        read_role_instructions,
        resolve_config,
        scrub_model_credentials,
    )
    from senpai_agent.operations import RoleTarget
    from senpai_agent.tools import (
        close_training_runtimes,
        training_runtime,
    )
    from senpai_agent.weave_monitoring import finish_weave_monitoring

    parser = argparse.ArgumentParser(
        description="Run the portable Senpai GitHub/OpenHands controller."
    )
    parser.add_argument("role", choices=("advisor", "student"))
    args = parser.parse_args(argv)
    role = args.role
    if env.get("SENPAI_ROLE") != role:
        raise RuntimeError(f"SENPAI_ROLE must be {role}")
    human_issues = env.get("SENPAI_ENABLE_HUMAN_ISSUES", "true").lower()
    if human_issues not in {"true", "false"}:
        raise RuntimeError("SENPAI_ENABLE_HUMAN_ISSUES must be true or false")

    max_turns = int(env.get("SENPAI_OPENHANDS_MAX_TURNS", "100000"))
    runner_config = resolve_config(
        parse_runner_args(["--max-turns", str(max_turns)]),
        env,
    )
    if runner_config.github_token is None:
        raise RuntimeError("controller worker requires GitHub credentials")
    scrub_model_credentials(os.environ, runner_config)
    reconcile_delegated_tasks(
        getattr(runner_config, "delegation_root_state_dir", None)
        or runner_config.state_dir,
        runner_config.state_dir / f"{role}-events.sqlite3",
    )
    github_mailbox = GitHubMailbox(
        repo=runner_config.github_repo,
        token=runner_config.github_token,
        role=role,
        advisor_branch=env["ADVISOR_BRANCH"],
        students=tuple(
            student.strip()
            for student in env.get("STUDENT_NAMES", "").split(",")
            if student.strip()
        ),
        student_name=env.get("STUDENT_NAME"),
        stale_wip_seconds=int(env.get("SENPAI_STALE_WIP_SECONDS", "7200")),
        trusted_actor=runner_config.github_trusted_actor,
        human_issues_enabled=human_issues == "true",
        feedback_path=(
            runner_config.state_dir / "github-feedback.json"
            if role == "student"
            else None
        ),
    )
    mailbox: Mailbox = github_mailbox
    conversation_selector = None
    reconcile = None
    job_supervisor, monitor_store = training_runtime(
        runner_config.workspace,
        runner_config.state_dir / "training",
        max_timeout_seconds=runner_config.training_max_timeout_seconds,
    )
    monitor_mailbox = MonitorMailbox(
        TrainingMonitorEngine(
            monitor_store,
            job_supervisor,
            WandbMetricSource(
                env["WANDB_ENTITY"],
                env["WANDB_PROJECT"],
                api_key=runner_config.command_secrets.get("WANDB_API_KEY"),
            ),
        ),
        monitor_store,
    )

    if role == "advisor":
        role_target = RoleTarget(
            research_tag=env["RESEARCH_TAG"],
            role="advisor",
        )
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalAdvisorMailbox(runner_config.state_dir / "advisor-events.sqlite3"),
            ContextResetMailbox(
                runner_config.state_dir / "context-resets.sqlite3",
                role_target,
            ),
            monitor_mailbox,
        )
    else:
        role_target = RoleTarget(
            research_tag=env["RESEARCH_TAG"],
            role="student",
            student=env["STUDENT_NAME"],
        )
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalStudentMailbox(runner_config.state_dir / "student-events.sqlite3"),
            ContextResetMailbox(
                runner_config.state_dir / "context-resets.sqlite3",
                role_target,
            ),
            monitor_mailbox,
        )
        registry = AssignmentConversationRegistry(
            runner_config.state_dir / "student-conversations.json"
        )
        conversation_selector = StudentConversationSelector(registry)
        reconcile = StudentWorkspaceReconciler(
            runner_config.workspace,
            repo=runner_config.github_repo,
            token=runner_config.github_token,
            active_mutable_job_ids=job_supervisor.active_mutable_job_ids,
        )

    full_prompt = _full_prompt(role, env)
    system_context = compose_system_instructions(
        read_role_instructions(runner_config.harness_file),
        read_role_instructions(runner_config.role_file),
    )
    continuation_context = (
        f"{system_context.strip()}\n\n"
        f"# Current research brief\n\n{full_prompt}"
    )
    inbox = PersistentInbox(
        runner_config.state_dir / "delivery-inbox.sqlite3",
        legacy_path=runner_config.state_dir / "pending-message-deliveries.json",
    )
    turns = OpenHandsTurnRunner(
        runner_config,
        full_prompt=full_prompt,
        github_mailbox=github_mailbox,
        active_poll_interval_seconds=float(
            env.get("SENPAI_ACTIVE_GITHUB_POLL_INTERVAL_S", "120")
        ),
    )
    controller = Controller(
        role=role,
        mailbox=mailbox,
        turns=turns,
        conversation_id=runner_config.conversation_id,
        system_context=continuation_context,
        conversation_state=ConversationStateLedger(
            runner_config.state_dir / "conversation-state.json"
        ),
        inbox=inbox,
        workspace_divergence_state=(
            WorkspaceDivergenceLedger(
                runner_config.state_dir / "workspace-divergence.json"
            )
            if role == "student"
            else None
        ),
        conversation_for_events=conversation_selector,
        reconcile=reconcile,
        progress=progress,
        operation_timeout_seconds=float(
            env.get("SENPAI_CONTROLLER_OPERATION_TIMEOUT_SECONDS", "300")
        ),
        turn_timeout_seconds=runner_config.timeout_seconds + 60,
        max_consecutive_turn_failures=int(
            env.get("SENPAI_CONTROLLER_MAX_CONSECUTIVE_TURN_FAILURES", "2")
        ),
        event_reminder_seconds=(
            float(env["SENPAI_EVENT_REMINDER_SECONDS"])
            if "SENPAI_EVENT_REMINDER_SECONDS" in env
            else None
        ),
        start_gate_path=(
            Path(env["SENPAI_START_GATE_PATH"])
            if env.get("SENPAI_START_GATE_PATH")
            else None
        ),
        launch_gate_path=(
            Path(env["SENPAI_LAUNCH_GATE_PATH"])
            if env.get("SENPAI_LAUNCH_GATE_PATH")
            else None
        ),
        start_gate_poll_seconds=float(env.get("SENPAI_START_GATE_POLL_SECONDS", "30")),
        full_prompt=full_prompt,
        poll_interval_seconds=_role_interval(
            env,
            role,
            "POLL_INTERVAL_S",
            600,
        ),
        jitter_seconds=_role_interval(
            env,
            role,
            "POLL_JITTER_S",
            120,
        ),
        next_monitor_poll_seconds=monitor_store.seconds_until_next_poll,
    )

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        controller.run()
    except KeyboardInterrupt:
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        inbox.close()
        close_training_runtimes()
        finish_weave_monitoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(controller_main())
