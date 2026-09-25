"""Portable Senpai control loop with GitHub as its only remote mailbox."""

from __future__ import annotations

import os
import random
import signal
import sys
import time
from base64 import b64decode
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import partial
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from senpai_agent.agent_markdown import strip_spdx_header
from senpai_agent.github.mailbox import ActiveGitHubWatcher, GitHubMailbox
from senpai_agent.inbox import (
    EXACT_ONCE_EVENT_KINDS,
    QUEUE_PRIORITY,
    STEERING_PRIORITIES,
    DeliveryState,
    InboxTurn,
    InboxTurnQuarantined,
    PersistentInbox,
)
from senpai_agent.local_events import LocalEvent, LocalEventStore
from senpai_agent.mailbox import (
    CompositeMailbox,
    ControllerEvent,
    LocalAdvisorMailbox,
    LocalStudentMailbox,
    Mailbox,
    StudentAssignmentAvailabilityMailbox,
)
from senpai_agent.monitor import (
    MonitorMailbox,
    TrainingMonitorEngine,
    WandbMetricSource,
)
from senpai_agent.PROMPTS import (
    CONTEXT_RECOVERY_PROMPT,
    CONTINUATION_CONTROLLER_PROMPT,
    INITIAL_CONTROLLER_PROMPT,
    OPERATOR_INSTRUCTIONS_PROMPT,
    render_prompt,
)
from senpai_agent.secrets import PRIVATE_CREDENTIAL_FD_ENVS, set_process_nondumpable
from senpai_agent.state import (
    AssignmentConversationRegistry,
    ConversationBatch,
    StartedConversationLedger,
    StudentConversationSelector,
    WorkspaceDivergenceLedger,
)
from senpai_agent.supervisor import LEASE_ENV, ProgressLease
from senpai_agent.workspace import StudentWorkspaceReconciler, WorkspaceDivergence


_EDGE_TRIGGERED_EVENT_KINDS = EXACT_ONCE_EVENT_KINDS | {
    "research_base_changed",
    "student_assignment_comment",
}
_ACTIVITY_LEASE_RENEWAL_SECONDS = 30
_LLM_PROVIDER_COOLDOWN_SECONDS = (30.0, 60.0, 120.0, 240.0, 300.0)


@dataclass(frozen=True, slots=True)
class TurnResult:
    exit_code: int
    delivered_event_keys: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    retryable: bool
    retry_after: float = 0


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


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    pending = [error]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(getattr(current, "_senpai_retried_provider_errors", ()))
        linked = getattr(current, "original_exception", None) or current.__cause__
        if isinstance(linked, BaseException):
            pending.append(linked)


def _retry_after_seconds(error: BaseException, now: float) -> float:
    value = getattr(error, "retry_after", None)
    if value is None:
        value = next(
            (
                header
                for headers in (
                    getattr(error, "litellm_response_headers", None),
                    getattr(error, "headers", None),
                    getattr(getattr(error, "response", None), "headers", None),
                )
                if headers
                for name, header in headers.items()
                if str(name).casefold() == "retry-after"
            ),
            None,
        )
    if value is None:
        return 0
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, parsedate_to_datetime(str(value)).timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return 0


def _provider_failure(error: Exception, now: float) -> ProviderFailure | None:
    from openai import OpenAIError
    from openhands.sdk.event.error_classification import FailureKind, classify_error
    from openhands.sdk.llm.exceptions import LLMError

    chain = tuple(_exception_chain(error))
    if not any(isinstance(current, (LLMError, OpenAIError)) for current in chain):
        return None
    kinds = {
        classify_error(type(current).__name__, str(current)).kind
        for current in chain
    }
    if kinds & {FailureKind.AUTH, FailureKind.QUOTA, FailureKind.CONFIG}:
        return ProviderFailure(retryable=False)
    if kinds & {FailureKind.AGENT_ACTION, FailureKind.INTERNAL}:
        return None
    if kinds & {FailureKind.RATE_LIMIT, FailureKind.TRANSIENT}:
        return ProviderFailure(
            retryable=True,
            retry_after=max(_retry_after_seconds(current, now) for current in chain),
        )
    return ProviderFailure(retryable=False)


def _provider_retry_delay(failures: int, retry_after: float) -> float:
    base = _LLM_PROVIDER_COOLDOWN_SECONDS[
        min(failures, len(_LLM_PROVIDER_COOLDOWN_SECONDS) - 1)
    ]
    return max(base, retry_after) + random.uniform(0, base * 0.2)


def _context_recovery_prompt(full_prompt: str, current_prompt: str) -> str:
    initial_context = (
        ""
        if not full_prompt or full_prompt in current_prompt
        else f"{full_prompt}\n\n"
    )
    return initial_context + render_prompt(
        CONTEXT_RECOVERY_PROMPT,
        CURRENT_PROMPT=current_prompt,
    )


def _activity_lease(
    progress: ProgressLease,
    timeout_seconds: float,
) -> Callable[[], None]:
    last_attempt = float("-inf")

    def renew() -> None:
        nonlocal last_attempt
        now = time.monotonic()
        if now - last_attempt < _ACTIVITY_LEASE_RENEWAL_SECONDS:
            return
        last_attempt = now
        try:
            progress.update("openhands-turn", timeout_seconds)
        except OSError as error:
            print(
                f"SENPAI_LEASE_UPDATE_ERROR {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    return renew


def _inference_state_lease(
    progress: ProgressLease,
) -> Callable[[float | None, float | None], None]:
    def publish(started_at: float | None, heartbeat_at: float | None) -> None:
        try:
            progress.update_llm_request(started_at, heartbeat_at)
        except OSError as error:
            print(
                f"SENPAI_LEASE_UPDATE_ERROR {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    return publish


class OpenHandsTurnRunner:
    def __init__(
        self,
        config: object,
        *,
        full_prompt: str,
        github_mailbox: Mailbox | None = None,
        active_poll_interval_seconds: float = 75,
        on_activity: Callable[[], None] | None = None,
        on_inference_state: (
            Callable[[float | None, float | None], None] | None
        ) = None,
    ):
        self.config = config
        self.full_prompt = full_prompt.strip()
        self.github_mailbox = github_mailbox
        self.active_poll_interval_seconds = active_poll_interval_seconds
        self.on_activity = on_activity
        self.on_inference_state = on_inference_state

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
        if (inbox is None) != (inbox_turn_id is None):
            raise ValueError("inbox and inbox turn ID must be provided together")

        def run_options() -> dict[str, object]:
            options: dict[str, object] = {}
            if inbox is not None and inbox_turn_id is not None:
                options.update(
                    inbox=inbox,
                    inbox_turn_id=inbox_turn_id,
                    recovery_prompt=_context_recovery_prompt(
                        self.full_prompt,
                        prompt,
                    ),
                )
            if self.on_activity is not None:
                options["on_activity"] = self.on_activity
            if self.on_inference_state is not None:
                options["on_inference_state"] = self.on_inference_state
            return options

        def run_turn() -> int:
            try:
                return run_openhands(prompt, config, **run_options())
            except Exception as error:
                if not _is_context_history_failure(error):
                    raise
                print(
                    "SENPAI_CONTEXT_RECOVERY "
                    f"conversation_id={conversation_id} "
                    f"error={type(error).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    return run_openhands(
                        _context_recovery_prompt(self.full_prompt, prompt),
                        config,
                        reset_context=True,
                        **run_options(),
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
                _student_live_event,
                conversation_id=conversation_id,
                registry=registry,
            )

        # A late watcher event may still be pending locally when the next
        # controller prompt carries that same GitHub event. The prompt is the
        # delivery path for this turn, so keep the event pump from repeating it.
        with LocalEventStore(store_path) as store:
            for event_key in event_keys:
                store.acknowledge(event_key)
        with ActiveGitHubWatcher(
            self.github_mailbox,
            store_path,
            known_keys=visible_event_keys | event_keys,
            poll_interval_seconds=self.active_poll_interval_seconds,
            map_event=map_event,
        ) as watcher:
            exit_code = run_turn()
        with LocalEventStore(store_path) as store:
            delivered = store.acknowledged(tuple(watcher.enqueued_keys))
        return TurnResult(
            exit_code=exit_code,
            delivered_event_keys=frozenset(delivered),
        )


def _student_live_event(
    event: ControllerEvent,
    *,
    conversation_id: UUID,
    registry: AssignmentConversationRegistry,
) -> LocalEvent | None:
    if event.kind == "student_pr_feedback":
        target = registry.for_assignment(
            str(event.payload["assignment_id"]),
            str(event.payload["revision_id"]),
        )
        if target != conversation_id:
            return None
    elif event.kind != "human_issue":
        return None
    return LocalEvent(
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
        started_conversations: StartedConversationLedger | None = None,
        inbox: PersistentInbox | None = None,
        workspace_divergence_state: WorkspaceDivergenceLedger | None = None,
        conversation_for_events: (
            Callable[[Sequence[ControllerEvent]], Sequence[ConversationBatch]] | None
        ) = None,
        reconcile: Callable[[Sequence[ControllerEvent]], None] | None = None,
        progress: ProgressLease | None = None,
        operation_timeout_seconds: float = 300,
        turn_timeout_seconds: float = 7260,
        max_consecutive_turn_failures: int = 2,
        event_reminder_seconds: float | None = None,
        start_gate_path: Path | None = None,
        launch_gate_path: Path | None = None,
        start_gate_poll_seconds: float = 30,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 600,
        jitter_seconds: float = 120,
    ):
        if min(poll_interval_seconds, jitter_seconds) < 0:
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
        self.started_conversations = started_conversations
        self.inbox = inbox or PersistentInbox()
        self.workspace_divergence_state = workspace_divergence_state
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.jitter_seconds = jitter_seconds
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
                    now = time.time()
                    provider_failure = _provider_failure(error, now)
                    if provider_failure is not None and provider_failure.retryable:
                        current = self.inbox.provider_cooldown()
                        delay = _provider_retry_delay(
                            0 if current is None else current.failure_count,
                            provider_failure.retry_after,
                        )
                        cooldown = self.inbox.defer_provider_retry(
                            turn.turn_id,
                            now + delay,
                        )
                        print(
                            "SENPAI_PROVIDER_COOLDOWN "
                            f"conversation_id={turn.conversation_id} "
                            f"turn_id={turn.turn_id} "
                            f"failure_count={cooldown.failure_count} "
                            f"retry_after_seconds={delay:g} "
                            f"error={type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    if provider_failure is not None:
                        raise
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
                self.inbox.clear_provider_cooldown()
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
            normal_delay = self.poll_interval_seconds + random.uniform(
                0,
                self.jitter_seconds,
            )
            provider_delay = self._provider_cooldown_remaining()
            if provider_delay is not None and (
                normal_delay <= 0 or provider_delay < normal_delay
            ):
                self._sleep("provider-cooldown", provider_delay)
            else:
                self._sleep("sleep", normal_delay)

    def _provider_cooldown_remaining(self) -> float | None:
        cooldown = self.inbox.provider_cooldown()
        if cooldown is None:
            return None
        remaining = cooldown.retry_at - time.time()
        return remaining if remaining > 0 else None

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
        for event in polled:
            if event.kind == "student_assignment_comment":
                self.inbox.require_event_payload(
                    self.conversation_id,
                    event.dedupe_key,
                    event.to_prompt(),
                )
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
                steering_priority = STEERING_PRIORITIES.get(event.kind)
                if steering_priority is not None:
                    self.inbox.steer(
                        conversation_id,
                        event.dedupe_key,
                        event.to_prompt(),
                        priority=steering_priority,
                        once=event.kind in EXACT_ONCE_EVENT_KINDS,
                    )
                else:
                    self.inbox.enqueue(
                        conversation_id,
                        event.dedupe_key,
                        event.to_prompt(),
                        priority=(
                            QUEUE_PRIORITY
                            if event.kind == "student_assignment"
                            else 0
                        ),
                        once=event.kind in EXACT_ONCE_EVENT_KINDS,
                    )

    def _next_ready_turn(
        self,
        excluded: set[UUID] | frozenset[UUID] = frozenset(),
    ) -> InboxTurn | None:
        if self._provider_cooldown_remaining() is not None:
            return None
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
            turn = self.inbox.next_turn(
                conversation_id,
                self._prompt((), continuing=continuing),
                legacy_prompt_identity=(
                    f"initial:{self.full_prompt}" if not continuing else None
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
    ) -> None:
        if self.progress is not None:
            self.progress.update(
                phase,
                timeout_seconds or self.operation_timeout_seconds,
                completed_turn=completed_turn,
            )

    def _sleep(self, phase: str, seconds: float) -> None:
        self._publish_progress(
            phase,
            max(seconds + self.operation_timeout_seconds, 1),
        )
        self.sleep(seconds)

    def _has_started(self, conversation_id: UUID) -> bool:
        return conversation_id in self._started or (
            self.started_conversations is not None
            and self.started_conversations.has_started(conversation_id)
        )

    def _mark_success(self, conversation_id: UUID) -> None:
        self._started.add(conversation_id)
        if self.started_conversations is not None:
            self.started_conversations.mark_started(conversation_id)

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
    ) -> str:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not continuing:
            return render_prompt(
                INITIAL_CONTROLLER_PROMPT,
                FULL_PROMPT=self.full_prompt,
                CURRENT_TIME=now,
            ).lstrip()
        return render_prompt(
            CONTINUATION_CONTROLLER_PROMPT,
            ROLE=self.role,
            CURRENT_TIME=now,
        )


def _full_prompt(env: Mapping[str, str]) -> str:
    encoded_extra = env.get("EXTRA_INSTRUCTIONS_B64")
    if not encoded_extra:
        return ""
    extra = b64decode(encoded_extra, validate=True).decode()
    return render_prompt(
        OPERATOR_INSTRUCTIONS_PROMPT,
        INSTRUCTIONS=strip_spdx_header(extra).strip(),
    )


def _role_interval(
    env: Mapping[str, str],
    role: Literal["advisor", "student"],
    suffix: str,
    default: float,
) -> float:
    role_key = f"SENPAI_{role.upper()}_{suffix}"
    shared_key = f"SENPAI_{suffix}"
    return float(env.get(role_key, env.get(shared_key, str(default))))


def _consume_private_credential_fds(env: MutableMapping[str, str]) -> None:
    for credential_name, fd_env in PRIVATE_CREDENTIAL_FD_ENVS.items():
        descriptor_value = env.pop(fd_env, None)
        if descriptor_value is None:
            continue
        try:
            descriptor = int(descriptor_value)
        except ValueError as error:
            raise RuntimeError(f"{fd_env} must be an integer") from error
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            value = stream.read().strip()
        if not value:
            raise RuntimeError(f"{fd_env} is empty")
        env[credential_name] = value


def controller_main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] = os.environ,
) -> int:
    import argparse

    env = os.environ if env is os.environ else dict(env)
    set_process_nondumpable()
    _consume_private_credential_fds(env)

    progress = ProgressLease(Path(env[LEASE_ENV])) if env.get(LEASE_ENV) else None
    if progress is not None:
        progress.update("startup", 300)

    from senpai_agent.delegation import reconcile_delegated_tasks
    from senpai_agent.weave_monitoring import (
        finish_weave_monitoring,
        initialize_weave_monitoring,
    )

    initialize_weave_monitoring(env)

    from senpai_agent.openhands_runner import (
        parse_runner_args,
        resolve_config,
        scrub_model_credentials,
    )
    from senpai_agent.tools import (
        close_training_runtimes,
        configure_training_credentials,
        training_runtime,
    )
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
    configure_training_credentials(runner_config.training_wandb_api_key)
    reconcile_delegated_tasks(
        getattr(runner_config, "delegation_root_state_dir", None)
        or runner_config.state_dir,
        runner_config.state_dir / f"{role}-events.sqlite3",
    )
    students = tuple(
        student.strip()
        for student in env.get("STUDENT_NAMES", "").split(",")
        if student.strip()
    )
    inbox = PersistentInbox(
        runner_config.state_dir / "delivery-inbox.sqlite3",
        legacy_path=runner_config.state_dir / "pending-message-deliveries.json",
    )
    github_mailbox = GitHubMailbox(
        repo=runner_config.github_repo,
        token=runner_config.github_token,
        role=role,
        advisor_branch=env["ADVISOR_BRANCH"],
        students=students,
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
    active_github_mailbox: Mailbox = github_mailbox
    conversation_selector = None
    reconcile = None

    if role == "advisor":
        advisor_event_store = runner_config.state_dir / "advisor-events.sqlite3"
        active_github_mailbox = StudentAssignmentAvailabilityMailbox(
            github_mailbox,
            inbox=inbox,
            conversation_id=runner_config.conversation_id,
            event_store_path=advisor_event_store,
        )
        mailbox = CompositeMailbox(
            active_github_mailbox,
            LocalAdvisorMailbox(advisor_event_store),
        )
    else:
        training, monitor_store = training_runtime(
            runner_config.workspace,
            runner_config.state_dir / "training",
        )
        metrics = WandbMetricSource(
            env["WANDB_ENTITY"],
            env["WANDB_PROJECT"],
            runner_config.wandb_api_key,
        )
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalStudentMailbox(runner_config.state_dir / "student-events.sqlite3"),
            MonitorMailbox(
                TrainingMonitorEngine(monitor_store, training, metrics),
                monitor_store,
            ),
        )
        registry = AssignmentConversationRegistry(
            runner_config.state_dir / "student-conversations.json"
        )
        conversation_selector = StudentConversationSelector(registry)
        reconcile = StudentWorkspaceReconciler(
            runner_config.workspace,
            repo=runner_config.github_repo,
            token=runner_config.github_token,
        )

    full_prompt = _full_prompt(env)
    turn_lease_seconds = runner_config.timeout_seconds + 60
    turns = OpenHandsTurnRunner(
        runner_config,
        full_prompt=full_prompt,
        github_mailbox=active_github_mailbox,
        active_poll_interval_seconds=float(
            env.get("SENPAI_ACTIVE_GITHUB_POLL_INTERVAL_S", "75")
        ),
        on_activity=(
            _activity_lease(progress, turn_lease_seconds)
            if progress is not None
            else None
        ),
        on_inference_state=(
            _inference_state_lease(progress) if progress is not None else None
        ),
    )
    controller = Controller(
        role=role,
        mailbox=mailbox,
        turns=turns,
        conversation_id=runner_config.conversation_id,
        started_conversations=StartedConversationLedger(
            runner_config.state_dir / "started-conversations.json"
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
        turn_timeout_seconds=turn_lease_seconds,
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
        configure_training_credentials(None)
        finish_weave_monitoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(controller_main())
