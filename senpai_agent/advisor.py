import argparse
import threading
import uuid
from collections.abc import Callable, Sequence, Set
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from openhands.sdk.conversation import ConversationExecutionStatus, ConversationState

from senpai_agent.hooks import queued_feedback_marker
from senpai_agent.inbox import (
    ADVISOR_ACTIVE_STEERING_PRIORITIES,
    QUEUE_PRIORITY,
    STEER_PRIORITY,
    DeliveryState,
    PersistentInbox,
    deliver_turn_messages,
    event_priority,
)
from senpai_agent.local_events import LocalEventStore

_TERMINAL_DELIVERY_STATUSES = frozenset(
    {
        ConversationExecutionStatus.FINISHED,
        ConversationExecutionStatus.ERROR,
        ConversationExecutionStatus.STUCK,
        ConversationExecutionStatus.DELETING,
    }
)
_STEERING_GRACE_SECONDS = 60.0
_STEERING_INTERRUPTION_NOTICE = (
    "Trusted human steering interrupted the active run. Active tools "
    "were given up to 60 seconds to finish; apply the next instruction before "
    "resuming displaced work."
)


def advisor_conversation_id(
    state_dir: Path,
    explicit_id: str | None = None,
) -> uuid.UUID:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "advisor-conversation-id"
    if explicit_id is None and path.exists():
        return uuid.UUID(path.read_text().strip())

    conversation_id = uuid.UUID(explicit_id) if explicit_id else uuid.uuid4()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{conversation_id}\n")
    temporary.replace(path)
    return conversation_id


class MessageConversation(Protocol):
    def send_message(self, message: str) -> None: ...


def _deliver_pending_events(
    store: LocalEventStore,
    conversation: MessageConversation,
    *,
    record_delivery: Callable[[str], None],
    already_delivered: Set[str] = frozenset(),
    parent_conversation_id: str | None = None,
) -> int:
    delivered = 0
    pending = store.pending()
    if parent_conversation_id is not None:
        pending = [
            event
            for event in pending
            if event.payload.get("parent_conversation_id") == parent_conversation_id
        ]
    for event in pending:
        if event.dedupe_key in already_delivered:
            continue
        conversation.send_message(event.to_user_message())
        record_delivery(event.dedupe_key)
        delivered += 1
    return delivered


def deliver_pending_events(
    store: LocalEventStore,
    conversation: MessageConversation,
    *,
    parent_conversation_id: str | None = None,
) -> int:
    return _deliver_pending_events(
        store,
        conversation,
        record_delivery=store.acknowledge,
        parent_conversation_id=parent_conversation_id,
    )


class AdvisorEventPump:
    def __init__(
        self,
        store: LocalEventStore,
        conversation: MessageConversation,
        *,
        poll_interval: float = 0.5,
        parent_conversation_id: str | None = None,
        inbox: PersistentInbox | None = None,
        conversation_id: str | uuid.UUID | None = None,
        steering_grace_seconds: float = _STEERING_GRACE_SECONDS,
    ):
        self._store = store
        self._conversation = conversation
        self._poll_interval = poll_interval
        self._parent_conversation_id = parent_conversation_id
        self._inbox = inbox
        self._queued_feedback_marker = (
            queued_feedback_marker(inbox.path.parent)
            if inbox is not None and inbox.path is not None
            else None
        )
        self._steering_grace_seconds = steering_grace_seconds
        if inbox is not None:
            if conversation_id is None:
                raise ValueError("inbox event pump requires a conversation ID")
            self._conversation_id = str(uuid.UUID(str(conversation_id)))
        else:
            self._conversation_id = str(
                conversation_id or getattr(conversation, "id", "local")
            )
        self._delivered_event_keys: set[str] = set()
        self._stop = threading.Event()
        self._steer_lock = threading.Lock()
        self._steer_turn_id: str | None = None
        self._steer_mode: int | None = None
        self._steer_generation = 0
        self._steer_ready = threading.Event()
        self._steer_interrupt_requested = False
        self._steer_interrupt_notice = False
        self._steer_paused = False
        self._boundary_thread: threading.Thread | None = None
        self._run_armed = False
        self._run_started = threading.Event()
        self._accept_steering = True
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="senpai-agent-event-pump",
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._store.pending_count():
                    self._deliver_if_safe()
                self._stop.wait(self._poll_interval)
        except BaseException as error:  # noqa: BLE001
            self._fail(error)

    def _fail(self, error: BaseException) -> None:
        with self._steer_lock:
            first_failure = self._error is None
            if first_failure:
                self._error = error
            should_interrupt = (
                first_failure and self._run_armed and self._run_started.is_set()
            )
            self._steer_ready.set()
        self._stop.set()
        if should_interrupt:
            self._conversation.interrupt()

    def _deliver_if_safe(self) -> int:
        if self._inbox is not None:
            return self._transfer_to_inbox()
        state = getattr(self._conversation, "state", None)
        if state is None:
            # Lightweight message adapters have no tool-action state to guard.
            return _deliver_pending_events(
                self._store,
                self._conversation,
                record_delivery=self._delivered_event_keys.add,
                already_delivered=self._delivered_event_keys,
                parent_conversation_id=self._parent_conversation_id,
            )
        with state:
            if state.execution_status in _TERMINAL_DELIVERY_STATUSES:
                return 0
            if ConversationState.get_unmatched_actions(state.active_branch()):
                return 0
            return _deliver_pending_events(
                self._store,
                self._conversation,
                record_delivery=self._delivered_event_keys.add,
                already_delivered=self._delivered_event_keys,
                parent_conversation_id=self._parent_conversation_id,
            )

    def _transfer_to_inbox(self) -> int:
        pending = self._store.pending()
        if self._parent_conversation_id is not None:
            pending = [
                event
                for event in pending
                if event.payload.get("parent_conversation_id")
                == self._parent_conversation_id
            ]
        transferred = 0
        start_mode: int | None = None
        steer_generation = 0
        steer_active_run = False
        for event in pending:
            mode = ADVISOR_ACTIVE_STEERING_PRIORITIES.get(event.kind)
            if mode is not None:
                with self._steer_lock:
                    if not self._accept_steering:
                        continue
                    steering = self._inbox.steer(
                        self._conversation_id,
                        event.dedupe_key,
                        event.to_inbox_message(),
                        priority=mode,
                    )
                    if steering is not None:
                        turn_id, state = steering
                        if state == DeliveryState.PENDING:
                            if self._steer_turn_id is None:
                                self._steer_generation += 1
                                steer_generation = self._steer_generation
                                start_mode = mode
                                self._steer_ready.clear()
                                self._steer_interrupt_requested = False
                                self._steer_interrupt_notice = False
                                self._steer_paused = False
                                steer_active_run = self._run_armed
                                self._steer_mode = mode
                            elif mode > self._steer_mode:
                                steer_generation = self._steer_generation
                                start_mode = mode
                                self._steer_ready.clear()
                                steer_active_run = self._run_armed
                                self._steer_mode = mode
                            self._steer_turn_id = turn_id
            else:
                self._inbox.enqueue(
                    self._conversation_id,
                    event.dedupe_key,
                    event.to_inbox_message(),
                    priority=event_priority(event.kind),
                )
            self._store.acknowledge(event.dedupe_key)
            transferred += 1
        if start_mode == STEER_PRIORITY:
            self._clear_queued_feedback_marker()
            self._interrupt_for_steer(steer_active_run)
        elif start_mode == QUEUE_PRIORITY:
            if self._queued_feedback_marker is not None:
                self._queued_feedback_marker.touch()
            self._queue_for_steer(steer_active_run, steer_generation)
        return transferred

    def _queue_for_steer(self, active_run: bool, generation: int) -> None:
        if not active_run:
            self._steer_ready.set()
            return
        self._boundary_thread = threading.Thread(
            target=self._run_boundary,
            args=(generation,),
            name="senpai-agent-steering-boundary",
            daemon=True,
        )
        self._boundary_thread.start()

    def _run_boundary(self, generation: int) -> None:
        try:
            self._pause_at_boundary(generation)
        except BaseException as error:  # noqa: BLE001
            self._fail(error)

    def _pause_at_boundary(self, generation: int) -> None:
        """Request the next safe agent-step boundary without blocking polling."""

        self._run_started.wait()
        state = getattr(self._conversation, "state", None)
        while not self._stop.is_set():
            if state is None:
                self._steer_ready.set()
                return
            if not state.acquire(timeout=self._poll_interval):
                continue
            retry = False
            try:
                with self._steer_lock:
                    if (
                        generation != self._steer_generation
                        or self._steer_mode != QUEUE_PRIORITY
                    ):
                        return
                    if not self._run_armed:
                        self._steer_ready.set()
                        return
                    if state.execution_status in (
                        ConversationExecutionStatus.IDLE,
                        ConversationExecutionStatus.PAUSED,
                        ConversationExecutionStatus.ERROR,
                    ):
                        retry = True
                    elif state.execution_status == ConversationExecutionStatus.RUNNING:
                        self._conversation.pause()
                        self._steer_paused = True
                    if not retry:
                        self._steer_ready.set()
                        return
            finally:
                state.release()
            self._stop.wait(self._poll_interval)

    def _interrupt_for_steer(self, active_run: bool) -> None:
        if not active_run:
            self._steer_ready.set()
            return
        self._run_started.wait()
        if self._stop.is_set():
            self._steer_ready.set()
            return
        state = getattr(self._conversation, "state", None)
        acquired = state is not None and state.acquire(
            timeout=self._steering_grace_seconds
        )
        interrupt_notice = not acquired
        if acquired:
            try:
                status = state.execution_status
                running = active_run and status not in (
                    ConversationExecutionStatus.FINISHED,
                    ConversationExecutionStatus.STUCK,
                )
                interrupt_notice = status == ConversationExecutionStatus.RUNNING
                if self._stop.is_set() or not running:
                    self._steer_ready.set()
                    return
                self._conversation.interrupt()
            finally:
                state.release()
        elif self._stop.is_set() or not active_run:
            self._steer_ready.set()
            return
        else:
            self._conversation.interrupt()
        with self._steer_lock:
            self._steer_interrupt_requested = True
            self._steer_interrupt_notice = interrupt_notice
        self._steer_ready.set()

    def run_started(self) -> None:
        with self._steer_lock:
            self._run_started.set()
            should_interrupt = self._error is not None and self._run_armed
        if should_interrupt:
            self._conversation.interrupt()

    def prepare_run(self) -> bool:
        """Arm one run, or deliver steering that arrived before it started."""

        with self._steer_lock:
            if self._error is not None:
                raise self._error
            if self._steer_turn_id is None:
                self._run_armed = True
                self._run_started.clear()
                return True
        self._deliver_steer()
        return False

    def finish_run(self) -> bool:
        """Close one run and deliver any steer after OpenHands cleanup."""

        with self._steer_lock:
            self._run_armed = False
            if self._steer_turn_id is None:
                self._accept_steering = False
                return False
        return self._deliver_steer()

    def _deliver_steer(self) -> bool:
        while True:
            if not self._steer_ready.wait(self._steering_grace_seconds + 1):
                raise TimeoutError("trusted input did not reach a steering boundary")
            with self._steer_lock:
                if self._error is not None:
                    raise self._error
                if not self._steer_ready.is_set():
                    continue
                turn_id = self._steer_turn_id
                interrupt_requested = self._steer_interrupt_requested
                interrupt_notice = self._steer_interrupt_notice
                paused = self._steer_paused
                self._steer_turn_id = None
                self._steer_mode = None
                self._steer_ready.clear()
                self._steer_interrupt_requested = False
                self._steer_interrupt_notice = False
                self._steer_paused = False
                self._clear_queued_feedback_marker()
                break
        assert turn_id is not None and self._inbox is not None
        state = getattr(self._conversation, "state", None)
        interrupted = (
            interrupt_requested
            and state is not None
            and state.execution_status == ConversationExecutionStatus.PAUSED
        )
        if interrupted and interrupt_notice:
            self._conversation.send_message(_STEERING_INTERRUPTION_NOTICE)
        deliver_turn_messages(self._conversation, self._inbox, turn_id)
        return (
            interrupt_requested
            or paused
            or (
                state is None
                or state.execution_status != ConversationExecutionStatus.PAUSED
            )
        )

    def __enter__(self) -> Self:
        self._clear_queued_feedback_marker()
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._run_started.set()
        self._thread.join()
        if self._boundary_thread is not None:
            self._boundary_thread.join()
        self._clear_queued_feedback_marker()
        # Failed turns may abandon their active branch; replay those events instead.
        if self._inbox is None and exc_type is None and self._error is None:
            state = getattr(self._conversation, "state", None)
            if (
                state is None
                or state.execution_status == ConversationExecutionStatus.FINISHED
            ):
                for key in sorted(self._delivered_event_keys):
                    self._store.acknowledge(key)
        if self._error is not None and _exc is not self._error:
            raise self._error from _exc

    def _clear_queued_feedback_marker(self) -> None:
        if self._queued_feedback_marker is not None:
            self._queued_feedback_marker.unlink(missing_ok=True)


def advisor_main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Inspect local Senpai advisor state")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pending = subparsers.add_parser("pending-count")
    pending.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "pending-count":
        with LocalEventStore(
            args.state_dir.expanduser().resolve() / "advisor-events.sqlite3"
        ) as store:
            print(store.pending_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(advisor_main())
