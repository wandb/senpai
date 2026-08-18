"""Create and run Senpai OpenHands conversations."""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from openhands.sdk import Agent, LocalConversation
from openhands.sdk.conversation import ConversationExecutionStatus, ConversationState
from openhands.sdk.event import ObservationEvent
from openhands.sdk.plugin import PluginSource

from senpai_agent.advisor import AdvisorEventPump
from senpai_agent.inbox import (
    DeliveryState,
    InboxTurn,
    PersistentInbox,
    deliver_turn_messages,
    events_after_turn_delivery,
    turn_has_finished_response,
)
from senpai_agent.local_events import LocalEventStore
from senpai_agent.openhands.config import RunnerConfig, local_event_db_path
from senpai_agent.PROMPTS import RECOVERED_ACTION_PROMPT


@dataclass(frozen=True)
class ConversationOutcome:
    status: ConversationExecutionStatus
    active_inbox_turn_id: str | None


def conversation_prompt_cache_key(config: RunnerConfig) -> str | None:
    if config.model.split("/", 1)[0].lower() != "openai":
        return None
    agent_kind = config.agent_name or ("child" if config.child else "main")
    return f"senpai:{config.role}:{agent_kind}"


def create_conversation(
    agent: Agent,
    config: RunnerConfig,
    event_callback: Callable[[object], None],
) -> LocalConversation:
    return LocalConversation(
        agent=agent,
        workspace=config.workspace,
        plugins=[PluginSource(source=str(config.plugin_dir))],
        persistence_dir=config.state_dir,
        conversation_id=config.conversation_id,
        callbacks=[] if config.child else [event_callback],
        max_iteration_per_run=config.max_turns,
        visualizer=None,
        secrets=dict(config.conversation_secrets),
        tags={"runtime": "senpai-openhands"},
        delete_on_close=config.child,
        prompt_cache_key=conversation_prompt_cache_key(config),
    )


@contextmanager
def graceful_interrupts(conversation: object) -> Iterator[Callable[[], bool]]:
    interrupted_by: list[int] = []

    def interrupt(signum: int, _frame: object) -> None:
        print(f"OPENHANDS_INTERRUPT signal={signum}", file=sys.stderr, flush=True)
        if not interrupted_by:
            interrupted_by.append(signum)
        conversation.interrupt()

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield lambda: bool(interrupted_by)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if interrupted_by:
        raise SystemExit(128 + interrupted_by[0])


async def arun_conversation(
    conversation: object,
    timeout_seconds: float,
    activity: Callable[[], float] | None = None,
    *,
    started: Callable[[], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Run until completion or one full timeout passes without activity."""

    task = asyncio.create_task(conversation.arun())

    async def cancel_run() -> None:
        conversation.interrupt()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    await asyncio.sleep(0)
    if started is not None:
        started()
    if stop_requested is not None and stop_requested():
        await cancel_run()
        return
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(0, deadline - time.monotonic()),
            )
            return
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            return
        except TimeoutError:
            if task.done():
                await task
                return
            renewed = (
                activity() + timeout_seconds if activity is not None else deadline
            )
            if renewed > time.monotonic():
                deadline = renewed
                continue
            print(
                f"OPENHANDS_TIMEOUT seconds={timeout_seconds:g}",
                file=sys.stderr,
                flush=True,
            )
            await cancel_run()
            return


def run_conversation(conversation: object, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        print(
            f"OPENHANDS_TIMEOUT seconds={max(timeout_seconds, 0):g}",
            file=sys.stderr,
            flush=True,
        )
        conversation.interrupt()
        return
    asyncio.run(arun_conversation(conversation, timeout_seconds))


async def arun_steerable_conversation(
    conversation: object,
    pump: AdvisorEventPump,
    timeout_seconds: float,
    activity: Callable[[], float] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    while stop_requested is None or not stop_requested():
        if not pump.prepare_run():
            continue
        if stop_requested is not None and stop_requested():
            return
        await arun_conversation(
            conversation,
            timeout_seconds,
            activity,
            started=pump.run_started,
            stop_requested=stop_requested,
        )
        if not pump.finish_run():
            return


def run_steerable_conversation(
    conversation: object,
    pump: AdvisorEventPump,
    timeout_seconds: float,
    activity: Callable[[], float] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    asyncio.run(
        arun_steerable_conversation(
            conversation,
            pump,
            timeout_seconds,
            activity,
            stop_requested,
        )
    )


def reject_recovered_actions(conversation: object) -> int:
    """Pair crash-orphaned actions so resuming cannot replay them implicitly."""

    state = getattr(conversation, "state", None)
    active_branch = getattr(state, "active_branch", None)
    if active_branch is None:
        return 0
    pending = ConversationState.get_unmatched_actions(active_branch())
    if not pending:
        return 0
    conversation.reject_pending_actions(reason=RECOVERED_ACTION_PROMPT)
    print(
        f"OPENHANDS_RECOVERED_ACTIONS rejected={len(pending)}",
        file=sys.stderr,
        flush=True,
    )
    return len(pending)


def _activate_inbox_turn(
    conversation: object,
    inbox: PersistentInbox,
    turn_id: str,
) -> InboxTurn:
    """Reset a recovery branch when needed, then deliver its canonical messages."""

    turn = inbox.turn(turn_id)
    if turn.context_reset_required:
        active_branch = tuple(conversation.state.active_branch())
        active_senders = {getattr(event, "sender", None) for event in active_branch}
        recovery_started = any(
            message.sender in active_senders for message in turn.messages
        )
        if active_branch and not recovery_started:
            preserved_events = len(conversation.state.events)
            conversation.navigate_to(None)
            print(
                "OPENHANDS_CONTEXT_RESET "
                f"conversation_id={turn.conversation_id} "
                f"preserved_events={preserved_events}",
                file=sys.stderr,
                flush=True,
            )
        inbox.record_context_reset(turn_id)
    return deliver_turn_messages(conversation, inbox, turn_id)


def _latest_completed_tool_event_id(
    conversation: object,
    turn: InboxTurn,
) -> str | None:
    for event in reversed(events_after_turn_delivery(conversation, turn)):
        if type(event) is ObservationEvent:
            event_id = getattr(event, "id", None)
            if isinstance(event_id, str) and event_id:
                return event_id
    return None


def run_turn(
    conversation: object,
    prompt: str,
    config: RunnerConfig,
    *,
    run_deadline: float | None = None,
    reset_context: bool = False,
    inbox: PersistentInbox | None = None,
    inbox_turn_id: str | None = None,
    recovery_prompt: str | None = None,
    activity: Callable[[], float] | None = None,
) -> ConversationOutcome:
    """Deliver and run one normal, recovered, or steerable conversation turn."""

    if (inbox is None) != (inbox_turn_id is None):
        raise ValueError("inbox and inbox_turn_id must be provided together")

    reject_recovered_actions(conversation)
    active_inbox_turn_id = inbox_turn_id
    if inbox is not None and active_inbox_turn_id is not None:
        active_inbox_turn_id = inbox.latest_turn(active_inbox_turn_id).turn_id
        if reset_context:
            recovery = inbox.recover_turn(
                active_inbox_turn_id,
                recovery_prompt or prompt,
                max_generations=config.inbox_max_recovery_generations,
            )
            active_inbox_turn_id = recovery.turn_id
    elif reset_context:
        preserved_events = len(conversation.state.events)
        conversation.navigate_to(None)
        print(
            "OPENHANDS_CONTEXT_RESET "
            f"conversation_id={config.conversation_id} "
            f"preserved_events={preserved_events}",
            file=sys.stderr,
            flush=True,
        )

    inference_required = True
    if inbox is None or active_inbox_turn_id is None:
        conversation.send_message(prompt)
    else:
        turn = _activate_inbox_turn(conversation, inbox, active_inbox_turn_id)
        if turn.state is DeliveryState.PROCESSED:
            inference_required = False
        elif turn_has_finished_response(conversation, turn):
            inbox.record_processed(active_inbox_turn_id)
            inference_required = False

    if inference_required:
        if inbox is not None and active_inbox_turn_id is not None:
            turn = inbox.turn(active_inbox_turn_id)
            inbox.record_progress(
                active_inbox_turn_id,
                _latest_completed_tool_event_id(conversation, turn),
            )
            if inbox.terminal_recovery_due(
                active_inbox_turn_id,
                max_attempts=config.inbox_max_stalled_attempts,
            ):
                stalled_turn_id = active_inbox_turn_id
                recovery = inbox.recover_turn(
                    stalled_turn_id,
                    recovery_prompt or prompt,
                    max_generations=config.inbox_max_recovery_generations,
                )
                active_inbox_turn_id = recovery.turn_id
                print(
                    "SENPAI_TERMINAL_TURN_RECOVERY "
                    f"conversation_id={config.conversation_id} "
                    f"stalled_turn_id={stalled_turn_id} "
                    f"recovery_turn_id={active_inbox_turn_id}",
                    file=sys.stderr,
                    flush=True,
                )
                recovery_turn = _activate_inbox_turn(
                    conversation,
                    inbox,
                    active_inbox_turn_id,
                )
                inbox.record_progress(
                    active_inbox_turn_id,
                    _latest_completed_tool_event_id(conversation, recovery_turn),
                )
            inbox.record_inference_attempt(active_inbox_turn_id)

        with graceful_interrupts(conversation) as stop_requested:
            if config.child:
                if run_deadline is None:
                    raise ValueError("child conversations require a run deadline")
                run_conversation(conversation, run_deadline - time.time())
            else:
                with LocalEventStore(local_event_db_path(config)) as event_store:
                    event_pump = AdvisorEventPump(
                        event_store,
                        conversation,
                        parent_conversation_id=(
                            str(config.conversation_id)
                            if config.role == "student"
                            else None
                        ),
                        inbox=inbox,
                        conversation_id=config.conversation_id,
                    )
                    with event_pump:
                        run_steerable_conversation(
                            conversation,
                            event_pump,
                            config.timeout_seconds,
                            activity,
                            stop_requested,
                        )

    status = conversation.state.execution_status
    if (
        inference_required
        and inbox is not None
        and active_inbox_turn_id is not None
        and status == ConversationExecutionStatus.FINISHED
    ):
        inbox.record_processed(active_inbox_turn_id)
    return ConversationOutcome(status, active_inbox_turn_id)
