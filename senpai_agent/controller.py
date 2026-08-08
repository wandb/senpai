"""Portable Senpai control loop with GitHub as its only remote mailbox."""

from __future__ import annotations

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
from senpai_agent.mailbox import (
    CompositeMailbox,
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
from senpai_agent.workspace import StudentWorkspaceReconciler, WorkspaceDivergence


_EDGE_TRIGGERED_EVENT_KINDS = frozenset({"research_base_changed"})


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
        active_poll_interval_seconds: float = 30,
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
    ) -> TurnResult:
        from senpai_agent.openhands_runner import local_event_db_path, run_openhands

        config = replace(
            self.config,
            conversation_id=conversation_id,
        )
        turn_deadline = time.monotonic() + config.timeout_seconds

        def run_turn() -> int:
            try:
                return run_openhands(prompt, config)
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
        with AdvisorEventStore(store_path) as store:
            for event_key in event_keys:
                store.acknowledge(event_key)
        with ActiveGitHubWatcher(
            self.github_mailbox,
            store_path,
            known_keys=event_keys,
            poll_interval_seconds=self.active_poll_interval_seconds,
            map_event=map_event,
        ) as watcher:
            exit_code = run_turn()
        with AdvisorEventStore(store_path) as store:
            delivered = store.acknowledged(tuple(watcher.observed_keys))
        return TurnResult(
            exit_code=exit_code,
            delivered_event_keys=frozenset(delivered),
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
        self.system_context = system_context.strip()
        self.conversation_state = conversation_state
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
        self._visible: dict[str, float] = {}
        self._deferred_until: dict[str, float] = {}
        self._workspace_divergence: dict[UUID, str] = {}

    def run(self, *, max_cycles: int | None = None) -> None:
        self._wait_for_start_gates()
        cycles = 0
        turn_failures = 0
        while max_cycles is None or cycles < max_cycles:
            self._publish_progress("poll")
            events = self._new_events(self.mailbox.poll())
            turn_failed = False
            while events:
                batches = self._event_batches(events)
                events = ()
                for batch in batches:
                    batch_events = batch.events
                    conversation_id = batch.conversation_id
                    workspace_divergence: WorkspaceDivergence | None = None
                    try:
                        if self.reconcile is not None:
                            self._publish_progress("reconcile")
                            try:
                                self.reconcile(batch_events)
                            except WorkspaceDivergence as conflict:
                                workspace_divergence = conflict
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
                                    event.kind == "student_assignment"
                                    for event in batch_events
                                ):
                                    self._clear_workspace_divergence(conversation_id)
                        continuing = self._has_started(conversation_id)
                        refresh_system_context = (
                            continuing
                            and self.conversation_state is not None
                            and not self.conversation_state.is_context_current(
                                conversation_id,
                                self.system_context,
                            )
                        )
                        prompt = self._prompt(
                            batch_events,
                            continuing=continuing,
                            refresh_system_context=refresh_system_context,
                        )
                        self._publish_progress(
                            "openhands-turn",
                            self.turn_timeout_seconds,
                        )
                        result = self.turns.run(
                            prompt,
                            conversation_id=conversation_id,
                            event_keys=frozenset(
                                event.dedupe_key for event in batch_events
                            ),
                        )
                    except ConversationRecoveryExhausted as error:
                        retry_delay = max(self.poll_interval_seconds, 600)
                        retry_at = time.monotonic() + retry_delay
                        deferred_keys = tuple(
                            event.dedupe_key for event in batch_events
                        )
                        for key in deferred_keys:
                            self._visible.pop(key, None)
                            self._deferred_until[key] = retry_at
                        print(
                            "SENPAI_TURN_DEFERRED "
                            f"conversation_id={conversation_id} "
                            f"event_keys={','.join(deferred_keys)} "
                            f"retry_after_seconds={retry_delay:g} "
                            f"error={error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    except Exception as error:  # noqa: BLE001
                        turn_failures += 1
                        for event in batch_events:
                            self._visible.pop(event.dedupe_key, None)
                        print(
                            f"SENPAI_TURN_EXCEPTION {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if turn_failures >= self.max_consecutive_turn_failures:
                            raise RuntimeError(
                                "controller exceeded its consecutive turn-failure "
                                f"limit ({self.max_consecutive_turn_failures})"
                            ) from error
                        turn_failed = True
                        continue
                    if result.exit_code == 0:
                        self._mark_success(conversation_id)
                        acknowledged = (
                            *(event.dedupe_key for event in batch_events),
                            *sorted(result.delivered_event_keys),
                        )
                        self.mailbox.acknowledge(
                            tuple(dict.fromkeys(acknowledged))
                        )
                        if workspace_divergence is not None:
                            self._record_workspace_divergence(
                                conversation_id,
                                workspace_divergence.event.dedupe_key,
                            )
                        self._publish_progress(
                            "turn-complete",
                            completed_turn=True,
                        )
                        delivered_at = time.monotonic()
                        for key in acknowledged:
                            self._visible[key] = delivered_at
                        continue
                    turn_failures += 1
                    for event in batch_events:
                        self._visible.pop(event.dedupe_key, None)
                    print(
                        "SENPAI_TURN_ERROR "
                        f"exit_code={result.exit_code} "
                        f"conversation_id={conversation_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if turn_failures >= self.max_consecutive_turn_failures:
                        raise RuntimeError(
                            "controller exceeded its consecutive turn-failure "
                            f"limit ({self.max_consecutive_turn_failures})"
                        )
                    turn_failed = True
                if turn_failed:
                    delay = min(
                        self.poll_interval_seconds,
                        2 ** min(turn_failures, 8),
                    )
                    self._sleep("turn-backoff", delay)
                    break
                turn_failures = 0
                # Post-turn reconciliation avoids waiting one heartbeat for work
                # that appeared while OpenHands was reasoning.
                try:
                    self._publish_progress("poll")
                    events = self._new_events(
                        self.mailbox.poll(),
                        allow_reminders=False,
                    )
                except Exception as error:  # noqa: BLE001
                    print(
                        f"SENPAI_POST_TURN_POLL_ERROR {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    events = ()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return
            if turn_failed:
                continue
            self._sleep(
                "sleep",
                self.poll_interval_seconds + random.uniform(0, self.jitter_seconds),
            )

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
        events: Sequence[ControllerEvent],
        *,
        continuing: bool,
        refresh_system_context: bool = False,
    ) -> str:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        event_prompt = "\n\n".join(event.to_prompt() for event in events)
        if not continuing:
            return (
                f"{self.full_prompt}\n\nCurrent time (UTC): {now}\n\n"
                f"# Current GitHub state\n\n{event_prompt}"
            )
        prompt = (
            f"Continue the {self.role} loop. Current time (UTC): {now}. "
            "GitHub now contains the following actionable state:\n\n"
            f"{event_prompt}"
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
    role_brief = (
        Template(read_agent_markdown(instructions)).safe_substitute(env).strip()
        if instructions.is_file()
        else "Follow the repository AGENTS.md, the assigned GitHub work, and the Senpai role charter."
    )
    prompt = (
        "# Research programme\n\n"
        f"{read_agent_markdown(program).strip()}\n\n"
        f"# {role.title()} task\n\n"
        f"{role_brief}"
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

    from senpai_agent.openhands_runner import (
        parse_runner_args,
        read_role_instructions,
        resolve_config,
        scrub_model_credentials,
    )
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

    if role == "advisor":
        mailbox = CompositeMailbox(
            github_mailbox,
            LocalAdvisorMailbox(runner_config.state_dir / "advisor-events.sqlite3"),
        )
    else:
        training, monitor_store = training_runtime(
            runner_config.workspace,
            runner_config.state_dir / "training",
            max_timeout_seconds=runner_config.training_max_timeout_seconds,
        )
        metrics = WandbMetricSource(
            env["WANDB_ENTITY"],
            env["WANDB_PROJECT"],
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

    full_prompt = _full_prompt(role, env)
    system_context = compose_system_instructions(
        read_role_instructions(runner_config.harness_file),
        read_role_instructions(runner_config.role_file),
    )
    continuation_context = (
        f"{system_context.strip()}\n\n"
        f"# Current research brief\n\n{full_prompt}"
    )
    turns = OpenHandsTurnRunner(
        runner_config,
        full_prompt=full_prompt,
        github_mailbox=github_mailbox,
        active_poll_interval_seconds=float(
            env.get("SENPAI_ACTIVE_GITHUB_POLL_INTERVAL_S", "30")
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
        close_training_runtimes()
        finish_weave_monitoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(controller_main())
