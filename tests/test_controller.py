import time
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from test_agent_markdown import HTML_HEADER

import senpai_agent.controller as controller_module
from senpai_agent.controller import (
    Controller,
    ConversationRecoveryExhausted,
    TurnResult,
    _full_prompt,
)
from senpai_agent.mailbox import ControllerEvent
from senpai_agent.state import ConversationStateLedger, WorkspaceDivergenceLedger
from senpai_agent.supervisor import ProgressLease, WorkerLease
from senpai_agent.workspace import WorkspaceDivergence, WorkspaceJobRunning

CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000001")


class Mailbox:
    def __init__(self, polls):
        self.polls = list(polls)
        self.calls = 0
        self.acknowledged = []

    def poll(self):
        self.calls += 1
        return self.polls.pop(0) if self.polls else ()

    def acknowledge(self, dedupe_keys):
        self.acknowledged.append(tuple(dedupe_keys))


class Turns:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def run(
        self,
        prompt,
        *,
        conversation_id,
        event_keys,
        visible_event_keys=frozenset(),
    ):
        self.calls.append(
            (prompt, conversation_id, event_keys, visible_event_keys)
        )
        outcome = self.outcomes.pop(0) if self.outcomes else TurnResult(exit_code=0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def controller(mailbox, turns, **overrides):
    return Controller(
        role=overrides.pop("role", "advisor"),
        mailbox=mailbox,
        turns=turns,
        conversation_id=overrides.pop("conversation_id", CONVERSATION_ID),
        full_prompt="programme",
        sleep=overrides.pop("sleep", lambda _seconds: None),
        poll_interval_seconds=overrides.pop("poll_interval_seconds", 600),
        jitter_seconds=overrides.pop("jitter_seconds", 0),
        **overrides,
    )


def review_event(number=17):
    return ControllerEvent(
        kind="review_ready",
        dedupe_key=f"review_ready:{number}:abc",
        payload={"number": number},
    )


def research_base_event(current_sha="def"):
    return ControllerEvent(
        kind="research_base_changed",
        dedupe_key=f"research_base_changed:17:abc:{current_sha}",
        payload={
            "number": 17,
            "required_base_sha": "abc",
            "current_base_sha": current_sha,
        },
    )


def test_first_turn_combines_programme_role_template_and_runtime_identity(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    instructions = workspace / "instructions"
    instructions.mkdir(parents=True)
    (workspace / "program.md").write_text(HTML_HEADER + "Minimize test error.")
    (instructions / "prompt-student.md").write_text(
        HTML_HEADER
        + "Work as $STUDENT_NAME on $ADVISOR_BRANCH in $WANDB_PROJECT. "
        + "Never expose $WANDB_API_KEY."
    )

    prompt = _full_prompt(
        "student",
        {
            "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
            "GH_REPO": "acme/widgets",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "cfd",
            "WANDB_API_KEY": "live-secret",
            "STUDENT_NAME": "fern",
            "EXTRA_INSTRUCTIONS_B64": b64encode(
                (HTML_HEADER + "Use typed tools.").encode()
            ).decode(),
        },
    )

    assert "# Research programme\n\nMinimize test error." in prompt
    assert "# Student task\n\nWork as fern on research in cfd." in prompt
    assert "$WANDB_API_KEY" in prompt
    assert "live-secret" not in prompt
    assert "# Additional launch instructions\n\nUse typed tools." in prompt
    assert "Role: student; repository: acme/widgets" in prompt
    assert "SPDX-" not in prompt


def test_first_turn_supports_repository_nested_program_without_role_overlay(
    tmp_path: Path,
):
    workspace = tmp_path / "target"
    (workspace / "senpai").mkdir(parents=True)
    (workspace / "senpai" / "program.md").write_text("Optimize MLX inference.")

    prompt = _full_prompt(
        "student",
        {
            "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
            "GH_REPO": "acme/mlxfast",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "mlxfast",
            "STUDENT_NAME": "fern",
            "EXTRA_INSTRUCTIONS_B64": b64encode(
                b"program.md can be found in senpai/program.md"
            ).decode(),
        },
    )

    assert "# Research programme\n\nOptimize MLX inference." in prompt
    assert "Follow the repository AGENTS.md" in prompt
    assert "program.md can be found in senpai/program.md" in prompt


def test_advisor_runtime_identity_includes_its_configured_name(tmp_path: Path):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "program.md").write_text("Improve the benchmark.")

    prompt = _full_prompt(
        "advisor",
        {
            "SENPAI_OPENHANDS_WORKSPACE": str(workspace),
            "GH_REPO": "acme/widgets",
            "ADVISOR_NAME": "aurora",
            "ADVISOR_BRANCH": "research",
            "WANDB_ENTITY": "acme",
            "WANDB_PROJECT": "cfd",
            "STUDENT_NAMES": "fern,tanjiro",
        },
    )

    assert "Advisor: aurora. Students: fern,tanjiro." in prompt


def test_empty_mailbox_does_not_start_a_model_turn():
    turns = Turns()

    controller(Mailbox([(), ()]), turns, role="student").run(max_cycles=2)

    assert turns.calls == []


def test_active_job_monitor_shortens_the_controller_heartbeat():
    sleeps = []

    controller(
        Mailbox([(), ()]),
        Turns(),
        sleep=sleeps.append,
        next_monitor_poll_seconds=lambda: 45,
    ).run(max_cycles=2)

    assert sleeps == [45]


def test_idle_job_monitor_keeps_the_controller_heartbeat():
    sleeps = []

    controller(
        Mailbox([(), ()]),
        Turns(),
        sleep=sleeps.append,
        next_monitor_poll_seconds=lambda: None,
    ).run(max_cycles=2)

    assert sleeps == [600]


@pytest.mark.parametrize(
    ("next_poll", "expected_phase"),
    [
        (lambda: float("nan"), "monitor-backoff"),
        (
            lambda: (_ for _ in ()).throw(RuntimeError("scheduler failed")),
            "monitor-backoff",
        ),
    ],
)
def test_invalid_monitor_schedule_uses_bounded_nonzero_backoff(
    next_poll,
    expected_phase,
):
    sleeps = []
    phases = []
    instance = controller(
        Mailbox([(), ()]),
        Turns(),
        sleep=sleeps.append,
        next_monitor_poll_seconds=next_poll,
        progress=SimpleNamespace(
            update=lambda phase, *_args, **_kwargs: phases.append(phase)
        ),
    )

    instance.run(max_cycles=2)

    assert sleeps == [5]
    assert expected_phase in phases


def test_due_monitor_never_creates_a_zero_sleep_hot_loop():
    sleeps = []

    controller(
        Mailbox([(), ()]),
        Turns(),
        sleep=sleeps.append,
        next_monitor_poll_seconds=lambda: 0,
    ).run(max_cycles=2)

    assert sleeps == [1]


def test_successful_turn_repolls_immediately_and_continues_without_full_brief():
    first = ControllerEvent(
        kind="idle_student",
        dedupe_key="idle_student:student-1",
        payload={"student": "student-1"},
    )
    second = review_event()
    mailbox = Mailbox([(first,), (second,), ()])
    turns = Turns()

    controller(mailbox, turns).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({first.dedupe_key}),
        frozenset({second.dedupe_key}),
    ]
    assert "programme" in turns.calls[0][0]
    assert "programme" not in turns.calls[1][0]
    assert mailbox.calls == 3


def test_post_turn_poll_at_reminder_boundary_only_delivers_new_state(
    monkeypatch,
):
    original = review_event(17)
    changed = review_event(18)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PostTurnMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            if self.calls == 1:
                return (original,)
            if self.calls == 2:
                clock[0] = 30
            return (original, changed)

    mailbox = PostTurnMailbox([])
    turns = Turns()

    controller_module.Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda _seconds: None,
        poll_interval_seconds=30,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=2)

    assert [call[2] for call in turns.calls] == [
        frozenset({original.dedupe_key}),
        frozenset({changed.dedupe_key}),
        frozenset({original.dedupe_key}),
    ]
    assert [call[3] for call in turns.calls] == [
        frozenset({original.dedupe_key}),
        frozenset({original.dedupe_key, changed.dedupe_key}),
        frozenset({original.dedupe_key, changed.dedupe_key}),
    ]


@pytest.mark.parametrize(
    "failure",
    [TurnResult(exit_code=19), RuntimeError("SDK transport failed")],
    ids=["nonzero-exit", "exception"],
)
def test_failed_turn_retries_the_same_unacknowledged_event_with_full_brief(failure):
    event = review_event()
    mailbox = Mailbox([(event,), (event,), ()])
    turns = Turns([failure, TurnResult(exit_code=0)])

    controller(mailbox, turns).run(max_cycles=2)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]
    assert all("programme" in call[0] for call in turns.calls)
    assert mailbox.acknowledged == [(event.dedupe_key,)]


def test_level_trigger_is_quiet_while_visible_and_wakes_after_reappearing():
    event = review_event()
    mailbox = Mailbox([(event,), (event,), (), (event,), ()])
    turns = Turns()

    controller(mailbox, turns).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]
    assert mailbox.acknowledged == [
        (event.dedupe_key,),
        (event.dedupe_key,),
    ]


def test_still_actionable_event_is_redelivered_after_one_poll_interval(
    monkeypatch,
):
    event = review_event()
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def __init__(self):
            super().__init__([])

        def poll(self):
            self.calls += 1
            return (event,)

    mailbox = PersistentMailbox()
    turns = Turns()

    controller_module.Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]


def test_older_visible_event_does_not_lose_its_reminder_to_another_turn(
    monkeypatch,
):
    older = review_event(17)
    newer = review_event(18)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (older,) if self.calls <= 2 else (older, newer)

    class WatcherAwareTurns(Turns):
        def run(
            self,
            prompt,
            *,
            conversation_id,
            event_keys,
            visible_event_keys=frozenset(),
        ):
            result = super().run(
                prompt,
                conversation_id=conversation_id,
                event_keys=event_keys,
                visible_event_keys=visible_event_keys,
            )
            if (
                event_keys == frozenset({newer.dedupe_key})
                and older.dedupe_key not in visible_event_keys
            ):
                return TurnResult(
                    exit_code=result.exit_code,
                    delivered_event_keys=frozenset({older.dedupe_key}),
                )
            return result

    turns = WatcherAwareTurns()
    Controller(
        role="advisor",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=20,
        jitter_seconds=0,
        event_reminder_seconds=30,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({older.dedupe_key}),
        frozenset({newer.dedupe_key}),
        frozenset({older.dedupe_key}),
    ]
    assert turns.calls[1][3] == frozenset(
        {older.dedupe_key, newer.dedupe_key}
    )


def test_fast_poll_defaults_to_ten_minute_level_trigger_reminders(monkeypatch):
    event = review_event()
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def __init__(self):
            super().__init__([])

        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="advisor",
        mailbox=PersistentMailbox(),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
    ).run(max_cycles=21)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]


def test_unchanged_research_base_change_does_not_repeat_on_reminder_cadence(
    monkeypatch,
):
    event = research_base_event()
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="advisor",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=600,
        jitter_seconds=0,
        event_reminder_seconds=600,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
    ]


def test_changed_research_base_sha_wakes_immediately():
    first = research_base_event("def")
    changed = research_base_event("fed")
    turns = Turns()

    controller(Mailbox([(first,), (changed,), ()]), turns).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({first.dedupe_key}),
        frozenset({changed.dedupe_key}),
    ]


def test_research_base_change_wakes_again_after_disappearing():
    event = research_base_event()
    turns = Turns()

    controller(
        Mailbox([(event,), (event,), (), (event,), ()]),
        turns,
    ).run(max_cycles=3)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]


def test_controller_main_does_not_derive_reminders_from_fast_polling(
    monkeypatch,
    tmp_path: Path,
):
    import senpai_agent.openhands_runner as runner_module
    import senpai_agent.tools as tools_module
    import senpai_agent.weave_monitoring as weave_module

    config = SimpleNamespace(
        github_token="token",
        github_repo="acme/widgets",
        github_trusted_actor=None,
        state_dir=tmp_path,
        workspace=tmp_path,
        training_max_timeout_seconds=1800,
        command_secrets={"WANDB_API_KEY": "wandb-key"},
        conversation_id=CONVERSATION_ID,
        timeout_seconds=3600,
        harness_file=tmp_path / "harness.md",
        role_file=tmp_path / "role.md",
    )
    monkeypatch.setattr(runner_module, "parse_runner_args", lambda _argv: object())
    monkeypatch.setattr(
        runner_module,
        "resolve_config",
        lambda _args, _env: config,
    )
    monkeypatch.setattr(runner_module, "scrub_model_credentials", lambda *_: None)
    monkeypatch.setattr(runner_module, "read_role_instructions", lambda _: "")
    monkeypatch.setattr(tools_module, "close_training_runtimes", lambda: None)
    monkeypatch.setattr(weave_module, "finish_weave_monitoring", lambda: None)
    monkeypatch.setattr(
        controller_module,
        "GitHubMailbox",
        lambda **_kwargs: Mailbox([]),
    )
    monkeypatch.setattr(
        controller_module,
        "compose_system_instructions",
        lambda *_: "",
    )
    monkeypatch.setattr(controller_module, "_full_prompt", lambda *_: "programme")

    created = []

    class CapturingController(Controller):
        def run(self, *, max_cycles=None):
            created.append(self)

    monkeypatch.setattr(controller_module, "Controller", CapturingController)

    assert (
        controller_module.controller_main(
            ["advisor"],
                {
                    "SENPAI_ROLE": "advisor",
                    "RESEARCH_TAG": "test-track",
                    "ADVISOR_BRANCH": "main",
                "SENPAI_POLL_INTERVAL_S": "30",
                "WANDB_ENTITY": "acme",
                "WANDB_PROJECT": "research",
            },
        )
        == 0
    )
    assert created[0].poll_interval_seconds == 30
    assert created[0].event_reminder_seconds == 600
    assert created[0].full_prompt == "programme"
    assert created[0].turns.full_prompt == "programme"
    assert created[0].system_context.endswith("# Current research brief\n\nprogramme")


def test_repeated_turn_failures_exit_to_the_supervisor_for_a_clean_restart():
    event = review_event()
    mailbox = Mailbox([(event,), (event,)])
    turns = Turns([RuntimeError("transport failed"), RuntimeError("still failed")])

    with pytest.raises(RuntimeError, match="turn-failure limit"):
        controller(
            mailbox,
            turns,
            max_consecutive_turn_failures=2,
        ).run(max_cycles=2)

    assert len(turns.calls) == 2


def test_exhausted_context_recovery_defers_then_retries_without_failure_streak(
    monkeypatch,
    capsys,
):
    event = review_event(17)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    mailbox = Mailbox([(event,), (event,), (event,), ()])
    turns = Turns(
        [
            ConversationRecoveryExhausted(
                CONVERSATION_ID,
                RuntimeError("clean branch also exceeded context"),
            ),
            TurnResult(exit_code=0),
        ]
    )

    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=600,
        jitter_seconds=0,
        max_consecutive_turn_failures=1,
    ).run(max_cycles=2)

    assert [call[2] for call in turns.calls] == [
        frozenset({event.dedupe_key}),
        frozenset({event.dedupe_key}),
    ]
    assert mailbox.acknowledged == [(event.dedupe_key,)]
    assert clock[0] == 600
    log = capsys.readouterr().err
    assert "SENPAI_TURN_DEFERRED" in log
    assert "retry_after_seconds=600" in log


def test_context_retry_deadline_survives_a_transiently_absent_event(
    monkeypatch,
):
    event = review_event(17)
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])
    mailbox = Mailbox([(event,), (), (event,), ()])
    turns = Turns(
        [
            ConversationRecoveryExhausted(
                CONVERSATION_ID,
                RuntimeError("clean branch also exceeded context"),
            ),
        ]
    )

    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=30,
        jitter_seconds=0,
        max_consecutive_turn_failures=1,
    ).run(max_cycles=2)

    assert len(turns.calls) == 1
    assert mailbox.acknowledged == []
    assert clock[0] == 30


def test_preserved_workspace_divergence_is_delivered_to_the_existing_turn():
    event = review_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    turns = Turns()

    def reconcile(_events):
        raise conflict

    controller(
        Mailbox([(event,), ()]),
        turns,
        role="student",
        reconcile=reconcile,
    ).run(max_cycles=1)

    assert turns.calls[0][2] == frozenset({event.dedupe_key, conflict.event.dedupe_key})
    assert "do not reset or discard local work" in turns.calls[0][0]


def student_assignment_event():
    return ControllerEvent(
        kind="student_assignment",
        dedupe_key="student_assignment:assignment-17:revision-2:base:head",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
        },
    )


def test_identical_workspace_divergence_does_not_rewake_on_reminders(
    monkeypatch,
):
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="student",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=1,
        jitter_seconds=0,
        event_reminder_seconds=1,
    ).run(max_cycles=3)

    assert len(turns.calls) == 1


def test_workspace_divergence_suppression_survives_a_controller_restart(tmp_path):
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    ledger = WorkspaceDivergenceLedger(tmp_path / "workspace-divergence.json")
    first_turns = Turns()
    second_turns = Turns()

    for turns in (first_turns, second_turns):
        controller(
            Mailbox([(event,), ()]),
            turns,
            role="student",
            reconcile=lambda _events: (_ for _ in ()).throw(conflict),
            workspace_divergence_state=ledger,
        ).run(max_cycles=1)

    assert len(first_turns.calls) == 1
    assert second_turns.calls == []


def test_unacknowledged_workspace_divergence_is_not_suppressed_after_restart(
    tmp_path,
):
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    ledger = WorkspaceDivergenceLedger(tmp_path / "workspace-divergence.json")

    class FailingAcknowledgementMailbox(Mailbox):
        def acknowledge(self, _dedupe_keys):
            raise RuntimeError("mailbox acknowledgement failed")

    with pytest.raises(RuntimeError, match="mailbox acknowledgement failed"):
        controller(
            FailingAcknowledgementMailbox([(event,)]),
            Turns(),
            role="student",
            reconcile=lambda _events: (_ for _ in ()).throw(conflict),
            workspace_divergence_state=ledger,
        ).run(max_cycles=1)

    assert ledger.current(CONVERSATION_ID) is None

    restarted_turns = Turns()
    controller(
        Mailbox([(event,), ()]),
        restarted_turns,
        role="student",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
        workspace_divergence_state=ledger,
    ).run(max_cycles=1)

    assert len(restarted_turns.calls) == 1


def test_new_feedback_passes_through_an_unchanged_workspace_divergence():
    assignment = student_assignment_event()
    feedback = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:issue_comment:17:101",
        payload={"message": "Use the accepted base."},
    )
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    turns = Turns()

    controller(
        Mailbox([(assignment,), (assignment, feedback), ()]),
        turns,
        role="student",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
    ).run(max_cycles=1)

    assert [call[2] for call in turns.calls] == [
        frozenset({assignment.dedupe_key, conflict.event.dedupe_key}),
        frozenset({feedback.dedupe_key, conflict.event.dedupe_key}),
    ]


def test_changed_workspace_divergence_wakes_again(monkeypatch):
    event = student_assignment_event()
    conflicts = iter(
        [
            WorkspaceDivergence(
                head_ref="student/candidate",
                expected_head="a" * 40,
                local_head="b" * 40,
            ),
            WorkspaceDivergence(
                head_ref="student/candidate",
                expected_head="a" * 40,
                local_head="c" * 40,
            ),
        ]
    )
    last = [None]
    clock = [0.0]
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    def reconcile(_events):
        last[0] = next(conflicts, last[0])
        raise last[0]

    class PersistentMailbox(Mailbox):
        def poll(self):
            self.calls += 1
            return (event,)

    turns = Turns()
    controller_module.Controller(
        role="student",
        mailbox=PersistentMailbox([]),
        turns=turns,
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        reconcile=reconcile,
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        poll_interval_seconds=1,
        jitter_seconds=0,
        event_reminder_seconds=1,
    ).run(max_cycles=2)

    assert len(turns.calls) == 2


def test_failed_workspace_divergence_turn_is_retried():
    event = student_assignment_event()
    conflict = WorkspaceDivergence(
        head_ref="student/candidate",
        expected_head="a" * 40,
        local_head="b" * 40,
    )
    turns = Turns([TurnResult(exit_code=19), TurnResult(exit_code=0)])

    controller(
        Mailbox([(event,), (event,), ()]),
        turns,
        role="student",
        reconcile=lambda _events: (_ for _ in ()).throw(conflict),
    ).run(max_cycles=2)

    assert len(turns.calls) == 2


def test_successful_turn_acknowledges_and_suppresses_mid_turn_feedback():
    assignment = ControllerEvent(
        kind="student_assignment",
        dedupe_key="student_assignment:assignment-17:revision-2",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
        },
    )
    feedback = ControllerEvent(
        kind="student_pr_feedback",
        dedupe_key="student_pr_feedback:issue_comment:17:101",
        payload={
            "assignment_id": "assignment-17",
            "revision_id": "revision-2",
        },
    )
    mailbox = Mailbox([(assignment,), (feedback,)])
    turns = Turns(
        [
            TurnResult(
                exit_code=0,
                delivered_event_keys=frozenset({feedback.dedupe_key}),
            )
        ]
    )

    controller(mailbox, turns, role="student").run(max_cycles=1)

    assert len(turns.calls) == 1
    assert mailbox.acknowledged == [
        (assignment.dedupe_key, feedback.dedupe_key),
    ]


def test_start_gate_wait_publishes_a_live_lease_before_polling(tmp_path: Path):
    gate = tmp_path / "start-gate"
    lease_path = tmp_path / "controller-lease.json"
    mailbox = Mailbox([()])
    observed = []

    def open_gate(seconds):
        observed.append((seconds, WorkerLease.read(lease_path)))
        gate.write_text("open")

    before = time.monotonic()
    Controller(
        role="advisor",
        mailbox=mailbox,
        turns=Turns(),
        conversation_id=CONVERSATION_ID,
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        start_gate_path=gate,
        start_gate_poll_seconds=7,
        sleep=open_gate,
        poll_interval_seconds=600,
        jitter_seconds=0,
    ).run(max_cycles=1)
    after = time.monotonic()

    seconds, lease = observed[0]
    assert seconds == 7
    assert lease.phase == "start-gate"
    assert before + 307 <= lease.deadline <= after + 307
    assert mailbox.calls == 1
    assert WorkerLease.read(lease_path).phase == "poll"


def test_controller_requires_start_and_launch_gates_before_polling(
    tmp_path: Path,
):
    start_gate = tmp_path / "start-gate"
    launch_gate = tmp_path / "launch-gate"
    lease_path = tmp_path / "controller-lease.json"
    mailbox = Mailbox([()])
    sleeps = []
    lease_phases = []

    def open_next_gate(seconds):
        sleeps.append(seconds)
        lease_phases.append(WorkerLease.read(lease_path).phase)
        gate = start_gate if not start_gate.is_file() else launch_gate
        gate.write_text("open")

    controller = Controller(
        role="advisor",
        mailbox=mailbox,
        turns=Turns(),
        conversation_id=UUID("00000000-0000-0000-0000-000000000087"),
        full_prompt="programme",
        progress=ProgressLease(lease_path),
        start_gate_path=start_gate,
        launch_gate_path=launch_gate,
        start_gate_poll_seconds=7,
        sleep=open_next_gate,
        poll_interval_seconds=600,
        jitter_seconds=0,
    )

    controller.run(max_cycles=1)

    assert sleeps == [7, 7]
    assert lease_phases == ["start-gate", "start-gate"]
    assert mailbox.calls == 1


def test_restart_continues_a_conversation_after_its_first_success(tmp_path: Path):
    conversation_id = UUID("00000000-0000-0000-0000-000000000004")
    state_path = tmp_path / "conversation-state.json"
    ConversationStateLedger(state_path).mark_success(conversation_id, "")
    turns = Turns()

    controller(
        Mailbox(
            [
                (
                    ControllerEvent(
                        kind="job_monitor",
                        dedupe_key="job:finished",
                        payload={"conversation_id": str(conversation_id)},
                    ),
                ),
                (),
            ]
        ),
        turns,
        role="student",
        conversation_id=conversation_id,
        conversation_state=ConversationStateLedger(state_path),
    ).run(max_cycles=1)

    assert turns.calls[0][1] == conversation_id
    assert "programme" not in turns.calls[0][0]


def test_workspace_lease_defers_only_checkout_events_in_a_mixed_batch():
    assignment = ControllerEvent(
        kind="student_assignment",
        dedupe_key="assignment:leased",
        payload={"assignment_id": "leased"},
    )
    monitor = ControllerEvent(
        kind="job_monitor",
        dedupe_key="job:regressed",
        payload={
            "conversation_id": str(CONVERSATION_ID),
            "summary": "metric regressed",
        },
    )
    mailbox = Mailbox([((assignment, monitor)), (assignment,)])
    turns = Turns()

    def reconcile(events):
        if any(event.kind == "student_assignment" for event in events):
            raise WorkspaceJobRunning(("job-17",))

    controller(
        mailbox,
        turns,
        role="student",
        reconcile=reconcile,
    ).run(max_cycles=1)

    assert len(turns.calls) == 1
    assert turns.calls[0][2] == frozenset({monitor.dedupe_key})
    assert "metric regressed" in turns.calls[0][0]
    assert mailbox.acknowledged == [(monitor.dedupe_key,)]


def test_turn_lease_uses_the_configured_hard_deadline(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"
    observed = []

    class LeaseReadingTurns(Turns):
        def run(self, *args, **kwargs):
            observed.append(WorkerLease.read(lease_path))
            return super().run(*args, **kwargs)

    before = time.monotonic()
    controller(
        Mailbox([(review_event(),)]),
        LeaseReadingTurns(),
        progress=ProgressLease(lease_path),
        operation_timeout_seconds=123,
        turn_timeout_seconds=456,
    ).run(max_cycles=1)
    after = time.monotonic()

    assert observed[0].phase == "openhands-turn"
    assert before + 456 <= observed[0].deadline <= after + 456


def test_only_an_acknowledged_turn_advances_supervisor_progress(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"
    event = review_event()
    mailbox = Mailbox([(event,), (event,), ()])

    controller(
        mailbox,
        Turns([TurnResult(exit_code=19), TurnResult(exit_code=0)]),
        progress=ProgressLease(lease_path),
    ).run(max_cycles=2)

    assert mailbox.acknowledged == [(event.dedupe_key,)]
    assert WorkerLease.read(lease_path).completed_turns == 1


def test_failed_acknowledgement_does_not_advance_supervisor_progress(tmp_path: Path):
    lease_path = tmp_path / "controller-lease.json"

    class FailingAcknowledgeMailbox(Mailbox):
        def acknowledge(self, _dedupe_keys):
            raise RuntimeError("durable acknowledgement failed")

    with pytest.raises(RuntimeError, match="durable acknowledgement failed"):
        controller(
            FailingAcknowledgeMailbox([(review_event(),)]),
            Turns(),
            progress=ProgressLease(lease_path),
        ).run(max_cycles=1)

    assert WorkerLease.read(lease_path).completed_turns == 0


def test_changed_system_context_is_injected_once_into_the_existing_conversation(
    tmp_path: Path,
):
    conversation_id = UUID("00000000-0000-0000-0000-000000000006")
    state = ConversationStateLedger(tmp_path / "conversation-state.json")
    state.mark_success(conversation_id, "old harness and role")
    turns = Turns()

    controller(
        Mailbox([(review_event(17),), (review_event(18),), ()]),
        turns,
        conversation_id=conversation_id,
        system_context="current harness and role",
        conversation_state=state,
    ).run(max_cycles=1)

    assert [call[1] for call in turns.calls] == [conversation_id, conversation_id]
    assert "# Updated Senpai system context" in turns.calls[0][0]
    assert "current harness and role" in turns.calls[0][0]
    assert "# Updated Senpai system context" not in turns.calls[1][0]
    assert state.is_context_current(conversation_id, "current harness and role")
