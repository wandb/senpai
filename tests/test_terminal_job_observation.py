import json
import subprocess
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

from senpai_agent.inbox import PersistentInbox
from senpai_agent.jobs import JobResult, JobState
from senpai_agent.monitor import (
    JobMonitorEngine,
    JobMonitorMailbox,
    JobMonitorSpec,
    JobMonitorStore,
    MonitorSignal,
)
from senpai_agent.tools import (
    CancelJobAction,
    CancelJobTool,
    JobSpec,
    RunJobAction,
    RunJobTool,
)


class FailedCancellationJob:
    def __init__(self, workspace: Path, result: JobResult):
        self.workspace = workspace
        self.result = result
        self.status_checks: list[str] = []
        self.cancelled: list[str] = []

    def run_job(self, _spec: JobSpec, **_options: object) -> JobResult:
        return self.result.model_copy(
            update={"state": JobState.RUNNING, "exit_code": None}
        )

    def get_job_status(self, job_id: str) -> JobResult:
        self.status_checks.append(job_id)
        return self.result

    def cancel_job(self, job_id: str) -> JobResult:
        self.cancelled.append(job_id)
        return self.result


def test_cancel_job_atomically_records_one_terminal_observation_during_active_turn(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    conversation_id = uuid.uuid4()
    result = JobResult(
        job_id="watcher-17",
        state=JobState.FAILED,
        exit_code=2,
        elapsed_seconds=1_740.936,
        log_path=str(tmp_path / "watcher.log"),
    )
    jobs = FailedCancellationJob(workspace, result)

    with JobMonitorStore(tmp_path / "job-monitors.sqlite3") as monitors:
        RunJobTool.create(jobs, monitors)[0].executor(
            RunJobAction(
                spec=JobSpec(
                    argv=("python", "watch_receipt.py"),
                    cwd=workspace,
                    timeout_seconds=1_800,
                    workspace_access="read_only",
                )
            ),
            SimpleNamespace(id=conversation_id),
        )

        observation = CancelJobTool.create(jobs, monitors)[0].executor(
            CancelJobAction(job_id=result.job_id),
            SimpleNamespace(id=conversation_id),
        )

        assert observation.job_id == result.job_id
        assert observation.state is JobState.FAILED
        assert observation.exit_code == 2
        assert monitors.active() == []
        assert monitors.emitted(result.job_id) == frozenset(
            {f"{result.job_id}:status:failed"}
        )

        row = monitors.connection.execute(
            """
            SELECT dedupe_key, job_id, signal_json, handled
            FROM monitor_signals
            WHERE job_id = ?
            """,
            (result.job_id,),
        ).fetchone()
        assert row is not None
        signal = json.loads(row[2])
        assert row[0] == f"{result.job_id}:status:failed"
        assert row[1] == result.job_id
        assert row[3] == 0
        assert signal["kind"] == "job_status"
        assert signal["job_id"] == result.job_id
        assert signal["state"] == "failed"
        assert signal["detail"] == "Job reached terminal state failed with exit code 2."

        assert JobMonitorEngine(monitors, jobs, SimpleNamespace()).poll() == ()
        CancelJobTool.create(jobs, monitors)[0].executor(
            CancelJobAction(job_id=result.job_id),
            SimpleNamespace(id=conversation_id),
        )
        assert monitors.connection.execute(
            "SELECT COUNT(*) FROM monitor_signals WHERE job_id = ?",
            (result.job_id,),
        ).fetchone()[0] == 1

        class NoPoll:
            def poll(self):
                return ()

        mailbox = JobMonitorMailbox(NoPoll(), monitors)
        with PersistentInbox(tmp_path / "inbox.sqlite3") as inbox:
            inbox.enqueue(conversation_id, "advisor-feedback:19", "Review PR #19")
            active = inbox.next_turn(
                conversation_id,
                "Continue the active conversation",
            )
            assert active is not None

            event = mailbox.poll()[0]
            assert inbox.enqueue(
                conversation_id,
                event.dedupe_key,
                event.to_prompt(),
            )
            assert not inbox.enqueue(
                conversation_id,
                event.dedupe_key,
                event.to_prompt(),
            )
            assert inbox.active_turn(conversation_id) == active
            assert (
                inbox.next_turn(conversation_id, "Do not start another turn")
                == active
            )
            assert [queued.dedupe_key for queued in mailbox.poll()] == [
                event.dedupe_key
            ]


def test_terminal_collection_and_watcher_race_records_one_observation(
    tmp_path: Path,
):
    database = tmp_path / "job-monitors.sqlite3"
    conversation_id = uuid.uuid4()
    result = JobResult(
        job_id="watcher-race",
        state=JobState.FAILED,
        exit_code=2,
        elapsed_seconds=1,
        log_path=str(tmp_path / "watcher.log"),
    )
    with JobMonitorStore(database) as store:
        store.register(
            JobMonitorSpec(
                job_id=result.job_id,
                conversation_id=conversation_id,
            )
        )

    barrier = threading.Barrier(2)
    outcomes: list[MonitorSignal | None] = []
    errors: list[BaseException] = []

    def watcher() -> None:
        try:
            with JobMonitorStore(database) as store:
                spec = store.due()[0]
                barrier.wait()
                outcomes.append(
                    store.record_terminal_and_complete(result, spec=spec)
                )
        except BaseException as error:  # assertion captures worker failure
            errors.append(error)

    def active_turn_collection() -> None:
        try:
            with JobMonitorStore(database) as store:
                barrier.wait()
                outcomes.append(store.record_terminal_and_complete(result))
        except BaseException as error:  # assertion captures worker failure
            errors.append(error)

    threads = [
        threading.Thread(target=watcher),
        threading.Thread(target=active_turn_collection),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sum(signal is not None for signal in outcomes) == 1
    with JobMonitorStore(database) as store:
        assert store.active() == []
        assert [signal.dedupe_key for signal in store.pending_signals()] == [
            "watcher-race:status:failed"
        ]
