import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from pydantic import SecretStr
from training_test_support import (
    assert_process_stopped,
    make_supervisor,
    run_python,
    wait_for_path,
    wait_for_terminal,
)

from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
    TrainingSupervisor,
)

TERM_IGNORING_SLEEP = (
    "import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(60)"
)


def test_training_preserves_target_python_and_project_imports(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    workspace, supervisor = make_supervisor(tmp_path)
    target_env = tmp_path / "target-env"
    subprocess.run(
        [sys.executable, "-P", "-m", "venv", "--without-pip", str(target_env)],
        check=True,
    )
    monkeypatch.setenv("SENPAI_TARGET_PYTHON_ENV", str(target_env))
    (workspace / "project_module.py").write_text("VALUE = 'project import'\n")
    output = workspace / "environment.json"
    script = workspace / "train.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from project_module import VALUE\n"
        f"Path({str(output)!r}).write_text(json.dumps({{"
        "'value': VALUE, 'prefix': sys.prefix, 'safe_path': sys.flags.safe_path, "
        "'uv_python': os.environ['UV_PYTHON'], "
        "'uv_project_environment': os.environ['UV_PROJECT_ENVIRONMENT'], "
        "'virtual_env': os.environ['VIRTUAL_ENV']}))\n"
    )

    running = supervisor.run_training(
        TrainingSpec(
            argv=["python", str(script)], cwd=str(workspace), timeout_seconds=30,
        )
    )
    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FINISHED
    assert json.loads(output.read_text()) == {
        "value": "project import",
        "prefix": str(target_env),
        "safe_path": False,
        "uv_python": str(target_env / "bin" / "python"),
        "uv_project_environment": str(target_env),
        "virtual_env": str(target_env),
    }


def test_training_timeout_honors_the_requested_deadline(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.4,
    )
    started = time.monotonic()
    running = run_python(
        supervisor,
        workspace,
        TERM_IGNORING_SLEEP,
        timeout_seconds=1,
    )
    launch_elapsed = time.monotonic() - started

    terminal = wait_for_terminal(supervisor, running.training_id)

    assert launch_elapsed < 0.5
    assert terminal.state is TrainingState.TIMED_OUT
    assert terminal.elapsed_seconds < 1.25
    assert time.monotonic() - started < 1.25


def test_timeout_stops_term_ignoring_descendants(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    child_pid_path = workspace / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{TERM_IGNORING_SLEEP!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    running = run_python(
        supervisor,
        workspace,
        parent_code,
        timeout_seconds=1,
    )

    wait_for_path(child_pid_path)
    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.TIMED_OUT
    assert_process_stopped(int(child_pid_path.read_text()))


def test_finished_training_stops_leftover_descendants(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=0.1,
    )
    child_pid_path = workspace / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys;"
        f"p=subprocess.Popen([sys.executable,'-c',{TERM_IGNORING_SLEEP!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))"
    )
    running = run_python(supervisor, workspace, parent_code)

    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FINISHED
    assert_process_stopped(int(child_pid_path.read_text()))


def test_close_cannot_extend_the_training_deadline(tmp_path: Path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        terminate_grace_seconds=1.5,
    )
    ready = workspace / "ready"
    started = time.monotonic()
    running = run_python(
        supervisor,
        workspace,
        (
            "import pathlib,signal,time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            f"pathlib.Path({str(ready)!r}).write_text('ready');"
            "time.sleep(60)"
        ),
        timeout_seconds=2,
    )
    wait_for_path(ready)
    while time.monotonic() - started < 0.7:
        time.sleep(0.01)

    supervisor.close()
    result = supervisor.get_training_status(running.training_id)

    assert result.state in {TrainingState.CANCELLED, TrainingState.TIMED_OUT}
    assert result.elapsed_seconds < 2.1
    assert time.monotonic() - started < 2.1


def test_restart_stops_a_verified_orphaned_process_group(tmp_path: Path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    state_dir.mkdir()
    child_pid_path = workspace / "orphan-child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{TERM_IGNORING_SLEEP!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    process = psutil.Popen(
        [sys.executable, "-c", parent_code],
        start_new_session=True,
    )
    wait_for_path(child_pid_path)
    orphan = TrainingResult(
        training_id="d7d0d19f-9961-4dac-b2ff-7382dc463674",
        state=TrainingState.RUNNING,
        pid=process.pid,
        process_group_id=process.pid,
        process_start_time=process.create_time(),
        exit_code=None,
        elapsed_seconds=12,
        log_path=str(state_dir / "orphan.log"),
    )
    (state_dir / f"{orphan.training_id}.json").write_text(orphan.model_dump_json())
    sidecar = state_dir / f"{orphan.training_id}.score.json"
    sidecar.write_text('{"metrics": {}, "passed": true, "score": 1.0}')

    try:
        supervisor = TrainingSupervisor(
            workspace=workspace,
            state_dir=state_dir,
            terminate_grace_seconds=0.1,
        )

        recovered = supervisor.get_training_status(orphan.training_id)
        assert recovered.state is TrainingState.CANCELLED
        assert "supervisor restarted" in recovered.error_tail
        assert sidecar.exists()
        assert process.wait(timeout=3) is not None
        assert_process_stopped(int(child_pid_path.read_text()))
    finally:
        if process.is_running():
            process.kill()
            process.wait()


def test_restart_does_not_signal_a_reused_pid(tmp_path: Path):
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    state_dir.mkdir()
    unrelated = psutil.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    orphan = TrainingResult(
        training_id="26a7194a-3bea-45b1-a2e5-cd20d99e3a31",
        state=TrainingState.RUNNING,
        pid=unrelated.pid,
        process_group_id=unrelated.pid,
        process_start_time=unrelated.create_time() - 10,
        exit_code=None,
        elapsed_seconds=12,
        log_path=str(state_dir / "orphan.log"),
    )
    (state_dir / f"{orphan.training_id}.json").write_text(orphan.model_dump_json())

    try:
        supervisor = TrainingSupervisor(workspace=workspace, state_dir=state_dir)

        recovered = supervisor.get_training_status(orphan.training_id)
        assert recovered.state is TrainingState.CANCELLED
        assert "no signal was sent" in recovered.error_tail
        assert unrelated.is_running()
    finally:
        unrelated.send_signal(signal.SIGKILL)
        unrelated.wait()


def test_training_masks_split_writer_output_before_log_and_result_persistence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WANDB_API_KEY", "research-secret")
    monkeypatch.setenv("WANDB_INFERENCE_API_KEY", "inference-secret")
    monkeypatch.setenv("SENPAI_WANDB_TRAINING_API_KEY", "stale-writer-secret")
    monkeypatch.setenv("WANDB_SERVICE", "parent-service")
    monkeypatch.setenv("WANDB_IDENTITY_TOKEN_FILE", "/parent/identity-token")
    workspace, supervisor = make_supervisor(
        tmp_path,
        wandb_api_key=SecretStr("student-writer-secret"),
        terminate_grace_seconds=0.1,
    )
    environment_path = workspace / "environment.json"
    prefix_written = workspace / "prefix-written"
    continue_output = workspace / "continue-output"
    running = run_python(
        supervisor,
        workspace,
        "import json,os,pathlib,sys,time;"
        "values={key: os.environ.get(key) for key in "
        "['WANDB_API_KEY','WANDB_INFERENCE_API_KEY','SENPAI_WANDB_TRAINING_API_KEY',"
        "'WANDB_SERVICE','WANDB_IDENTITY_TOKEN_FILE']};"
        f"pathlib.Path({str(environment_path)!r}).write_text(json.dumps(values));"
        "key=os.environ['WANDB_API_KEY'].encode();"
        "os.write(1,b'failed with key='+key[:7]);"
        f"pathlib.Path({str(prefix_written)!r}).touch();\n"
        f"while not pathlib.Path({str(continue_output)!r}).exists(): time.sleep(.01)\n"
        "os.write(2,key[7:]+b'\\n');"
        "os.write(1,b'https://wandb.ai/team/project/runs/writer-run\\n');"
        "sys.exit(1)",
    )
    log = Path(running.log_path)
    try:
        wait_for_path(prefix_written)
        deadline = time.monotonic() + 3
        while log.read_bytes() != b"failed with key=" and time.monotonic() < deadline:
            time.sleep(0.02)
        # The reader has consumed the first write, but keeps the possible key
        # prefix in memory until the next write completes or disproves it.
        assert log.read_bytes() == b"failed with key="
        continue_output.touch()
        result = wait_for_terminal(supervisor, running.training_id)
    finally:
        supervisor.close()

    assert result.state is TrainingState.FAILED
    assert json.loads(environment_path.read_text()) == {
        "WANDB_API_KEY": "student-writer-secret",
        "WANDB_INFERENCE_API_KEY": None,
        "SENPAI_WANDB_TRAINING_API_KEY": None,
        "WANDB_SERVICE": None,
        "WANDB_IDENTITY_TOKEN_FILE": None,
    }
    expected = (
        "failed with key=<secret-hidden>\n"
        "https://wandb.ai/team/project/runs/writer-run\n"
    )
    assert log.read_text() == expected
    assert result.error_tail == expected
    assert result.wandb_run_ids == ("writer-run",)
    persisted = tmp_path / "state" / f"{running.training_id}.json"
    assert "student-writer-secret" not in persisted.read_text()


def test_training_masks_a_partial_writer_key_at_natural_eof(tmp_path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        wandb_api_key=SecretStr("0123456789abcdef0123456789abcdef01234567"),
        terminate_grace_seconds=0.1,
    )
    running = run_python(
        supervisor,
        workspace,
        "import os,sys;"
        "os.write(1,b'partial writer='+os.environ['WANDB_API_KEY'].encode()[:24]);"
        "sys.exit(1)",
    )
    try:
        result = wait_for_terminal(supervisor, running.training_id)
    finally:
        supervisor.close()

    expected = "partial writer=<secret-hidden>"
    assert result.state is TrainingState.FAILED
    assert Path(running.log_path).read_text() == expected
    assert result.error_tail == expected
    persisted = tmp_path / "state" / f"{running.training_id}.json"
    assert json.loads(persisted.read_text())["error_tail"] == expected


def test_cancel_does_not_wait_for_a_detached_output_writer(tmp_path):
    workspace, supervisor = make_supervisor(
        tmp_path,
        wandb_api_key=SecretStr("student-writer-secret"),
        terminate_grace_seconds=0.1,
    )
    child_pid = workspace / "detached.pid"
    writer = (
        "import os,threading\n"
        "def emit():\n"
        " while True: os.write(1,os.environ['WANDB_API_KEY'].encode()*256)\n"
        "for _ in range(4): threading.Thread(target=emit,daemon=True).start()\n"
        "emit()\n"
    )
    running = run_python(
        supervisor,
        workspace,
        "import pathlib,subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable,'-c',{writer!r}],start_new_session=True);"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid));"
        "time.sleep(60)",
        timeout_seconds=30,
    )
    cancellation = threading.Thread(
        target=supervisor.cancel_training, args=(running.training_id,)
    )
    try:
        wait_for_path(child_pid)
        log = Path(running.log_path)
        deadline = time.monotonic() + 3
        while log.stat().st_size < 64 * 1024 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert log.stat().st_size >= 64 * 1024
        cancellation.start()
        cancellation.join(timeout=3)
        finished_on_time = not cancellation.is_alive()
    finally:
        if child_pid.exists():
            try:
                os.killpg(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        supervisor.close()
        if cancellation.ident is not None:
            cancellation.join()

    assert finished_on_time
    assert (
        supervisor.get_training_status(running.training_id).state
        is TrainingState.CANCELLED
    )
    output = Path(running.log_path).read_bytes()
    assert b"student-writer-secret" not in output
    assert output.endswith(b"<secret-hidden>")


def test_output_capture_failure_cannot_report_success(tmp_path, monkeypatch):
    workspace, supervisor = make_supervisor(tmp_path, terminate_grace_seconds=0.1)
    original_open = Path.open

    def fail_log_write(path, mode="r", *args, **kwargs):
        if path.suffix == ".log" and mode == "wb":
            raise OSError("fixture disk full")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_log_write)
    running = run_python(supervisor, workspace, "import time;time.sleep(.05)")
    try:
        result = wait_for_terminal(supervisor, running.training_id)
    finally:
        supervisor.close()

    assert result.state is TrainingState.FAILED
    assert result.error_tail == "Training output capture failed (OSError)."
