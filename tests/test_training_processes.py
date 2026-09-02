import json
import signal
import sys
import time
from pathlib import Path

import psutil
from pydantic import SecretStr

from senpai_agent.training import (
    TrainingResult,
    TrainingSpec,
    TrainingState,
    TrainingSupervisor,
)
from training_test_support import (
    assert_process_stopped,
    make_supervisor,
    run_python,
    wait_for_path,
    wait_for_terminal,
)


TERM_IGNORING_SLEEP = (
    "import signal,time;"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
    "time.sleep(60)"
)


def test_training_receives_only_its_per_student_wandb_key(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("WANDB_API_KEY", "controller-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("GITHUB_TOKEN", "github-key")
    workspace, supervisor = make_supervisor(
        tmp_path,
        wandb_api_key=SecretStr("student-training-key"),
    )
    output = workspace / "environment.json"
    code = (
        "import json,os,pathlib;"
        "keys=('WANDB_API_KEY','EXA_API_KEY','GITHUB_TOKEN');"
        f"pathlib.Path({str(output)!r}).write_text(json.dumps(dict("
        "(name,os.environ.get(name)) for name in keys)))"
    )

    result = run_python(supervisor, workspace, code)
    terminal = wait_for_terminal(supervisor, result.training_id)
    assert terminal.state is TrainingState.FINISHED
    assert json.loads(output.read_text()) == {
        "WANDB_API_KEY": "student-training-key",
        "EXA_API_KEY": None,
        "GITHUB_TOKEN": None,
    }


def test_training_preserves_normal_project_python_imports(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    workspace, supervisor = make_supervisor(tmp_path)
    target_env = tmp_path / "target-env"
    (target_env / "bin").mkdir(parents=True)
    monkeypatch.setenv("SENPAI_TARGET_PYTHON_ENV", str(target_env))
    (workspace / "project_module.py").write_text("VALUE = 'project-import'\n")
    output = workspace / "import-result.txt"
    script = workspace / "train.py"
    script.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from project_module import VALUE\n"
        f"Path({str(output)!r}).write_text(json.dumps({{"
        "'value': VALUE, 'uv_python': os.environ['UV_PYTHON'], "
        "'uv_project_environment': os.environ['UV_PROJECT_ENVIRONMENT'], "
        "'virtual_env': os.environ['VIRTUAL_ENV']}))\n"
    )

    running = supervisor.run_training(
        TrainingSpec(
            argv=[sys.executable, str(script)],
            cwd=str(workspace),
            timeout_seconds=30,
        )
    )
    terminal = wait_for_terminal(supervisor, running.training_id)

    assert terminal.state is TrainingState.FINISHED
    assert json.loads(output.read_text()) == {
        "value": "project-import",
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
