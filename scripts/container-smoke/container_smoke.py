"""Offline Docker smoke helpers; all credentials and target repositories are synthetic."""
from __future__ import annotations
import base64
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

STATE = Path('/state')
WORKSPACE = Path('/workspace/senpai')
TARGET = WORKSPACE / 'target'
PYTHON = '/opt/senpai-venv/bin/python'


def git(*args: str, cwd: Path = TARGET) -> str:
    return subprocess.check_output(['/usr/bin/git', '-C', str(cwd), *args], text=True).strip()


def seed() -> None:
    import tarfile
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with tarfile.open('/smoke/source.tar') as archive:
        archive.extractall(WORKSPACE, filter='data')
    for repo, branch in ((WORKSPACE, 'runner'), (TARGET, 'advisor')):
        repo.mkdir(parents=True, exist_ok=True)
        git('init', '-q', '-b', branch, cwd=repo)
        git('config', 'user.name', 'Synthetic Smoke', cwd=repo)
        git('config', 'user.email', 'smoke@example.invalid', cwd=repo)
        if repo == TARGET:
            (repo / 'program.md').write_text('Synthetic offline smoke research policy.\n')
            (repo / 'research.py').write_text('VALUE = 1\n')
        git('add', '.', cwd=repo)
        git('commit', '-qm', 'synthetic smoke fixture', cwd=repo)
        git('remote', 'add', 'origin', 'https://github.com/smoke/target.git', cwd=repo)
    from senpai_agent.program_context import load_program_system_prompt, encode_program_system_prompt
    program = load_program_system_prompt(TARGET, 'program.md')
    (STATE / 'program.b64').write_text(encode_program_system_prompt(program))
    (STATE / 'program-commit').write_text(program.source_commit)
    (STATE / 'program-digest').write_text(program.content_sha256)
    from senpai_agent import launch_context
    launch_context.LAUNCH_CONTEXT_TEMPLATE = WORKSPACE / 'system_instructions/SENPAI-LAUNCH-CONTEXT.md'
    launch = launch_context.render_launch_context(role='advisor', github_repo='smoke/target',
        wandb_entity='synthetic', wandb_project='offline', backend='docker-smoke',
        nodes_per_student=1, gpus_per_student_node=0, timeout_minutes=1, max_epochs=1, tag='offline-smoke',
        advisor_branch='advisor', target_base='advisor', students=['synthetic-student'])
    (STATE / 'launch.b64').write_text(base64.b64encode(launch.encode()).decode())
    git('switch', '-qc', 'student/in-progress')
    (TARGET / 'research.py').write_text('VALUE = 2  # intentionally uncommitted\n')
    (TARGET / 'keep-untracked.txt').write_text('Retain this research work.\n')
    (WORKSPACE / 'sitecustomize.py').write_text("raise RuntimeError('workspace startup code executed')\n")
    (STATE / 'checkout.json').write_text(json.dumps({
        'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
        'research': (TARGET / 'research.py').read_text(),
        'untracked': (TARGET / 'keep-untracked.txt').read_text(),
    }))
    for root in (Path('/state'), Path('/workspace'), Path('/home/senpai')):
        for path in [root, *root.rglob('*')]:
            os.chown(path, 10001, 10001, follow_symlinks=False)
    print('synthetic fixtures ready')


def poison_target() -> None:
    python = Path('/home/senpai/.venvs/senpai-target/bin/python')
    python.unlink()
    python.write_text('#!/bin/sh\ntouch /state/target-python-executed\nexit 88\n')
    python.chmod(0o755)
    (TARGET / 'program.md').write_text('Mutated workspace policy must not replace the launch snapshot.\n')
    print('target interpreter tripwire installed and workspace policy mutated')


def worker() -> None:
    from senpai_agent.secrets import GITHUB_TOKEN_FD_ENV, PRIVATE_CREDENTIAL_FD_ENVS, set_process_nondumpable
    from senpai_agent.supervisor import ProgressLease, LEASE_ENV
    set_process_nondumpable()
    assert ctypes.CDLL(None).prctl(3, 0, 0, 0, 0) == 0, "process remained dumpable"
    for name, expected in ((GITHUB_TOKEN_FD_ENV, 'synthetic-github'),
                           (PRIVATE_CREDENTIAL_FD_ENVS['EXA_API_KEY'], 'synthetic-exa'),
                           (PRIVATE_CREDENTIAL_FD_ENVS['WANDB_API_KEY'], 'synthetic-wandb')):
        descriptor = int(os.environ.pop(name))
        with os.fdopen(descriptor, 'r') as stream:
            assert stream.read() == expected, name
        try:
            os.fstat(descriptor)
        except OSError:
            pass
        else:
            raise AssertionError('credential descriptor remained open')
    assert not {'GITHUB_TOKEN', 'GH_TOKEN', 'WANDB_API_KEY', 'EXA_API_KEY'} & os.environ.keys()
    from senpai_agent.system_instructions import decode_system_instructions, SYSTEM_INSTRUCTIONS_FILE_ENV, SYSTEM_INSTRUCTIONS_SHA256_ENV
    instructions = decode_system_instructions(Path(os.environ[SYSTEM_INSTRUCTIONS_FILE_ENV]).read_text().strip(), os.environ[SYSTEM_INSTRUCTIONS_SHA256_ENV])
    assert instructions.program.content.strip() == 'Synthetic offline smoke research policy.'
    direct = subprocess.Popen([PYTHON, '-c', 'import time; time.sleep(300)'])
    child = os.fork()
    if child == 0:
        os.setsid()
        grandchild = os.fork()
        if grandchild != 0:
            os._exit(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (STATE / 'detached-pid').write_text(str(os.getpid()))
        while True:
            time.sleep(1)
    os.waitpid(child, 0)
    (STATE / 'direct-pid').write_text(str(direct.pid))
    ProgressLease(Path(os.environ[LEASE_ENV])).update('synthetic-worker', 20, completed_turn=True)
    deadline = time.monotonic() + 15
    while not (STATE / 'finish').exists():
        assert time.monotonic() < deadline, 'health observer did not finish'
        time.sleep(.05)
    raise SystemExit(23)


def status(url: str) -> int:
    try:
        result = urlopen(url, timeout=1)
    except HTTPError as error:
        result = error
    with result:
        return result.status


def pid1() -> None:
    assert os.getpid() == 1, 'this test requires a real container PID 1'
    import psutil
    import senpai_agent
    from senpai_agent.secrets import set_process_nondumpable
    from senpai_agent.supervisor import (
        WorkerSupervisor, SupervisorConfig, serve_lease_health,
        _consume_github_token, _consume_private_credential_files, prepare_system_context_environment,
    )
    set_process_nondumpable()
    assert ctypes.CDLL(None).prctl(3, 0, 0, 0, 0) == 0, "process remained dumpable"
    import senpai_agent.kubernetes_executor
    assert os.environ["SENPAI_IMAGE_REVISION"] == os.environ["SENPAI_REPO_REVISION"]
    worker_environment = prepare_system_context_environment("advisor", STATE / "system-context", os.environ)
    digest = worker_environment["SENPAI_SYSTEM_INSTRUCTIONS_SHA256"]
    persisted_digest = STATE / "snapshot-digest"
    if persisted_digest.exists():
        assert persisted_digest.read_text() == digest, "restart changed the immutable full prompt"
    else:
        persisted_digest.write_text(digest)
    assert Path(senpai_agent.__file__).is_relative_to('/opt/senpai-venv'), senpai_agent.__file__
    for path in ('/opt/senpai-venv', '/opt/senpai-agent-definitions', '/opt/senpai-plugin'):
        assert not os.access(path, os.W_OK), path
    expected = json.loads((STATE / 'checkout.json').read_text())
    assert expected == {
        'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
        'research': (TARGET / 'research.py').read_text(),
        'untracked': (TARGET / 'keep-untracked.txt').read_text(),
    }, 'bootstrap changed retained research work'
    assert not (STATE / 'target-python-executed').exists(), 'bootstrap executed target Python'
    starts = int((STATE / 'starts').read_text()) + 1 if (STATE / 'starts').exists() else 1
    (STATE / 'starts').write_text(str(starts))
    target_bin = Path(os.environ['SENPAI_TARGET_PYTHON_ENV']) / 'bin'
    sentinel = target_bin / 'custom-research-command'
    if starts == 1:
        sentinel.write_text('#!/bin/sh\nprintf retained-command\\n\n')
        sentinel.chmod(0o755)
        from openhands.tools.terminal import TerminalAction
        from senpai_agent.tools import SenpaiTerminalTool
        (TARGET / 'project_module.py').write_text("VALUE = 'local-project-ok'\n")
        (TARGET / 'check_terminal.py').write_text(
            "import sys\nfrom project_module import VALUE\n"
            "assert sys.prefix == '/home/senpai/.venvs/senpai-target'\nprint(VALUE)\n")
        state = SimpleNamespace(workspace=SimpleNamespace(working_dir=str(TARGET)), env_observation_persistence_dir=None)
        executor = SenpaiTerminalTool.create(state, role='advisor')[0].executor
        try:
            assert executor.is_pooled, 'Linux tmux backend was not selected'
            observed = executor(TerminalAction(command='python check_terminal.py', timeout=20))
            assert observed.exit_code == 0 and 'local-project-ok' in observed.text, observed.text
        finally:
            executor.close()
    else:
        assert sentinel.read_text() == '#!/bin/sh\nprintf retained-command\\n\n', 'bootstrap replaced a custom target command'
    github = _consume_github_token(os.environ)
    private = _consume_private_credential_files(os.environ)
    for name in ('SENPAI_GITHUB_TOKEN_FILE', 'SENPAI_WANDB_API_KEY_FILE', 'SENPAI_EXA_API_KEY_FILE'):
        assert not Path(os.environ[name]).exists(), name
    for name in ('finish', 'direct-pid', 'detached-pid'):
        (STATE / name).unlink(missing_ok=True)
    lease = STATE / 'lease.json'
    lease.unlink(missing_ok=True)
    errors = []
    with serve_lease_health(lease, host='127.0.0.1', port=0) as server:
        url = f'http://127.0.0.1:{server.server_port}/healthz'
        assert status(url) == 503
        def observe():
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if status(url) == 200 and (STATE / 'detached-pid').exists():
                        (STATE / 'finish').write_text('health was observed')
                        return
                    time.sleep(.05)
                raise AssertionError('worker never became healthy')
            except BaseException as error:
                errors.append(error)
        monitor = threading.Thread(target=observe)
        monitor.start()
        supervisor = WorkerSupervisor(command=(PYTHON, '-P', __file__, 'worker'), lease_path=lease,
            config=SupervisorConfig(startup_timeout_seconds=10, check_interval_seconds=.05, terminate_grace_seconds=2),
            environment=worker_environment, github_token=github, private_credentials=private)
        github = None
        private.clear()
        worker_environment.clear()
        result = supervisor.run()
        monitor.join(timeout=16)
        assert not monitor.is_alive() and not errors, errors
        assert result == 23, result
        assert not supervisor.environment and supervisor.github_token is None and not supervisor.private_credentials
        assert status(url) == 503
        for name in ('direct-pid', 'detached-pid'):
            assert not psutil.pid_exists(int((STATE / name).read_text())), f'{name} survived cleanup'
    print(json.dumps({'status':'ok','container_start':starts,'health':'503 -> 200 -> 503',
        'credential_fds':'consumed and closed','cleanup':'direct and detached children stopped',
        'checkout':'branch, HEAD and dirty work preserved','program_snapshot':'fixed across workspace mutation and restart','target_python':'not executed during bootstrap',
        'tmux':'passed' if starts == 1 else 'first start only'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    {'seed': seed, 'poison': poison_target, 'worker': worker, 'pid1': pid1}[sys.argv[1]]()
