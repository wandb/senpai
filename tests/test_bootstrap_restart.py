"""Exercise same-pod bootstrap against retained emptyDir contents."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from git_workflow_support import commit_file, git, repository
from launch_test_support import launch_args, render_role
from senpai_agent.program_context import (
    encode_program_system_prompt,
    load_program_system_prompt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bootstrap_runtime(tmp_path, role):
    container = tmp_path / "container"
    home = container / "home/senpai"
    runner = container / "workspace/senpai"
    target = runner / "target"
    target_env = home / ".venvs/senpai-target"
    record = tmp_path / "starts.jsonl"
    recorder = tmp_path / "record_startup.py"
    recorder.write_text('''
import json, os, subprocess
from pathlib import Path
from senpai_agent.supervisor import _consume_github_token, _consume_private_credential_files
names = ("SENPAI_GITHUB_TOKEN_FILE", "SENPAI_WANDB_API_KEY_FILE", "SENPAI_EXA_API_KEY_FILE")
paths = [Path(os.environ[name]) for name in names]
assert all(name not in os.environ for name in ("GITHUB_TOKEN", "GH_TOKEN", "WANDB_API_KEY", "EXA_API_KEY"))
assert len({path.parent for path in paths}) == 1
assert paths[0].parent.stat().st_mode & 0o777 == 0o700
assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
assert _consume_github_token(os.environ).get_secret_value() == "github-fixture"
services = _consume_private_credential_files(os.environ)
assert {name: value.get_secret_value() for name, value in services.items()} == {"WANDB_API_KEY": "wandb-fixture", "EXA_API_KEY": "exa-fixture"}
assert all(not path.exists() for path in paths)
safe_directories = subprocess.check_output(["git", "config", "--global", "--get-all", "safe.directory"], text=True).splitlines()
with Path(os.environ["START_RECORD"]).open("a") as output:
    output.write(json.dumps({"handoff_dir": str(paths[0].parent), "safe_directories": safe_directories}) + "\\n")
''')
    tools = tmp_path / "tools"
    tools.mkdir()
    for name, body in (
        ("gh", 'test "$1 $2" = "repo set-default"\n'),
        ("nvidia-smi", "printf 'fixture GPU\\n'\n"),
    ):
        executable = tools / name
        executable.write_text("#!/bin/sh\n" + body)
        executable.chmod(0o755)

    def container_paths(script):
        for path in ("/workspace", "/var/lib/senpai", "/tmp/senpai"):
            script = script.replace(path, str(container / path.lstrip("/")))
        uv = shutil.which("uv")
        assert uv is not None, "the bootstrap contract requires uv"
        return script.replace("/usr/local/bin/uv", shlex.quote(uv))

    source_root = tmp_path / "runner-source"
    source_root.mkdir()
    source, _remote, _revision = repository(source_root)
    (source / "k8s").mkdir()
    entrypoint = container_paths((ROOT / "k8s" / f"entrypoint-{role}.sh").read_text())
    entrypoint = entrypoint.replace(
        f'exec "$SENPAI_PYTHON" -P -m senpai_agent.supervisor {role}',
        'exec "$SENPAI_PYTHON" -P "$STARTUP_RECORDER"',
    )
    revision = commit_file(source, f"k8s/entrypoint-{role}.sh", entrypoint, "entrypoint")
    target_source_root = tmp_path / "target-source"
    target_source_root.mkdir()
    target_source, target_remote, _target_revision = repository(target_source_root)
    target_baseline = commit_file(
        target_source, "program.md", "Preserve research work across restarts.\n", "program"
    )
    git(target_source, "push", "origin", "experiment-7")
    if role == "student":
        git(target_source, "push", "origin", "HEAD:refs/heads/research")
    program = load_program_system_prompt(target_source, "program.md", target_baseline)
    program_context = tmp_path / "program-context.b64"
    program_context.write_text(encode_program_system_prompt(program))

    configmap, deployment, _secret = render_role(
        role, launch_args(problem_dir="target/", advisor_branch="research"),
        program=program,
    )
    pod = yaml.safe_load(deployment)["spec"]["template"]["spec"]
    app = pod["containers"][0]
    retained_names = {volume["name"] for volume in pod["volumes"] if "emptyDir" in volume}
    retained_paths = {
        mount["name"]: container / mount["mountPath"].lstrip("/")
        for mount in app["volumeMounts"] if mount["name"] in retained_names
    }
    bootstrap = container_paths(app["args"][0])
    environment = {
        **os.environ,
        **yaml.safe_load(configmap)["data"],
        "HOME": str(home),
        "PATH": f"{tools}:{os.environ['PATH']}",
        "PYTHONPATH": str(ROOT),
        "SENPAI_PYTHON": sys.executable,
        "SENPAI_PLUGIN": str(ROOT / "plugins/senpai"),
        "SENPAI_AGENT_DIR": str(ROOT / ".agents/agents"),
        "SENPAI_PROGRAM_CONTEXT_FILE": str(program_context),
        "SENPAI_REPO_URL": str(source),
        "SENPAI_REPO_REVISION": revision,
        "SENPAI_IMAGE_REVISION": revision,
        "TARGET_REPO_URL": str(target_remote),
        "TARGET_REPO_BRANCH": "experiment-7",
        "PROBLEM_DIR": "target",
        "GH_HISTORY_SCOPE": "branch",
        "ADVISOR_BRANCH": "research",
        "GH_REPO": "acme/target",
        "RESEARCH_TAG": "restart",
        "STUDENT_NAME": "fern",
        "STUDENT_NAMES": "fern",
        "GITHUB_TOKEN": "github-fixture",
        "GH_TOKEN": "github-alias-fixture",
        "WANDB_API_KEY": "wandb-fixture",
        "EXA_API_KEY": "exa-fixture",
        "START_RECORD": str(record),
        "STARTUP_RECORDER": str(recorder),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }

    def start():
        home.mkdir(parents=True, exist_ok=True)
        (container / "tmp").mkdir(exist_ok=True)
        for path in retained_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            ["bash", "-c", bootstrap], env=environment,
            capture_output=True, text=True, timeout=30,
        )

    return SimpleNamespace(
        start=start, container=container, runner=runner, target=target,
        target_env=target_env, record=record, environment=environment,
        retained_paths=retained_paths, target_baseline=target_baseline,
    )


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_fresh_container_preserves_target_work_and_dependencies(tmp_path, bootstrap_runtime):
    runtime = bootstrap_runtime
    container, runner, target = runtime.container, runtime.runner, runtime.target
    target_env, record = runtime.target_env, runtime.record
    environment, retained_paths = runtime.environment, runtime.retained_paths
    target_baseline = runtime.target_baseline

    def start():
        result = runtime.start()
        assert result.returncode == 0, result.stdout + result.stderr

    # A mounted emptyDir cannot be removed and recreated during clone fallback.
    target.mkdir(parents=True)
    mounted_directory = os.open(target, os.O_RDONLY)
    try:
        start()
        assert target.stat().st_ino == os.fstat(mounted_directory).st_ino
    finally:
        os.close(mounted_directory)
    git(target, "checkout", "-b", "fern/unpublished")
    unpublished = commit_file(target, "unpublished.txt", "local experiment\n", "local")
    (target / "model.py").write_text("dirty model\n")
    (target / "untracked.txt").write_text("untracked experiment\n")
    status = git(target, "status", "--short")
    assert git(target, "rev-parse", "origin/research") == target_baseline
    wheel = tmp_path / "restart_dependency-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("restart_dependency.py", "VALUE = 'installed before restart'\n")
        archive.writestr("restart_dependency-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: restart-dependency\nVersion: 1.0\n")
        archive.writestr("restart_dependency-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("restart_dependency-1.0.dist-info/RECORD", "")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(target_env / "bin/python"), "--no-deps", "--no-index", str(wheel)],
        env=environment, capture_output=True, text=True, check=True,
    )
    target_site = Path(sysconfig.get_path("purelib", vars={"base": str(target_env)}))
    exposure = tmp_path / "target-startup-executed"
    (target_site / "restart_probe.pth").write_text(
        f"import pathlib; pathlib.Path({str(exposure)!r}).touch()\n"
    )

    # Only declared emptyDir volumes survive the simulated container replacement.
    pod_storage = tmp_path / "pod-volumes"
    pod_storage.mkdir()
    for name, path in retained_paths.items():
        path.rename(pod_storage / name)
    shutil.rmtree(container)
    for name, path in retained_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        (pod_storage / name).rename(path)
    start()

    assert git(target, "rev-parse", "HEAD") == unpublished
    assert git(target, "branch", "--show-current") == "fern/unpublished"
    assert git(target, "status", "--short") == status
    assert (target / "model.py").read_text() == "dirty model\n"
    assert (target / "untracked.txt").read_text() == "untracked experiment\n"
    assert not exposure.exists(), "trusted restart executed target-controlled startup"
    installed = subprocess.check_output(
        [str(target_env / "bin/python"), "-P", "-c", "import restart_dependency; print(restart_dependency.VALUE)"],
        env=environment, text=True,
    )
    assert installed.strip() == "installed before restart"
    starts = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(starts) == 2
    assert starts[0]["handoff_dir"] != starts[1]["handoff_dir"]
    assert {str(runner.resolve()), str(target.resolve())} <= set(starts[1]["safe_directories"])


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_incomplete_checkout_fails_without_changing_retained_work(bootstrap_runtime):
    runtime = bootstrap_runtime
    target = runtime.target
    target.mkdir(parents=True)
    git(target, "init")
    git(target, "remote", "add", "origin", runtime.environment["TARGET_REPO_URL"])
    git(target, "fetch", "origin", "experiment-7:refs/heads/recovered")
    git(target, "symbolic-ref", "HEAD", "refs/heads/incomplete")
    (target / "untracked.txt").write_text("untracked recovery work\n")
    (target / "staged.txt").write_text("staged recovery work\n")
    git(target, "add", "staged.txt")
    original_refs = git(target, "show-ref")
    original_index = git(target, "ls-files", "--stage")
    original_head = (target / ".git/HEAD").read_text()

    result = runtime.start()

    assert result.returncode != 0
    assert "has no valid HEAD commit" in result.stderr
    assert "inspect and repair the retained checkout" in result.stderr
    assert not runtime.record.exists()
    assert git(target, "show-ref") == original_refs
    assert git(target, "ls-files", "--stage") == original_index
    assert (target / ".git/HEAD").read_text() == original_head
    assert (target / "untracked.txt").read_text() == "untracked recovery work\n"
    assert (target / "staged.txt").read_text() == "staged recovery work\n"


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_role_startup_isolates_target_uv_commands_from_agent_environment(
    tmp_path: Path, role: str
):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text()
    startup = entrypoint[entrypoint.index("export IS_SANDBOX=1"):]
    uv = shutil.which("uv")
    assert uv is not None, "the bootstrap contract requires uv"
    startup = startup.replace("/usr/local/bin/uv", shlex.quote(uv))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").write_text("#!/bin/sh\nexit 99\n")
    (fake_bin / "python").chmod(0o755)
    python = tmp_path / "runner-python"
    # Execute real bootstrap probes and setup, then record the supervisor handoff.
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:4] != ['-P', '-m', 'senpai_agent.supervisor']:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "print(json.dumps({'args': sys.argv[1:], 'environment': "
        "{key: os.environ.get(key) for key in "
        "('UV_PROJECT_ENVIRONMENT', 'UV_PYTHON', 'VIRTUAL_ENV', 'SENPAI_PYTHON')}}))\n"
    )
    python.chmod(0o755)

    completed = subprocess.run(
        ["bash", "-e", "-c", startup],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
            "PYTHONPATH": str(ROOT),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "LOGDIR": str(tmp_path),
            "WORKDIR": str(tmp_path),
            "TARGET_WORKDIR": str(tmp_path / "target"),
            "GIT_ASKPASS_FILE": str(tmp_path / "askpass"),
            "SENPAI_GITHUB_TOKEN_FILE": str(tmp_path / "token"),
            "NODES_PER_STUDENT": "1",
            "SENPAI_PYTHON": str(python),
            "UV_PROJECT_ENVIRONMENT": "/opt/senpai-venv",
            "UV_PYTHON": "/opt/senpai-venv/bin/python",
            "VIRTUAL_ENV": "/opt/senpai-venv",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    launched = json.loads(completed.stdout)
    assert launched["args"] == ["-P", "-m", "senpai_agent.supervisor", role]
    assert launched["environment"] == {
        "UV_PROJECT_ENVIRONMENT": None,
        "UV_PYTHON": None,
        "VIRTUAL_ENV": None,
        "SENPAI_PYTHON": str(python),
    }


def test_kubectl_proxy_uses_agent_python_inside_target_uv_environment(tmp_path: Path):
    entrypoint = (ROOT / "k8s" / "entrypoint-student.sh").read_text()
    proxy_setup = entrypoint[
        entrypoint.index('    proxy_dir="$LOGDIR/bin"'):
        entrypoint.index('    export PATH="$proxy_dir:$PATH"')
    ]
    runner_python = tmp_path / "runner-python"
    runner_python.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    runner_python.chmod(0o755)
    target_bin = tmp_path / "target-venv" / "bin"
    target_bin.mkdir(parents=True)
    (target_bin / "python").write_text("#!/bin/sh\nexit 99\n")
    (target_bin / "python").chmod(0o755)
    environment = {
        **os.environ,
        "LOGDIR": str(tmp_path),
        "SENPAI_PYTHON": str(runner_python),
        "PATH": f"{target_bin}:{os.environ['PATH']}",
    }
    subprocess.run(["bash", "-c", proxy_setup], env=environment, check=True)

    completed = subprocess.run(
        [str(tmp_path / "bin" / "kubectl"), "apply", "-f", "-"],
        env=environment, capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout) == [
        "-P", "-m", "senpai_agent.kubernetes_executor", "kubectl", "apply", "-f", "-",
    ]
