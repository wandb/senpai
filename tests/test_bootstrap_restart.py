"""Exercise same-pod bootstrap against retained emptyDir contents."""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest
import yaml

from git_workflow_support import commit_file, git, repository
from launch_test_support import launch_args, render_role

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_fresh_container_preserves_target_work_and_dependencies(tmp_path, role):
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
names = ["SENPAI_GITHUB_TOKEN_FILE", "SENPAI_WANDB_API_KEY_FILE", "SENPAI_EXA_API_KEY_FILE"]
expected_services = {"WANDB_API_KEY": "wandb-fixture", "EXA_API_KEY": "exa-fixture"}
if os.environ["SENPAI_ROLE"] == "student":
    names.append("SENPAI_WANDB_TRAINING_API_KEY_FILE")
    expected_services["SENPAI_WANDB_TRAINING_API_KEY"] = "writer-fixture"
paths = [Path(os.environ[name]) for name in names]
assert all(name not in os.environ for name in ("GITHUB_TOKEN", "GH_TOKEN", "WANDB_API_KEY", "EXA_API_KEY", "SENPAI_WANDB_TRAINING_API_KEY"))
assert len({path.parent for path in paths}) == 1
assert paths[0].parent.stat().st_mode & 0o777 == 0o700
assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
assert _consume_github_token(os.environ).get_secret_value() == "github-fixture"
services = _consume_private_credential_files(os.environ)
assert {name: value.get_secret_value() for name, value in services.items()} == expected_services
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
        return script

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
    target_source, target_remote, target_baseline = repository(target_source_root)
    if role == "student":
        git(target_source, "push", "origin", "HEAD:refs/heads/research")

    _configmap, deployment, _secret = render_role(
        role, launch_args(problem_dir="target/", advisor_branch="research")
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
        "HOME": str(home),
        "PATH": f"{tools}:{os.environ['PATH']}",
        "PYTHONPATH": str(ROOT),
        "SENPAI_PYTHON": sys.executable,
        "SENPAI_PLUGIN": str(ROOT / "plugins/senpai"),
        "SENPAI_AGENT_DIR": str(ROOT / ".agents/agents"),
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
    if role == "student":
        environment["SENPAI_WANDB_TRAINING_API_KEY"] = "writer-fixture"

    def start():
        home.mkdir(parents=True, exist_ok=True)
        (container / "tmp").mkdir(exist_ok=True)
        for path in retained_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["bash", "-c", bootstrap], env=environment,
            capture_output=True, text=True, timeout=30,
        )
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
