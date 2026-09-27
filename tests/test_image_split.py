import json
import os
import re
import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
TEMPLATE_TOKEN = re.compile(r"\{\{[A-Z0-9_]+\}\}")
BINARY_ONLY_RUNTIME_PACKAGES = (
    "cryptography",
    "jiter",
    "litellm",
    "pycparser",
    "pydantic-core",
    "rpds-py",
)


def load_kubernetes_template(name: str) -> dict:
    """Render Go-template tokens before asking PyYAML to parse the manifest."""
    template = (ROOT / "k8s" / name).read_text(encoding="utf-8")
    template = template.replace(
        "{{MODEL_PROVIDER_ENV}}", "        - name: MODEL_API_KEY"
    )
    template = template.replace("{{CUSTOM_SECRET_ENV_REFS}}", "")
    return yaml.safe_load(TEMPLATE_TOKEN.sub("fixture", template))


def container_for(manifest: dict) -> dict:
    return manifest["spec"]["template"]["spec"]["containers"][0]


def named_items(items: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in items}


def test_advisor_dockerfile_prunes_the_training_stack():
    dockerfile = (ROOT / "Dockerfile.advisor").read_text(encoding="utf-8")
    lowered = dockerfile.lower()

    assert dockerfile.startswith("FROM python:3.13-slim")
    assert "uv export --locked" in dockerfile
    assert "--prune torch" in dockerfile
    assert "--prune torchvision" in dockerfile
    assert "--prune torch-geometric" in dockerfile
    assert "coreweave/ml-containers" not in lowered
    assert "nvidia_" not in lowered
    assert "senpai-gpu-smoke-test" not in lowered
    assert "import torch" not in lowered
    assert "@anthropic-ai/claude-code" not in lowered


def test_student_dockerfile_declares_the_cuda_training_runtime():
    dockerfile = (ROOT / "Dockerfile.student").read_text(encoding="utf-8")
    lowered = dockerfile.lower()

    assert "coreweave/ml-containers" in lowered
    assert "uv export --locked" in dockerfile
    assert "openhands.sdk" in dockerfile
    assert 'torch.__version__.startswith("2.13.")' in dockerfile
    assert "NVIDIA_VISIBLE_DEVICES=all" in dockerfile
    assert "senpai-gpu-smoke-test" in dockerfile
    assert "@anthropic-ai/claude-code" not in lowered


def test_lock_targets_linux_and_macos_without_the_unused_notebook_stack():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["uv"]["environments"] == [
        "sys_platform == 'linux'",
        "sys_platform == 'darwin'",
    ]
    assert "jupyter" not in project["project"]["optional-dependencies"]["dev"]

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_names = {package["name"] for package in lock["package"]}
    assert locked_names.isdisjoint({"fastjsonschema", "pyreadline3", "pywinpty"})


def test_role_images_refuse_source_builds_for_flagged_registry_packages():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        for package in BINARY_ONLY_RUNTIME_PACKAGES:
            assert f"--only-binary {package}" in dockerfile


def test_both_role_images_run_as_the_same_explicit_non_root_user():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "USER 10001:10001" in dockerfile
        assert "HOME=/home/senpai" in dockerfile
        assert "PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright" in dockerfile
        assert 'ln -s "$chromium_path" /usr/local/bin/chromium' in dockerfile
        assert "mkdir -p /workspace/senpai /workspaces /var/lib/senpai" in dockerfile
        assert dockerfile.rindex("ENV HOME=/home/senpai") < dockerfile.index(
            "USER 10001:10001"
        )
        assert dockerfile.index("USER 10001:10001") < dockerfile.index(
            "RUN HOME=/var/lib/senpai/home senpai-browser-smoke-test"
        )


def test_both_images_do_not_spawn_credential_bearing_health_processes():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "HEALTHCHECK" not in dockerfile
        assert "senpai-container-health" not in dockerfile


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_images_install_the_runner_and_protect_runtime_assets(role: str):
    dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")
    root_setup = dockerfile.split("USER 10001:10001", 1)[0].replace("\\\n", " ")

    assert (
        'uv pip install --python "$SENPAI_PYTHON" --no-deps --compile-bytecode /tmp/senpai'
        in root_setup
    )
    assert '"pip==25.3"' in root_setup
    assert "SENPAI_PYTHON=/opt/senpai-venv/bin/python" in dockerfile
    assert "UV_PYTHON=/opt/senpai-venv/bin/python" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT" not in dockerfile
    for source, destination in (
        (".agents/agents", "/opt/senpai-agent-definitions"),
        ("plugins/senpai", "/opt/senpai-plugin"),
    ):
        assert f"COPY {source} {destination}" in root_setup
    for command in ("chown -R root:root", "chmod -R a-w"):
        protected = root_setup.split(command, 1)[1].split("&&", 1)[0]
        assert "/opt/senpai-venv" in protected
        assert '"$SENPAI_AGENT_DIR"' in protected
        assert '"$SENPAI_PLUGIN"' in protected
        if role == "student":
            assert '"$UV_PYTHON_INSTALL_DIR"' in protected
    for user_ownership in root_setup.split("chown -R 10001:10001")[1:]:
        writable = user_ownership.split("&&", 1)[0]
        assert "/opt/senpai-venv" not in writable
        assert '"$SENPAI_AGENT_DIR"' not in writable
        assert '"$SENPAI_PLUGIN"' not in writable
        assert '"$UV_PYTHON_INSTALL_DIR"' not in writable


def test_both_images_record_the_exact_source_revision():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "ARG SENPAI_SOURCE_REVISION=unknown" in dockerfile
        assert (
            'LABEL org.opencontainers.image.revision="${SENPAI_SOURCE_REVISION}"'
            in dockerfile
        )
        assert 'SENPAI_IMAGE_REVISION="${SENPAI_SOURCE_REVISION}"' in dockerfile


def test_build_workflow_builds_all_images_from_the_exact_checked_out_commit():
    source = (ROOT / ".github" / "workflows" / "build.yaml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    events = yaml.load(source, Loader=yaml.BaseLoader)["on"]
    build = workflow["jobs"]["build"]
    roles = build["strategy"]["matrix"]["role"]
    steps = {step["name"]: step for step in build["steps"]}

    assert set(roles) == {"advisor", "student", "cutoff"}
    assert events["pull_request"] == {}
    assert workflow["env"]["SOURCE_REVISION"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )
    assert build["env"]["IMAGE_NAME"] == ("${{ github.repository }}-${{ matrix.role }}")
    assert steps["Checkout"]["with"]["ref"] == "${{ env.SOURCE_REVISION }}"
    assert steps["Extract metadata"]["with"]["images"] == (
        "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}"
    )
    assert (
        "type=raw,value=sha-${{ env.SOURCE_REVISION }}"
        in steps["Extract metadata"]["with"]["tags"]
    )
    build_inputs = steps["Build and push"]["with"]
    assert build_inputs["file"] == "Dockerfile.${{ matrix.role }}"
    assert (
        "SENPAI_SOURCE_REVISION=${{ env.SOURCE_REVISION }}"
        in build_inputs["build-args"]
    )


def test_runtime_workflow_uses_the_lockfile_uv_and_exa_versions():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "test.yaml").read_text(encoding="utf-8")
    )
    steps = {step["name"]: step for step in workflow["jobs"]["runtime"]["steps"]}

    assert steps["Install uv and Python"]["with"]["version"] == "0.10.9"
    install = steps["Install runtime test dependencies"]["run"]
    assert "uv lock --check" in install
    assert "exa-py @ https://github.com/exa-labs/exa-py/archive/" in install


def test_role_state_is_pod_local_and_separate_from_the_dataset_pvc():
    advisor = load_kubernetes_template("advisor-deployment.yaml")
    student = load_kubernetes_template("student-deployment.yaml")

    advisor_container = container_for(advisor)
    advisor_mounts = named_items(advisor_container["volumeMounts"])
    advisor_volumes = named_items(advisor["spec"]["template"]["spec"]["volumes"])
    assert advisor_mounts["state"]["mountPath"] == "/var/lib/senpai"
    assert advisor_volumes["state"]["emptyDir"] == {}
    assert advisor_volumes["dataset"]["persistentVolumeClaim"] == {
        "claimName": "fixture"
    }
    assert "serve-events" not in advisor_container["args"][0]

    student_container = container_for(student)
    student_mounts = named_items(student_container["volumeMounts"])
    student_volumes = named_items(student["spec"]["template"]["spec"]["volumes"])
    assert student_mounts["state"]["mountPath"] == "/var/lib/senpai"
    assert student_volumes["state"]["emptyDir"] == {}
    assert "student_logs" not in student_container["args"][0]


@pytest.mark.parametrize(
    ("role", "logdir"),
    [
        ("advisor", 'LOGDIR="/var/lib/senpai/$RESEARCH_TAG/advisor"'),
        ("student", 'LOGDIR="/var/lib/senpai"'),
    ],
)
def test_entrypoints_delegate_runtime_lifecycle_to_the_python_supervisor(
    role: str,
    logdir: str,
):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )
    deployment = load_kubernetes_template(f"{role}-deployment.yaml")
    container = container_for(deployment)

    assert logdir in entrypoint
    assert "serve-events" not in entrypoint
    assert "envsubst" not in entrypoint
    assert (
        f'SENPAI_OPENHANDS_ROLE_FILE="$WORKDIR/system_instructions/{role.upper()}.md"'
        in entrypoint
    )
    assert f'exec "$SENPAI_PYTHON" -P -m senpai_agent.supervisor {role}' in entrypoint
    assert "uv pip install" not in entrypoint
    assert "agent-context.sh" not in entrypoint
    assert "PYTHONSAFEPATH" not in entrypoint
    assert "wait_for_senpai_start_gate" not in entrypoint
    trust_runner = 'git config --global --add safe.directory "$WORKDIR"'
    assert entrypoint.index(trust_runner) < entrypoint.index(
        'install_senpai_git_guard "$WORKDIR"'
    )
    assert "readinessProbe" not in container
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}


@pytest.mark.parametrize("role", ["advisor", "student"])
@pytest.mark.parametrize(
    "target_state",
    ["fresh", "workspace_module", "existing_interpreter", "existing_site"],
)
def test_target_environment_setup_does_not_execute_target_code(
    tmp_path: Path,
    role: str,
    target_state: str,
):
    home = tmp_path / "home"
    workspace = tmp_path / "target"
    workspace.mkdir()
    target_env = home / ".venvs" / "senpai-target"
    target_site = Path(
        sysconfig.get_path("purelib", vars={"base": str(target_env)})
    )
    exposure = tmp_path / "target-code-executed"
    hostile_code = f"from pathlib import Path; Path({str(exposure)!r}).touch()\n"
    if target_state == "workspace_module":
        (workspace / "venv.py").write_text(hostile_code)
    elif target_state == "existing_interpreter":
        target_python = target_env / "bin" / "python"
        target_python.parent.mkdir(parents=True)
        target_site.mkdir(parents=True)
        target_python.write_text('#!/bin/sh\ntouch "$EXPOSURE_PATH"\n')
        target_python.chmod(0o755)
    elif target_state == "existing_site":
        target_site.mkdir(parents=True)
        (target_site / "untrusted.pth").write_text(
            f"import pathlib; pathlib.Path({str(exposure)!r}).touch()\n"
        )

    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text()
    setup = entrypoint[entrypoint.index("export SENPAI_TARGET_PYTHON_ENV=") :]
    setup = setup.split('cd "$WORKDIR"', 1)[0]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH"}
    }
    environment.update(
        HOME=str(home),
        SENPAI_PYTHON=sys.executable,
        EXPOSURE_PATH=str(exposure),
        # The image imports an installed package; this source-level harness
        # explicitly supplies the trusted package under test.
        PYTHONPATH=str(ROOT),
    )

    completed = subprocess.run(
        ["bash", "-e", "-c", setup],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert not exposure.exists(), "trusted bootstrap executed target-controlled code"
    assert completed.returncode == 0, completed.stderr
    assert (target_site / "senpai-runtime.pth").read_text().strip() == (
        sysconfig.get_path("purelib")
    )
    if target_state != "fresh":
        return

    wheel = tmp_path / "target_example-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("target_example.py", "VALUE = 'target dependency'\n")
        archive.writestr(
            "target_example-0.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: target-example\nVersion: 0.0.0\n",
        )
        archive.writestr(
            "target_example-0.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("target_example-0.0.0.dist-info/RECORD", "")
    target_python = target_env / "bin" / "python"
    subprocess.run(
        [
            "uv", "pip", "install", "--python", str(target_python),
            "--no-deps", "--no-index", str(wheel),
        ],
        env={**environment, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.check_output(
        [
            str(target_python), "-P", "-c",
            "import json,pydantic,target_example; "
            "print(json.dumps([pydantic.__file__, target_example.__file__, "
            "target_example.VALUE]))",
        ],
        cwd=workspace,
        env=environment,
        text=True,
    )
    runtime_package, target_package, value = json.loads(result)
    assert Path(runtime_package).is_relative_to(sysconfig.get_path("purelib"))
    assert Path(target_package).is_relative_to(target_site)
    assert value == "target dependency"


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_target_packages_and_shared_console_scripts_use_target_environment(
    tmp_path: Path, role: str,
):
    runtime = tmp_path / "runtime's environment"
    subprocess.run([sys.executable, "-P", "-m", "venv", str(runtime)], check=True)
    runtime_python = runtime / "bin/python"
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()

    def wheel(name, version, requires=(), module="", console_scripts=()):
        path = wheel_dir / f"{name}-{version}-py3-none-any.whl"
        info = f"{name}-{version}.dist-info"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{name}.py", f"VERSION = {version!r}\n" + module)
            archive.writestr(
                f"{info}/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
            )
            archive.writestr(
                f"{info}/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr(f"{info}/RECORD", "")
            if console_scripts:
                archive.writestr(
                    f"{info}/entry_points.txt",
                    "[console_scripts]\n"
                    + "".join(f"{script} = {name}:main\n" for script in console_scripts),
                )
        return path

    shared = wheel(
        "shared_example", "1.0",
        module=(
            "def main():\n"
            "    import json, sys, target_addon\n"
            "    print(json.dumps([sys.executable, sys.prefix, sys.argv[1:], "
            "target_addon.VERSION]))\n"
        ),
        console_scripts=("shared-example", "existing-tool", "linked-tool"),
    )
    subprocess.run(
        [str(runtime_python), "-m", "pip", "install", "--no-index", str(shared)],
        check=True, capture_output=True,
    )
    home = tmp_path / "user's home"
    target = home / ".venvs/senpai-target"
    target_bin = target / "bin"
    target_bin.mkdir(parents=True)
    existing_script = target_bin / "existing-tool"
    existing_script.write_text("#!/bin/sh\nprintf 'retained target script\\n'\n")
    existing_script.chmod(0o755)
    existing_link = target_bin / "linked-tool"
    existing_link.symlink_to("missing-target-tool")
    setup = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text()
    setup = setup[setup.index("export SENPAI_TARGET_PYTHON_ENV=") :]
    setup = setup.split('cd "$WORKDIR"', 1)[0]
    environment = {"PATH": os.environ["PATH"], "HOME": str(home),
                   "SENPAI_PYTHON": str(runtime_python), "PYTHONPATH": str(ROOT)}
    subprocess.run(["bash", "-e", "-c", setup], env=environment, check=True)
    target_python = target / "bin/python"
    target_site = Path(sysconfig.get_path("purelib", vars={"base": str(target)}))
    addon = wheel("target_addon", "1.0", ("shared-example>=1.0",))
    # No index or find-links: resolving the addon's dependency must reuse the
    # installed shared distribution, not fetch or copy it into the target.
    subprocess.run(
        [str(target_python), "-m", "pip", "install", "--no-index", str(addon)],
        env=environment, check=True, capture_output=True,
    )
    assert not (target_site / "shared_example.py").exists()
    assert (target_site / "target_addon.py").is_file()

    target_environment = {
        **environment,
        "PATH": f"{target_bin}:{runtime / 'bin'}:{environment['PATH']}",
    }
    arguments = ["two words", "quote'and\"double", "$(printf unsafe); *", ""]
    result = subprocess.run(
        ["shared-example", *arguments],
        env=target_environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        str(target_python), str(target), arguments, "1.0",
    ]
    subprocess.run(["bash", "-e", "-c", setup], env=environment, check=True)
    assert subprocess.check_output(
        [str(existing_script)], env=target_environment, text=True,
    ) == "retained target script\n"
    assert existing_link.is_symlink()
    assert existing_link.readlink() == Path("missing-target-tool")

    upgrade = wheel("shared_example", "2.0")
    subprocess.run(
        [str(target_python), "-m", "pip", "install", "--no-index", str(upgrade)],
        env=environment, check=True, capture_output=True,
    )
    for python, expected in ((runtime_python, "1.0"), (target_python, "2.0")):
        version = subprocess.check_output(
            [str(python), "-P", "-c", "import shared_example; print(shared_example.VERSION)"],
            env=environment, text=True,
        ).strip()
        assert version == expected


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_roles_clear_a_stale_lease_before_bootstrap(role: str):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )
    container = container_for(load_kubernetes_template(f"{role}-deployment.yaml"))
    bootstrap = container["args"][0]
    lease = "openhands_state/controller-lease.json"

    assert entrypoint.index(lease) < entrypoint.index("git clone")
    assert bootstrap.index(lease) < bootstrap.index("git init /workspace/senpai")


def test_bootstrap_git_credentials_are_not_exposed_in_process_arguments():
    for role in ("advisor", "student"):
        deployment = load_kubernetes_template(f"{role}-deployment.yaml")
        container = container_for(deployment)
        bootstrap = container["args"][0]

        assert "${GITHUB_TOKEN}@github.com" not in bootstrap
        assert "GIT_ASKPASS" in bootstrap
        assert "mkdir -p /workspace" in bootstrap
        assert "SENPAI_GITHUB_TOKEN_FILE" in bootstrap
        assert "unset GITHUB_TOKEN GH_TOKEN" in bootstrap
        assert "exec bash" in bootstrap


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_role_probes_query_http_without_starting_a_process(role: str):
    container = container_for(load_kubernetes_template(f"{role}-deployment.yaml"))
    assert container["startupProbe"] == {
        "httpGet": {"path": "/healthz", "port": 8080},
        "periodSeconds": 10,
        "timeoutSeconds": 5,
        "failureThreshold": 60,
    }
    assert container["livenessProbe"] == {
        "httpGet": {"path": "/healthz", "port": 8080},
        "periodSeconds": 30,
        "timeoutSeconds": 5,
        "failureThreshold": 5,
        "terminationGracePeriodSeconds": 75,
    }


def test_role_entrypoints_default_openhands_turns_to_two_hours_of_inactivity():
    for name in ("entrypoint-advisor.sh", "entrypoint-student.sh"):
        entrypoint = (ROOT / "k8s" / name).read_text()
        assert 'SENPAI_OPENHANDS_TIMEOUT_SECONDS:-7200' in entrypoint


def test_role_pods_enforce_non_root_process_isolation():
    for role in ("advisor", "student"):
        deployment = load_kubernetes_template(f"{role}-deployment.yaml")
        pod = deployment["spec"]["template"]["spec"]
        container = container_for(deployment)

        assert pod["securityContext"] == {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "fsGroupChangePolicy": "OnRootMismatch",
            "seccompProfile": {"type": "RuntimeDefault"},
        }
        assert container["securityContext"] == {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }
        assert (
            pod["terminationGracePeriodSeconds"]
            > container["livenessProbe"]["terminationGracePeriodSeconds"]
        )


def test_runtime_git_auth_uses_ephemeral_askpass_not_a_credential_store():
    guard = (ROOT / "plugins" / "senpai" / "scripts" / "git-guard.sh").read_text(
        encoding="utf-8"
    )
    assert "GIT_ASKPASS" in guard
    assert "GIT_TERMINAL_PROMPT" in guard
    assert ".git-credentials" not in guard
    assert 'credential.helper "store' not in guard
    assert "x-access-token:%s@github.com" not in guard

    for role in ("advisor", "student"):
        entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
            encoding="utf-8"
        )
        assert 'GIT_ASKPASS_FILE="/tmp/senpai-git-askpass"' in entrypoint
        assert ".git-credentials" not in entrypoint
        assert 'credential.helper "store' not in entrypoint


def test_entrypoint_umask_is_configurable_but_token_creation_stays_private():
    for role in ("advisor", "student"):
        entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
            encoding="utf-8"
        )

        assert 'umask "${SENPAI_UMASK:-0022}"' in entrypoint
        assert (
            "(umask 077; printf '%s' \"$GITHUB_TOKEN\" > "
            '"$SENPAI_GITHUB_TOKEN_FILE")'
        ) in entrypoint


@pytest.mark.parametrize("role", ["advisor", "student"])
@pytest.mark.parametrize("services", ["absent", "environment", "files"])
def test_entrypoint_hands_off_credentials_without_requiring_optional_services(
    tmp_path: Path, role: str, services: str
):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text()
    handoff = entrypoint[
        entrypoint.index('CREDENTIAL_HANDOFF_DIR=""') : entrypoint.index(
            'rm -f "$GIT_ASKPASS_FILE"'
        )
    ].replace("/tmp/senpai-supervisor.", str(tmp_path / "handoff."))
    handoff += '''
test "${GITHUB_TOKEN+x}${GH_TOKEN+x}${WANDB_API_KEY+x}${EXA_API_KEY+x}" = ""
printf '%s\\n' "$SENPAI_GITHUB_TOKEN_FILE" "${SENPAI_WANDB_API_KEY_FILE:-}" "${SENPAI_EXA_API_KEY_FILE:-}"
umask > "$UMASK_OUTPUT"
'''
    credential_names = ("GITHUB_TOKEN", "WANDB_API_KEY", "EXA_API_KEY")
    env = {
        key: value for key, value in os.environ.items()
        if key not in credential_names
        and key not in {f"SENPAI_{name}_FILE" for name in credential_names}
    }
    env |= {
        "GITHUB_TOKEN": "github-fixture",
        "GH_TOKEN": "github-alias-fixture",
        "UMASK_OUTPUT": str(tmp_path / "umask"),
    }
    expected = ["github-fixture"]
    if services != "absent":
        expected += ["wandb-fixture", "exa-fixture"]
    if services == "environment":
        env |= {"WANDB_API_KEY": "wandb-fixture", "EXA_API_KEY": "exa-fixture"}
    elif services == "files":
        existing = tmp_path / "provided"
        existing.mkdir(mode=0o700)
        for name, value in zip(credential_names, expected, strict=True):
            path = existing / name.lower()
            path.write_text(value)
            path.chmod(0o600)
            env[f"SENPAI_{name}_FILE"] = str(path)

    result = subprocess.run(
        ["bash", "-e", "-c", "umask 0022\n" + handoff],
        env=env, capture_output=True, text=True, check=True,
    )

    paths = result.stdout.splitlines()
    assert len(paths) == 3
    assert bool(paths[1]) == bool(paths[2]) == (services != "absent")
    active_paths = [Path(path) for path in paths if path]
    assert len({path.parent for path in active_paths}) == 1
    assert active_paths[0].parent.stat().st_mode & 0o777 == 0o700
    for path, value in zip(active_paths, expected, strict=True):
        assert path.read_text() == value
        assert path.stat().st_mode & 0o777 == 0o600
    if services == "files":
        assert active_paths[0].parent == existing
    assert int((tmp_path / "umask").read_text().strip(), 8) == 0o22


def test_manifests_expose_no_advisor_service_or_callback_credentials():
    deployment = load_kubernetes_template("advisor-deployment.yaml")
    student = load_kubernetes_template("student-deployment.yaml")

    advisor_container = container_for(deployment)
    advisor_env = named_items(advisor_container["env"])
    student_env = named_items(container_for(student)["env"])
    assert "ports" not in advisor_container
    assert not {
        "SENPAI_ADVISOR_EVENT_TOKEN",
        "SENPAI_ADVISOR_NOTIFY_URL",
        "SENPAI_ADVISOR_NOTIFY_TOKEN",
    } & (set(advisor_env) | set(student_env))
    assert not (ROOT / "k8s" / "advisor-service.yaml").exists()


def test_launch_configuration_names_only_the_two_role_images():
    config = yaml.safe_load((ROOT / "senpai.yaml").read_text(encoding="utf-8"))

    assert "advisor_image" in config
    assert "student_image" in config
    assert "control_image" not in config
    assert "image" not in config
