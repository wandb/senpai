import re
import tomllib
from pathlib import Path

import pytest
import yaml

from k8s.launch import legacy_runner_bootstrap

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
    replacements = {
        "{{MODEL_PROVIDER_ENV}}": "- name: MODEL_API_KEY",
        "{{REPAIR_PROTOCOL_ANNOTATION}}": "",
        "{{SUPERVISOR_ACCESS_LABEL}}": "",
        "{{ROLE_INIT_CONTAINERS}}": "",
        "{{ROLE_BOOTSTRAP}}": legacy_runner_bootstrap(),
        "{{ROLE_WORKSPACE_MOUNTS}}": (
            "        - name: workspace\n          mountPath: /workspace"
        ),
        "{{ROLE_WORKSPACE_VOLUMES}}": (
            "      - name: workspace\n        emptyDir: {}"
        ),
        "{{REPAIR_CONTAINER}}": "",
        "{{REPAIR_VOLUMES}}": "",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
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
    assert "import importlib.metadata, openhands.sdk, sys, torch" in dockerfile
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
        assert "mkdir -p /workspace /workspaces /var/lib/senpai" in dockerfile
        assert dockerfile.rindex("ENV HOME=/home/senpai") < dockerfile.index(
            "USER 10001:10001"
        )
        assert dockerfile.index("USER 10001:10001") < dockerfile.index(
            "RUN HOME=/var/lib/senpai/home senpai-browser-smoke-test"
        )


def test_both_images_expose_the_controller_lease_as_their_healthcheck():
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text(encoding="utf-8")

        assert "HEALTHCHECK" in dockerfile
        assert "CMD /usr/local/bin/senpai-container-health" in dockerfile


def test_role_images_install_an_immutable_isolated_control_runtime():
    runner = (ROOT / "scripts" / "senpai-run-controller.sh").read_text()
    assert "exec /opt/senpai-venv/bin/python -I -m" in runner
    for role in ("advisor", "student"):
        dockerfile = (ROOT / f"Dockerfile.{role}").read_text()
        assert "COPY senpai_agent /tmp/senpai/senpai_agent" in dockerfile
        assert 'uv pip install --python "$SENPAI_PYTHON" --no-deps /tmp/senpai' in dockerfile
        assert 'assert "/opt/senpai-venv/" in senpai_agent.__file__' in dockerfile
        assert "chown -R 10001:10001 /opt/senpai-venv" not in dockerfile
        assert 'chown -R 10001:10001 /opt/senpai-venv "$UV_PYTHON_INSTALL_DIR"' not in dockerfile
        assert "SENPAI_SKIP_EDITABLE_INSTALL=1" in dockerfile


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
    assert f"exec /usr/local/bin/senpai-run-controller {role}" in entrypoint
    assert "wait_for_senpai_start_gate" not in entrypoint
    trust_runner = 'git config --global safe.directory "$WORKDIR"'
    assert entrypoint.index(trust_runner) < entrypoint.index(
        'install_senpai_git_guard "$WORKDIR"'
    )
    assert "readinessProbe" not in container
    assert container["livenessProbe"]["exec"]["command"][:2] == [
        "/bin/sh",
        "-c",
    ]
    assert (
        "/usr/local/bin/senpai-container-health"
        in container["livenessProbe"]["exec"]["command"][2]
    )
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}


@pytest.mark.parametrize("role", ["advisor", "student"])
def test_roles_clear_a_stale_lease_before_bootstrap(role: str):
    entrypoint = (ROOT / "k8s" / f"entrypoint-{role}.sh").read_text(
        encoding="utf-8"
    )
    container = container_for(load_kubernetes_template(f"{role}-deployment.yaml"))
    bootstrap = container["args"][0]
    lease = "openhands_state/controller-lease.json"

    assert entrypoint.index(lease) < entrypoint.index("SENPAI_BOOTSTRAP_STARTED_PATH")
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

    for role in ("advisor", "student"):
        container = container_for(load_kubernetes_template(f"{role}-deployment.yaml"))
        assert (
            "/usr/local/bin/senpai-container-health"
            in container["startupProbe"]["exec"]["command"][2]
        )
        assert container["startupProbe"]["failureThreshold"] == 60


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
        assert container["livenessProbe"]["terminationGracePeriodSeconds"] >= 75


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
        assert (
            'SENPAI_GITHUB_TOKEN_FILE="/tmp/senpai-supervisor-github-token"'
            in entrypoint
        )
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
