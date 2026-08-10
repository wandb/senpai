import base64
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "tests" / "kubernetes" / "canary.py"
REVISION = "a" * 40
IMAGE = f"senpai-kubernetes-canary:sha-{REVISION}"
ADVISOR_IMAGE = f"senpai-kubernetes-canary-advisor:sha-{REVISION}"
STUDENT_IMAGE = f"senpai-kubernetes-canary-student:sha-{REVISION}"


def render(phase: str) -> list[dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(CANARY),
            "render",
            "--phase",
            phase,
            "--namespace",
            "senpai-ci-123",
            "--other-namespace",
            "senpai-ci-123-other",
            "--tag",
            "ci-123",
            "--image",
            IMAGE,
            "--revision",
            REVISION,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def render_with_role_images(phase: str) -> list[dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(CANARY),
            "render",
            "--phase",
            phase,
            "--namespace",
            "senpai-ci-123",
            "--other-namespace",
            "senpai-ci-123-other",
            "--tag",
            "ci-123",
            "--advisor-image",
            ADVISOR_IMAGE,
            "--student-image",
            STUDENT_IMAGE,
            "--revision",
            REVISION,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def by_kind_and_name(documents: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        (document["kind"], document["metadata"]["name"]): document
        for document in documents
    }


def supervisor_bundle(documents: list[dict]) -> tuple[str, str, dict]:
    deployment = next(
        document
        for document in documents
        if document["kind"] == "Deployment"
        and document["metadata"]["name"] == "senpai-supervisor-ci-123"
    )
    pod = deployment["spec"]["template"]["spec"]
    control = next(
        container
        for container in pod["containers"]
        if container["name"] == "supervisor-control"
    )
    return (
        control["env"][0]["valueFrom"]["secretKeyRef"]["name"],
        control["envFrom"][0]["configMapRef"]["name"],
        deployment,
    )


def test_initial_canary_manifest_uses_dummy_credentials_and_real_boundaries():
    """
    Requirement: the PR canary exercises the production container authority split
    without receiving any live credential.
    Interface: YAML emitted by `tests/kubernetes/canary.py render`.
    """

    documents = render("initial")
    resources = by_kind_and_name(documents)

    assert ("Namespace", "senpai-ci-123") in resources
    assert ("Namespace", "senpai-ci-123-other") in resources
    assert ("Deployment", "senpai-advisor-ci-123") in resources
    assert ("Deployment", "senpai-ci-123-fern") in resources
    assert ("Deployment", "senpai-supervisor-ci-123") in resources
    assert ("PersistentVolumeClaim", "senpai-ci-ci-123-supervisor-state") in resources
    assert resources[(
        "PersistentVolumeClaim",
        "senpai-ci-ci-123-supervisor-state",
    )]["metadata"]["annotations"] == {
        "senpai.wandb.com/sqlite-safe": "true"
    }
    policy = resources[("NetworkPolicy", "senpai-supervisor-egress-ci-123")]
    assert policy["spec"]["podSelector"]["matchLabels"] == {
        "research-tag": "ci-123",
        "senpai-supervisor-access": "true",
    }
    unrestricted = [
        rule["to"][0]["ipBlock"]
        for rule in policy["spec"]["egress"]
        if "ports" not in rule
    ]
    assert {block["cidr"] for block in unrestricted} == {"0.0.0.0/0", "::/0"}
    assert {value for block in unrestricted for value in block["except"]} == {
        "169.254.0.0/16",
        "fe80::/10",
        "fd00:ec2::254/128",
    }
    dns_exceptions = [
        rule for rule in policy["spec"]["egress"] if "ports" in rule
    ]
    assert {rule["to"][0]["ipBlock"]["cidr"] for rule in dns_exceptions} == {
        "169.254.0.0/16",
        "fe80::/10",
    }
    assert all(
        {(port["protocol"], port["port"]) for port in rule["ports"]}
        == {("TCP", 53), ("UDP", 53)}
        for rule in dns_exceptions
    )

    for document in documents:
        if document["kind"] != "Secret":
            continue
        decoded = {
            name: base64.b64decode(value).decode()
            for name, value in document["data"].items()
        }
        assert decoded
        assert all(value.startswith("SENPAI_CI_DUMMY_") for value in decoded.values())

    advisor = resources[("Deployment", "senpai-advisor-ci-123")]
    advisor_pod = advisor["spec"]["template"]["spec"]
    assert {container["name"] for container in advisor_pod["containers"]} == {
        "advisor",
        "repair",
    }
    repair = next(
        container
        for container in advisor_pod["containers"]
        if container["name"] == "repair"
    )
    assert "env" not in repair and "envFrom" not in repair
    assert advisor_pod["automountServiceAccountToken"] is False
    assert advisor["spec"]["template"]["metadata"]["labels"][
        "senpai-supervisor-access"
    ] == "true"
    student = resources[("Deployment", "senpai-ci-123-fern")]
    student_pod = student["spec"]["template"]["spec"]
    assert {container["name"] for container in student_pod["containers"]} == {
        "student",
        "repair",
    }
    assert not any(
        "nvidia.com/gpu" in resources
        for container in student_pod["containers"]
        for resources in container.get("resources", {}).values()
    )

    embedded_source = (
        "set -eu; cp -R /opt/senpai/. /workspace/senpai; "
        "test -f /workspace/senpai/tests/kubernetes/canary.py"
    )
    for pod in (advisor_pod, student_pod):
        source = next(
            container
            for container in pod["initContainers"]
            if container["name"] == "source"
        )
        assert source["args"][0].startswith(embedded_source)
        assert "env" not in source
        assert source["envFrom"] == [
            {"configMapRef": {"name": source["envFrom"][0]["configMapRef"]["name"]}}
        ]
    advisor_source = next(
        container
        for container in advisor_pod["initContainers"]
        if container["name"] == "source"
    )
    assert "envsubst" in advisor_source["args"][0]
    assert "SENPAI_IMMUTABLE_ADVISOR_GUIDANCE_FILE" in advisor_source["args"][0]
    assert advisor_pod["containers"][0]["envFrom"]

    _secret, _config, supervisor = supervisor_bundle(documents)
    supervisor_pod = supervisor["spec"]["template"]["spec"]
    assert {container["name"] for container in supervisor_pod["containers"]} == {
        "supervisor-control",
        "supervisor-shell",
    }
    supervisor_source = next(
        container
        for container in supervisor_pod["initContainers"]
        if container["name"] == "source"
    )
    assert supervisor_source["args"] == [embedded_source]
    assert "env" not in supervisor_source
    assert supervisor_source["envFrom"] == [
        {
            "configMapRef": {
                "name": supervisor_source["envFrom"][0]["configMapRef"]["name"]
            }
        }
    ]
    shell = next(
        container
        for container in supervisor_pod["containers"]
        if container["name"] == "supervisor-shell"
    )
    assert "envFrom" not in shell
    assert all(
        fragment not in entry["name"]
        for entry in shell["env"]
        for fragment in ("TOKEN", "KEY", "SECRET", "CREDENTIAL")
    )
    assert "service-account" not in {
        mount["name"] for mount in shell["volumeMounts"]
    }
    assert supervisor_pod["automountServiceAccountToken"] is False
    assert supervisor["spec"]["template"]["metadata"]["labels"][
        "senpai-supervisor-access"
    ] == "true"
    assert all(
        container["image"] == IMAGE
        for document in documents
        if document["kind"] in {"Deployment", "Pod"}
        for container in document["spec"].get(
            "containers",
            document["spec"].get("template", {}).get("spec", {}).get("containers", []),
        )
    )


def test_canary_renders_advisor_and_student_from_their_actual_distinct_images():
    """
    Requirement: the production canary must exercise the student image as a
    student, rather than silently running both roles from the advisor image.
    Interface: role-specific canary renderer arguments and emitted Deployments.
    """

    resources = by_kind_and_name(render_with_role_images("initial"))
    advisor = resources[("Deployment", "senpai-advisor-ci-123")]
    student = resources[("Deployment", "senpai-ci-123-fern")]
    supervisor = resources[("Deployment", "senpai-supervisor-ci-123")]

    assert {
        container["image"]
        for container in advisor["spec"]["template"]["spec"]["containers"]
    } == {ADVISOR_IMAGE}
    assert {
        container["image"]
        for container in student["spec"]["template"]["spec"]["containers"]
    } == {STUDENT_IMAGE}
    assert {
        container["image"]
        for container in supervisor["spec"]["template"]["spec"]["containers"]
    } == {ADVISOR_IMAGE}


def test_pull_request_loads_and_smokes_the_actual_student_production_image():
    """
    Requirement: each role image is built only by its existing matrix job, but
    pull requests must load the real student image and exercise its immutable
    runtime plus repair lifecycle before the lightweight Kind topology can pass.
    Interface: the pull-request image workflow and student smoke script.
    """

    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "build.yaml").read_text()
    )
    steps = workflow["jobs"]["build"]["steps"]
    image_build = next(step for step in steps if step["name"] == "Build and push")
    student_smoke = next(
        step for step in steps if step["name"] == "Run student image production smoke"
    )
    canary = next(
        step for step in steps if step["name"] == "Run Kubernetes production canary"
    )

    assert image_build["with"]["load"] == (
        "${{ github.event_name == 'pull_request' && matrix.role != 'cutoff' }}"
    )
    assert "github.event_name == 'pull_request'" in student_smoke["if"]
    assert "matrix.role == 'student'" in student_smoke["if"]
    assert student_smoke["env"]["STUDENT_IMAGE"] == (
        "ghcr.io/${{ env.IMAGE_NAME }}:sha-${{ env.SOURCE_REVISION }}"
    )
    assert student_smoke["run"] == "scripts/test-student-image-smoke.sh"

    smoke = (ROOT / "scripts" / "test-student-image-smoke.sh").read_text()
    assert "/opt/senpai-venv/bin/python -I" in smoke
    assert "senpai_agent" in smoke
    assert "test ! -w" in smoke
    assert "/usr/local/bin/senpai-repair-executor" in smoke
    assert "/tmp/senpai-repair-executor-smoke/executor.sock" in smoke
    assert "@senpai-repair-executor" not in smoke
    assert " serve " in smoke
    assert " client " in smoke

    assert canary["env"]["ADVISOR_CANARY_IMAGE"] != canary["env"][
        "STUDENT_CANARY_IMAGE"
    ]
    canary_script = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()
    assert '--advisor-image "$ADVISOR_CANARY_IMAGE"' in canary_script
    assert '--student-image "$STUDENT_CANARY_IMAGE"' in canary_script
    assert (
        'load docker-image --name "$CLUSTER" "$ADVISOR_CANARY_IMAGE"'
        in canary_script
    )
    assert (
        'load docker-image --name "$CLUSTER" "$STUDENT_CANARY_IMAGE"'
        in canary_script
    )


def test_live_role_repairs_prove_the_target_mount_is_a_real_git_worktree():
    """
    Requirement: a directory merely named `.git` must not let the production
    canary claim that supervisor repairs reached the cloned target repository.
    Interface: live advisor and student repair assertions in the canary script.
    """

    script = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()

    assert script.count(
        'test "$(git rev-parse --show-toplevel)" = /repair/workspace'
    ) >= 2
    assert "test ! -e /run/senpai-repair-executor" in script
    assert 'repair_socket=$(repair_socket_for "$deployment")' in script
    assert "@senpai-repair-executor" not in script
    helper = (ROOT / "tests" / "kubernetes" / "canary.py").read_text()
    assert "cp -R /opt/senpai/. /workspace/senpai" in helper
    assert "cp -a /opt/senpai/. /workspace/senpai" not in helper


def test_live_canary_covers_terminal_and_durable_operation_recovery():
    """The cluster gate must cover the supervisor's crash-sensitive boundaries."""

    script = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()

    assert "probe-terminal-wakes" in script
    assert "drop-repair-reply" in script
    assert "seed-interrupted-operations" in script
    assert "assert-interrupted-operations" in script
    assert script.count("--operation-id") >= 8
    assert "--status" in script
    assert "same operation ID accepted a different command" in script
    assert "repair replay executed twice" in script


def test_failure_diagnostics_preserve_current_and_previous_container_logs():
    """A restart must not erase the exception emitted by the prior container."""

    script = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()
    cleanup = script.split("cleanup() {", 1)[1].split("trap cleanup EXIT", 1)[0]

    assert "container-restarts.tsv" in script
    assert "containerStatuses" in script
    assert '{"advisor", "student", "supervisor"}' in script
    assert 'if (( restarts > 0 )); then' in script
    assert '"$prefix.current.log"' in script
    assert '"$prefix.previous.log"' in script
    assert "--previous --timestamps=true --tail=2000" in script
    assert cleanup.index("collect_diagnostics") < cleanup.index(
        "delete deployment,pod"
    )


def test_live_canary_proves_role_quiescence_and_poisoned_root_recovery():
    """The cluster gate must reproduce the production-only pause/restart risks."""

    script = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()
    helper = (ROOT / "tests" / "kubernetes" / "canary.py").read_text()

    assert "GENERATION_ENV" in helper
    assert "signal.SIG_IGN" in helper
    assert "start_new_session=True" in helper
    assert 'server.bind(("127.0.0.1", {ROLE_LISTENER_PORT}))' in helper
    assert script.count("http://127.0.0.1:18765/") >= 3
    assert "wait_for_role_listener_advance" in script
    assert "wake-stale456" in script
    assert "operation-stale456" in script
    assert script.count("kill -TERM 1") >= 2
    assert "wait_for_container_restart" in script
    assert "senpai_agent.isolated_terminal_health" in script
    assert "ROLE_SHELL=/usr/local/bin/senpai-role-shell" in script


def test_docs_state_shared_pod_loopback_and_fail_closed_repair_gate():
    """Do not overclaim sidecar isolation; make repair quiescence explicit."""

    readme = (ROOT / "README.md").read_text()
    canary = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()
    normalized_readme = " ".join(readme.split())

    assert "share the Pod network namespace" in readme
    assert "loopback" in readme
    assert "do not isolate loopback" in readme
    assert "proves that no TCP listener remains" in normalized_readme
    assert "an unproven pause refuses the repair" in normalized_readme
    assert "disableDefaultCNI: true" in canary
    assert "rollout status -n kube-system daemonset/calico-node" in canary
    assert "senpai-metadata-decoy" in canary
    assert "verify_sha256" in canary
    assert "sha256sum --check" not in canary


def test_failed_upgrade_is_supervisor_only_and_retains_versioned_rollback_bundles():
    """
    Requirement: a supervisor-only bad release changes both immutable bundle names,
    never reapplies role resources, and gives Kubernetes a failing revision to undo.
    Interface: initial and upgrade YAML emitted by the canary renderer.
    """

    initial = render("initial")
    upgrade = render("broken-upgrade")
    first_secret, first_config, _first_deployment = supervisor_bundle(initial)
    next_secret, next_config, next_deployment = supervisor_bundle(upgrade)

    assert (first_secret, first_config) != (next_secret, next_config)
    assert max(map(len, (first_secret, first_config, next_secret, next_config))) <= 63
    assert not any(
        document["kind"] in {"Namespace", "PersistentVolume", "PersistentVolumeClaim"}
        or (
            document["kind"] == "Deployment"
            and document["metadata"]["name"] == "senpai-advisor-ci-123"
        )
        or document["metadata"]["name"] == "senpai-launch-secrets-ci-123"
        for document in upgrade
    )
    bundles = {
        document["metadata"]["name"]: document
        for document in (*initial, *upgrade)
        if document["kind"] in {"Secret", "ConfigMap"}
        and "supervisor" in document["metadata"]["name"]
    }
    assert bundles[first_secret]["immutable"] is True
    assert bundles[first_config]["immutable"] is True
    assert bundles[next_secret]["immutable"] is True
    assert bundles[next_config]["immutable"] is True
    control = next(
        container
        for container in next_deployment["spec"]["template"]["spec"]["containers"]
        if container["name"] == "supervisor-control"
    )
    assert "exit 42" in control["args"][0]

    canary = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()
    assert "SupervisorRollback.capture" in canary
    assert "k8s/supervisor_rollback.py restore" in canary
    assert "persistent_state_rolled_back" in canary
    assert "ABSENT_SERVICE_ACCOUNT" in canary
    assert "assert-interrupted-operations" in canary
    assert "rollout undo" not in canary


def test_pull_request_canary_has_no_live_secret_or_checkout_credential_path():
    """
    Requirement: untrusted pull-request code receives only dummy sentinels and
    cannot inherit the checkout credential in its Git configuration.
    Interface: the build workflow, thin Dockerfile, and canary lifecycle script.
    """

    workflow_text = (ROOT / ".github" / "workflows" / "build.yaml").read_text()
    workflow = yaml.safe_load(workflow_text)
    assert "pull_request_target" not in workflow_text
    build_steps = workflow["jobs"]["build"]["steps"]
    checkout = next(step for step in build_steps if step["name"] == "Checkout")
    assert checkout["with"]["persist-credentials"] is False
    canary = next(
        step for step in build_steps if step["name"] == "Run Kubernetes production canary"
    )
    assert "secrets." not in yaml.safe_dump(canary)

    dockerfile = (ROOT / "tests" / "kubernetes" / "Dockerfile").read_text()
    assert "COPY . " not in dockerfile
    assert ".git" not in "\n".join(
        line for line in dockerfile.splitlines() if line.lstrip().startswith("COPY")
    )
    assert ".env" not in "\n".join(
        line for line in dockerfile.splitlines() if line.lstrip().startswith("COPY")
    )

    script = (ROOT / "scripts" / "test-kubernetes-canary.sh").read_text()
    assert (
        "kindest/node:v1.34.8@sha256:"
        "02722c2dedddcfc00febf5d27fbeb9b7b2c14294c82109ff4a85d89ac9ba3256"
    ) in script
    install_step = next(
        step for step in build_steps if step["name"] == "Install pinned Kind"
    )
    install_script = install_step["run"]
    assert "/v0.32.0/kind-linux-amd64" in install_script
    assert (
        "50030de23cf40a18505f20426f6a8506bedf13c6e509244bd1fa9463721b0f54"
        in install_script
    )
    assert "disableDefaultCNI: true" in script
    assert "CALICO_VERSION=v3.32.1" in script
    assert "projectcalico/calico/$CALICO_VERSION/manifests/calico.yaml" in script
    assert "a1df919d9721cf667accdc3e72848911b0cb25cfab7d2478ad0c996302c95744" in script
    for digest in (
        "sha256:bb1567e3ed81e2e8414e9a68f186e1f7ffd4067a4871a9ae90896793af0190dd",
        "sha256:18008f781c869376dbbc4dfb1ffe3afb46f7897887d4f20e080c420ac44a6612",
        "sha256:7f874b3f0b540c2b523aea9961ef5e2f43b0af9056a47874c916d6cf348168d3",
    ):
        assert digest in script
    assert "169.254.169.254/32" in script
    assert "supervisor shell reached the metadata decoy" in script
    assert "repair sidecar reached the metadata decoy" in script
    assert 'delete cluster --name "$CLUSTER"' in script
    assert "SUPERVISOR_STATE_SENTINEL" in script
    assert "SENPAI_CI_DUMMY_" not in workflow_text
