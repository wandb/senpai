import base64
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "tests" / "kubernetes" / "canary.py"
REVISION = "a" * 40
IMAGE = f"senpai-kubernetes-canary:sha-{REVISION}"


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
    assert ("Deployment", "senpai-supervisor-ci-123") in resources

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

    _secret, _config, supervisor = supervisor_bundle(documents)
    supervisor_pod = supervisor["spec"]["template"]["spec"]
    assert {container["name"] for container in supervisor_pod["containers"]} == {
        "supervisor-control",
        "supervisor-shell",
    }
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
    assert all(
        container["image"] == IMAGE
        for document in documents
        if document["kind"] in {"Deployment", "Pod"}
        for container in document["spec"].get(
            "containers",
            document["spec"].get("template", {}).get("spec", {}).get("containers", []),
        )
    )


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
        "kindest/node:v1.33.4@sha256:"
        "25a6018e48dfcaee478f4a59af81157a437f15e6e140bf103f85a2e7cd0cbbf2"
    ) in script
    assert 'delete cluster --name "$CLUSTER"' in script
    assert "SUPERVISOR_STATE_SENTINEL" in script
    assert "SENPAI_CI_DUMMY_" not in workflow_text
