"""End-to-end test for the senpai Docker image on a k8s cluster.

Deploys a test pod, verifies the weave plugin works, checks that a
claude session produces a trace in Weave, then tears down the pod.

Usage:
    uv run pytest tests/test_docker_image.py -v -s
"""

import subprocess
import time
import uuid
from pathlib import Path

import pytest
import weave
from weave.trace.weave_client import CallsFilter

ENTITY = "wandb-applied-ai-team"
PROJECT = "senpai-v1"
POD_NAME = "senpai-image-test"
IMAGE = "ghcr.io/wandb/senpai:latest"
REPO_URL = "https://github.com/wandb/senpai.git"
REPO_BRANCH = "main"
POD_TEMPLATE = Path(__file__).parent / "test-pod.yaml"
STARTUP_TIMEOUT = 120  # seconds to wait for pod + plugin install
CLAUDE_TIMEOUT = 60
TAG = "test"


def kubectl(*args: str, timeout: int = 30, input: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True, text=True, timeout=timeout, input=input,
    )
    return result.stdout.strip()


def kubectl_check(*args: str, timeout: int = 30, input: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True, text=True, timeout=timeout, input=input,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def wait_for_pod(name: str, timeout: int = STARTUP_TIMEOUT):
    """Poll until pod is Running or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = kubectl("get", "pod", name, "-o", "jsonpath={.status.phase}")
        if out == "Running":
            return
        time.sleep(5)
    raise TimeoutError(f"Pod {name} not running after {timeout}s")


def _build_configmap() -> str:
    """Generate the senpai-config-test ConfigMap YAML."""
    return "\n".join([
        "apiVersion: v1", "kind: ConfigMap", "metadata:",
        "  name: senpai-config-test",
        "  labels:",
        f"    app: senpai",
        f"    role: test",
        f"    research-tag: {TAG}",
        "data:",
        f'  REPO_URL: "{REPO_URL}"',
        f'  REPO_BRANCH: "{REPO_BRANCH}"',
        f'  RESEARCH_TAG: "{TAG}"',
        f'  WANDB_ENTITY: "{ENTITY}"',
        f'  WANDB_PROJECT: "{PROJECT}"',
    ])


def _render_pod_template() -> str:
    """Render the test-pod.yaml template with IMAGE and RESEARCH_TAG."""
    text = POD_TEMPLATE.read_text()
    return text.replace("{{IMAGE}}", IMAGE).replace("{{RESEARCH_TAG}}", TAG)


@pytest.fixture(scope="module")
def test_pod():
    """Create configmap + test pod, wait for it, yield, then clean up."""
    kubectl("delete", "pod,configmap", "-l", f"research-tag={TAG}", "--ignore-not-found", timeout=120)
    time.sleep(2)

    # Apply configmap then pod
    kubectl_check("apply", "-f", "-", input=_build_configmap())
    kubectl_check("apply", "-f", "-", input=_render_pod_template())
    wait_for_pod(POD_NAME)
    # Wait for plugin install script to finish
    time.sleep(30)
    yield POD_NAME

    kubectl("delete", "pod,configmap", "-l", f"research-tag={TAG}", "--ignore-not-found", timeout=120)


def test_tools_installed(test_pod):
    """All baked-in tools are available."""
    for cmd in ["claude --version", "gh --version", "node --version", "uv --version", "yq --version", "weave-claude-plugin --version"]:
        out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=15)
        assert out, f"`{cmd}` returned empty output"


def test_weave_plugin_ready(test_pod):
    """Plugin status shows 'Ready to trace'."""
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", "weave-claude-plugin status", timeout=15)
    assert "Ready to trace" in out


def test_claude_creates_trace(test_pod):
    """Running claude with a unique prompt produces a matching Weave trace."""
    marker = f"senpai-test-{uuid.uuid4().hex[:8]}"
    prompt = f"Reply with exactly this string and nothing else: {marker}"

    out = kubectl_check(
        "exec", test_pod, "--",
        "bash", "-c", f'CLAUDE_CODE_ALLOW_ROOT=1 claude -p "{prompt}"',
        timeout=CLAUDE_TIMEOUT,
    )
    assert marker in out, f"Expected {marker!r} in output, got: {out!r}"

    # Give the daemon a moment to flush the trace
    time.sleep(10)

    # Query Weave for recent traces and find the one with our marker
    client = weave.init(f"{ENTITY}/{PROJECT}")
    calls = list(client.get_calls(
        filter=CallsFilter(trace_roots_only=True),
        limit=10,
        sort_by=[{"field": "started_at", "direction": "desc"}],
    ))
    assert calls, "No traces found in Weave"

    matched = [c for c in calls if marker in str(c.inputs)]
    assert matched, f"No trace found containing marker {marker!r}"
