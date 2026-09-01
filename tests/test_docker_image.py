"""End-to-end test for the senpai Docker image on a k8s cluster.

Deploys a test pod, verifies the image runtime, then tears down the pod.

Usage:
    SENPAI_TEST_STUDENT_IMAGE=ghcr.io/wandb/senpai-student:sha-<full-commit> \
    SENPAI_TEST_REPO_REVISION=<full-commit> \
      uv run pytest tests/test_docker_image.py -v -s
"""

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest

ENTITY = "wandb-applied-ai-team"
PROJECT = "senpai-v1"
RUN_ID = uuid.uuid4().hex[:8]
POD_NAME = f"senpai-image-test-{RUN_ID}"
CONFIGMAP_NAME = f"{POD_NAME}-config"
IMAGE = os.environ.get("SENPAI_TEST_STUDENT_IMAGE", "")
SENPAI_REPO_URL = os.environ.get(
    "SENPAI_TEST_REPO_URL", "https://github.com/wandb/senpai.git"
)
SENPAI_REPO_REVISION = os.environ.get("SENPAI_TEST_REPO_REVISION", "")
POD_TEMPLATE = Path(__file__).parent / "test-pod.yaml"
STARTUP_TIMEOUT = 120
TAG = f"image-test-{RUN_ID}"
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def kubectl(*args: str, timeout: int = 30, input: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input,
        check=False,
    )
    return result.stdout.strip()


def kubectl_check(*args: str, timeout: int = 30, input: str | None = None) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def wait_for_pod(name: str, timeout: int = STARTUP_TIMEOUT):
    """Poll until the test container is ready or has failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        raw_pod = kubectl("get", "pod", name, "-o", "json")
        if not raw_pod:
            time.sleep(5)
            continue
        pod = json.loads(raw_pod)
        phase = pod["status"].get("phase")
        statuses = pod["status"].get("containerStatuses", [])
        if any(status["ready"] for status in statuses):
            return
        if phase == "Failed":
            logs = kubectl("logs", name)
            raise RuntimeError(f"Pod {name} failed during startup:\n{logs}")
        time.sleep(5)
    raise TimeoutError(f"Pod {name} not ready after {timeout}s")


def _require_immutable_test_inputs() -> None:
    if not IMAGE or not SENPAI_REPO_REVISION:
        pytest.skip(
            "set SENPAI_TEST_STUDENT_IMAGE and SENPAI_TEST_REPO_REVISION"
        )
    if not FULL_COMMIT.fullmatch(SENPAI_REPO_REVISION):
        pytest.fail(
            "SENPAI_TEST_REPO_REVISION must be a full lowercase commit SHA"
        )
    if "@sha256:" not in IMAGE and not IMAGE.endswith(
        f":sha-{SENPAI_REPO_REVISION}"
    ):
        pytest.fail(
            "SENPAI_TEST_STUDENT_IMAGE must use a digest or the matching "
            "sha-<full-commit> tag"
        )


def _build_configmap() -> str:
    """Generate the per-run test ConfigMap YAML."""
    return "\n".join(
        [
            "apiVersion: v1",
            "kind: ConfigMap",
            "metadata:",
            f"  name: {CONFIGMAP_NAME}",
            "  labels:",
            "    app: senpai",
            "    role: test",
            f"    research-tag: {TAG}",
            "data:",
            f'  SENPAI_REPO_URL: "{SENPAI_REPO_URL}"',
            f'  SENPAI_REPO_REVISION: "{SENPAI_REPO_REVISION}"',
            f'  RESEARCH_TAG: "{TAG}"',
            f'  WANDB_ENTITY: "{ENTITY}"',
            f'  WANDB_PROJECT: "{PROJECT}"',
        ]
    )


def _render_pod_template() -> str:
    """Render the pod template for this isolated test run."""
    text = POD_TEMPLATE.read_text()
    replacements = {
        "{{POD_NAME}}": POD_NAME,
        "{{CONFIGMAP_NAME}}": CONFIGMAP_NAME,
        "{{IMAGE}}": IMAGE,
        "{{RESEARCH_TAG}}": TAG,
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


@pytest.fixture(scope="module")
def test_pod():
    """Create configmap + test pod, wait for it, yield, then clean up."""
    _require_immutable_test_inputs()
    kubectl(
        "delete",
        "pod,configmap",
        "-l",
        f"research-tag={TAG}",
        "--ignore-not-found",
        timeout=120,
    )
    time.sleep(2)

    try:
        kubectl_check("apply", "-f", "-", input=_build_configmap())
        kubectl_check("apply", "-f", "-", input=_render_pod_template())
        wait_for_pod(POD_NAME)
        yield POD_NAME
    finally:
        kubectl(
            "delete",
            "pod,configmap",
            "-l",
            f"research-tag={TAG}",
            "--ignore-not-found",
            timeout=120,
        )


def test_tools_installed(test_pod):
    """Student runtime tools are available without advisor-only kubectl."""
    for cmd in ["gh --version", "uv --version"]:
        out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=15)
        assert out, f"`{cmd}` returned empty output"
    assert (
        kubectl(
            "exec",
            test_pod,
            "--",
            "bash",
            "-c",
            "! command -v kubectl",
            timeout=15,
        )
        == ""
    )


def test_legacy_weave_claude_plugin_removed(test_pod):
    """The retired Claude Code tracing plugin is absent."""
    cmd = "! command -v weave-claude-plugin && test ! -e ~/.weave_claude_plugin && echo ok"
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=15)
    assert out == "ok"


def test_python_runtime_and_training_dependencies(test_pod):
    """The student image exposes its pinned Python and training stack."""
    cmd = (
        "python - <<'PY'\n"
        "import importlib.metadata\n"
        "import sys\n"
        "import numpy\n"
        "import openhands.sdk\n"
        "import exa_py\n"
        "import torch\n"
        "import torch_geometric\n"
        "import weave_openhands\n"
        "import yaml\n"
        "assert sys.version_info[:2] == (3, 13)\n"
        "assert torch.__version__.startswith('2.13.')\n"
        "assert torch.version.cuda.startswith('13.')\n"
        "assert importlib.metadata.version('openhands-sdk') == '1.40.0'\n"
        "assert importlib.metadata.version('weave-openhands') == '0.1.0'\n"
        "print('ok')\n"
        "PY"
    )
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=20)
    assert "ok" in out


def test_cuda_runtime_on_a_real_gpu(test_pod):
    """PyTorch can execute a kernel through the host NVIDIA runtime."""
    out = kubectl_check("exec", test_pod, "--", "senpai-gpu-smoke-test", timeout=30)
    result = json.loads(out)

    assert result["status"] == "ok"
    assert result["devices"]


def test_openhands_plugin_loads_workflow_skills_without_exa_mcp(test_pod):
    """The native plugin carries workflow skills without an Exa MCP server."""
    cmd = (
        "python - <<'PY'\n"
        "from openhands.sdk.plugin import Plugin\n"
        "plugin = Plugin.load('/workspaces/senpai/plugins/senpai')\n"
        "skills = {skill.name for skill in plugin.skills}\n"
        "assert 'assign-experiment' in skills\n"
        "assert 'maintain-research-state' in skills\n"
        "assert {'poll-for-work', 'survey-prs'}.isdisjoint(skills)\n"
        "assert not plugin.mcp_config\n"
        "assert not __import__('pathlib').Path("
        "'/workspaces/senpai/plugins/senpai/.mcp.json').exists()\n"
        "print('ok')\n"
        "PY"
    )
    out = kubectl_check("exec", test_pod, "--", "bash", "-c", cmd, timeout=20)
    assert "ok" in out


def test_openhands_browser_toolset_runs_chromium(test_pod):
    """BrowserToolSet starts Chromium and reads a deterministic local page."""
    out = kubectl_check(
        "exec",
        test_pod,
        "--",
        "senpai-browser-smoke-test",
        timeout=60,
    )
    assert "senpai-browser-ok" in out
