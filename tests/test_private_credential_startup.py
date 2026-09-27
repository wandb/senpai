"""Exercise credential handoffs across real Python process startup."""

import json
import os
import select
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from openhands_support import runtime_env

from senpai_agent.delegation import DelegationRequest, OpenHandsChildProcess
from senpai_agent.openhands_runner import (
    delegation_config,
    parse_runner_args,
    resolve_config,
)
from senpai_agent.secrets import PRIVATE_CREDENTIAL_FD_ENVS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    sys.platform != "linux" or os.getuid() == 0,
    reason="requires Linux procfs inspection by a non-root user",
)
def test_nondumpable_child_denies_same_uid_process_inspection():
    """Readiness follows hardening; the exec-to-prctl race remains outside this guarantee."""

    startup = """
import ctypes
import json
import os
import sys
from pathlib import Path
from senpai_agent import secrets

assert Path(secrets.__file__).resolve() == Path(os.environ["PYTHONPATH"]) / "senpai_agent/secrets.py"
secrets.set_process_nondumpable()
descriptor = int(sys.argv[1])
assert os.pread(descriptor, 100, 0) == b"private-model-key"
dumpable = ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)
print(json.dumps({"dumpable": dumpable, "uid": os.getuid()}), flush=True)
sys.stdin.read(1)
"""
    with tempfile.TemporaryFile() as credentials:
        credentials.write(b"private-model-key")
        credentials.flush()
        descriptor = credentials.fileno()
        process = subprocess.Popen(
            (sys.executable, "-P", "-c", startup, str(descriptor)),
            env={"PYTHONPATH": str(ROOT), "PRIVATE_MODEL_KEY": "private-model-key"},
            pass_fds=(descriptor,),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    try:
        assert select.select([process.stdout], [], [], 10)[0], "child did not become ready"
        ready = json.loads(process.stdout.readline())
        assert ready == {"dumpable": 0, "uid": os.getuid()}
        for path in (
            Path(f"/proc/{process.pid}/environ"),
            Path(f"/proc/{process.pid}/fd/{descriptor}"),
        ):
            with pytest.raises(PermissionError):
                path.read_bytes()
        _stdout, stderr = process.communicate("\n", timeout=5)
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_real_child_startup_resolves_private_models_and_preserves_services(
    tmp_path, monkeypatch
):
    environment = runtime_env(tmp_path)
    environment.update(
        {
            "SENPAI_OPENHANDS_FRONTIER_MODEL": "openai/gpt-5.6-sol",
            "WANDB_API_KEY": "wandb-service-key",
            "EXA_API_KEY": "exa-service-key",
            "WANDB_ENTITY": "test-entity",
            "WANDB_PROJECT": "test-project",
            "SENPAI_CUSTOM_SECRET_ENV_NAMES": "PRIVATE_AUTH",
            "PRIVATE_AUTH": "custom-service-key",
        }
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    parent = resolve_config(parse_runner_args(["--max-turns", "1"]), environment)
    child = OpenHandsChildProcess(
        delegation_config(parent),
        DelegationRequest(
            task_id=str(uuid.uuid4()),
            parent_conversation_id=str(parent.conversation_id),
            parent_context=(),
            agent="explore",
            model="fast",
            search_mode=None,
        ),
    )
    command = child.command
    # Replace only model execution; retain the real exec, CLI, FD consumer and
    # config resolution so this test needs no model service.
    startup = """
import json
import os
import sys
from pathlib import Path
import weave_openhands

transforms = []
def initialize(*args, **kwargs):
    transforms.append(kwargs["content_transform"])
weave_openhands.init = initialize
weave_openhands.finish = lambda: None
descriptor = int(os.environ["SENPAI_MODEL_CREDENTIALS_FD"])
import senpai_agent.openhands_runner as runner
assert Path(runner.__file__).resolve() == Path(os.environ["PYTHONPATH"]) / "senpai_agent/openhands_runner.py"

def inspect_runtime(prompt, config):
    assert "Inspect the private handoff." in prompt
    assert config.child
    assert config.github_token is None
    assert config.api_key.get_secret_value() == "anthropic-key"
    assert config.smart_api_key.get_secret_value() == "anthropic-key"
    assert config.fast_api_key.get_secret_value() == "anthropic-key"
    assert config.frontier_api_key.get_secret_value() == "openai-key"
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SENPAI_MODEL_CREDENTIALS_FD", "GITHUB_TOKEN"):
        assert name not in os.environ
    expected = {"WANDB_API_KEY": "wandb-service-key", "EXA_API_KEY": "exa-service-key", "PRIVATE_AUTH": "custom-service-key"}
    assert config.conversation_secrets == expected
    assert {name: os.environ[name] for name in expected} == expected
    assert len(transforms) == 1
    for credential in ("anthropic-key", "openai-key", *expected.values()):
        assert transforms[0]("private " + credential) == "private <secret-hidden>"
    try:
        os.fstat(descriptor)
    except OSError:
        pass
    else:
        raise AssertionError("model handoff FD remained open")
    print("OPENHANDS_RESULT " + json.dumps({"status": "finished", "result": "private startup verified"}))
    return 0

runner.run_openhands = inspect_runtime
raise SystemExit(runner.main(sys.argv[1:]))
"""
    monkeypatch.setattr(
        OpenHandsChildProcess,
        "command",
        property(lambda _self: (command[0], "-P", "-c", startup, *command[4:])),
    )

    assert child.run("Inspect the private handoff.", 30) == "private startup verified"


def test_controller_restores_private_services_before_tracing_import(tmp_path):
    descriptors = []
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "WANDB_ENTITY": "test-entity",
        "WANDB_PROJECT": "test-project",
    }
    for name in PRIVATE_CREDENTIAL_FD_ENVS:
        environment.pop(name, None)
    for credential, fd_name in PRIVATE_CREDENTIAL_FD_ENVS.items():
        read_fd, write_fd = os.pipe()
        os.write(write_fd, f"private-{credential}".encode())
        os.close(write_fd)
        descriptors.append(read_fd)
        environment[fd_name] = str(read_fd)
    startup = """
import os
import sys
from pathlib import Path
import weave_openhands

observed = []
def initialize(*args, **kwargs):
    observed.append({name: os.environ.get(name) for name in ("WANDB_API_KEY", "EXA_API_KEY")})
weave_openhands.init = initialize
weave_openhands.finish = lambda: None
descriptors = [int(os.environ[name]) for name in ("SENPAI_WANDB_API_KEY_FD", "SENPAI_EXA_API_KEY_FD")]
from senpai_agent import controller
assert Path(controller.__file__).resolve() == Path(os.environ["PYTHONPATH"]) / "senpai_agent/controller.py"
assert "senpai_agent.openhands_runner" not in sys.modules
try:
    controller.controller_main(["--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("controller help did not exit")
assert observed == [{"WANDB_API_KEY": "private-WANDB_API_KEY", "EXA_API_KEY": "private-EXA_API_KEY"}]
for name in ("SENPAI_WANDB_API_KEY_FD", "SENPAI_EXA_API_KEY_FD"):
    assert name not in os.environ
for descriptor in descriptors:
    try:
        os.fstat(descriptor)
    except OSError:
        pass
    else:
        raise AssertionError("service handoff FD remained open")
"""
    try:
        result = subprocess.run(
            (sys.executable, "-P", "-c", startup),
            env=environment,
            cwd=tmp_path,
            pass_fds=tuple(descriptors),
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)

    assert result.returncode == 0, result.stdout + result.stderr


def test_controller_rejects_empty_service_handoff_and_closes_its_fd(monkeypatch):
    from senpai_agent import controller

    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    monkeypatch.setattr(controller, "set_process_nondumpable", lambda: None)

    with pytest.raises(RuntimeError, match="SENPAI_EXA_API_KEY_FD is empty"):
        controller.controller_main(
            ["--help"],
            {"SENPAI_EXA_API_KEY_FD": str(read_fd)},
        )

    with pytest.raises(OSError):
        os.fstat(read_fd)
