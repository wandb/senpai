import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "k8s"))

import launch  # noqa: E402
import launch_helpers  # noqa: E402

REVISION = "a" * 40
ADVISOR_IMAGE = f"ghcr.io/wandb/senpai-advisor:sha-{REVISION}"
STUDENT_IMAGE = f"ghcr.io/wandb/senpai-student:sha-{REVISION}"
EXECUTOR_IMAGE = f"ghcr.io/wandb/senpai-executor@sha256:{'b' * 64}"


def launch_args(**overrides) -> launch.Args:
    values = {
        "tag": "test-track",
        "target_repo_url": "https://github.com/example/problem.git",
        "names": "fern",
        "advisor": True,
        "advisor_image": ADVISOR_IMAGE,
        "student_image": STUDENT_IMAGE,
        "executor_image": EXECUTOR_IMAGE,
        "senpai_repo_revision": REVISION,
    }
    values.update(overrides)
    return launch.Args(**values)


def run_launch(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "k8s" / "launch.py"),
            "--dry_run",
            "--tag",
            "image-split",
            "--target_repo_url",
            "https://github.com/example/problem.git",
            "--n_students",
            "1",
            "--advisor_image",
            ADVISOR_IMAGE,
            "--student_image",
            STUDENT_IMAGE,
            "--executor_image",
            EXECUTOR_IMAGE,
            "--senpai_repo_revision",
            REVISION,
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def render_role_manifest(
    role: str,
    args: launch.Args | None = None,
    *,
    program: launch.ProgramSystemPrompt | None = None,
) -> tuple[str, str]:
    args = launch_args() if args is None else args
    program = program or launch.ProgramSystemPrompt(
        program_path=args.program_path or "program.md",
        source_commit=REVISION,
        content="Test launch research policy.",
    )
    program_secret_name, program_secret = launch_helpers.render_program_context_secret(
        args.tag, launch.encode_program_system_prompt(program)
    )
    secret_name = f"senpai-launch-secrets-{args.tag}"
    providers = launch.deployed_model_providers(args)
    secret = launch_helpers.render_launch_secret(
        args.tag,
        "github",
        "exa",
        "wandb",
        anthropic_api_key="anthropic" if "anthropic" in providers else None,
        openai_api_key="openai" if "openai" in providers else None,
        custom_secrets={
            name: f"{name.lower()}-secret"
            for name in args.custom_secret_env_names
        },
    )
    template = (ROOT / "k8s" / f"{role}-deployment.yaml").read_text()
    if role == "student":
        manifest = launch.render_student(
            template,
            "fern",
            args.tag,
            secret_name,
            secret,
            args,
            program=program,
            program_secret_name=program_secret_name,
            program_secret=program_secret,
        )
    else:
        manifest = launch.render_advisor(
            template,
            args.tag,
            ["fern"],
            secret_name,
            secret,
            args,
            program=program,
            program_secret_name=program_secret_name,
            program_secret=program_secret,
        )
    return manifest, secret


def render_role(
    role: str,
    args: launch.Args | None = None,
    *,
    program: launch.ProgramSystemPrompt | None = None,
) -> tuple[str, str, str]:
    manifest, secret = render_role_manifest(role, args, program=program)
    documents = manifest.split("\n---\n")
    return documents[0], documents[-1], secret
