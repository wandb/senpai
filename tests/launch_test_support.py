import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "k8s"))

import launch  # noqa: E402
import launch_helpers  # noqa: E402

from senpai_agent.program_context import encode_program_system_prompt  # noqa: E402

REVISION = "a" * 40
ADVISOR_IMAGE = f"ghcr.io/wandb/senpai-advisor:sha-{REVISION}"
STUDENT_IMAGE = f"ghcr.io/wandb/senpai-student:sha-{REVISION}"


def launch_args(**overrides) -> launch.Args:
    values = {
        "tag": "test-track",
        "target_repo_url": "https://github.com/example/problem.git",
        "names": "fern",
        "advisor": True,
        "advisor_image": ADVISOR_IMAGE,
        "student_image": STUDENT_IMAGE,
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
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def render_role(role: str, args: launch.Args | None = None) -> tuple[str, str, str]:
    args = launch_args() if args is None else args
    program = launch.ProgramSystemPrompt(
        program_path=args.program_path or "program.md",
        source_commit=REVISION,
        content="Test launch research policy.",
    )
    providers = launch.deployed_model_providers(args)
    secret_name, secret = launch_helpers.render_launch_secret(
        args.tag,
        "github",
        "exa",
        "wandb",
        anthropic_api_key="anthropic" if "anthropic" in providers else None,
        openai_api_key="openai" if "openai" in providers else None,
        wandb_inference_api_key=(
            "wandb-inference" if "wandb" in providers else None
        ),
        custom_secrets={
            name: f"{name.lower()}-secret"
            for name in args.custom_secret_env_names
        },
    )
    program_secret_name, program_secret = (
        launch_helpers.render_program_context_secret(
            args.tag,
            encode_program_system_prompt(program),
        )
    )
    wandb_secret_name, wandb_secret = launch_helpers.render_student_wandb_secret(
        args.tag,
        "fern",
        "wandb-fern",
        "viewer-fern",
    )
    template = (ROOT / "k8s" / f"{role}-deployment.yaml").read_text()
    if role == "student":
        manifest = launch.render_student(
            template,
            "fern",
            args.tag,
            secret_name,
            secret,
            program_secret_name,
            program_secret,
            wandb_secret_name,
            wandb_secret,
            "viewer-fern",
            "viewer-controller",
            "viewer-inference" if "wandb" in providers else None,
            args,
            program,
        )
    else:
        manifest = launch.render_advisor(
            template,
            args.tag,
            ["fern"],
            secret_name,
            secret,
            program_secret_name,
            program_secret,
            "viewer-controller",
            "viewer-inference" if "wandb" in providers else None,
            args,
            program,
        )
    configmap, deployment = manifest.split("\n---\n", 1)
    return configmap, deployment, secret
