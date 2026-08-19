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
    secret_name = f"senpai-launch-secrets-{args.tag}"
    providers = launch.deployed_model_providers(args)
    secret = launch_helpers.render_launch_secret(
        args.tag,
        "github",
        "exa" if args.web_search else None,
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
        )
    else:
        manifest = launch.render_advisor(
            template,
            args.tag,
            ["fern"],
            secret_name,
            secret,
            args,
        )
    configmap, deployment = manifest.split("\n---\n", 1)
    return configmap, deployment, secret
