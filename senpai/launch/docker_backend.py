# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Docker Compose backend with file-mounted workload secrets."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

from .specs import RoleSpec

CONTAINER_COMMAND = """set -e
for secret_path in /run/secrets/*; do
  secret_name="${secret_path##*/}"
  export "${secret_name}=$(cat "$secret_path")"
done
repo_auth_url="$(printf '%s' "$REPO_URL" | sed "s#https://github.com/#https://${GITHUB_TOKEN}@github.com/#")"
git clone --branch "$REPO_BRANCH" --single-branch --depth 1 --no-tags "$repo_auth_url" /workspace/senpai
cd /workspace/senpai
git remote set-url origin "$REPO_URL"
exec bash "k8s/entrypoint-${SENPAI_ROLE}.sh"
"""


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "-", value.lower()).strip("-_")


def _gpu_assignments(args, role_specs: list[RoleSpec]) -> dict[str, list[str]]:
    students = [spec for spec in role_specs if spec.role == "student"]
    requested = args.gpus_per_student * len(students)
    configured = [gpu.strip() for gpu in args.docker_gpu_ids.split(",") if gpu.strip()]
    if args.gpus_per_student == 0:
        if configured:
            raise SystemExit(
                "ERROR: --docker_gpu_ids requires --gpus_per_student greater than 0"
            )
        return {}
    gpu_ids = configured or [str(index) for index in range(requested)]
    if len(gpu_ids) != requested:
        raise SystemExit(
            f"ERROR: Docker launch needs {requested} GPU IDs for {len(students)} students; "
            f"got {len(gpu_ids)} from --docker_gpu_ids"
        )
    assignments = {}
    for index, spec in enumerate(students):
        start = index * args.gpus_per_student
        assignments[spec.key] = gpu_ids[start : start + args.gpus_per_student]
    return assignments


def render_compose(args, role_specs: list[RoleSpec]) -> str:
    secret_names = sorted({name for spec in role_specs for name in spec.secrets})
    gpu_ids = _gpu_assignments(args, role_specs)
    services = {}
    for spec in role_specs:
        service = {
            "image": args.image,
            # Compose treats `$` as interpolation; `$$` passes it to the shell.
            "command": ["bash", "-lc", CONTAINER_COMMAND.replace("$", "$$")],
            "environment": {**spec.env, "SENPAI_ROLE": spec.role},
            "secrets": list(secret_names),
            "labels": {
                "app": "senpai",
                "role": spec.role,
                "research-tag": args.tag,
            },
        }
        if spec.role == "student":
            service["labels"]["student"] = spec.name
        if assigned := gpu_ids.get(spec.key):
            service["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": assigned,
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            }
        if args.docker_dataset_path:
            dataset_path = Path(args.docker_dataset_path).expanduser().resolve()
            service["volumes"] = [f"{dataset_path}:{args.pvc_mount_path}"]
        services[_safe_name(spec.key)] = service

    compose = {
        "services": services,
        "secrets": {name: {"environment": name} for name in secret_names},
    }
    return yaml.safe_dump(compose, sort_keys=False)


def launch_docker(args, role_specs: list[RoleSpec]) -> None:
    compose = render_compose(args, role_specs)
    if args.dry_run:
        print(compose, end="")
        return

    run_dir = Path(args.docker_run_root).expanduser().resolve() / args.tag
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    compose_path = run_dir / "compose.yaml"
    compose_path.write_text(compose)
    compose_path.chmod(0o600)

    secret_env = os.environ.copy()
    for spec in role_specs:
        secret_env.update(spec.secrets)

    project = _safe_name(f"senpai-{args.tag}")
    subprocess.run(
        [
            "docker",
            "compose",
            "--project-name",
            project,
            "--file",
            str(compose_path),
            "up",
            "--detach",
            "--force-recreate",
        ],
        check=True,
        env=secret_env,
    )
    print(f"\nLaunched Docker project {project}")
    print(f"\nMonitor:\n  docker compose -p {project} -f {compose_path} logs -f")
    print(f"\nStop:\n  docker compose -p {project} -f {compose_path} down")
