# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Backend-independent advisor and student launch specifications."""

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from senpai_agent.launch_context import render_launch_context

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
CONTAINER_STATE_ROOT = "/var/lib/senpai"
CONTAINER_GATE_ROOT = "/senpai-launch"
CONTAINER_WORKDIR = "/workspace/senpai"
CONTAINER_USER_ID = 10001
CONTAINER_IMAGE_GROUP_ID = 10001
CONTAINER_RESERVED_PATHS = (
    CONTAINER_STATE_ROOT,
    CONTAINER_GATE_ROOT,
    CONTAINER_WORKDIR,
)


@dataclass(frozen=True)
class RoleSpec:
    """Environment and secrets needed to start one Senpai role."""

    role: str
    name: str
    env: dict[str, str]
    secrets: dict[str, str]

    @property
    def key(self) -> str:
        return "advisor" if self.role == "advisor" else f"student-{self.name}"


def validate_identifier(kind: str, value: str) -> None:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{kind} {value!r} must be 1-63 characters using only letters, "
            "numbers, '.', '_', or '-', and must start with a letter or number"
        )


def validate_role_specs(backend: str, tag: str, role_specs: list[RoleSpec]) -> None:
    """Validate the role identities shared by non-Kubernetes launchers."""
    validate_identifier(f"{backend} tag", tag)
    if not role_specs:
        raise ValueError(f"{backend} launch has no advisor or students")

    keys: set[str] = set()
    advisor_count = 0
    for spec in role_specs:
        if spec.role not in {"advisor", "student"}:
            raise ValueError(f"unsupported {backend} role {spec.role!r}")
        if spec.role == "advisor":
            advisor_count += 1
            validate_identifier(f"{backend} advisor name", spec.name)
        else:
            validate_identifier(f"{backend} student name", spec.name)
        if spec.key in keys:
            raise ValueError(f"duplicate {backend} role {spec.key!r}")
        keys.add(spec.key)
    if advisor_count > 1:
        raise ValueError(f"{backend} launch can contain at most one advisor")


def validate_writable_parent(path: Path, label: str) -> None:
    """Fail before launch mutations when a local directory cannot be created."""
    parent = path
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"{label} cannot be created: {path}")


def validate_pvc_mount_path(value: str, reserved_paths: tuple[str, ...]) -> None:
    """Require a stable absolute mount target outside backend-owned paths."""
    mount_path = PurePosixPath(value)
    reserved = tuple(PurePosixPath(path) for path in reserved_paths)
    if (
        not mount_path.is_absolute()
        or ".." in mount_path.parts
        or any(
            mount_path == path
            or mount_path in path.parents
            or path in mount_path.parents
            for path in reserved
        )
    ):
        choices = ", ".join(reserved_paths)
        raise ValueError(
            f"pvc_mount_path must be an absolute path outside {choices}"
        )


def target_repo_slug(url: str) -> str:
    """Extract an owner/repo slug from an HTTPS or SSH GitHub URL."""
    return url.split("github.com", 1)[-1].lstrip(":/").removesuffix(".git")


def build_extra_instructions(args, tag: str, student_names: list[str]) -> str:
    """Render backend-neutral, authoritative context for both agent roles."""
    return render_launch_context(
        backend=args.backend,
        gpus_per_student=args.gpus_per_student,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        students=student_names,
        extra_instructions=args.extra_instructions,
    )


def _encoded_extra_instructions(args, tag: str, student_names: list[str]) -> str:
    return base64.b64encode(
        build_extra_instructions(args, tag, student_names).encode()
    ).decode()


def _common_env(args, tag: str) -> dict[str, str]:
    return {
        "REPO_URL": args.repo_url,
        "REPO_REVISION": args.repo_revision,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "RESEARCH_TAG": tag,
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": "online",
        "ADVISOR_BRANCH": args.advisor_branch,
        "GH_HISTORY_SCOPE": args.gh_history_scope,
        "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }


def role_model_config(args, role: str) -> dict[str, str]:
    """Return the OpenHands model profiles shared by every launch backend."""
    model = args.advisor_model if role == "advisor" else args.student_model
    reasoning_effort = (
        args.advisor_reasoning_effort
        if role == "advisor"
        else args.student_reasoning_effort
    )
    return {
        "SENPAI_OPENHANDS_MODEL": model,
        "SENPAI_OPENHANDS_REASONING_EFFORT": reasoning_effort,
        "SENPAI_OPENHANDS_SMART_MODEL": args.smart_model,
        "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": args.smart_reasoning_effort,
        "SENPAI_OPENHANDS_FAST_MODEL": args.fast_model,
        "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": args.fast_reasoning_effort,
        "SENPAI_OPENHANDS_FRONTIER_MODEL": args.frontier_model,
        "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": args.frontier_reasoning_effort,
        "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS": str(
            args.local_condenser_max_events
        ),
        "SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS": str(
            args.local_condenser_max_tokens
        ),
        "SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS": str(
            args.local_condenser_target_events
        ),
    }


def build_student_spec(
    args,
    tag: str,
    student_name: str,
    secrets: dict[str, str],
) -> RoleSpec:
    env = _common_env(args, tag)
    env.update(role_model_config(args, "student"))
    env.update(
        {
            "STUDENT_NAME": student_name,
            "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
            "SENPAI_MAX_EPOCHS": str(args.max_epochs),
            "EXTRA_INSTRUCTIONS_B64": _encoded_extra_instructions(
                args, tag, [student_name]
            ),
        }
    )
    return RoleSpec("student", student_name, env, secrets)


def build_advisor_spec(
    args,
    tag: str,
    student_names: list[str],
    secrets: dict[str, str],
) -> RoleSpec:
    env = _common_env(args, tag)
    env.update(role_model_config(args, "advisor"))
    env.update(
        {
            "ADVISOR_NAME": args.advisor_name,
            "STUDENT_NAMES": ",".join(student_names),
            "SENPAI_STALE_WIP_SECONDS": str(args.stale_wip_seconds),
            "EXTRA_INSTRUCTIONS_B64": _encoded_extra_instructions(
                args, tag, student_names
            ),
        }
    )
    return RoleSpec("advisor", args.advisor_name, env, secrets)
