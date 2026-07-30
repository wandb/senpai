# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run Senpai roles as Docker containers on one host."""

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .specs import (
    RoleSpec,
    validate_identifier,
    validate_role_specs,
    validate_writable_parent,
)

CONTAINER_RUN_ROOT = "/senpai-run"
CONTAINER_STATUS_ROOT = "/senpai-status"
CONTAINER_WORKDIR = "/workspace/senpai"
RUN_LABEL = "com.wandb.senpai.run"
ROLE_LABEL = "com.wandb.senpai.role"


@dataclass(frozen=True)
class DockerRolePlan:
    """Host paths and Docker command for one role."""

    spec: RoleSpec
    container_name: str
    role_root: Path
    state_root: Path
    workdir: Path
    env_file: Path
    ready_file: Path
    cid_file: Path
    status_dir: Path | None
    devices: tuple[str, ...]
    command: tuple[str, ...]


@dataclass(frozen=True)
class DockerLaunchPlan:
    """A validated Docker launch with all host paths resolved."""

    run_root: Path
    status_root: Path
    roles: tuple[DockerRolePlan, ...]


def _path_beneath(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Docker launch path escapes run root: {path}")
    return path


def _run_root(args) -> Path:
    base = Path(args.docker_run_root).expanduser().resolve()
    if base.exists() and not base.is_dir():
        raise ValueError(f"Docker run root is not a directory: {base}")
    candidate = base / args.tag
    if candidate.is_symlink():
        raise RuntimeError(
            f"Docker run already exists at {candidate}; use a new --tag"
        )
    run_root = candidate.resolve()
    if not run_root.is_relative_to(base) or run_root == base:
        raise ValueError(f"Docker launch path escapes configured run root: {run_root}")
    return run_root


def _check_run_root_available(run_root: Path) -> None:
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(
            f"Docker run already exists at {run_root}; use a new --tag"
        )


def _student_gpu_map(raw: str) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    for raw_assignment in raw.split(","):
        assignment = raw_assignment.strip()
        if not assignment:
            continue
        if ":" not in assignment:
            raise ValueError(
                f"invalid GPU assignment {assignment!r}; expected student:gpu"
            )
        student, raw_devices = assignment.split(":", 1)
        student = student.strip()
        validate_identifier("GPU assignment student", student)
        devices = [
            device.strip() for device in raw_devices.split("+") if device.strip()
        ]
        if not devices or any(not device.isdecimal() for device in devices):
            raise ValueError(
                f"invalid GPU assignment {assignment!r}; GPU IDs must be "
                "non-negative indices"
            )
        if student in assignments:
            raise ValueError(f"duplicate GPU assignment for student {student!r}")
        assignments[student] = devices
    return assignments


def _gpu_assignments(
    args,
    role_specs: list[RoleSpec],
    available_gpu_ids: list[str] | None,
) -> dict[str, list[str]]:
    students = [spec.name for spec in role_specs if spec.role == "student"]
    explicit = _student_gpu_map(args.docker_student_gpu_ids)
    if args.gpus_per_student == 0:
        if explicit:
            raise ValueError(
                "--docker_student_gpu_ids requires --gpus_per_student greater than 0"
            )
        return {}
    if not students:
        if explicit:
            raise ValueError("Docker GPU assignments require at least one student")
        return {}

    if explicit:
        missing = [student for student in students if student not in explicit]
        extra = [student for student in explicit if student not in students]
        if missing:
            raise ValueError(
                f"missing Docker GPU assignments for: {', '.join(missing)}"
            )
        if extra:
            raise ValueError(
                "GPU assignments name students outside this launch: "
                + ", ".join(extra)
            )
        wrong_count = {
            student: devices
            for student, devices in explicit.items()
            if len(devices) != args.gpus_per_student
        }
        if wrong_count:
            details = ", ".join(
                f"{student}:{'+'.join(devices)}"
                for student, devices in wrong_count.items()
            )
            raise ValueError(
                f"--gpus_per_student={args.gpus_per_student}, but these "
                f"assignments differ: {details}"
            )
        assignments = explicit
    else:
        required = len(students) * args.gpus_per_student
        gpu_ids = (
            available_gpu_ids
            if available_gpu_ids is not None
            else [str(index) for index in range(required)]
        )
        if len(gpu_ids) < required:
            raise RuntimeError(
                f"Docker launch needs {required} GPUs, but only "
                f"{len(gpu_ids)} are visible"
            )
        assignments = {
            student: gpu_ids[offset : offset + args.gpus_per_student]
            for student, offset in zip(
                students,
                range(0, required, args.gpus_per_student),
                strict=True,
            )
        }

    device_owners: dict[str, str] = {}
    duplicates = []
    for student, devices in assignments.items():
        for device in devices:
            if device in device_owners:
                duplicates.append(
                    f"{device} assigned to {device_owners[device]} and {student}"
                )
            device_owners[device] = student
    if duplicates:
        raise ValueError(
            "Docker GPU assignments must be exclusive: " + "; ".join(duplicates)
        )

    if available_gpu_ids is not None:
        unavailable = [
            device
            for devices in assignments.values()
            for device in devices
            if device not in available_gpu_ids
        ]
        if unavailable:
            raise RuntimeError(
                "Docker GPU indices are not visible: "
                + ", ".join(dict.fromkeys(unavailable))
            )
    return assignments


def _container_name(tag: str, role_key: str) -> str:
    return f"senpai-{tag}-{role_key}"


def _docker_gpu_args(devices: tuple[str, ...]) -> list[str]:
    if not devices:
        return []
    return ["--gpus", f"device={','.join(devices)}"]


def _role_values(spec: RoleSpec) -> dict[str, str]:
    values = {
        **spec.env,
        **{key: value for key, value in spec.secrets.items() if value},
    }
    values.update(
        {
            "HOME": f"{CONTAINER_RUN_ROOT}/home",
            "SENPAI_BACKEND": "docker",
            "SENPAI_GIT_CREDENTIAL_FILE": (
                f"{CONTAINER_RUN_ROOT}/secrets/git-credentials"
            ),
            "SENPAI_LAUNCH_GATE_PATH": f"{CONTAINER_STATUS_ROOT}/.launch",
            "SENPAI_LOGDIR": f"{CONTAINER_RUN_ROOT}/logs/iterations",
            "SENPAI_READY_FILE": f"{CONTAINER_RUN_ROOT}/ready",
            "SENPAI_RUN_ROOT": CONTAINER_RUN_ROOT,
            "SENPAI_STATUS_DIR": CONTAINER_STATUS_ROOT,
            "SENPAI_WORKDIR": CONTAINER_WORKDIR,
        }
    )
    return values


def _env_file_text(values: dict[str, str]) -> str:
    lines = []
    for key in sorted(values):
        value = values[key]
        if "\n" in value or "\r" in value:
            raise ValueError(f"Docker environment value {key} contains a newline")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _docker_command(
    args,
    spec: RoleSpec,
    state_root: Path,
    status_root: Path,
    status_dir: Path | None,
    workdir: Path,
    env_file: Path,
    cid_file: Path,
    devices: tuple[str, ...],
) -> list[str]:
    command = [
        "docker",
        "run",
        "--detach",
        "--init",
        "--restart",
        "unless-stopped",
        "--name",
        _container_name(args.tag, spec.key),
        "--cidfile",
        str(cid_file),
        "--label",
        f"{RUN_LABEL}={args.tag}",
        "--label",
        f"{ROLE_LABEL}={spec.key}",
        "--workdir",
        CONTAINER_WORKDIR,
        "--env-file",
        str(env_file),
        "--volume",
        f"{workdir}:{CONTAINER_WORKDIR}",
        "--volume",
        f"{state_root}:{CONTAINER_RUN_ROOT}",
        "--volume",
        f"{status_root}:{CONTAINER_STATUS_ROOT}:ro",
    ]
    if status_dir is not None:
        command.extend(
            [
                "--volume",
                f"{status_dir}:{CONTAINER_STATUS_ROOT}/{spec.name}",
            ]
        )
    if args.docker_data_dir:
        data_dir = Path(args.docker_data_dir).expanduser().resolve()
        command.extend(
            [
                "--volume",
                f"{data_dir}:{args.pvc_mount_path}",
            ]
        )
    if spec.role == "student":
        command.extend(["--shm-size", args.docker_shm_size])
        command.extend(_docker_gpu_args(devices))
    script = (
        "k8s/entrypoint-advisor.sh"
        if spec.role == "advisor"
        else "k8s/entrypoint-student.sh"
    )
    command.extend([args.image, "bash", script])
    return command


def plan_docker(
    args,
    role_specs: list[RoleSpec],
    available_gpu_ids: list[str] | None = None,
) -> DockerLaunchPlan:
    """Build a Docker launch plan without writing files or calling Docker."""
    validate_role_specs("Docker", args.tag, role_specs)
    if args.gpus_per_student < 0:
        raise ValueError("--gpus_per_student must be non-negative")
    if args.docker_ready_timeout_s <= 0:
        raise ValueError("--docker_ready_timeout_s must be greater than 0")

    run_root = _run_root(args)
    _check_run_root_available(run_root)
    run_base = run_root.parent
    validate_writable_parent(run_base, "Docker run root")
    status_root = _path_beneath(run_root, "status")
    if args.docker_data_dir:
        data_dir = Path(args.docker_data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            raise ValueError(f"Docker data directory does not exist: {data_dir}")
        if data_dir.is_relative_to(run_base) or run_base.is_relative_to(data_dir):
            raise ValueError(
                f"Docker data directory overlaps private run state: {data_dir}"
            )
        mount_path = PurePosixPath(args.pvc_mount_path)
        reserved = tuple(
            PurePosixPath(path)
            for path in (CONTAINER_RUN_ROOT, CONTAINER_STATUS_ROOT, CONTAINER_WORKDIR)
        )
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
            raise ValueError(
                "Docker pvc_mount_path must be an absolute path outside "
                f"{CONTAINER_RUN_ROOT}, {CONTAINER_STATUS_ROOT}, and "
                f"{CONTAINER_WORKDIR}"
            )

    assignments = _gpu_assignments(args, role_specs, available_gpu_ids)
    roles = []
    for spec in role_specs:
        role_root = _path_beneath(run_root, "roles", spec.key)
        state_root = _path_beneath(role_root, "state")
        workdir = _path_beneath(role_root, "workdir")
        env_file = _path_beneath(role_root, "role.env")
        ready_file = _path_beneath(state_root, "ready")
        cid_file = _path_beneath(role_root, "container.cid")
        status_dir = (
            _path_beneath(status_root, spec.name)
            if spec.role == "student"
            else None
        )
        devices = tuple(assignments.get(spec.name, ()))
        command = _docker_command(
            args,
            spec,
            state_root,
            status_root,
            status_dir,
            workdir,
            env_file,
            cid_file,
            devices,
        )
        roles.append(
            DockerRolePlan(
                spec=spec,
                container_name=_container_name(args.tag, spec.key),
                role_root=role_root,
                state_root=state_root,
                workdir=workdir,
                env_file=env_file,
                ready_file=ready_file,
                cid_file=cid_file,
                status_dir=status_dir,
                devices=devices,
                command=tuple(command),
            )
        )
    return DockerLaunchPlan(run_root, status_root, tuple(roles))


def _prepare_runner_workdir(repo_url: str, workdir: Path, branch: str) -> None:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            repo_url,
            str(workdir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not clone runner branch {branch!r} from {repo_url}: "
            f"{result.stderr.strip()}"
        )


def _check_docker() -> None:
    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed or is not on PATH")
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Docker daemon is unavailable: {result.stderr.strip()}")


def _check_runner_source(repo_url: str, branch: str) -> None:
    result = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            repo_url,
            f"refs/heads/{branch}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Runner branch {branch!r} is not available from {repo_url}: "
            f"{result.stderr.strip()}"
        )


def _docker_gpu_indices(image: str) -> list[str]:
    cuda_smoke = """\
import subprocess
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")
for index in range(torch.cuda.device_count()):
    assert torch.ones(1, device=f"cuda:{index}").item() == 1
print(
    subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    ),
    end="",
)
"""
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            image,
            "-c",
            cuda_smoke,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "Docker cannot run PyTorch CUDA with the Senpai image: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    indices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not index.isdecimal() for index in indices) or len(set(indices)) != len(
        indices
    ):
        raise RuntimeError(f"Unexpected GPU indices from the Senpai image: {indices}")
    return indices


def _check_image(image: str) -> None:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            image,
            "-c",
            "true",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Docker cannot run the Senpai image {image!r}: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def _check_container_names_available(names: list[str]) -> None:
    for name in names:
        result = subprocess.run(
            ["docker", "container", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            raise RuntimeError(
                f"Docker container {name!r} already exists; "
                "remove it or use a new --tag"
            )


def preflight_docker(args, role_specs: list[RoleSpec]) -> DockerLaunchPlan:
    """Validate a real Docker launch without changing host or GitHub state."""
    preview = plan_docker(args, role_specs)
    _check_runner_source(args.repo_url, args.repo_branch)
    _check_docker()
    _check_container_names_available(
        [role.container_name for role in preview.roles]
    )
    needs_gpus = any(role.devices for role in preview.roles)
    check = "image and GPU access" if needs_gpus else "image"
    print(
        f"Checking Docker {check} with {args.image!r}; "
        "the first pull can take several minutes.",
        flush=True,
    )
    if needs_gpus:
        gpu_ids = _docker_gpu_indices(args.image)
    else:
        _check_image(args.image)
        gpu_ids = None
    return plan_docker(args, role_specs, gpu_ids)


def _create_role_files(args, role: DockerRolePlan) -> None:
    role.state_root.mkdir(parents=True)
    (role.state_root / "home").mkdir()
    (role.state_root / "logs" / "iterations").mkdir(parents=True)
    (role.state_root / "secrets").mkdir()
    if role.status_dir is not None:
        role.status_dir.mkdir(parents=True)
    _prepare_runner_workdir(args.repo_url, role.workdir, args.repo_branch)
    role.env_file.write_text(
        _env_file_text(_role_values(role.spec)),
        encoding="utf-8",
    )
    role.env_file.chmod(0o600)


def _container_state(name: str) -> dict:
    result = subprocess.run(
        ["docker", "container", "inspect", "--format", "{{json .State}}", name],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Docker container {name!r} disappeared during startup")
    return json.loads(result.stdout)


def _wait_until_ready(plan: DockerLaunchPlan, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        pending = []
        for role in plan.roles:
            name = role.container_name
            state = _container_state(name)
            status = state["Status"]
            if state.get("Restarting") or status in {
                "dead",
                "exited",
                "removing",
                "restarting",
            }:
                raise RuntimeError(
                    f"Docker container {name!r} failed before becoming ready "
                    f"(status={status}, exit={state.get('ExitCode')})"
                )
            if not role.ready_file.is_file():
                pending.append(name)
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            names = ", ".join(pending)
            raise RuntimeError(
                f"Timed out after {timeout_s:g}s waiting for Docker roles: {names}"
            )
        time.sleep(min(0.5, remaining))


def _container_logs(container_ids: list[str]) -> str:
    sections = []
    for container_id in container_ids:
        result = subprocess.run(
            ["docker", "logs", "--tail", "80", container_id],
            capture_output=True,
            text=True,
        )
        logs = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and logs:
            sections.append(f"--- Docker logs {container_id[:12]} ---\n{logs}")
    return "\n".join(sections)


def _remove_containers(container_ids: list[str]) -> list[str]:
    failures = []
    for container_id in reversed(container_ids):
        result = subprocess.run(
            ["docker", "rm", "--force", container_id],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            failures.append(
                f"{container_id[:12]}: {result.stderr.strip() or 'docker rm failed'}"
            )
    return failures


def _created_container_ids(
    started_ids: list[str],
    attempted_roles: list[DockerRolePlan],
) -> list[str]:
    ids = [container_id for container_id in started_ids if container_id]
    ids.extend(
        role.cid_file.read_text().strip()
        for role in attempted_roles
        if role.cid_file.is_file()
    )
    return list(dict.fromkeys(container_id for container_id in ids if container_id))


def _print_plan(args, plan: DockerLaunchPlan) -> None:
    print(f"Docker Senpai run: {plan.run_root}")
    print(f"Runner source: {args.repo_url} ({args.repo_branch})")
    print(f"Image: {args.image}")
    for role in plan.roles:
        print(f"\n--- Docker {role.spec.key} ---")
        print(f"workdir: {role.workdir}")
        print(f"state:   {role.state_root}")
        print(f"env:     {role.env_file} (credentials redacted)")
        print(f"command: {shlex.join(role.command)}")


def launch_docker(
    args,
    role_specs: list[RoleSpec],
    plan: DockerLaunchPlan | None = None,
) -> None:
    """Launch all roles together and open their shared gate once ready."""
    if args.dry_run:
        _print_plan(args, plan or plan_docker(args, role_specs))
        return

    plan = plan or preflight_docker(args, role_specs)
    _check_run_root_available(plan.run_root)
    plan.run_root.mkdir(parents=True)
    plan.run_root.chmod(0o700)
    plan.status_root.mkdir()
    attempted_roles: list[DockerRolePlan] = []
    started_ids: list[str] = []
    try:
        for role in plan.roles:
            _create_role_files(args, role)
        for role in plan.roles:
            attempted_roles.append(role)
            result = subprocess.run(
                role.command,
                cwd=role.workdir,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Docker failed to start {role.spec.key}: "
                    f"{result.stderr.strip()}"
                )
            container_id = result.stdout.strip()
            started_ids.append(container_id)
            print(
                f"Launched {role.spec.key}: container={role.container_name} "
                f"id={container_id[:12]}"
            )

        _wait_until_ready(plan, args.docker_ready_timeout_s)
        (plan.status_root / ".launch").touch(exist_ok=False)
    except BaseException as error:
        created_ids = _created_container_ids(started_ids, attempted_roles)
        logs = ""
        logs = _container_logs(created_ids)
        cleanup_failures = _remove_containers(created_ids)
        if not cleanup_failures:
            try:
                shutil.rmtree(plan.run_root)
            except OSError as cleanup_error:
                cleanup_failures.append(
                    f"could not remove run state at {plan.run_root}: {cleanup_error}"
                )
        if isinstance(error, Exception):
            detail = f"\n{logs}" if logs else ""
            if cleanup_failures:
                detail += (
                    f"\nDocker rollback was incomplete; inspect run state at "
                    f"{plan.run_root}:\n" + "\n".join(cleanup_failures)
                )
            raise RuntimeError(f"{error}{detail}") from error
        if cleanup_failures:
            print(
                f"Docker rollback was incomplete; inspect run state at "
                f"{plan.run_root}",
            )
        raise

    names = [role.container_name for role in plan.roles]
    print("\nAll roles are ready; launch gate opened.")
    print("\nStatus:")
    print(f"  docker ps --all --filter label={RUN_LABEL}={args.tag}")
    print("\nLogs:")
    print(f"  docker logs --follow {names[0]}")
    print("\nStop and remove:")
    print(f"  docker rm --force {' '.join(names)}")
