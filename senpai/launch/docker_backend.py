# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run Senpai roles as Docker containers on one host."""

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .specs import (
    CONTAINER_GATE_ROOT,
    CONTAINER_IMAGE_GROUP_ID,
    CONTAINER_RESERVED_PATHS,
    CONTAINER_STATE_ROOT,
    CONTAINER_USER_ID,
    CONTAINER_WORKDIR,
    RoleSpec,
    validate_identifier,
    validate_pvc_mount_path,
    validate_role_specs,
    validate_writable_parent,
)

RUN_LABEL = "com.wandb.senpai.run"
ROLE_LABEL = "com.wandb.senpai.role"
DEFAULT_DOCKER_RUN_ROOT = "~/.senpai/runs"
DOCKER_PULL_ATTEMPTS = 3


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
    devices: tuple[str, ...]
    command: tuple[str, ...]


@dataclass(frozen=True)
class DockerLaunchPlan:
    """A validated Docker launch with all host paths resolved."""

    run_root: Path
    gate_root: Path
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
    value = f"device={','.join(devices)}"
    if len(devices) > 1:
        value = f'"{value}"'
    return ["--gpus", value]


def _role_image(args, spec: RoleSpec) -> str:
    return args.advisor_image if spec.role == "advisor" else args.student_image


def _supplemental_group_args(paths: list[Path]) -> list[str]:
    """Add each mounted path's group, using its nearest existing parent in plans."""
    group_ids: dict[int, None] = {}
    for path in paths:
        existing = path
        while not existing.exists():
            existing = existing.parent
        group_ids[existing.stat().st_gid] = None
    return [value for gid in group_ids for value in ("--group-add", str(gid))]


def _role_values(spec: RoleSpec) -> dict[str, str]:
    values = {
        **spec.env,
        **{key: value for key, value in spec.secrets.items() if value},
    }
    values.update(
        {
            "HOME": f"{CONTAINER_STATE_ROOT}/home",
            "SENPAI_BACKEND": "docker",
            "SENPAI_LAUNCH_GATE_PATH": f"{CONTAINER_GATE_ROOT}/.launch",
            "SENPAI_UMASK": "0002",
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
    gate_root: Path,
    workdir: Path,
    env_file: Path,
    cid_file: Path,
    devices: tuple[str, ...],
) -> list[str]:
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else None
    mount_sources = [workdir, state_root, gate_root]
    if data_dir is not None:
        mount_sources.append(data_dir)
    data_identity = (
        [
            "--user",
            f"{CONTAINER_USER_ID}:{data_dir.stat().st_gid}",
            "--group-add",
            str(CONTAINER_IMAGE_GROUP_ID),
        ]
        if data_dir is not None
        else []
    )
    command = [
        "docker",
        "run",
        "--detach",
        "--init",
        "--restart",
        "unless-stopped",
        "--stop-timeout",
        "90",
        *data_identity,
        *_supplemental_group_args(mount_sources),
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
        f"{state_root}:{CONTAINER_STATE_ROOT}",
        "--volume",
        f"{gate_root}:{CONTAINER_GATE_ROOT}:ro",
    ]
    if data_dir is not None:
        command.extend(
            [
                "--volume",
                f"{data_dir}:{args.pvc_mount_path}",
            ]
        )
    if spec.role == "student":
        if args.gpus_per_student > 0:
            command.extend(
                [
                    "--cpus",
                    str(args.cpu_per_gpu * args.gpus_per_student),
                    "--memory",
                    f"{args.memory_gi_per_gpu * args.gpus_per_student}g",
                ]
            )
        command.extend(["--shm-size", args.docker_shm_size])
        command.extend(_docker_gpu_args(devices))
    script = (
        "k8s/entrypoint-advisor.sh"
        if spec.role == "advisor"
        else "k8s/entrypoint-student.sh"
    )
    command.extend([_role_image(args, spec), "bash", script])
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
    gate_root = _path_beneath(run_root, "gate")
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            raise ValueError(f"Docker data directory does not exist: {data_dir}")
        if data_dir in {Path(data_dir.anchor), Path.home().resolve()}:
            raise ValueError("Docker data directory must not be / or the home directory")
        if data_dir.is_relative_to(run_base) or run_base.is_relative_to(data_dir):
            raise ValueError(
                f"Docker data directory overlaps private run state: {data_dir}"
            )
        validate_pvc_mount_path(
            args.pvc_mount_path,
            CONTAINER_RESERVED_PATHS,
        )
    elif args.start_gate_path:
        raise ValueError(
            "Docker --start_gate_path requires --data_dir so the gate "
            "is visible inside every role"
        )

    assignments = _gpu_assignments(args, role_specs, available_gpu_ids)
    roles = []
    for spec in role_specs:
        role_root = _path_beneath(run_root, "roles", spec.key)
        state_root = _path_beneath(role_root, "state")
        workdir = _path_beneath(role_root, "workdir")
        env_file = _path_beneath(role_root, "role.env")
        ready_parts = (
            (args.tag, "advisor", "openhands_state", "controller-lease.json")
            if spec.role == "advisor"
            else ("openhands_state", "controller-lease.json")
        )
        ready_file = _path_beneath(state_root, *ready_parts)
        cid_file = _path_beneath(role_root, "container.cid")
        devices = tuple(assignments.get(spec.name, ()))
        command = _docker_command(
            args,
            spec,
            state_root,
            gate_root,
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
                devices=devices,
                command=tuple(command),
            )
        )
    return DockerLaunchPlan(run_root, gate_root, tuple(roles))


def _prepare_runner_workdir(repo_url: str, workdir: Path, revision: str) -> None:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            repo_url,
            str(workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not clone runner source from {repo_url}: "
            f"{result.stderr.strip()}"
        )
    result = subprocess.run(
        ["git", "-C", str(workdir), "checkout", "--detach", revision],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not check out runner revision {revision!r} from {repo_url}: "
            f"{result.stderr.strip()}"
        )


def _check_docker() -> None:
    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed or is not on PATH")
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"Docker daemon is unavailable: {result.stderr.strip()}")


def _check_runner_source(repo_url: str, revision: str) -> None:
    with tempfile.TemporaryDirectory(prefix="senpai-source-check-") as tmp:
        checkout = Path(tmp) / "runner"
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                repo_url,
                str(checkout),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone.returncode:
            raise RuntimeError(
                f"Runner revision {revision!r} cannot be checked out from "
                f"{repo_url}: {clone.stderr.strip()}"
            )

        reference = (
            revision
            if re.fullmatch(r"[0-9a-f]{40}", revision)
            else f"refs/remotes/origin/{revision}"
        )
        for action, command in (
            (
                "resolve",
                [
                    "git",
                    "-C",
                    str(checkout),
                    "cat-file",
                    "-e",
                    f"{reference}^{{commit}}",
                ],
            ),
            (
                "check out",
                ["git", "-C", str(checkout), "checkout", "--detach", reference],
            ),
        ):
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"Runner revision {revision!r} cannot be checked out from "
                    f"{repo_url}: could not {action} it: {detail}"
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
        check=False,
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


def _pull_image(image: str) -> None:
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode == 0:
        return

    inspect_detail = "\n".join(
        output
        for output in (inspect.stderr.strip(), inspect.stdout.strip())
        if output
    ) or "<no error detail>"
    if not any(
        missing in inspect_detail for missing in ("No such image", "No such object")
    ):
        raise RuntimeError(
            f"Docker cannot inspect cached image {image!r}: {inspect_detail}"
        )

    for attempt in range(1, DOCKER_PULL_ATTEMPTS + 1):
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return

        detail = "\n".join(
            output
            for output in (result.stderr.strip(), result.stdout.strip())
            if output
        ) or "<no error detail>"
        connection_reset = "connection reset by peer" in detail.lower()
        if not connection_reset:
            raise RuntimeError(f"Docker cannot pull {image!r}: {detail}")
        if attempt < DOCKER_PULL_ATTEMPTS:
            print(
                f"Docker registry connection reset while pulling {image!r}; "
                f"retrying ({attempt + 1}/{DOCKER_PULL_ATTEMPTS}).",
                flush=True,
            )
            continue
        raise RuntimeError(
            f"Docker cannot pull {image!r} after {DOCKER_PULL_ATTEMPTS} "
            f"connection-reset attempts: {detail}"
        )


def _check_image(image: str, revision: str) -> None:
    _pull_image(image)
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--entrypoint",
            "/bin/bash",
            image,
            "-c",
            'test "$SENPAI_IMAGE_REVISION" = "$1"',
            "senpai-image-check",
            revision,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Docker cannot run {image!r} at source revision {revision}: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def _check_data_mount(role: str, image: str, data_dir: Path, mount_path: str) -> None:
    group_id = data_dir.stat().st_gid
    access_check = r"""
test -r "$1" && test -x "$1" && test -w "$1" &&
test -z "$(find "$1" -type d \( ! -readable -o ! -executable \) -print -quit)" &&
test -z "$(find "$1" -type f ! -readable -print -quit)"
"""
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{CONTAINER_USER_ID}:{group_id}",
            "--group-add",
            str(CONTAINER_IMAGE_GROUP_ID),
            "--volume",
            f"{data_dir}:{mount_path}:rw",
            "--entrypoint",
            "/bin/bash",
            image,
            "-c",
            access_check,
            "senpai-data-check",
            mount_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Docker {role} image {image!r} cannot read, traverse, and write "
            f"data directory {data_dir} as its runtime user with host group GID "
            f"{group_id}: {detail}. Grant that group directory r/w/x access "
            "through group ownership or an ACL; Senpai did not modify user data."
        )


def _check_container_names_available(names: list[str]) -> None:
    for name in names:
        result = subprocess.run(
            ["docker", "container", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            raise RuntimeError(
                f"Docker container {name!r} already exists; "
                "remove it or use a new --tag"
            )


def preflight_docker(args, role_specs: list[RoleSpec]) -> DockerLaunchPlan:
    """Validate a real Docker launch without changing host or GitHub state."""
    preview = plan_docker(args, role_specs)
    _check_runner_source(args.repo_url, args.repo_revision)
    _check_docker()
    _check_container_names_available(
        [role.container_name for role in preview.roles]
    )
    has_advisor = any(role.spec.role == "advisor" for role in preview.roles)
    has_students = any(role.spec.role == "student" for role in preview.roles)
    needs_gpus = any(role.devices for role in preview.roles)
    check = "images and GPU access" if needs_gpus else "images"
    print(
        f"Checking Docker {check}; the first pull can take several minutes.",
        flush=True,
    )
    if has_advisor:
        _check_image(args.advisor_image, args.repo_revision)
    if has_students:
        _check_image(args.student_image, args.repo_revision)
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if has_advisor:
            _check_data_mount(
                "advisor",
                args.advisor_image,
                data_dir,
                args.pvc_mount_path,
            )
        if has_students:
            _check_data_mount(
                "student",
                args.student_image,
                data_dir,
                args.pvc_mount_path,
            )
    if needs_gpus:
        gpu_ids = _docker_gpu_indices(args.student_image)
    else:
        gpu_ids = None
    return plan_docker(args, role_specs, gpu_ids)


def _share_with_container_group(path: Path) -> None:
    """Keep host ownership while granting the container's supplemental group access."""
    for root, directories, files in os.walk(path):
        root_path = Path(root)
        root_path.chmod(root_path.stat().st_mode | 0o2070)
        for name in directories:
            directory = root_path / name
            if not directory.is_symlink():
                directory.chmod(directory.stat().st_mode | 0o2070)
        for name in files:
            file = root_path / name
            if not file.is_symlink():
                file.chmod(file.stat().st_mode | 0o060)


def _create_role_files(args, role: DockerRolePlan) -> None:
    role.state_root.mkdir(parents=True)
    (role.state_root / "home").mkdir()
    _prepare_runner_workdir(args.repo_url, role.workdir, args.repo_revision)
    _share_with_container_group(role.state_root)
    _share_with_container_group(role.workdir)
    role.env_file.write_text(
        _env_file_text(_role_values(role.spec)),
        encoding="utf-8",
    )
    role.env_file.chmod(0o600)


def _container_details(name: str) -> dict:
    result = subprocess.run(
        ["docker", "container", "inspect", "--format", "{{json .}}", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Docker container {name!r} disappeared during startup")
    return json.loads(result.stdout)


def _container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "container", "inspect", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    detail = result.stderr.strip() or result.stdout.strip()
    if "No such container" in detail or "No such object" in detail:
        return False
    raise RuntimeError(
        f"Could not determine whether Docker container {name!r} exists: "
        f"{detail or '<no error detail>'}. Run state was preserved."
    )


def _container_state(name: str) -> dict:
    return _container_details(name)["State"]


def _lease_is_waiting_at_gate(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            int(value["pid"]) > 0
            and value["phase"] == "start-gate"
            and float(value["deadline"]) > 0
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


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
            if not _lease_is_waiting_at_gate(role.ready_file):
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
            check=False,
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
            check=False,
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


def _write_manifest(args, plan: DockerLaunchPlan) -> None:
    path = plan.run_root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "roles": [
                    {
                        "key": role.spec.key,
                        "container": role.container_name,
                    }
                    for role in plan.roles
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _load_manifest(tag: str, run_root: str) -> tuple[Path, dict]:
    validate_identifier("Docker tag", tag)
    base = Path(run_root).expanduser().resolve()
    path = (base / tag).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError(f"Docker run path escapes configured root: {path}")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No Docker Senpai run {tag!r} found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tag") != tag:
        raise RuntimeError(f"Docker run manifest tag does not match {tag!r}")
    return path, manifest


def status_docker(tag: str, run_root: str = "~/.senpai/runs") -> None:
    """Print container and OpenHands supervisor health for one Docker run."""
    _, manifest = _load_manifest(tag, run_root)
    for role in manifest["roles"]:
        details = _container_details(role["container"])
        state = details["State"]
        health = state.get("Health", {}).get("Status", "none")
        print(
            f"{role['key']}: container={role['container']} "
            f"state={state['Status']} health={health} "
            f"restarts={details['RestartCount']}"
        )


def logs_docker(
    tag: str,
    run_root: str = "~/.senpai/runs",
    *,
    role_key: str = "",
    follow: bool = False,
    tail: int = 200,
) -> None:
    """Print bounded logs for one role, optionally following the stream."""
    if tail < 1:
        raise ValueError("Docker log tail must be at least 1")
    _, manifest = _load_manifest(tag, run_root)
    roles = manifest["roles"]
    role = next(
        (value for value in roles if value["key"] == role_key),
        roles[0] if not role_key and roles else None,
    )
    if role is None:
        choices = ", ".join(value["key"] for value in roles)
        raise ValueError(f"Unknown Docker role {role_key!r}; choose one of: {choices}")
    command = ["docker", "logs", "--tail", str(tail)]
    if follow:
        command.append("--follow")
    command.append(role["container"])
    subprocess.run(command, check=True)


def terminate_docker(tag: str, run_root: str = "~/.senpai/runs") -> None:
    """Gracefully stop one recorded run, remove its containers and credentials."""
    path, manifest = _load_manifest(tag, run_root)
    names = [role["container"] for role in manifest["roles"]]
    existing = [name for name in names if _container_exists(name)]
    if existing:
        subprocess.run(["docker", "stop", "--time", "90", *existing], check=True)
        subprocess.run(["docker", "rm", *existing], check=True)
    shutil.rmtree(path)
    print(f"Stopped Docker Senpai run {tag!r} and removed its private run state.")


def _print_plan(args, plan: DockerLaunchPlan) -> None:
    print(f"Docker Senpai run: {plan.run_root}")
    print(f"Runner source: {args.repo_url} ({args.repo_revision})")
    print(f"Advisor image: {args.advisor_image}")
    print(f"Student image: {args.student_image}")
    for role in plan.roles:
        print(f"\n--- Docker {role.spec.key} ---")
        print(f"workdir: {role.workdir}")
        print(f"state:   {role.state_root}")
        print(f"env:     {role.env_file} (credentials redacted)")
        print(f"command: {shlex.join(role.command)}")


def _runtime_command(args, role: DockerRolePlan, gate_root: Path) -> list[str]:
    return _docker_command(
        args,
        role.spec,
        role.state_root,
        gate_root,
        role.workdir,
        role.env_file,
        role.cid_file,
        role.devices,
    )


def _lifecycle_command(args, action: str, *options: str) -> str:
    command = ["python3", "k8s/docker.py", action, args.tag, *options]
    run_root = Path(args.docker_run_root).expanduser().resolve()
    default_root = Path(DEFAULT_DOCKER_RUN_ROOT).expanduser().resolve()
    if run_root != default_root:
        command.extend(["--run-root", str(run_root)])
    return shlex.join(command)


def launch_docker(
    args,
    role_specs: list[RoleSpec],
    plan: DockerLaunchPlan | None = None,
    *,
    show_lifecycle: bool = True,
) -> None:
    """Launch all roles together and open their shared gate once ready."""
    if args.dry_run:
        _print_plan(args, plan or plan_docker(args, role_specs))
        return

    plan = plan or preflight_docker(args, role_specs)
    _check_run_root_available(plan.run_root)
    plan.run_root.mkdir(parents=True)
    plan.run_root.chmod(0o700)
    plan.gate_root.mkdir()
    _share_with_container_group(plan.gate_root)
    _write_manifest(args, plan)
    attempted_roles: list[DockerRolePlan] = []
    started_ids: list[str] = []
    try:
        for role in plan.roles:
            _create_role_files(args, role)
        for role in plan.roles:
            attempted_roles.append(role)
            result = subprocess.run(
                _runtime_command(args, role, plan.gate_root),
                cwd=role.workdir,
                capture_output=True,
                text=True,
                check=False,
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
        launch_gate = plan.gate_root / ".launch"
        launch_gate.touch(exist_ok=False)
        launch_gate.chmod(0o640)
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

    print("\nAll roles are ready; launch gate opened.")
    if not show_lifecycle:
        return
    print("\nStatus:")
    print(f"  {_lifecycle_command(args, 'status')}")
    print("\nLogs:")
    print(f"  {_lifecycle_command(args, 'logs', '--follow')}")
    print("\nStop and remove private run state:")
    print(f"  {_lifecycle_command(args, 'terminate')}")
