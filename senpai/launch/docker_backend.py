# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run Senpai roles as Docker containers on one host."""

import shlex
import shutil
import subprocess
from pathlib import Path

from .specs import RoleSpec

CONTAINER_RUN_ROOT = "/senpai-run"
CONTAINER_WORKDIR = "/workspace/senpai"
RUN_LABEL = "com.wandb.senpai.run"
ROLE_LABEL = "com.wandb.senpai.role"


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
        devices = [
            device.strip() for device in raw_devices.split("+") if device.strip()
        ]
        if not student or not devices:
            raise ValueError(
                f"invalid GPU assignment {assignment!r}; expected student:gpu"
            )
        if student in assignments:
            raise ValueError(f"duplicate GPU assignment for student {student!r}")
        assignments[student] = devices
    return assignments


def _validate_gpu_assignments(args, role_specs: list[RoleSpec]) -> dict[str, list[str]]:
    students = [spec.name for spec in role_specs if spec.role == "student"]
    assignments = _student_gpu_map(args.docker_student_gpu_ids)
    if args.gpus_per_student == 0:
        if assignments:
            raise ValueError(
                "--docker_student_gpu_ids requires --gpus_per_student greater than 0"
            )
        return assignments
    if not students:
        return assignments
    if not assignments:
        raise ValueError(
            "Docker GPU launches require --docker_student_gpu_ids, for example fern:0,tanjiro:1"
        )

    missing = [student for student in students if student not in assignments]
    extra = [student for student in assignments if student not in students]
    if missing:
        raise ValueError(f"missing Docker GPU assignments for: {', '.join(missing)}")
    if extra:
        raise ValueError(
            f"GPU assignments name students outside this launch: {', '.join(extra)}"
        )

    wrong_count = {
        student: devices
        for student, devices in assignments.items()
        if len(devices) != args.gpus_per_student
    }
    if wrong_count:
        details = ", ".join(
            f"{student}:{'+'.join(devices)}" for student, devices in wrong_count.items()
        )
        raise ValueError(
            f"--gpus_per_student={args.gpus_per_student}, but these assignments differ: {details}"
        )

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
    return assignments


def _container_name(tag: str, role_key: str) -> str:
    def safe(value: str) -> str:
        return "".join(
            char if char.isalnum() or char in "_.-" else "-" for char in value
        )

    return f"senpai-{safe(tag)[:60]}-{safe(role_key)[:60]}"


def _docker_gpu_args(devices: list[str]) -> list[str]:
    if not devices:
        return []
    selection = (
        f"device={devices[0]}" if len(devices) == 1 else f'"device={",".join(devices)}"'
    )
    return ["--gpus", selection]


def _role_values(spec: RoleSpec) -> dict[str, str]:
    values = {
        **spec.env,
        **{key: value for key, value in spec.secrets.items() if value},
    }
    values.update(
        {
            "HOME": f"{CONTAINER_RUN_ROOT}/home/{spec.key}",
            "SENPAI_BACKEND": "docker",
            "SENPAI_GIT_CREDENTIAL_FILE": f"{CONTAINER_RUN_ROOT}/secrets/{spec.key}.git-credentials",
            "SENPAI_LOGDIR": f"{CONTAINER_RUN_ROOT}/logs/{spec.key}/iterations",
            "SENPAI_RUN_ROOT": CONTAINER_RUN_ROOT,
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
    run_root: Path,
    spec: RoleSpec,
    workdir: Path,
    env_file: Path,
    devices: list[str],
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
        f"{run_root}:{CONTAINER_RUN_ROOT}",
    ]
    if args.docker_data_dir:
        command.extend(
            [
                "--volume",
                f"{Path(args.docker_data_dir).expanduser().resolve()}:{args.pvc_mount_path}",
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


def _runner_source(args) -> Path:
    if args.docker_runner_source:
        return Path(args.docker_runner_source).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _prepare_runner_workdir(source: Path, workdir: Path, branch: str) -> None:
    if (workdir / ".git").exists():
        return
    if workdir.exists() and any(workdir.iterdir()):
        raise RuntimeError(
            f"Docker role workdir exists but is not a Git checkout: {workdir}"
        )
    workdir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            str(source),
            str(workdir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not clone runner branch {branch!r} from {source}: {result.stderr.strip()}"
        )


def _check_docker() -> None:
    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed or is not on PATH")
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Docker daemon is unavailable: {result.stderr.strip()}")


def _check_container_names_available(names: list[str]) -> None:
    for name in names:
        result = subprocess.run(
            ["docker", "container", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            raise RuntimeError(
                f"Docker container {name!r} already exists; remove it or use a new --tag"
            )


def launch_docker(args, role_specs: list[RoleSpec]) -> None:
    """Launch all roles as detached Docker containers."""
    if not role_specs:
        raise ValueError("Docker launch has no advisor or students")
    gpu_assignments = _validate_gpu_assignments(args, role_specs)
    run_root = Path(args.docker_run_root).expanduser().resolve() / args.tag
    source = _runner_source(args)
    if not source.is_dir():
        raise ValueError(f"Docker runner source does not exist: {source}")
    if args.docker_data_dir and not Path(args.docker_data_dir).expanduser().is_dir():
        raise ValueError(
            f"Docker data directory does not exist: {args.docker_data_dir}"
        )

    print(f"Docker Senpai run: {run_root}")
    print(f"Runner source: {source} ({args.repo_branch})")
    print(f"Image: {args.image}")

    commands = []
    for spec in role_specs:
        workdir = run_root / "workdirs" / spec.key
        env_file = run_root / "env" / f"{spec.key}.env"
        devices = gpu_assignments.get(spec.name, []) if spec.role == "student" else []
        command = _docker_command(args, run_root, spec, workdir, env_file, devices)
        commands.append((spec, workdir, env_file, command))

    if args.dry_run:
        for spec, workdir, env_file, command in commands:
            print(f"\n--- Docker {spec.key} ---")
            print(f"workdir: {workdir}")
            print(f"env:     {env_file} (credentials redacted)")
            print(f"command: {shlex.join(command)}")
        return

    _check_docker()
    names = [_container_name(args.tag, spec.key) for spec in role_specs]
    _check_container_names_available(names)
    run_root.mkdir(parents=True, exist_ok=True)
    run_root.chmod(0o700)

    for spec, workdir, env_file, _ in commands:
        _prepare_runner_workdir(source, workdir, args.repo_branch)
        (run_root / "home" / spec.key).mkdir(parents=True, exist_ok=True)
        (run_root / "logs" / spec.key / "iterations").mkdir(parents=True, exist_ok=True)
        (run_root / "secrets").mkdir(parents=True, exist_ok=True)
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(_env_file_text(_role_values(spec)), encoding="utf-8")
        env_file.chmod(0o600)

    for spec, workdir, _, command in commands:
        result = subprocess.run(command, cwd=workdir, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                f"Docker failed to start {spec.key}: {result.stderr.strip()}"
            )
        print(
            f"Launched {spec.key}: container={_container_name(args.tag, spec.key)} id={result.stdout.strip()[:12]}"
        )

    print("\nStatus:")
    print(f"  docker ps --all --filter label={RUN_LABEL}={args.tag}")
    print("\nLogs:")
    print(f"  docker logs --follow {names[0]}")
    print("\nStop and remove:")
    print(f"  docker rm --force {' '.join(names)}")
