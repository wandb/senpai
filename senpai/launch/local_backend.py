# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Local backend for running Senpai roles on one machine."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .specs import RoleSpec

CONTAINER_WORKDIR = "/workspace/senpai"
CONTAINER_RUN_ROOT = "/senpai-run"


def _student_gpu_map(raw: str) -> dict[str, list[str]]:
    gpu_map: dict[str, list[str]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid --local_student_gpu_ids entry {item!r}; expected name:gpu")
        name, devices = item.split(":", 1)
        name = name.strip()
        device_list = [device.strip() for device in devices.replace("|", "+").split("+") if device.strip()]
        if not name or not device_list:
            raise ValueError(f"invalid --local_student_gpu_ids entry {item!r}; expected name:gpu")
        gpu_map[name] = device_list
    return gpu_map


def _validate_local_gpu_assignments(args, role_specs: list[RoleSpec]) -> None:
    students = [spec.name for spec in role_specs if spec.role == "student"]
    gpu_map = _student_gpu_map(args.local_student_gpu_ids)
    if not students:
        return
    if args.gpus_per_student == 0:
        if gpu_map:
            raise RuntimeError("--local_student_gpu_ids requires --gpus_per_student greater than 0")
        return
    if not gpu_map:
        raise RuntimeError(
            "local GPU launches require --local_student_gpu_ids when "
            "--gpus_per_student is greater than 0"
        )

    missing = [name for name in students if name not in gpu_map]
    if missing:
        raise RuntimeError(
            "missing GPU assignments for local students: "
            f"{', '.join(missing)}; pass --local_student_gpu_ids name:gpu"
        )

    extra = [name for name in gpu_map if name not in students]
    if extra:
        raise RuntimeError(f"unknown local GPU assignment students: {', '.join(extra)}")

    wrong_count = {
        name: devices
        for name, devices in gpu_map.items()
        if len(devices) != args.gpus_per_student
    }
    if wrong_count:
        details = ", ".join(f"{name}:{'+'.join(devices)}" for name, devices in wrong_count.items())
        raise RuntimeError(
            f"--gpus_per_student={args.gpus_per_student} but assignments have a different count: {details}"
        )

    owners: dict[str, str] = {}
    duplicates = []
    for name, devices in gpu_map.items():
        for device in devices:
            if device in owners:
                duplicates.append(f"{device} assigned to {owners[device]} and {name}")
            owners[device] = name
    if duplicates:
        raise RuntimeError("local GPU assignments must be exclusive: " + "; ".join(duplicates))


def role_command(spec: RoleSpec) -> list[str]:
    script = "k8s/entrypoint-advisor.sh" if spec.role == "advisor" else "k8s/entrypoint-student.sh"
    return ["bash", script]


def local_run_root(args) -> Path:
    return Path(args.local_run_root).expanduser().resolve() / args.tag


def local_runner_source(args) -> Path:
    source = args.local_runner_source or "."
    return Path(source).expanduser().resolve()


def _env_file_text(values: dict[str, str], *, redact: bool = False) -> str:
    lines = []
    for key in sorted(values):
        value = "<redacted>" if redact else values[key]
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def _docker_env_file_text(values: dict[str, str]) -> str:
    lines = []
    for key in sorted(values):
        value = values[key]
        value = value.replace("\n", "\\n")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _write_env_file(path: Path, values: dict[str, str], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_env_file_text(values), encoding="utf-8")
    path.chmod(mode)


def _write_docker_env_file(path: Path, values: dict[str, str], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_docker_env_file_text(values), encoding="utf-8")
    path.chmod(mode)


def _prepare_runner_workdir(source: Path, workdir: Path) -> None:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    if (workdir / ".git").exists():
        subprocess.run(["git", "-C", str(workdir), "fetch", "--all", "--prune"], check=True)
        subprocess.run(["git", "-C", str(workdir), "pull", "--ff-only"], check=True)
        return
    if workdir.exists() and any(workdir.iterdir()):
        raise RuntimeError(f"local role workdir exists but is not a git checkout: {workdir}")
    subprocess.run(["git", "clone", str(source), str(workdir)], check=True)


def _role_values(
    args,
    run_root: Path,
    spec: RoleSpec,
    workdir: Path,
    log_dir: Path,
    *,
    containerized: bool = False,
) -> dict[str, str]:
    values = dict(spec.env)
    values.update({key: value for key, value in spec.secrets.items() if value})
    workdir_value = CONTAINER_WORKDIR if containerized else str(workdir)
    run_root_value = CONTAINER_RUN_ROOT if containerized else str(run_root)
    backend = "local-container" if containerized else "local"
    values.update(
        {
            "SENPAI_BACKEND": backend,
            "SENPAI_WORKDIR": workdir_value,
            "SENPAI_LOGDIR": f"{run_root_value}/logs/{spec.key}/iterations",
            "SENPAI_GIT_CREDENTIAL_FILE": f"{run_root_value}/secrets/{spec.key}.git-credentials",
            "HOME": f"{run_root_value}/home/{spec.key}",
        }
    )
    if args.local_skip_install:
        values["SENPAI_SKIP_INSTALL"] = "1"
    if args.local_disable_hivemind:
        values["SENPAI_DISABLE_HIVEMIND"] = "1"
    if spec.role == "advisor":
        values["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpus_per_student == 0:
        values["CUDA_VISIBLE_DEVICES"] = ""
    elif args.local_student_gpu_ids:
        gpu_by_student = _student_gpu_map(args.local_student_gpu_ids)
        values["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_by_student.get(spec.name, []))
    return values


def _role_env(args, run_root: Path, spec: RoleSpec, workdir: Path, log_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(_role_values(args, run_root, spec, workdir, log_dir))
    return env


def _container_name(tag: str, key: str) -> str:
    safe = "".join(char if char.isalnum() or char in "_.-" else "-" for char in f"senpai-{tag}-{key}")
    return safe[:128]


def _docker_gpu_args(values: dict[str, str]) -> list[str]:
    devices = values.get("CUDA_VISIBLE_DEVICES", "")
    if not devices:
        return []
    return ["--gpus", f"device={devices}"]


def _docker_command(
    args,
    run_root: Path,
    spec: RoleSpec,
    workdir: Path,
    env_file: Path,
    secret_file: Path,
    values: dict[str, str],
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        _container_name(args.tag, spec.key),
        "--workdir",
        CONTAINER_WORKDIR,
        "--env-file",
        str(env_file),
        "--env-file",
        str(secret_file),
        "--volume",
        f"{workdir}:{CONTAINER_WORKDIR}",
        "--volume",
        f"{run_root}:{CONTAINER_RUN_ROOT}",
    ]
    command.extend(_docker_gpu_args(values))
    command.append(args.local_container_image)
    command.extend(role_command(spec))
    return command


def launch_local(args, role_specs: list[RoleSpec]) -> None:
    _validate_local_gpu_assignments(args, role_specs)
    run_root = local_run_root(args)
    source = local_runner_source(args)
    container_image = getattr(args, "local_container_image", "")
    print(f"Local Senpai run: {run_root}")
    print(f"Runner source: {source}")
    if container_image:
        print(f"Container image: {container_image}")

    if args.dry_run:
        for spec in role_specs:
            workdir = run_root / "workdirs" / spec.key
            values = _role_values(args, run_root, spec, workdir, run_root / "logs", containerized=bool(container_image))
            print(f"\n--- Local {spec.key} ---")
            print(f"workdir: {workdir}")
            print(f"log:     {run_root / 'logs' / f'{spec.key}.log'}")
            print(f"pid:     {run_root / 'pids' / f'{spec.key}.pid'}")
            if container_image:
                env_file = run_root / "env" / f"{spec.key}.docker.env"
                secret_file = run_root / "secrets" / f"{spec.key}.docker.env"
                docker_command = _docker_command(args, run_root, spec, workdir, env_file, secret_file, values)
                print("command:", " ".join(shlex.quote(part) for part in docker_command))
            else:
                print("command:", " ".join(role_command(spec)))
            print(_env_file_text(values, redact=True), end="")
        return

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "pids").mkdir(exist_ok=True)
    (run_root / "env").mkdir(exist_ok=True)
    (run_root / "secrets").mkdir(exist_ok=True)

    for spec in role_specs:
        workdir = run_root / "workdirs" / spec.key
        log_file = run_root / "logs" / f"{spec.key}.log"
        pid_file = run_root / "pids" / f"{spec.key}.pid"
        if pid_file.exists():
            raise RuntimeError(f"{spec.key} already has a pid file: {pid_file}")

        _prepare_runner_workdir(source, workdir)
        _write_env_file(run_root / "env" / f"{spec.key}.env", spec.env)
        _write_env_file(run_root / "secrets" / f"{spec.key}.env", spec.secrets, mode=0o600)
        values = _role_values(args, run_root, spec, workdir, run_root / "logs", containerized=bool(container_image))
        Path(run_root / "home" / spec.key).mkdir(parents=True, exist_ok=True)
        Path(run_root / "logs" / spec.key / "iterations").mkdir(parents=True, exist_ok=True)
        if container_image:
            env_file = run_root / "env" / f"{spec.key}.docker.env"
            secret_file = run_root / "secrets" / f"{spec.key}.docker.env"
            _write_docker_env_file(env_file, {key: values[key] for key in values if key not in spec.secrets})
            _write_docker_env_file(
                secret_file,
                {key: value for key, value in spec.secrets.items() if value},
                mode=0o600,
            )
            command = _docker_command(args, run_root, spec, workdir, env_file, secret_file, values)
            env = os.environ.copy()
        else:
            command = role_command(spec)
            env = _role_env(args, run_root, spec, workdir, run_root / "logs")

        log = log_file.open("ab")
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        print(f"Launched local {spec.key}: pid={process.pid}, log={log_file}")

    print(f"\nStatus:")
    print(f"  ps -p $(cat {run_root}/pids/*.pid)")
    print(f"\nLogs:")
    print(f"  tail -f {run_root}/logs/*.log")
    print(f"\nStop:")
    print(f"  kill $(cat {run_root}/pids/*.pid)")


def stop_local(tag: str, run_root: str = "~/.senpai/runs") -> None:
    root = Path(run_root).expanduser().resolve() / tag
    for pid_file in sorted((root / "pids").glob("*.pid")):
        pid = pid_file.read_text(encoding="utf-8").strip()
        if pid:
            subprocess.run(["kill", pid], check=False)
        pid_file.unlink(missing_ok=True)


def status_local(tag: str, run_root: str = "~/.senpai/runs") -> None:
    root = Path(run_root).expanduser().resolve() / tag
    if not root.exists():
        print(f"No local Senpai run found: {root}")
        return
    for pid_file in sorted((root / "pids").glob("*.pid")):
        pid = pid_file.read_text(encoding="utf-8").strip()
        state = "unknown"
        if pid and shutil.which("ps"):
            result = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True)
            state = result.stdout.strip() or "not-running"
        print(f"{pid_file.stem}: pid={pid or '<missing>'} state={state}")
