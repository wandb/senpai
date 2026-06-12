# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Local process backend for running Senpai roles on one machine."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .specs import RoleSpec


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


def _write_env_file(path: Path, values: dict[str, str], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_env_file_text(values), encoding="utf-8")
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


def _role_env(args, run_root: Path, spec: RoleSpec, workdir: Path, log_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(spec.env)
    env.update({key: value for key, value in spec.secrets.items() if value})
    env.update(
        {
            "SENPAI_BACKEND": "local",
            "SENPAI_WORKDIR": str(workdir),
            "SENPAI_LOGDIR": str(log_dir / spec.key / "iterations"),
            "SENPAI_GIT_CREDENTIAL_FILE": str(run_root / "secrets" / f"{spec.key}.git-credentials"),
            "HOME": str(run_root / "home" / spec.key),
        }
    )
    if args.local_skip_install:
        env["SENPAI_SKIP_INSTALL"] = "1"
    if args.local_disable_hivemind:
        env["SENPAI_DISABLE_HIVEMIND"] = "1"
    if spec.role == "advisor":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif args.local_student_gpu_ids:
        gpu_by_student = dict(
            item.split(":", 1)
            for item in args.local_student_gpu_ids.split(",")
            if ":" in item
        )
        env["CUDA_VISIBLE_DEVICES"] = gpu_by_student.get(spec.name, "")
    elif args.gpus_per_student == 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def launch_local(args, role_specs: list[RoleSpec]) -> None:
    run_root = local_run_root(args)
    source = local_runner_source(args)
    print(f"Local Senpai run: {run_root}")
    print(f"Runner source: {source}")

    if args.dry_run:
        for spec in role_specs:
            workdir = run_root / "workdirs" / spec.key
            print(f"\n--- Local {spec.key} ---")
            print(f"workdir: {workdir}")
            print(f"log:     {run_root / 'logs' / f'{spec.key}.log'}")
            print(f"pid:     {run_root / 'pids' / f'{spec.key}.pid'}")
            print("command:", " ".join(role_command(spec)))
            print(_env_file_text({**spec.env, **spec.secrets}, redact=True), end="")
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
        role_env = _role_env(args, run_root, spec, workdir, run_root / "logs")
        Path(role_env["HOME"]).mkdir(parents=True, exist_ok=True)
        Path(role_env["SENPAI_LOGDIR"]).mkdir(parents=True, exist_ok=True)

        log = log_file.open("ab")
        process = subprocess.Popen(
            role_command(spec),
            cwd=workdir,
            env=role_env,
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
