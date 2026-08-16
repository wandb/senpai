# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Run Senpai roles as persistent system LaunchDaemons on Apple Silicon."""

from __future__ import annotations

import grp
import hashlib
import json
import os
import platform
import plistlib
import pwd
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from senpai_agent.supervisor import lease_is_healthy

from .specs import (
    RoleSpec,
    validate_identifier,
    validate_role_specs,
    validate_writable_parent,
)

DEFAULT_NATIVE_RUN_ROOT = "~/.senpai/native"
DEFAULT_NATIVE_TMUX_ROOT = "~/.senpai/t"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
LAUNCHD_PREFIX = "com.wandb.senpai"
LAUNCHD_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]+")
LAUNCH_DAEMON_ROOT = Path("/Library/LaunchDaemons")
LAUNCHCTL = "/bin/launchctl"
CAFFEINATE = "/usr/bin/caffeinate"
INSTALL = "/usr/bin/install"
REMOVE = "/bin/rm"
MISSING_SERVICE_MARKERS = (
    "could not find service",
    "service is not loaded",
    "no such process",
)
DARWIN_UNIX_SOCKET_PATH_MAX = 104
MACOS_COMMAND_PATHS = (
    "/opt/homebrew/bin",
    "/opt/homebrew/opt/gettext/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)


@dataclass(frozen=True)
class NativeRolePlan:
    """Private paths and launchd identity for one role."""

    spec: RoleSpec
    label: str
    role_root: Path
    home: Path
    workdir: Path
    log_root: Path
    state_root: Path
    tmp_root: Path
    tmux_root: Path
    descriptor: Path
    plist: Path
    launchd_plist: Path
    stdout_log: Path
    stderr_log: Path
    lease: Path


@dataclass(frozen=True)
class NativeLaunchPlan:
    """A validated native launch rooted in one private directory."""

    tag: str
    run_root: Path
    launch_gate: Path
    source_root: Path
    domain: str
    user_name: str
    group_name: str
    roles: tuple[NativeRolePlan, ...]


def _path_beneath(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Native launch path escapes run root: {path}")
    return path


def _launchd_plist_path(label: str) -> Path:
    if not LAUNCHD_LABEL.fullmatch(label):
        raise ValueError(f"Invalid native LaunchDaemon label: {label!r}")
    return LAUNCH_DAEMON_ROOT / f"{label}.plist"


def _configured_run_root(args) -> Path:
    return Path(
        getattr(args, "native_run_root", DEFAULT_NATIVE_RUN_ROOT)
    ).expanduser().resolve()


def _run_root(tag: str, configured_root: str | Path) -> Path:
    validate_identifier("Native tag", tag)
    base = Path(configured_root).expanduser().resolve()
    if base.exists() and not base.is_dir():
        raise ValueError(f"Native run root is not a directory: {base}")
    candidate = base / tag
    if candidate.is_symlink():
        raise RuntimeError(
            f"Native run already exists at {candidate}; use a new --tag"
        )
    path = candidate.resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f"Native run path escapes configured root: {path}")
    return path


def _check_run_root_available(run_root: Path) -> None:
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"Native run already exists at {run_root}; use a new --tag")


def _native_tmux_base() -> Path:
    return Path(DEFAULT_NATIVE_TMUX_ROOT).expanduser().resolve()


def _tmux_root(run_root: Path, role_key: str) -> Path:
    """Return a stable, short socket root for one role."""
    run_root = Path(run_root).expanduser().resolve()
    identity = os.fsencode(run_root) + b"\0" + role_key.encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    base = _native_tmux_base()
    root = _path_beneath(base, digest)
    socket = root / f"tmux-{os.getuid()}" / "openhands"
    if len(os.fsencode(socket)) >= DARWIN_UNIX_SOCKET_PATH_MAX:
        raise ValueError(
            "Native home is too long for the macOS tmux socket path"
        )
    return root


def plan_native(args, role_specs: list[RoleSpec]) -> NativeLaunchPlan:
    """Resolve a native launch without writing files or starting services."""
    validate_role_specs("Native", args.tag, role_specs)
    timeout = float(getattr(args, "native_ready_timeout_s", 600))
    if timeout <= 0:
        raise ValueError("native_ready_timeout_s must be greater than 0")

    run_root = _run_root(args.tag, _configured_run_root(args))
    if run_root.is_relative_to(SOURCE_ROOT):
        raise ValueError("native_run_root must be outside the Senpai source checkout")
    _check_run_root_available(run_root)
    validate_writable_parent(run_root.parent, "Native run root")
    validate_writable_parent(_native_tmux_base(), "Native tmux root")
    launch_gate = _path_beneath(run_root, "launch-gate")
    user_name = pwd.getpwuid(os.getuid()).pw_name
    group_name = grp.getgrgid(os.getgid()).gr_name
    roles = []
    for spec in role_specs:
        role_root = _path_beneath(run_root, "roles", spec.key)
        state_root = _path_beneath(role_root, "state")
        roles.append(
            NativeRolePlan(
                spec=spec,
                label=f"{LAUNCHD_PREFIX}.{args.tag}.{spec.key}",
                role_root=role_root,
                home=_path_beneath(role_root, "home"),
                workdir=_path_beneath(role_root, "workspace"),
                log_root=_path_beneath(role_root, "logs"),
                state_root=state_root,
                tmp_root=_path_beneath(role_root, "tmp"),
                tmux_root=_tmux_root(run_root, spec.key),
                descriptor=_path_beneath(role_root, "role.json"),
                plist=_path_beneath(role_root, "service.plist"),
                launchd_plist=_launchd_plist_path(
                    f"{LAUNCHD_PREFIX}.{args.tag}.{spec.key}"
                ),
                stdout_log=_path_beneath(role_root, "logs", "stdout.log"),
                stderr_log=_path_beneath(role_root, "logs", "stderr.log"),
                lease=_path_beneath(
                    state_root, "openhands_state", "controller-lease.json"
                ),
            )
        )
    if len({role.tmux_root for role in roles}) != len(roles):
        raise RuntimeError("Native role tmux socket roots collided")
    return NativeLaunchPlan(
        tag=args.tag,
        run_root=run_root,
        launch_gate=launch_gate,
        source_root=SOURCE_ROOT,
        domain="system",
        user_name=user_name,
        group_name=group_name,
        roles=tuple(roles),
    )


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _sudo_run(command: list[str]) -> subprocess.CompletedProcess:
    return _run(["sudo", "-n", *command])


def _native_path() -> str:
    """Include keg-only gettext for non-interactive launchd and SSH sessions."""
    inherited = os.environ.get("PATH", "").split(os.pathsep)
    paths = (str(Path(sys.executable).parent), *MACOS_COMMAND_PATHS, *inherited)
    return os.pathsep.join(dict.fromkeys(path for path in paths if path))


def _command_error(action: str, result: subprocess.CompletedProcess) -> RuntimeError:
    detail = result.stderr.strip() or result.stdout.strip() or "no error detail"
    return RuntimeError(f"{action}: {detail}")


def _check_source_revision(source_root: Path, revision: str) -> None:
    head = _run(["git", "-C", str(source_root), "rev-parse", "HEAD"])
    if head.returncode:
        raise _command_error("Could not inspect the current Senpai checkout", head)
    if head.stdout.strip() != revision:
        raise RuntimeError(
            f"Current Senpai checkout is {head.stdout.strip()}, expected {revision}"
        )
    commit = _run(
        ["git", "-C", str(source_root), "cat-file", "-e", f"{revision}^{{commit}}"]
    )
    if commit.returncode:
        raise RuntimeError(f"Runner revision {revision!r} is not available locally")


def _job_state(domain: str, label: str) -> tuple[str, int | None] | None:
    result = _sudo_run([LAUNCHCTL, "print", f"{domain}/{label}"])
    if result.returncode:
        detail = (result.stderr + result.stdout).lower()
        if any(marker in detail for marker in MISSING_SERVICE_MARKERS):
            return None
        raise _command_error(f"Could not inspect launchd service {label!r}", result)
    state_match = re.search(r"(?m)^\s*state = (\S+)", result.stdout)
    pid_match = re.search(r"(?m)^\s*pid = (\d+)", result.stdout)
    return (
        state_match.group(1) if state_match else "loaded",
        int(pid_match.group(1)) if pid_match else None,
    )


def preflight_native(args, role_specs: list[RoleSpec]) -> NativeLaunchPlan:
    """Validate the native host and exact local source revision without mutation."""
    plan = plan_native(args, role_specs)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Native Senpai requires an Apple Silicon macOS host")
    missing = [
        command
        for command in (
            "sudo",
            "launchctl",
            "caffeinate",
            "git",
            "gh",
            "envsubst",
        )
        if shutil.which(command, path=_native_path()) is None
    ]
    if missing:
        raise RuntimeError(
            "Native Senpai is missing required commands: " + ", ".join(missing)
        )
    _check_source_revision(plan.source_root, args.senpai_repo_revision)
    for script in ("entrypoint-advisor.sh", "entrypoint-student.sh"):
        if not (plan.source_root / "k8s" / script).is_file():
            raise RuntimeError(f"Native Senpai source is missing k8s/{script}")
    sudo = _sudo_run(["true"])
    if sudo.returncode:
        raise _command_error(
            "Native Senpai requires non-interactive sudo",
            sudo,
        )
    domain = _sudo_run([LAUNCHCTL, "print", plan.domain])
    if domain.returncode:
        raise _command_error(
            f"Native launchd domain {plan.domain!r} is unavailable",
            domain,
        )
    loaded = [
        role.label
        for role in plan.roles
        if _job_state(plan.domain, role.label)
    ]
    if loaded:
        raise RuntimeError(
            "Native launchd services already exist: " + ", ".join(loaded)
        )
    installed = [
        str(role.launchd_plist)
        for role in plan.roles
        if role.launchd_plist.exists() or role.launchd_plist.is_symlink()
    ]
    if installed:
        raise RuntimeError(
            "Native LaunchDaemon plists already exist: " + ", ".join(installed)
        )
    occupied_tmux_roots = [
        str(role.tmux_root)
        for role in plan.roles
        if role.tmux_root.exists() or role.tmux_root.is_symlink()
    ]
    if occupied_tmux_roots:
        raise RuntimeError(
            "Native tmux socket roots already exist: "
            + ", ".join(occupied_tmux_roots)
        )
    return plan


def _prepare_runner_workdir(source_root: Path, workdir: Path, revision: str) -> None:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    clone = _run(
        [
            "git",
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            str(source_root),
            str(workdir),
        ]
    )
    if clone.returncode:
        raise _command_error(
            f"Could not clone current Senpai checkout to {workdir}", clone
        )
    checkout = _run(
        ["git", "-C", str(workdir), "checkout", "--detach", revision]
    )
    if checkout.returncode:
        raise _command_error(
            f"Could not check out runner revision {revision!r}", checkout
        )


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def _write_private_json(path: Path, value: object) -> None:
    descriptor = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(descriptor)


def _role_environment(plan: NativeLaunchPlan, role: NativeRolePlan) -> dict[str, str]:
    pythonpath = str(role.workdir)
    if inherited := os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + inherited
    return {
        **role.spec.env,
        "BROWSER_USE_DISABLE_EXTENSIONS": "1",
        "HOME": str(role.home),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PATH": _native_path(),
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
        "SENPAI_BACKEND": "native",
        "SENPAI_BOOTSTRAP_STARTED_PATH": str(role.state_root / "bootstrap-started"),
        "SENPAI_GITHUB_TOKEN_FILE": str(role.state_root / "github-token"),
        "SENPAI_GIT_ASKPASS_FILE": str(role.state_root / "git-askpass"),
        "SENPAI_LAUNCH_GATE_PATH": str(plan.launch_gate),
        "SENPAI_LOGDIR": str(role.state_root),
        "SENPAI_PYTHON": sys.executable,
        "SENPAI_SKIP_EDITABLE_INSTALL": "1",
        "SENPAI_TMPDIR": str(role.tmp_root),
        "SENPAI_UMASK": "0077",
        "SENPAI_WORKDIR": str(role.workdir),
        "TMUX_TMPDIR": str(role.tmux_root),
        "TMPDIR": str(role.tmp_root),
    }


def _launchd_plist(plan: NativeLaunchPlan, role: NativeRolePlan) -> dict:
    return {
        "Label": role.label,
        "ProgramArguments": [
            CAFFEINATE,
            "-is",
            sys.executable,
            str(role.workdir / "k8s" / "native.py"),
            "run-role",
            str(role.descriptor),
        ],
        "WorkingDirectory": str(role.workdir),
        "UserName": plan.user_name,
        "GroupName": plan.group_name,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Interactive",
        "ThrottleInterval": 5,
        "StandardOutPath": str(role.stdout_log),
        "StandardErrorPath": str(role.stderr_log),
    }


def _create_role_files(args, plan: NativeLaunchPlan, role: NativeRolePlan) -> None:
    _private_directory(role.role_root)
    for path in (
        role.home,
        role.log_root,
        role.state_root,
        role.tmp_root,
    ):
        _private_directory(path)
    _prepare_runner_workdir(
        plan.source_root,
        role.workdir,
        args.senpai_repo_revision,
    )
    role.workdir.chmod(0o700)
    for path in (role.stdout_log, role.stderr_log):
        path.touch(mode=0o600, exist_ok=False)
        path.chmod(0o600)
    entrypoint = role.workdir / "k8s" / f"entrypoint-{role.spec.role}.sh"
    if not entrypoint.is_file():
        raise RuntimeError(f"Native role source is missing {entrypoint}")
    _write_private_json(
        role.descriptor,
        {
            "entrypoint": str(entrypoint),
            "environment": _role_environment(plan, role),
            "role": role.spec.role,
            "secrets": role.spec.secrets,
            "token_file": str(role.state_root / "github-token"),
            "workdir": str(role.workdir),
        },
    )
    with role.plist.open("wb") as output:
        plistlib.dump(_launchd_plist(plan, role), output, sort_keys=True)
    role.plist.chmod(0o600)


def _create_tmux_root(role: NativeRolePlan) -> None:
    _private_directory(role.tmux_root.parent)
    try:
        role.tmux_root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError(
            f"Native tmux socket root already exists: {role.tmux_root}"
        ) from error


def _write_manifest(plan: NativeLaunchPlan) -> None:
    _write_private_json(
        plan.run_root / "manifest.json",
        {
            "domain": plan.domain,
            "tag": plan.tag,
            "roles": [
                {
                    "key": role.spec.key,
                    "label": role.label,
                    "lease": str(role.lease),
                    "plist": str(role.launchd_plist),
                    "stderr": str(role.stderr_log),
                    "stdout": str(role.stdout_log),
                    "tmux_root": str(role.tmux_root),
                }
                for role in plan.roles
            ],
        },
    )


def _bootstrap_role(plan: NativeLaunchPlan, role: NativeRolePlan) -> None:
    install = _sudo_run(
        [
            INSTALL,
            "-o",
            "root",
            "-g",
            "wheel",
            "-m",
            "0644",
            str(role.plist),
            str(role.launchd_plist),
        ]
    )
    if install.returncode:
        raise _command_error(
            f"Could not install native role {role.spec.key!r}",
            install,
        )
    result = _sudo_run(
        [LAUNCHCTL, "bootstrap", plan.domain, str(role.launchd_plist)]
    )
    if result.returncode:
        raise _command_error(f"Could not start native role {role.spec.key!r}", result)


def _bootout_role(domain: str, label: str) -> None:
    result = _sudo_run([LAUNCHCTL, "bootout", f"{domain}/{label}"])
    if result.returncode:
        raise _command_error(f"Could not stop native role {label!r}", result)


def _remove_launchd_plist(path: Path) -> None:
    result = _sudo_run([REMOVE, "-f", str(path)])
    if result.returncode:
        raise _command_error(f"Could not remove native LaunchDaemon {path}", result)


def _uninstall_service(domain: str, label: str, plist: Path) -> None:
    errors = []
    try:
        if _job_state(domain, label) is not None:
            _bootout_role(domain, label)
    except RuntimeError as error:
        errors.append(str(error))
    try:
        _remove_launchd_plist(plist)
    except RuntimeError as error:
        errors.append(str(error))
    if errors:
        raise RuntimeError("\n".join(errors))


def _uninstall_role(plan: NativeLaunchPlan, role: NativeRolePlan) -> None:
    _uninstall_service(plan.domain, role.label, role.launchd_plist)


def _uninstall_recorded_role(domain: str, role: dict) -> None:
    label = role["label"]
    expected = _launchd_plist_path(label)
    recorded = Path(role["plist"])
    if recorded != expected:
        raise RuntimeError(
            f"Native manifest has unexpected LaunchDaemon path for {label!r}: "
            f"{recorded}"
        )
    _uninstall_service(domain, label, expected)


def _remove_tmux_root(path: str | Path, expected: Path) -> None:
    base = _native_tmux_base()
    recorded = Path(path)
    root = recorded.resolve()
    if (
        recorded.is_symlink()
        or recorded.parent.resolve() != base
        or root.parent != base
        or not re.fullmatch(r"[0-9a-f]{16}", root.name)
        or root != expected.resolve()
    ):
        raise RuntimeError(f"Native manifest has unexpected tmux root: {path}")
    if root.exists():
        shutil.rmtree(root)


def _lease_is_waiting_at_gate(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            int(value["pid"]) > 0
            and value["phase"] == "start-gate"
            and float(value["deadline"]) > time.monotonic()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def _role_failure_logs(roles: list[NativeRolePlan]) -> str:
    sections = []
    for role in roles:
        output = "\n".join(
            filter(None, (_tail(role.stdout_log), _tail(role.stderr_log)))
        )
        if output:
            sections.append(f"--- Native logs {role.spec.key} ---\n{output}")
    return "\n".join(sections)


def _wait_until_ready(plan: NativeLaunchPlan, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        pending = []
        for role in plan.roles:
            state = _job_state(plan.domain, role.label)
            if state is None:
                raise RuntimeError(
                    f"Native role {role.spec.key!r} disappeared before becoming ready"
                )
            if not (
                _lease_is_waiting_at_gate(role.lease)
                and lease_is_healthy(role.lease)
            ):
                pending.append(role.spec.key)
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Timed out after {timeout_s:g}s waiting for native roles: "
                + ", ".join(pending)
            )
        time.sleep(min(0.5, remaining))


def _print_plan(args, plan: NativeLaunchPlan) -> None:
    print(f"Native Senpai run: {plan.run_root}")
    print(f"Runner source: {plan.source_root} ({args.senpai_repo_revision})")
    print(f"launchd domain: {plan.domain}")
    for role in plan.roles:
        print(f"\n--- Native {role.spec.key} ---")
        print(f"service:   {role.label}")
        print(f"plist:     {role.launchd_plist} (root:wheel 0644)")
        print(f"home:      {role.home}")
        print(f"workspace: {role.workdir}")
        print(f"logs:      {role.log_root}")
        print(f"state:     {role.state_root} (credentials redacted)")


def _lifecycle_command(args, action: str, *options: str) -> str:
    command = ["python3", "k8s/native.py", action, args.tag, *options]
    configured = _configured_run_root(args)
    default = Path(DEFAULT_NATIVE_RUN_ROOT).expanduser().resolve()
    if configured != default:
        command.extend(["--run-root", str(configured)])
    return shlex.join(command)


def launch_native(
    args,
    role_specs: list[RoleSpec],
    plan: NativeLaunchPlan | None = None,
    *,
    show_lifecycle: bool = True,
    open_gate: bool = True,
) -> None:
    """Start native roles, wait for their leases, then optionally open the gate."""
    if getattr(args, "dry_run", False):
        _print_plan(args, plan or plan_native(args, role_specs))
        return

    plan = plan or preflight_native(args, role_specs)
    _check_run_root_available(plan.run_root)
    _private_directory(plan.run_root)
    attempted: list[NativeRolePlan] = []
    created: list[NativeRolePlan] = []
    try:
        for role in plan.roles:
            _create_role_files(args, plan, role)
            _create_tmux_root(role)
            created.append(role)
        _write_manifest(plan)
        for role in plan.roles:
            attempted.append(role)
            _bootstrap_role(plan, role)
            print(f"Launched {role.spec.key}: service={role.label}")
        _wait_until_ready(
            plan,
            float(getattr(args, "native_ready_timeout_s", 600)),
        )
        if open_gate:
            plan.launch_gate.touch(exist_ok=False)
            plan.launch_gate.chmod(0o600)
    except BaseException as error:
        cleanup_errors = []
        for role in reversed(attempted):
            try:
                _uninstall_role(plan, role)
            except RuntimeError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        logs = _role_failure_logs(list(plan.roles))
        if not cleanup_errors:
            for role in created:
                try:
                    _remove_tmux_root(role.tmux_root, role.tmux_root)
                except (OSError, RuntimeError) as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            if not cleanup_errors:
                try:
                    shutil.rmtree(plan.run_root)
                except OSError as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
        if isinstance(error, Exception):
            detail = f"\n{logs}" if logs else ""
            if cleanup_errors:
                detail += (
                    f"\nNative rollback was incomplete; inspect {plan.run_root}:\n"
                    + "\n".join(cleanup_errors)
                )
            raise RuntimeError(f"{error}{detail}") from error
        raise

    if open_gate:
        print("\nAll native roles are ready; launch gate opened.")
    else:
        print(
            "\nAll native roles are ready; launch gate remains closed for the "
            "fleet coordinator."
        )
    if not show_lifecycle:
        return
    print("\nStatus:")
    print(f"  {_lifecycle_command(args, 'status')}")
    print("\nLogs:")
    print(f"  {_lifecycle_command(args, 'logs', '--follow')}")
    print("\nStop and remove private run state:")
    print(f"  {_lifecycle_command(args, 'terminate')}")


def _load_manifest(
    tag: str,
    run_root: str = DEFAULT_NATIVE_RUN_ROOT,
) -> tuple[Path, dict]:
    path = _run_root(tag, run_root)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No native Senpai run {tag!r} found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tag") != tag:
        raise RuntimeError(f"Native run manifest tag does not match {tag!r}")
    if manifest.get("domain") != "system":
        raise RuntimeError("Native run manifest does not use the system launchd domain")
    return path, manifest


def status_native(tag: str, run_root: str = DEFAULT_NATIVE_RUN_ROOT) -> None:
    """Print launchd state and the existing OpenHands controller-lease health."""
    _, manifest = _load_manifest(tag, run_root)
    for role in manifest["roles"]:
        job = _job_state(manifest["domain"], role["label"])
        state, pid = job if job is not None else ("unloaded", None)
        health = "healthy" if lease_is_healthy(Path(role["lease"])) else "unhealthy"
        print(
            f"{role['key']}: service={role['label']} state={state} "
            f"pid={pid or '-'} health={health}"
        )


def _select_role(manifest: dict, role_key: str) -> dict:
    roles = manifest["roles"]
    role = next(
        (value for value in roles if value["key"] == role_key),
        roles[0] if not role_key and roles else None,
    )
    if role is None:
        choices = ", ".join(value["key"] for value in roles)
        raise ValueError(f"Unknown native role {role_key!r}; choose one of: {choices}")
    return role


def logs_native(
    tag: str,
    run_root: str = DEFAULT_NATIVE_RUN_ROOT,
    *,
    role_key: str = "",
    follow: bool = False,
    tail: int = 200,
) -> None:
    """Print bounded launchd stdout and stderr for one role."""
    if tail < 1:
        raise ValueError("Native log tail must be at least 1")
    _, manifest = _load_manifest(tag, run_root)
    role = _select_role(manifest, role_key)
    paths = [Path(role[name]) for name in ("stdout", "stderr")]
    existing = [str(path) for path in paths if path.is_file()]
    if not existing:
        raise RuntimeError(f"No native logs exist yet for {role['key']!r}")
    command = ["tail", "-n", str(tail)]
    if follow:
        command.append("-f")
    subprocess.run([*command, *existing], check=True)


def terminate_native(tag: str, run_root: str = DEFAULT_NATIVE_RUN_ROOT) -> None:
    """Unload recorded services and remove their private workspaces and secrets."""
    path, manifest = _load_manifest(tag, run_root)
    errors = []
    for role in reversed(manifest["roles"]):
        try:
            _uninstall_recorded_role(manifest["domain"], role)
            if tmux_root := role.get("tmux_root"):
                _remove_tmux_root(
                    tmux_root,
                    _tmux_root(path, role["key"]),
                )
        except (OSError, RuntimeError) as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError(
            "Native cleanup was incomplete; private state was preserved:\n"
            + "\n".join(errors)
        )
    shutil.rmtree(path)
    print(f"Stopped native Senpai run {tag!r} and removed its private run state.")


def _require_private_file(path: Path, label: str) -> None:
    if not path.is_file() or (path.stat().st_mode & 0o777) != 0o600:
        raise RuntimeError(f"{label} must be a mode-0600 regular file: {path}")


def _write_secret(path: Path, value: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(value)
    temporary.chmod(0o600)
    temporary.replace(path)


def run_role(path: Path) -> None:
    """Load one private descriptor and replace this process with its entrypoint."""
    _require_private_file(path, "Native role descriptor")
    value = json.loads(path.read_text(encoding="utf-8"))
    environment = dict(value["environment"])
    secrets = {key: secret for key, secret in value["secrets"].items() if secret}
    github_token = secrets.pop("GITHUB_TOKEN", "") or secrets.pop("GH_TOKEN", "")
    if github_token:
        token_file = Path(value["token_file"])
        _write_secret(token_file, github_token)
        environment["SENPAI_GITHUB_TOKEN_FILE"] = str(token_file)
    environment.update(secrets)
    environment.pop("GITHUB_TOKEN", None)
    environment.pop("GH_TOKEN", None)
    os.chdir(value["workdir"])
    os.execve(
        "/bin/bash",
        ["bash", value["entrypoint"]],
        environment,
    )


def run_from_payload(action: str, path: Path) -> None:
    """Consume one private JSON payload, then preflight or launch its roles."""
    if action not in {"preflight", "launch"}:
        raise ValueError(f"unsupported native payload action: {action}")
    _require_private_file(path, "Native launch payload")
    raw = path.read_text(encoding="utf-8")
    path.unlink()
    payload = json.loads(raw)
    args = SimpleNamespace(**payload["args"])
    role_specs = [RoleSpec(**values) for values in payload["roles"]]
    plan = preflight_native(args, role_specs)
    if action == "launch":
        launch_native(
            args,
            role_specs,
            plan,
            show_lifecycle=False,
            open_gate=bool(payload.get("open_gate", False)),
        )


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(
            "usage: python -m senpai.launch.native_backend "
            "{preflight,launch,preflight-payload,launch-payload,run-role} PATH"
        )
    action, raw_path = sys.argv[1:]
    path = Path(raw_path)
    if action == "run-role":
        run_role(path)
    elif action in {"preflight", "launch", "preflight-payload", "launch-payload"}:
        run_from_payload(action.removesuffix("-payload"), path)
    else:
        sys.exit(f"unsupported native action: {action}")


if __name__ == "__main__":
    main()
