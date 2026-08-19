#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch and score the bounded two-target Senpai evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SCRIPT = ROOT / "k8s" / "launch.py"
CUTOFF_SCRIPT = ROOT / "scripts" / "arm_senpai_cluster_cutoff.sh"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
load_dotenv(ROOT / ".env", override=False)

MODEL = "openai/gpt-5.6-luna"
REASONING_EFFORT = "high"
DEFAULT_TRAINING_TIMEOUT_MINUTES = 20.0
DEFAULT_TOTAL_TIMEOUT_HOURS = 6.0
READINESS_TIMEOUT_MINUTES = 30.0
CUTOFF_START_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 30
POLL_JITTER_SECONDS = 0
STALE_WIP_SECONDS = 1800
MAX_RUN_ID_LENGTH = 27

NANOGPT_TARGET_LOSS = 3.28
NANOGPT_SIGNIFICANCE_DELTA = 0.004
NANOGPT_BENCHMARK = "modded-nanogpt-track-3-optimization"
TANDEM_SPLITS = (
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
)


@dataclass(frozen=True)
class Target:
    name: str
    label: str
    repo_url: str
    base_branch: str
    base_revision: str
    primary_metric: str
    metric_unit: str


TARGETS = (
    Target(
        name="nanogpt",
        label="Modded NanoGPT",
        repo_url="https://github.com/morganmcg1/modded-nanogpt-senpai",
        base_branch="master",
        base_revision="0ba525c8d6dbdbdf7a69b4ddb129658527f9212b",
        primary_metric="speedrun/final_first_step_to_target",
        metric_unit="steps",
    ),
    Target(
        name="tandemfoil",
        label="TandemFoilSet Balanced",
        repo_url="https://github.com/morganmcg1/TandemFoilSet-Balanced",
        base_branch="codex/eval-wandb-group",
        base_revision="58161c30627bb67d204020fd4281f0098ecde6fc",
        primary_metric="test_avg/mae_surf_p",
        metric_unit="MAE",
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"eval-{timestamp}-{uuid.uuid4().hex[:6]}"


def validate_run_id(value: str) -> str:
    if len(value) > MAX_RUN_ID_LENGTH or re.fullmatch(
        r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", value
    ) is None:
        raise ValueError(
            f"run ID must be at most {MAX_RUN_ID_LENGTH} lowercase letters, "
            "digits, or hyphens"
        )
    return value


def positive_number(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def resolve_senpai_revision(config: Mapping[str, Any]) -> str:
    require_config(config, "advisor_image", "student_image")
    configured = str(config.get("senpai_repo_revision", "")).strip()
    images = (str(config["advisor_image"]), str(config["student_image"]))
    for image in images:
        if re.fullmatch(r"\S+:sha-[0-9a-f]{40}", image) is None and re.fullmatch(
            r"\S+@sha256:[0-9a-f]{64}", image
        ) is None:
            raise ValueError(
                "advisor/student images must use an immutable digest or "
                ":sha-<40-character-commit> tag"
            )
    tagged_revisions = {
        match.group(1)
        for image in images
        if (match := re.search(r":sha-([0-9a-f]{40})$", image))
    }
    candidates = tagged_revisions | ({configured} if configured else set())
    if len(candidates) != 1:
        raise ValueError(
            "advisor/student images must identify one Senpai revision; digest "
            "images require senpai_repo_revision"
        )
    revision = candidates.pop()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("senpai_repo_revision must be a full lowercase commit SHA")
    return revision


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"eval config must be a YAML mapping: {path}")
    return value


def require_config(config: Mapping[str, Any], *names: str) -> None:
    missing = [name for name in names if not config.get(name)]
    if missing:
        raise ValueError(
            "eval config is missing required values: " + ", ".join(missing)
        )


def build_manifest(
    config: Mapping[str, Any],
    run_id: str,
    *,
    web_search: bool,
    dry_run: bool,
    training_timeout_minutes: float = DEFAULT_TRAINING_TIMEOUT_MINUTES,
    total_timeout_hours: float = DEFAULT_TOTAL_TIMEOUT_HOURS,
    cutoff_image: str | None = None,
) -> dict[str, Any]:
    require_config(
        config,
        "wandb_entity",
        "wandb_project",
        "pvc_claim_name",
        "pvc_mount_path",
        "advisor_image",
        "student_image",
    )
    training_timeout_minutes = positive_number(
        training_timeout_minutes, "training timeout"
    )
    if round(training_timeout_minutes * 60) < 1:
        raise ValueError("training timeout must be at least one second")
    total_timeout_hours = positive_number(total_timeout_hours, "total timeout")
    senpai_revision = resolve_senpai_revision(config)
    evaluator_revision = git_revision()
    cutoff_image = cutoff_image or (
        f"ghcr.io/wandb/senpai-cutoff:sha-{evaluator_revision}"
    )
    started_at_epoch = int(time.time())
    deadline_epoch = started_at_epoch + math.ceil(total_timeout_hours * 3600)
    mount = str(config["pvc_mount_path"]).rstrip("/")
    gate = f"{mount}/senpai-evals/{run_id}/start-gate"
    suffix = run_id[-12:]
    targets = []
    for target in TARGETS:
        tag = f"{run_id}-{target.name}"
        targets.append(
            {
                **asdict(target),
                "research_tag": tag,
                "wandb_group": f"{run_id}/{target.name}",
                "advisor_branch": f"senpai-eval/{run_id}/{target.name}",
                "student_name": f"eval-{suffix}-{target.name}",
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "starting",
        "created_at": datetime.fromtimestamp(started_at_epoch, UTC).isoformat(),
        "started_at_epoch": started_at_epoch,
        "deadline_epoch": deadline_epoch,
        "deadline_at": datetime.fromtimestamp(deadline_epoch, UTC).isoformat(),
        "dry_run": dry_run,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "web_search": web_search,
        "training_timeout_minutes": training_timeout_minutes,
        "total_timeout_hours": total_timeout_hours,
        "readiness_timeout_minutes": READINESS_TIMEOUT_MINUTES,
        "senpai_repo_url": config.get(
            "senpai_repo_url", "https://github.com/wandb/senpai.git"
        ),
        "senpai_repo_revision": senpai_revision,
        "evaluator_revision": evaluator_revision,
        "evaluator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "advisor_image": config["advisor_image"],
        "student_image": config["student_image"],
        "cutoff_image": cutoff_image,
        "wandb_entity": config["wandb_entity"],
        "wandb_project": config["wandb_project"],
        "kube_context": config.get("kube_context", ""),
        "namespace": config.get("namespace", "default"),
        "pvc_claim_name": config["pvc_claim_name"],
        "pvc_mount_path": config["pvc_mount_path"],
        "start_gate_path": gate,
        "cutoff_job": cutoff_job_name(run_id),
        "targets": targets,
    }


def target_instructions(
    target: Mapping[str, Any], training_timeout_minutes: float
) -> str:
    common = f"""This is a bounded Senpai agent evaluation.

Optimize the target repository's declared primary metric and publish normal,
reproducible experiment evidence. Every training invocation has a hard
{training_timeout_minutes:g}-minute wall-clock ceiling. Budget setup,
validation, test evaluation, and W&B synchronization inside that ceiling.
Use run_training for every GPU execution.

Every W&B training run must use the exact group from $WANDB_RUN_GROUP. Do not
run debug or reduced-data evaluations. Prefer one decisive experiment at a
time, and preserve the target's metric and data-split contract.
"""
    if target["name"] == "nanogpt":
        return common + """
Use records/track_3_optimization/train_gpt_simple.py with --num_trials 1 and
--wandb_group "$WANDB_RUN_GROUP". The scored metric is the nonnegative
speedrun/final_first_step_to_target, and the final validation loss must pass
the repository's statistical-significance gate. A best intermediate loss is
not a substitute for the final loss.
"""
    return common + """
Run train.py with --wandb_group "$WANDB_RUN_GROUP". Leave --debug and
--skip_test disabled. A scored run must finish all four held-out test splits;
best_val_avg/mae_surf_p is diagnostic and cannot replace
test_avg/mae_surf_p.
"""


def target_launch_config(
    base: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(base)
    config.update(
        {
            "tag": target["research_tag"],
            "target_repo_url": target["repo_url"],
            "target_repo_branch": target["base_branch"],
            "target_repo_revision": target["base_revision"],
            "advisor_branch": target["advisor_branch"],
            "program_path": "",
            "advisor": True,
            "names": target["student_name"],
            "n_students": 1,
            "student_prefix": "",
            "advisor_model": MODEL,
            "advisor_reasoning_effort": REASONING_EFFORT,
            "student_model": MODEL,
            "student_reasoning_effort": REASONING_EFFORT,
            "smart_model": MODEL,
            "smart_reasoning_effort": REASONING_EFFORT,
            "fast_model": MODEL,
            "fast_reasoning_effort": REASONING_EFFORT,
            "frontier_model": MODEL,
            "frontier_reasoning_effort": REASONING_EFFORT,
            "human_issues": False,
            "web_search": manifest["web_search"],
            "wandb_run_group": target["wandb_group"],
            "timeout_minutes": manifest["training_timeout_minutes"],
            "poll_interval_s": POLL_INTERVAL_SECONDS,
            "poll_jitter_s": POLL_JITTER_SECONDS,
            "stale_wip_seconds": STALE_WIP_SECONDS,
            "gh_history_scope": "fresh",
            "start_gate_path": manifest["start_gate_path"],
            "extra_instructions": target_instructions(
                target, manifest["training_timeout_minutes"]
            ),
            "dry_run": manifest["dry_run"],
            "preflight_only": False,
        }
    )
    return config


def run_checked(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    print(f"+ {shlex.join(argv)}", flush=True)
    subprocess.run(argv, cwd=ROOT, env=env, check=True)


def current_kube_context(config: Mapping[str, Any], *, dry_run: bool) -> str:
    if context := str(config.get("kube_context", "")).strip():
        return context
    if dry_run:
        return "dry-run"
    result = subprocess.run(
        ["kubectl", "config", "current-context"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def cutoff_job_name(run_id: str) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    return f"senpai-cutoff-{(safe or 'senpai-eval')[:45]}"


def kubectl_command(manifest: Mapping[str, Any], *args: str) -> list[str]:
    argv = ["kubectl"]
    if context := str(manifest.get("kube_context", "")).strip():
        argv.extend(("--context", context))
    argv.extend(("--namespace", str(manifest["namespace"]), *args))
    return argv


def arm_cutoff(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    target_tags = ",".join(
        target["research_tag"] for target in manifest["targets"]
    )
    argv = [
        str(CUTOFF_SCRIPT),
        "--run-slug",
        manifest["run_id"],
        "--tags-csv",
        target_tags,
        "--expected-pods",
        str(2 * len(manifest["targets"])),
        "--expected-deployments",
        str(2 * len(manifest["targets"])),
        "--readiness-timeout-minutes",
        str(manifest["readiness_timeout_minutes"]),
        "--budget-hours",
        str(manifest["total_timeout_hours"]),
        "--deadline-epoch",
        str(manifest["deadline_epoch"]),
        "--pvc-claim",
        manifest["pvc_claim_name"],
        "--pvc-mount-path",
        manifest["pvc_mount_path"],
        "--pvc-log-root",
        f"{str(manifest['pvc_mount_path']).rstrip('/')}/senpai-evals",
        "--start-gate-path",
        manifest["start_gate_path"],
        "--image",
        str(manifest["cutoff_image"]),
    ]
    if manifest["dry_run"]:
        argv.append("--dry-run")
    environment = {
        **os.environ,
        "CONTEXT": current_kube_context(config, dry_run=manifest["dry_run"]),
        "NAMESPACE": str(config.get("namespace", "default")),
    }
    run_checked(argv, env=environment)


def wait_for_cutoff_ready(manifest: Mapping[str, Any]) -> None:
    selector = f"app=senpai-cutoff,run-slug={manifest['run_id']}"
    timeout = f"{CUTOFF_START_TIMEOUT_SECONDS}s"
    for condition in ("create", "condition=Ready"):
        run_checked(
            kubectl_command(
                manifest,
                "wait",
                f"--for={condition}",
                f"--timeout={timeout}",
                "pod",
                "-l",
                selector,
            )
        )


def run_best_effort(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=ROOT, text=True, check=False)


def cleanup_eval_resources(manifest: Mapping[str, Any]) -> bool:
    tags = ",".join(target["research_tag"] for target in manifest["targets"])
    selector = f"app=senpai,research-tag in ({tags})"
    target_commands = (
        kubectl_command(
            manifest,
            "delete",
            "deployments,configmaps,secrets",
            "-l",
            selector,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=300s",
        ),
        kubectl_command(
            manifest,
            "delete",
            "pods",
            "-l",
            selector,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=180s",
        ),
    )
    if any(run_best_effort(command).returncode for command in target_commands):
        print("Target cleanup failed; leaving the cutoff job armed.", file=sys.stderr)
        return False
    job = str(manifest["cutoff_job"])
    cleanup_commands = (
        kubectl_command(
            manifest,
            "delete",
            "job",
            job,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=180s",
        ),
        kubectl_command(
            manifest,
            "delete",
            "configmap",
            f"{job}-script",
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout=60s",
        ),
    )
    return not any(
        run_best_effort(command).returncode for command in cleanup_commands
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_path(results_dir: Path, run_id: str) -> Path:
    return results_dir / f"{run_id}.json"


def launch_eval(
    config_path: Path,
    results_dir: Path,
    *,
    run_id: str,
    web_search: bool,
    cutoff_image: str | None,
    dry_run: bool,
    training_timeout_minutes: float = DEFAULT_TRAINING_TIMEOUT_MINUTES,
    total_timeout_hours: float = DEFAULT_TOTAL_TIMEOUT_HOURS,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    manifest = build_manifest(
        config,
        validate_run_id(run_id),
        web_search=web_search,
        dry_run=dry_run,
        training_timeout_minutes=training_timeout_minutes,
        total_timeout_hours=total_timeout_hours,
        cutoff_image=cutoff_image,
    )
    path = manifest_path(results_dir, run_id)
    if path.exists():
        raise FileExistsError(f"eval run ID already exists: {run_id}")
    write_json(path, manifest)
    resources_started = False
    try:
        with tempfile.TemporaryDirectory(prefix="senpai-eval-") as directory:
            temporary_root = Path(directory)
            launch_paths = []
            for target in manifest["targets"]:
                launch_config = target_launch_config(config, manifest, target)
                instructions_path = (
                    temporary_root / f"{target['name']}.instructions.md"
                )
                instructions_path.write_text(
                    launch_config["extra_instructions"], encoding="utf-8"
                )
                launch_config["extra_instructions"] = str(instructions_path)
                target_config_path = temporary_root / f"{target['name']}.yaml"
                target_config_path.write_text(
                    yaml.safe_dump(launch_config, sort_keys=False),
                    encoding="utf-8",
                )
                launch_paths.append(target_config_path)

            if not dry_run:
                for target_config_path in launch_paths:
                    preflight = load_yaml(target_config_path)
                    preflight.update(dry_run=False, preflight_only=True)
                    preflight_path = target_config_path.with_suffix(
                        ".preflight.yaml"
                    )
                    preflight_path.write_text(
                        yaml.safe_dump(preflight, sort_keys=False),
                        encoding="utf-8",
                    )
                    run_checked(
                        [
                            sys.executable,
                            str(LAUNCH_SCRIPT),
                            "--config_path",
                            str(preflight_path),
                        ]
                    )

            resources_started = not dry_run
            arm_cutoff(config, manifest)
            if not dry_run:
                manifest["cutoff_status"] = "armed"
                manifest["updated_at"] = utc_now()
                write_json(path, manifest)
                wait_for_cutoff_ready(manifest)
                if time.time() >= manifest["deadline_epoch"]:
                    raise TimeoutError("total eval deadline elapsed before launch")
                manifest["cutoff_status"] = "ready"
                manifest["cutoff_ready_at"] = utc_now()
                manifest["status"] = "launching"
                write_json(path, manifest)

            for target_config_path in launch_paths:
                if not dry_run and time.time() >= manifest["deadline_epoch"]:
                    raise TimeoutError("total eval deadline elapsed during launch")
                run_checked(
                    [
                        sys.executable,
                        str(LAUNCH_SCRIPT),
                        "--config_path",
                        str(target_config_path),
                    ]
                )
            if not dry_run and time.time() >= manifest["deadline_epoch"]:
                raise TimeoutError("total eval deadline elapsed during launch")
    except BaseException as error:
        if resources_started:
            manifest["cleanup_status"] = (
                "complete"
                if cleanup_eval_resources(manifest)
                else "cutoff_left_armed"
            )
        manifest["status"] = "launch_failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = utc_now()
        write_json(path, manifest)
        raise
    manifest["status"] = "dry_run" if dry_run else "launched"
    if not dry_run:
        manifest["launched_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    write_json(path, manifest)
    return manifest


def cutoff_job_logs(manifest: Mapping[str, Any], job: str) -> str:
    pods_argv = kubectl_command(
        manifest,
        "get",
        "pods",
        "-l",
        f"batch.kubernetes.io/job-name={job}",
        "-o",
        "json",
    )
    print(f"+ {shlex.join(pods_argv)}", flush=True)
    result = subprocess.run(
        pods_argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    pods = sorted(
        json.loads(result.stdout).get("items", []),
        key=lambda pod: (
            pod.get("metadata", {}).get("creationTimestamp", ""),
            pod.get("metadata", {}).get("name", ""),
        ),
    )
    logs = []
    for pod in pods:
        name = pod.get("metadata", {}).get("name")
        if not name:
            continue
        logs_argv = kubectl_command(
            manifest,
            "logs",
            f"pod/{name}",
            "--container=cutoff",
        )
        print(f"+ {shlex.join(logs_argv)}", flush=True)
        logs.append(
            subprocess.run(
                logs_argv,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        )
    return "\n".join(logs)


def refresh_cutoff_status(manifest: dict[str, Any]) -> dict[str, Any]:
    job = str(manifest["cutoff_job"])
    argv = kubectl_command(manifest, "get", "job", job, "-o", "json")
    print(f"+ {shlex.join(argv)}", flush=True)
    result = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    job_state = json.loads(result.stdout)
    conditions = job_state.get("status", {}).get("conditions", [])
    complete = next(
        (
            condition
            for condition in conditions
            if condition.get("type") == "Complete"
            and condition.get("status") == "True"
        ),
        None,
    )
    failed = any(
        condition.get("type") == "Failed" and condition.get("status") == "True"
        for condition in conditions
    )
    manifest["cutoff_status"] = (
        "complete" if complete else "failed" if failed else "running"
    )
    if complete:
        manifest["status"] = "completed"
        manifest["cutoff_completed_at"] = complete.get(
            "lastTransitionTime", utc_now()
        )

    logs = cutoff_job_logs(manifest, job)
    if reasons := re.findall(r"Cutoff armed: ARM_REASON=(\S+)", logs):
        manifest["cutoff_arm_reason"] = reasons[-1]
    polls = re.findall(
        r"Ready gate poll: ready=(\d+)/(\d+), pods=(\d+)/(\d+), "
        r"deployments=(\d+)/(\d+)",
        logs,
    )
    if polls:
        ready, expected_ready, pods, expected_pods, deployments, expected_deployments = (
            map(int, polls[-1])
        )
        manifest["cutoff_last_ready_counts"] = {
            "ready_pods": ready,
            "expected_ready_pods": expected_ready,
            "pods": pods,
            "expected_pods": expected_pods,
            "deployments": deployments,
            "expected_deployments": expected_deployments,
        }
    manifest["updated_at"] = utc_now()
    return manifest


def wait_for_cutoff(manifest: dict[str, Any]) -> dict[str, Any]:
    timeout_seconds = max(1, int(manifest["deadline_epoch"] - time.time() + 600))
    run_checked(
        kubectl_command(
            manifest,
            "wait",
            "--for=condition=complete",
            f"--timeout={timeout_seconds}s",
            f"job/{manifest['cutoff_job']}",
        )
    )
    return refresh_cutoff_status(manifest)


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def nonnegative_integer(value: object) -> int | None:
    number = finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def run_summary(run: object) -> dict[str, Any]:
    summary = getattr(run, "summary", {})
    if isinstance(summary, Mapping):
        return dict(summary)
    value = getattr(summary, "_json_dict", {})
    return dict(value) if isinstance(value, Mapping) else {}


def run_identity(run: object) -> dict[str, Any]:
    return {
        "run_id": str(getattr(run, "id", "unknown")),
        "name": str(getattr(run, "name", "")),
        "url": str(getattr(run, "url", "")),
        "state": str(getattr(run, "state", "unknown")),
    }


def trial_zero(row: Mapping[str, Any]) -> bool:
    return nonnegative_integer(row.get("trial")) == 0


def scan_history(run: object, keys: Sequence[str]) -> list[Mapping[str, Any]]:
    return list(run.scan_history(keys=list(keys)))


def score_nanogpt_run(
    run: object,
) -> tuple[dict[str, Any] | None, str | None]:
    config = dict(getattr(run, "config", {}) or {})
    summary = run_summary(run)
    if config.get("benchmark") != NANOGPT_BENCHMARK:
        return None, "wrong benchmark contract"
    if finite_number(config.get("num_trials")) != 1:
        return None, "num_trials must equal 1"

    validation = []
    for row in scan_history(run, ("trial", "val/step", "val/loss")):
        loss = finite_number(row.get("val/loss"))
        if loss is None:
            continue
        if not trial_zero(row):
            return None, "validation trial must equal integer 0"
        step = nonnegative_integer(row.get("val/step"))
        if step is None:
            return None, "validation steps must be nonnegative integers"
        validation.append((step, loss))
    if not validation:
        return None, "missing final validation history"
    final_step, final_loss = max(validation, key=lambda item: item[0])
    crossing_steps = [
        step for step, loss in validation if loss <= NANOGPT_TARGET_LOSS
    ]
    if not crossing_steps:
        return None, "target loss was not reached in validation history"
    derived_first_step = min(crossing_steps)

    markers = []
    for row in scan_history(
        run,
        (
            "trial",
            "speedrun/final_first_step_to_target",
            "speedrun/final_reached_target",
        ),
    ):
        if (
            finite_number(row.get("speedrun/final_first_step_to_target")) is None
            and finite_number(row.get("speedrun/final_reached_target")) is None
        ):
            continue
        if not trial_zero(row):
            return None, "final marker trial must equal integer 0"
        markers.append(row)
    marker = markers[-1] if markers else summary
    reported_first_step = nonnegative_integer(
        marker.get("speedrun/final_first_step_to_target")
    )
    reached = finite_number(marker.get("speedrun/final_reached_target"))
    if reported_first_step is None or reached != 1:
        return None, "missing successful final target marker"
    if reported_first_step != derived_first_step:
        return None, "final target marker disagrees with validation history"

    gate_margin = (
        NANOGPT_TARGET_LOSS - final_loss
    ) - NANOGPT_SIGNIFICANCE_DELTA
    if gate_margin < -1e-12:
        return None, "final validation loss failed the significance gate"

    diagnostics = {
        "final_val_loss": final_loss,
        "final_val_step": final_step,
        "reported_first_step_to_target": reported_first_step,
        "significance_gate_margin": gate_margin,
    }
    if (runtime := finite_number(summary.get("_runtime"))) is not None:
        diagnostics["wandb_runtime_seconds"] = runtime
    return (
        {
            **run_identity(run),
            "score": derived_first_step,
            "metric": "speedrun/final_first_step_to_target",
            "unit": "steps",
            "diagnostics": diagnostics,
        },
        None,
    )


def enabled(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def score_tandem_run(
    run: object,
) -> tuple[dict[str, Any] | None, str | None]:
    config = dict(getattr(run, "config", {}) or {})
    if enabled(config.get("debug", False)):
        return None, "debug run"
    if enabled(config.get("skip_test", False)):
        return None, "test evaluation was skipped"
    summary = run_summary(run)
    split_values = {}
    for split in TANDEM_SPLITS:
        key = f"test/{split}/mae_surf_p"
        value = finite_number(summary.get(key))
        if value is None:
            return None, f"missing finite {key}"
        if value < 0:
            return None, f"{key} must be nonnegative"
        split_values[key] = value
    reported = finite_number(summary.get("test_avg/mae_surf_p"))
    if reported is None:
        return None, "missing finite test_avg/mae_surf_p"
    if reported < 0:
        return None, "test_avg/mae_surf_p must be nonnegative"
    recomputed = sum(split_values.values()) / len(split_values)
    if not math.isclose(reported, recomputed, rel_tol=1e-6, abs_tol=1e-6):
        return None, "test average is inconsistent with the four splits"

    diagnostics: dict[str, Any] = {
        "recomputed_test_avg": recomputed,
        **split_values,
    }
    for key in (
        "best_val_avg/mae_surf_p",
        "best_epoch",
        "total_train_minutes",
        "_runtime",
    ):
        if (value := finite_number(summary.get(key))) is not None:
            output_key = "wandb_runtime_seconds" if key == "_runtime" else key
            diagnostics[output_key] = value
    return (
        {
            **run_identity(run),
            "score": reported,
            "metric": "test_avg/mae_surf_p",
            "unit": "MAE",
            "diagnostics": diagnostics,
        },
        None,
    )


def score_target_runs(
    target: Mapping[str, Any],
    runs: Iterable[object],
) -> dict[str, Any]:
    scorer = score_nanogpt_run if target["name"] == "nanogpt" else score_tandem_run
    valid = []
    rejected = []
    states: Counter[str] = Counter()
    for run in runs:
        identity = run_identity(run)
        states[identity["state"]] += 1
        score, reason = scorer(run)
        if score is not None:
            valid.append(score)
        else:
            rejected.append({**identity, "reason": reason or "unscored"})
    valid.sort(key=lambda item: item["score"])
    return {
        "name": target["name"],
        "label": target["label"],
        "group": target["wandb_group"],
        "primary_metric": target["primary_metric"],
        "direction": "minimize",
        "total_runs": sum(states.values()),
        "eligible_runs": len(valid),
        "states": dict(sorted(states.items())),
        "best": valid[0] if valid else None,
        "valid_runs": valid,
        "rejected_runs": rejected,
    }


def format_score(value: object) -> str:
    number = finite_number(value)
    return "—" if number is None else f"{number:.6g}"


def render_markdown(
    manifest: Mapping[str, Any],
    target_results: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# Senpai eval report: {manifest['run_id']}",
        "",
        f"- Model: `{manifest['model']}` at `{manifest['reasoning_effort']}`",
        f"- Senpai revision: `{manifest['senpai_repo_revision']}`",
        f"- Built-in web search: `{'on' if manifest['web_search'] else 'off'}`",
        f"- Per-training hard timeout: `{manifest['training_timeout_minutes']:g} minutes`",
        f"- Total eval timeout: `{manifest['total_timeout_hours']:g} hours`",
        f"- Cutoff: `{manifest.get('cutoff_status', 'unknown')}` "
        f"(`{manifest.get('cutoff_arm_reason', 'unknown')}`)",
        f"- W&B project: `{manifest['wandb_entity']}/{manifest['wandb_project']}`",
        "",
        "| Target | Eligible / total | Primary metric | Best | W&B run | State |",
        "|---|---:|---|---:|---|---|",
    ]
    for result in target_results:
        best = result["best"]
        if best:
            run_link = (
                f"[{best['run_id']}]({best['url']})"
                if best["url"]
                else best["run_id"]
            )
            score = format_score(best["score"])
            state = best["state"]
        else:
            run_link = score = state = "—"
        lines.append(
            f"| {result['label']} | {result['eligible_runs']} / "
            f"{result['total_runs']} | `{result['primary_metric']}` | "
            f"{score} | {run_link} | {state} |"
        )

    for result in target_results:
        lines.extend(("", f"## {result['label']}", ""))
        if best := result["best"]:
            lines.append("Best-run diagnostics:")
            lines.append("")
            for key, value in best["diagnostics"].items():
                lines.append(f"- `{key}`: {format_score(value)}")
        else:
            lines.append("No run satisfied the target's scoring contract.")
        if result["rejected_runs"]:
            lines.extend(("", "Unscored runs:", ""))
            for rejected in result["rejected_runs"]:
                lines.append(
                    f"- `{rejected['run_id']}` ({rejected['state']}): "
                    f"{rejected['reason']}"
                )
    return "\n".join(lines) + "\n"


def log_report_to_wandb(
    manifest: Mapping[str, Any],
    target_results: Sequence[Mapping[str, Any]],
    markdown: str,
) -> str:
    digest = hashlib.sha256(manifest["run_id"].encode()).hexdigest()[:16]
    run = wandb.init(
        entity=manifest["wandb_entity"],
        project=manifest["wandb_project"],
        id=f"eval-{digest}",
        resume="allow",
        name=f"{manifest['run_id']}-report",
        group=manifest["run_id"],
        job_type="senpai-agent-eval-report",
        tags=["senpai-agent-eval", "report"],
        config={
            "eval_run_id": manifest["run_id"],
            "model": manifest["model"],
            "reasoning_effort": manifest["reasoning_effort"],
            "web_search": manifest["web_search"],
            "training_timeout_minutes": manifest["training_timeout_minutes"],
            "total_timeout_hours": manifest["total_timeout_hours"],
            "senpai_repo_url": manifest["senpai_repo_url"],
            "senpai_repo_revision": manifest["senpai_repo_revision"],
            "evaluator_revision": manifest["evaluator_revision"],
            "evaluator_sha256": manifest["evaluator_sha256"],
            "advisor_image": manifest["advisor_image"],
            "student_image": manifest["student_image"],
            "cutoff_image": manifest["cutoff_image"],
            "created_at": manifest["created_at"],
            "launched_at": manifest.get("launched_at"),
            "deadline_at": manifest["deadline_at"],
            "cutoff_arm_reason": manifest.get("cutoff_arm_reason", "unknown"),
            "cutoff_ready_at": manifest.get("cutoff_ready_at"),
            "cutoff_completed_at": manifest.get("cutoff_completed_at"),
            "cutoff_last_ready_counts": manifest.get(
                "cutoff_last_ready_counts"
            ),
            "targets": [
                {
                    "name": target["name"],
                    "repo_url": target["repo_url"],
                    "base_branch": target["base_branch"],
                    "base_revision": target["base_revision"],
                    "wandb_group": target["wandb_group"],
                }
                for target in manifest["targets"]
            ],
        },
    )
    metrics: dict[str, int | float] = {}
    for result in target_results:
        prefix = f"eval/{result['name']}"
        metrics[f"{prefix}/total_runs"] = result["total_runs"]
        metrics[f"{prefix}/eligible_runs"] = result["eligible_runs"]
        if result["best"]:
            metrics[f"{prefix}/primary"] = result["best"]["score"]
            for key, value in result["best"]["diagnostics"].items():
                if (number := finite_number(value)) is not None:
                    metrics[f"{prefix}/diagnostic/{key}"] = number
    metrics["eval/scored_targets"] = sum(
        result["best"] is not None for result in target_results
    )
    metrics["eval/total_targets"] = len(target_results)
    run.log(metrics)
    run.summary["report_markdown"] = markdown
    url = run.url
    run.finish()
    return url


def report_eval(
    manifest: Mapping[str, Any],
    results_dir: Path,
    *,
    log_wandb: bool,
    api: object | None = None,
) -> tuple[dict[str, Any], str]:
    if log_wandb:
        if manifest.get("status") != "completed":
            raise RuntimeError(
                "refusing to publish an aggregate W&B report before the cutoff "
                "job completes"
            )
        if manifest.get("cutoff_arm_reason") not in {
            "all_ready",
            "readiness_timeout",
            "total_deadline",
        }:
            raise RuntimeError(
                "refusing to publish without a recorded cutoff arm reason"
            )
    api = api or wandb.Api()
    project = f"{manifest['wandb_entity']}/{manifest['wandb_project']}"
    target_results = []
    for target in manifest["targets"]:
        runs = api.runs(project, filters={"group": target["wandb_group"]})
        target_results.append(score_target_runs(target, runs))
    markdown = render_markdown(manifest, target_results)
    report = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "generated_at": utc_now(),
        "eval_status": manifest.get("status", "unknown"),
        "provenance": {
            "senpai_repo_url": manifest["senpai_repo_url"],
            "senpai_repo_revision": manifest["senpai_repo_revision"],
            "evaluator_revision": manifest["evaluator_revision"],
            "evaluator_sha256": manifest["evaluator_sha256"],
            "advisor_image": manifest["advisor_image"],
            "student_image": manifest["student_image"],
            "cutoff_image": manifest["cutoff_image"],
            "created_at": manifest["created_at"],
            "launched_at": manifest.get("launched_at"),
            "deadline_at": manifest["deadline_at"],
            "cutoff_completed_at": manifest.get("cutoff_completed_at"),
            "cutoff_status": manifest.get("cutoff_status", "unknown"),
            "cutoff_arm_reason": manifest.get("cutoff_arm_reason", "unknown"),
            "cutoff_last_ready_counts": manifest.get(
                "cutoff_last_ready_counts"
            ),
            "target_revisions": {
                target["name"]: target["base_revision"]
                for target in manifest["targets"]
            },
        },
        "targets": target_results,
    }
    report_path = results_dir / f"{manifest['run_id']}.report.json"
    markdown_path = results_dir / f"{manifest['run_id']}.report.md"
    write_json(report_path, report)
    markdown_path.write_text(markdown, encoding="utf-8")
    if log_wandb:
        report["wandb_report_run"] = log_report_to_wandb(
            manifest,
            target_results,
            markdown,
        )
        write_json(report_path, report)
    return report, markdown


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser(
        "launch",
        help="launch both targets and arm the cluster-side cutoff",
    )
    launch.add_argument("--config-path", type=Path, default=ROOT / "senpai.local.yaml")
    launch.add_argument("--run-id", default=None)
    launch.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    launch.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Senpai's built-in browser and external-search facilities",
    )
    launch.add_argument("--cutoff-image", default=None)
    launch.add_argument(
        "--training-timeout-minutes",
        type=float,
        default=DEFAULT_TRAINING_TIMEOUT_MINUTES,
        help="hard ceiling for each training invocation (default: 20)",
    )
    launch.add_argument(
        "--total-timeout-hours",
        type=float,
        default=DEFAULT_TOTAL_TIMEOUT_HOURS,
        help="absolute launch-to-cleanup deadline (default: 6)",
    )
    launch.add_argument(
        "--wait",
        action="store_true",
        help="wait for the cutoff, then write and log the report",
    )
    launch.add_argument("--dry-run", action="store_true")

    report = subparsers.add_parser("report", help="score and report one launched eval")
    report.add_argument("--run-id", required=True)
    report.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    report.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the aggregate report back to W&B",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "launch":
        if args.wait and args.dry_run:
            raise ValueError("--wait cannot be combined with --dry-run")
        run_id = validate_run_id(args.run_id or default_run_id())
        config_path = args.config_path.expanduser().resolve()
        results_dir = args.results_dir.expanduser().resolve()
        manifest = launch_eval(
            config_path,
            results_dir,
            run_id=run_id,
            web_search=args.web_search,
            cutoff_image=args.cutoff_image,
            dry_run=args.dry_run,
            training_timeout_minutes=args.training_timeout_minutes,
            total_timeout_hours=args.total_timeout_hours,
        )
        print(f"Eval {run_id} {manifest['status']}.", flush=True)
        print(
            "Report command: "
            + shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "report",
                    "--run-id",
                    run_id,
                    "--results-dir",
                    str(results_dir),
                ]
            ),
            flush=True,
        )
        if args.wait:
            manifest = wait_for_cutoff(manifest)
            write_json(manifest_path(results_dir, run_id), manifest)
            _, markdown = report_eval(manifest, results_dir, log_wandb=True)
            print(markdown)
        return 0

    results_dir = args.results_dir.expanduser().resolve()
    path = manifest_path(results_dir, validate_run_id(args.run_id))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") in {"launched", "launching"}:
        try:
            refresh_cutoff_status(manifest)
            write_json(path, manifest)
        except subprocess.CalledProcessError:
            if args.wandb:
                raise
    _, markdown = report_eval(manifest, results_dir, log_wandb=args.wandb)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
