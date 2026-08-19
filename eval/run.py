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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median, pstdev, pvariance
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wandb
import yaml
from dotenv import load_dotenv
from pydantic import SecretStr

from eval.adjudication import GitHubReads, adjudicate_trial, freeze_advisor_head
from k8s.launch_helpers import resolve_github_token
from senpai_agent.github.http import GitHubReader

EVALUATOR_PATH = Path(__file__).resolve()
ADJUDICATOR_PATH = ROOT / "eval" / "adjudication.py"
LAUNCH_SCRIPT = ROOT / "k8s" / "launch.py"
CUTOFF_SCRIPT = ROOT / "scripts" / "arm_senpai_cluster_cutoff.sh"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"
load_dotenv(ROOT / ".env", override=False)

MODEL = "openai/gpt-5.6-luna"
REASONING_EFFORT = "high"
DEFAULT_TRAINING_TIMEOUT_MINUTES = 20.0
DEFAULT_TOTAL_TIMEOUT_HOURS = 6.0
DEFAULT_N_TRIALS = 3
DEFAULT_WANDB_ENTITY = "wandb-applied-ai-team"
DEFAULT_WANDB_PROJECT = "senpai_eval"
READINESS_TIMEOUT_MINUTES = 30.0
CUTOFF_START_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 30
POLL_JITTER_SECONDS = 0
STALE_WIP_SECONDS = 1800
MAX_RUN_ID_LENGTH = 27

NANOGPT_TARGET_LOSS = 3.28
NANOGPT_SIGNIFICANCE_DELTA = 0.004
NANOGPT_BENCHMARK = "modded-nanogpt-track-3-optimization"
NANOGPT_VAL_TOKENS = 10_485_760
NANOGPT_SHARD_TOKENS = 100_000_000
NANOGPT_SHARD_BYTES = 1_024 + 2 * NANOGPT_SHARD_TOKENS
NANOGPT_TRAIN_SHARDS = [
    f"fineweb_train_{index:06d}.bin" for index in range(1, 21)
]
NANOGPT_VAL_SHARDS = ["fineweb_val_000000.bin"]
NANOGPT_DATA_CONTRACT = {
    "train_shards": NANOGPT_TRAIN_SHARDS,
    "val_shards": NANOGPT_VAL_SHARDS,
    "tokens_per_shard": NANOGPT_SHARD_TOKENS,
    "bytes_per_shard": NANOGPT_SHARD_BYTES,
    "val_tokens": NANOGPT_VAL_TOKENS,
}
NANOGPT_METRIC_CONTRACT = {
    "primary": "speedrun/final_first_step_to_target",
    "validation": "val/loss",
    "direction": "minimize",
    "target": NANOGPT_TARGET_LOSS,
    "significance_rule": (
        "(target - mean_loss) * sqrt(num_trials) >= stat_sig_delta"
    ),
}
TANDEM_SPLITS = (
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
)
TANDEM_VAL_SPLITS = tuple(split.replace("test_", "val_", 1) for split in TANDEM_SPLITS)
TANDEM_METRIC_CONTRACT = {
    "primary": "test_avg/mae_surf_p",
    "selection": "val_avg/mae_surf_p",
    "direction": "minimize",
    "test_splits": list(TANDEM_SPLITS),
}
TANDEM_PROTECTED_HASHES = {
    "split_manifest_sha256": (
        "5b9bf301f0a7f0f415133333fa9be4e6a321ca8ee0d01a6ae06443dffc5261de"
    ),
    "scoring_source_sha256": (
        "81ebbc4f72c58121826a157446817428bc8b10716bef55cb142e353154df8871"
    ),
    "loader_source_sha256": (
        "7640dc3c7b7c914e11d9e94f6e2ea8026274ffd8acf9f71bb8c1178dc01973f0"
    ),
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Target:
    name: str
    branch_slug: str
    label: str
    repo_url: str
    base_branch: str
    base_revision: str
    primary_metric: str
    metric_unit: str


TARGETS = (
    Target(
        name="nanogpt",
        branch_slug="nano",
        label="Modded NanoGPT",
        repo_url="https://github.com/morganmcg1/modded-nanogpt-senpai",
        base_branch="codex/eval-wandb-contract",
        base_revision="1f487927967e0c9973822117f2131280e2e40d04",
        primary_metric="speedrun/final_first_step_to_target",
        metric_unit="steps",
    ),
    Target(
        name="tandemfoil",
        branch_slug="foil",
        label="TandemFoilSet Balanced",
        repo_url="https://github.com/morganmcg1/TandemFoilSet-Balanced",
        base_branch="codex/eval-wandb-group",
        base_revision="21afbf128e8ca267d0d0e72efc91856dcf2c2cbf",
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


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_report_sources(manifest: Mapping[str, Any]) -> None:
    mismatches = [
        name
        for name, path in (
            ("evaluator_sha256", EVALUATOR_PATH),
            ("adjudicator_sha256", ADJUDICATOR_PATH),
        )
        if manifest.get(name) != file_sha256(path)
    ]
    if mismatches:
        raise RuntimeError(
            "report source hash mismatch: " + ", ".join(mismatches)
        )


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
    n_trials: int = DEFAULT_N_TRIALS,
    cutoff_image: str | None = None,
) -> dict[str, Any]:
    require_config(
        config,
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
    n_trials = positive_integer(n_trials, "n_trials")
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
    seed_base = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)
    for target_ordinal, target in enumerate(TARGETS):
        trials = []
        for trial_index in range(n_trials):
            trial_name = f"trial-{trial_index + 1:02d}"
            seed_offset = target_ordinal * n_trials + trial_index
            trial_seed = (seed_base + seed_offset) % (2**31 - 1)
            advisor_branch = (
                f"senpai-eval/{run_id}/{target.branch_slug}-t{trial_index + 1:02d}"
            )
            if len(advisor_branch) > 50:
                raise ValueError(
                    "n_trials makes advisor routing labels exceed 50 characters"
                )
            trials.append(
                {
                    "trial_index": trial_index,
                    "trial_name": trial_name,
                    "trial_seed": trial_seed,
                    "research_tag": (
                        f"{run_id}-{target.branch_slug}-t{trial_index + 1:02d}"
                    ),
                    "wandb_group": f"{run_id}/{target.name}/{trial_name}",
                    "advisor_branch": advisor_branch,
                    "student_name": (
                        f"eval-{suffix}-{target.branch_slug}-t{trial_index + 1:02d}"
                    ),
                    "adjudication": {
                        "status": "pending",
                        "selected_run_id": None,
                        "evidence": {},
                    },
                }
            )
        targets.append(
            {
                **asdict(target),
                "trials": trials,
            }
        )
    return {
        "schema_version": 2,
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
        "n_trials": n_trials,
        "readiness_timeout_minutes": READINESS_TIMEOUT_MINUTES,
        "senpai_repo_url": config.get(
            "senpai_repo_url", "https://github.com/wandb/senpai.git"
        ),
        "senpai_repo_revision": senpai_revision,
        "evaluator_revision": evaluator_revision,
        "evaluator_dirty": git_dirty(),
        "evaluator_sha256": file_sha256(EVALUATOR_PATH),
        "adjudicator_sha256": file_sha256(ADJUDICATOR_PATH),
        "advisor_image": config["advisor_image"],
        "student_image": config["student_image"],
        "cutoff_image": cutoff_image,
        "wandb_entity": DEFAULT_WANDB_ENTITY,
        "wandb_project": DEFAULT_WANDB_PROJECT,
        "kube_context": config.get("kube_context", ""),
        "namespace": config.get("namespace", "default"),
        "custom_secret_env_names": list(
            config.get("custom_secret_env_names", [])
        ),
        "pvc_claim_name": config["pvc_claim_name"],
        "pvc_mount_path": config["pvc_mount_path"],
        "start_gate_path": gate,
        "cutoff_job": cutoff_job_name(run_id),
        "targets": targets,
    }


def iter_trial_specs(
    manifest: Mapping[str, Any],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for target in manifest["targets"]:
        for trial in target["trials"]:
            yield target, trial


def target_launch_config(
    base: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(base)
    config.update(
        {
            "tag": trial["research_tag"],
            "target_repo_url": target["repo_url"],
            "target_repo_branch": target["base_branch"],
            "target_repo_revision": target["base_revision"],
            "advisor_branch": trial["advisor_branch"],
            "program_path": "",
            "advisor": True,
            "names": trial["student_name"],
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
            "wandb_entity": manifest["wandb_entity"],
            "wandb_project": manifest["wandb_project"],
            "wandb_run_group": trial["wandb_group"],
            "trial_index": trial["trial_index"],
            "trial_seed": trial["trial_seed"],
            "timeout_minutes": manifest["training_timeout_minutes"],
            "poll_interval_s": POLL_INTERVAL_SECONDS,
            "poll_jitter_s": POLL_JITTER_SECONDS,
            "stale_wip_seconds": STALE_WIP_SECONDS,
            "gh_history_scope": "fresh",
            "start_gate_path": manifest["start_gate_path"],
            "extra_instructions": "",
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
        trial["research_tag"] for _target, trial in iter_trial_specs(manifest)
    )
    launch_count = sum(1 for _target, _trial in iter_trial_specs(manifest))
    argv = [
        str(CUTOFF_SCRIPT),
        "--run-slug",
        manifest["run_id"],
        "--tags-csv",
        target_tags,
        "--expected-pods",
        str(2 * launch_count),
        "--expected-deployments",
        str(2 * launch_count),
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
    tags = ",".join(
        trial["research_tag"] for _target, trial in iter_trial_specs(manifest)
    )
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


def run_parallel(commands: Sequence[Sequence[str]]) -> None:
    failures = []
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_checked, command) for command in commands]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(error)
    if failures:
        raise failures[0]


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
    n_trials: int = DEFAULT_N_TRIALS,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    manifest = build_manifest(
        config,
        validate_run_id(run_id),
        web_search=web_search,
        dry_run=dry_run,
        training_timeout_minutes=training_timeout_minutes,
        total_timeout_hours=total_timeout_hours,
        n_trials=n_trials,
        cutoff_image=cutoff_image,
    )
    path = manifest_path(results_dir, run_id)
    if path.exists():
        raise FileExistsError(f"eval run ID already exists: {run_id}")
    write_json(path, manifest)
    resources_started = False
    try:
        with tempfile.TemporaryDirectory(prefix="senpai-eval-") as directory:
            launch_paths = []
            for target, trial in iter_trial_specs(manifest):
                launch_config = target_launch_config(
                    config, manifest, target, trial
                )
                target_config_path = (
                    Path(directory)
                    / f"{target['name']}-{trial['trial_name']}.yaml"
                )
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

            if not dry_run and time.time() >= manifest["deadline_epoch"]:
                raise TimeoutError("total eval deadline elapsed during launch")
            launch_commands = [
                [
                    sys.executable,
                    str(LAUNCH_SCRIPT),
                    "--config_path",
                    str(target_config_path),
                ]
                for target_config_path in launch_paths
            ]
            if dry_run:
                for command in launch_commands:
                    run_checked(command)
            else:
                run_parallel(launch_commands)
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
    identity = {
        "run_id": str(getattr(run, "id", "unknown")),
        "name": str(getattr(run, "name", "")),
        "url": str(getattr(run, "url", "")),
        "state": str(getattr(run, "state", "unknown")),
    }
    for attribute in ("group", "job_type", "commit", "created_at"):
        if value := getattr(run, attribute, None):
            identity[attribute] = str(value)
    if tags := getattr(run, "tags", None):
        identity["tags"] = list(tags)
    return identity


def nonempty_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value)


def nonempty_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
    )


def config_contract_error(
    run: object,
    manifest: Mapping[str, Any],
    target: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> str | None:
    """Return why a W&B run cannot represent one full eval trial."""

    if str(getattr(run, "state", "unknown")) != "finished":
        return "W&B run did not finish"
    if str(getattr(run, "group", "")) != trial["wandb_group"]:
        return "W&B run object has the wrong group"
    config = dict(getattr(run, "config", {}) or {})
    expected = {
        "wandb_entity": manifest["wandb_entity"],
        "wandb_project": manifest["wandb_project"],
        "wandb_group": trial["wandb_group"],
        "wandb_run_group": trial["wandb_group"],
        "senpai_trial_index": trial["trial_index"],
        "senpai_trial_seed": trial["trial_seed"],
        "senpai_timeout_minutes": manifest["training_timeout_minutes"],
        "git_dirty": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            return f"config {key} does not match the eval contract"
    if COMMIT_RE.fullmatch(str(config.get("git_commit", ""))) is None:
        return "config git_commit is not a full commit SHA"

    if target["name"] == "nanogpt":
        return nanogpt_config_contract_error(config, run_summary(run))
    return tandem_config_contract_error(config, run_summary(run))


def shard_manifest_valid(
    value: object,
    *,
    expected_names: Sequence[str],
) -> bool:
    if not nonempty_sequence(value):
        return False
    names = [str(item.get("name", "")) for item in value if isinstance(item, Mapping)]
    return names == list(expected_names) and all(
        isinstance(item, Mapping)
        and item.get("bytes") == NANOGPT_SHARD_BYTES
        for item in value
    )


def nanogpt_config_contract_error(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str | None:
    expected = {
        "benchmark": NANOGPT_BENCHMARK,
        "run_kind": "full-training",
        "num_trials": 1,
        "val_tokens": NANOGPT_VAL_TOKENS,
        "target_val_loss": NANOGPT_TARGET_LOSS,
        "stat_sig_delta": NANOGPT_SIGNIFICANCE_DELTA,
        "data_contract": NANOGPT_DATA_CONTRACT,
        "metric_contract": NANOGPT_METRIC_CONTRACT,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            return f"config {key} does not match the NanoGPT contract"
    if SHA256_RE.fullmatch(str(config.get("source_sha256", ""))) is None:
        return "config source_sha256 is missing"
    if config.get("seed") != config.get("senpai_trial_seed"):
        return "config seed does not match the eval seed"
    if not nonempty_mapping(config.get("model_config")):
        return "config model_config is missing"
    if not nonempty_sequence(config.get("optimizer_groups")):
        return "config optimizer_groups is missing"
    train_shards = config.get("train_shards")
    val_shards = config.get("val_shards")
    if not shard_manifest_valid(
        train_shards,
        expected_names=NANOGPT_TRAIN_SHARDS,
    ):
        return "config train_shards does not match the full data contract"
    if not shard_manifest_valid(
        val_shards,
        expected_names=NANOGPT_VAL_SHARDS,
    ):
        return "config val_shards does not match the full data contract"
    expected_summary = {
        "eval/completed": True,
        "eval/data_contract_satisfied": True,
        "eval/all_trials_reached_target": True,
        "eval/ranking_eligible": True,
        "eval/train_shard_count": len(train_shards),
        "eval/val_shard_count": len(val_shards),
        "eval/primary_metric_name": NANOGPT_METRIC_CONTRACT["primary"],
        "eval/primary_metric_direction": "minimize",
        "speedrun/statistically_valid": True,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            return f"summary {key} does not match the NanoGPT contract"
    return None


def tandem_config_contract_error(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str | None:
    expected = {
        "debug": False,
        "skip_test": False,
        "splits_dir": "/mnt/new-pvc/datasets/tandemfoil/splits_v2",
        "metric_contract": TANDEM_METRIC_CONTRACT,
        "train_samples": 1499,
        "val_samples": {split: 100 for split in TANDEM_VAL_SPLITS},
        "materialized_split_manifest_sha256": TANDEM_PROTECTED_HASHES[
            "split_manifest_sha256"
        ],
        "data_contract_satisfied": True,
        **TANDEM_PROTECTED_HASHES,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            return f"config {key} does not match the TandemFoil contract"
    if config.get("seed") != config.get("senpai_trial_seed"):
        return "config seed does not match the eval seed"
    if SHA256_RE.fullmatch(str(config.get("training_source_sha256", ""))) is None:
        return "config training_source_sha256 is missing"
    for key in ("model_config", "optimizer_config", "scheduler_config"):
        if not nonempty_mapping(config.get(key)):
            return f"config {key} is missing"
    expected_summary = {
        "eval/completed": True,
        "eval/ranking_eligible": True,
        "eval/data_contract_satisfied": True,
        "eval/full_test_splits": len(TANDEM_SPLITS),
        "eval/primary_metric_name": TANDEM_METRIC_CONTRACT["primary"],
        "eval/primary_metric_direction": "minimize",
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            return f"summary {key} does not match the TandemFoil contract"
    return None


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


def resolve_adjudication(
    trial: Mapping[str, Any],
    valid_runs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    value = trial.get("adjudication", {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{trial['trial_name']} adjudication must be a mapping")
    status = value.get("status", "pending")
    if status not in {"pending", "accepted", "rejected"}:
        raise ValueError(
            f"{trial['trial_name']} has unknown adjudication status: {status}"
        )
    evidence = value.get("evidence", {})
    if not isinstance(evidence, Mapping):
        raise ValueError(
            f"{trial['trial_name']} adjudication evidence must be a mapping"
        )
    selected_run_id = value.get("selected_run_id")
    selected = next(
        (run for run in valid_runs if run["run_id"] == selected_run_id),
        None,
    )
    if status == "accepted" and selected is None:
        raise ValueError(
            f"{trial['trial_name']} accepted adjudication must select an eligible run"
        )
    if status != "accepted" and selected_run_id is not None:
        raise ValueError(
            f"{trial['trial_name']} {status} adjudication cannot select a run"
        )
    return (
        {
            "status": status,
            "selected_run_id": selected_run_id,
            "evidence": dict(evidence),
        },
        selected,
    )


def score_trial_runs(
    manifest: Mapping[str, Any],
    target: Mapping[str, Any],
    trial: Mapping[str, Any],
    runs: Iterable[object],
) -> dict[str, Any]:
    scorer = score_nanogpt_run if target["name"] == "nanogpt" else score_tandem_run
    valid = []
    rejected = []
    states: Counter[str] = Counter()
    for run in runs:
        identity = run_identity(run)
        states[identity["state"]] += 1
        reason = config_contract_error(run, manifest, target, trial)
        score = None
        if reason is None:
            score, reason = scorer(run)
        if score is not None:
            score["commit_sha"] = str(
                dict(getattr(run, "config", {}) or {})["git_commit"]
            )
            valid.append(score)
        else:
            rejected.append({**identity, "reason": reason or "unscored"})
    valid.sort(key=lambda item: item["score"])
    adjudication, final_result = resolve_adjudication(trial, valid)
    return {
        "trial_index": trial["trial_index"],
        "trial_name": trial["trial_name"],
        "trial_seed": trial["trial_seed"],
        "research_tag": trial["research_tag"],
        "wandb_group": trial["wandb_group"],
        "advisor_branch": trial["advisor_branch"],
        "student_name": trial["student_name"],
        "total_runs": sum(states.values()),
        "eligible_runs": len(valid),
        "states": dict(sorted(states.items())),
        "raw_candidate": valid[0] if valid else None,
        "adjudication_status": adjudication["status"],
        "adjudication_evidence": adjudication["evidence"],
        "selected": final_result,
        "candidate_runs": valid,
        "rejected_runs": rejected,
    }


def score_distribution(final_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(result["score"]) for result in final_results]
    if not scores:
        return {
            "count": 0,
            "scores": [],
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "population_variance": None,
            "population_stddev": None,
            "coefficient_of_variation": None,
        }
    mean_score = fmean(scores)
    standard_deviation = pstdev(scores)
    return {
        "count": len(scores),
        "scores": scores,
        "mean": mean_score,
        "median": median(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "population_variance": pvariance(scores),
        "population_stddev": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / mean_score if mean_score else None
        ),
    }


def aggregate_target_trials(
    target: Mapping[str, Any],
    trial_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(trial["adjudication_status"] for trial in trial_results)
    adjudicated_trials = statuses["accepted"] + statuses["rejected"]
    states: Counter[str] = Counter()
    for trial in trial_results:
        states.update(trial["states"])
    final_results = [
        {
            "trial_index": trial["trial_index"],
            "trial_name": trial["trial_name"],
            "trial_seed": trial["trial_seed"],
            "research_tag": trial["research_tag"],
            "wandb_group": trial["wandb_group"],
            "advisor_branch": trial["advisor_branch"],
            "student_name": trial["student_name"],
            **trial["selected"],
        }
        for trial in trial_results
        if trial["selected"] is not None
    ]
    return {
        "name": target["name"],
        "label": target["label"],
        "primary_metric": target["primary_metric"],
        "metric_unit": target["metric_unit"],
        "direction": "minimize",
        "trial_count": len(trial_results),
        "accepted_trials": len(final_results),
        "adjudicated_trials": adjudicated_trials,
        "adjudication_statuses": dict(sorted(statuses.items())),
        "run_states": dict(sorted(states.items())),
        "total_runs": sum(trial["total_runs"] for trial in trial_results),
        "eligible_runs": sum(
            trial["eligible_runs"] for trial in trial_results
        ),
        "final_results": final_results,
        "distribution": score_distribution(final_results),
        "trials": list(trial_results),
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
        f"- Independent trials per target: `{manifest['n_trials']}`",
        f"- Cutoff: `{manifest.get('cutoff_status', 'unknown')}` "
        f"(`{manifest.get('cutoff_arm_reason', 'unknown')}`)",
        f"- W&B project: `{manifest['wandb_entity']}/{manifest['wandb_project']}`",
        "",
        "| Target | Accepted / adjudicated / trials | Eligible / total runs | "
        "Primary metric | Mean | Stddev | Range |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for result in target_results:
        distribution = result["distribution"]
        score_range = (
            f"{format_score(distribution['minimum'])}–"
            f"{format_score(distribution['maximum'])}"
            if distribution["count"]
            else "—"
        )
        lines.append(
            f"| {result['label']} | {result['accepted_trials']} / "
            f"{result['adjudicated_trials']} / {result['trial_count']} | "
            f"{result['eligible_runs']} / "
            f"{result['total_runs']} | `{result['primary_metric']}` | "
            f"{format_score(distribution['mean'])} | "
            f"{format_score(distribution['population_stddev'])} | "
            f"{score_range} |"
        )

    for result in target_results:
        lines.extend(("", f"## {result['label']}", ""))
        lines.extend(
            (
                "| Trial | Adjudication | Eligible / total | Raw candidate | Final result |",
                "|---|---|---:|---:|---:|",
            )
        )
        for trial in result["trials"]:
            raw = trial["raw_candidate"]
            final = trial["selected"]
            lines.append(
                f"| {trial['trial_name']} | "
                f"{trial['adjudication_status']} | "
                f"{trial['eligible_runs']} / {trial['total_runs']} | "
                f"{format_score(raw['score'] if raw else None)} | "
                f"{format_score(final['score'] if final else None)} |"
            )
        if not result["final_results"]:
            lines.extend(
                (
                    "",
                    "No trial has an accepted final result. Raw metric minima "
                    "remain candidates only.",
                )
            )
        for trial in result["trials"]:
            lines.extend(("", f"### {trial['trial_name']}", ""))
            lines.append(f"- W&B group: `{trial['wandb_group']}`")
            lines.append(f"- Advisor branch: `{trial['advisor_branch']}`")
            lines.append(f"- Seed: `{trial['trial_seed']}`")
            if evidence := trial["adjudication_evidence"]:
                lines.append(f"- Evidence: `{json.dumps(evidence, sort_keys=True)}`")
            if trial["rejected_runs"]:
                lines.append("- Unscored runs:")
                for rejected in trial["rejected_runs"]:
                    lines.append(
                        f"  - `{rejected['run_id']}` ({rejected['state']}): "
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
            "n_trials": manifest["n_trials"],
            "training_timeout_minutes": manifest["training_timeout_minutes"],
            "total_timeout_hours": manifest["total_timeout_hours"],
            "senpai_repo_url": manifest["senpai_repo_url"],
            "senpai_repo_revision": manifest["senpai_repo_revision"],
            "evaluator_revision": manifest["evaluator_revision"],
            "evaluator_dirty": manifest["evaluator_dirty"],
            "evaluator_sha256": manifest["evaluator_sha256"],
            "adjudicator_sha256": manifest["adjudicator_sha256"],
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
                    "trials": [
                        {
                            "trial_index": trial["trial_index"],
                            "trial_name": trial["trial_name"],
                            "trial_seed": trial["trial_seed"],
                            "research_tag": trial["research_tag"],
                            "wandb_group": trial["wandb_group"],
                            "advisor_branch": trial["advisor_branch"],
                            "student_name": trial["student_name"],
                            "adjudication": trial["adjudication"],
                        }
                        for trial in target["trials"]
                    ],
                }
                for target in manifest["targets"]
            ],
        },
    )
    logged: dict[str, Any] = {}
    for result in target_results:
        prefix = f"eval/{result['name']}"
        distribution = result["distribution"]
        logged[f"{prefix}/trials_total"] = result["trial_count"]
        logged[f"{prefix}/trials_accepted"] = result["accepted_trials"]
        logged[f"{prefix}/trials_accepted_fraction"] = (
            result["accepted_trials"] / result["trial_count"]
        )
        logged[f"{prefix}/trials_adjudicated"] = result["adjudicated_trials"]
        logged[f"{prefix}/trials_adjudicated_fraction"] = (
            result["adjudicated_trials"] / result["trial_count"]
        )
        logged[f"{prefix}/total_runs"] = result["total_runs"]
        logged[f"{prefix}/eligible_runs"] = result["eligible_runs"]
        logged[f"{prefix}/eligible_run_fraction"] = (
            result["eligible_runs"] / result["total_runs"]
            if result["total_runs"]
            else 0.0
        )
        for status, count in result["adjudication_statuses"].items():
            logged[f"{prefix}/adjudication/{status}"] = count
        for state, count in result["run_states"].items():
            logged[f"{prefix}/run_state/{state}"] = count
        for name, value in distribution.items():
            if name == "scores" or value is None:
                continue
            logged[f"{prefix}/distribution/{name}"] = value

        rows = []
        scatter_rows = []
        for trial in result["trials"]:
            final = trial["selected"]
            raw = trial["raw_candidate"]
            rows.append(
                [
                    trial["trial_index"],
                    trial["trial_name"],
                    trial["trial_seed"],
                    trial["adjudication_status"],
                    trial["wandb_group"],
                    trial["advisor_branch"],
                    trial["eligible_runs"],
                    trial["total_runs"],
                    raw["score"] if raw else None,
                    final["score"] if final else None,
                    final["run_id"] if final else None,
                    final["url"] if final else None,
                    json.dumps(
                        trial["adjudication_evidence"], sort_keys=True
                    ),
                ]
            )
            if final:
                logged[
                    f"{prefix}/trial/{trial['trial_index']}/final_primary"
                ] = final["score"]
                scatter_rows.append(
                    [trial["trial_index"], final["score"], final["run_id"]]
                )
        table = wandb.Table(
            columns=[
                "trial_index",
                "trial_name",
                "trial_seed",
                "adjudication_status",
                "wandb_group",
                "advisor_branch",
                "eligible_runs",
                "total_runs",
                "raw_candidate_score",
                "final_score",
                "selected_run_id",
                "selected_run_url",
                "adjudication_evidence",
            ],
            data=rows,
        )
        logged[f"{prefix}/trial_results"] = table
        if scatter_rows:
            scatter_table = wandb.Table(
                columns=["trial_index", "score", "selected_run_id"],
                data=scatter_rows,
            )
            logged[f"{prefix}/score_scatter"] = wandb.plot.scatter(
                scatter_table,
                "trial_index",
                "score",
                title=f"{result['label']} accepted final scores",
            )
    logged["eval/targets_with_accepted_results"] = sum(
        result["accepted_trials"] > 0 for result in target_results
    )
    logged["eval/targets_fully_adjudicated"] = sum(
        result["adjudicated_trials"] == result["trial_count"]
        for result in target_results
    )
    logged["eval/total_targets"] = len(target_results)
    logged["eval/trials_total"] = sum(
        result["trial_count"] for result in target_results
    )
    logged["eval/trials_accepted"] = sum(
        result["accepted_trials"] for result in target_results
    )
    logged["eval/trials_adjudicated"] = sum(
        result["adjudicated_trials"] for result in target_results
    )
    run.log(logged)
    run.summary["report_markdown"] = markdown
    url = run.url
    run.finish()
    return url


def eval_github_reader(manifest: Mapping[str, Any]) -> GitHubReader:
    token = resolve_github_token(
        ROOT / ".env",
        tuple(manifest.get("custom_secret_env_names", [])),
    )
    return GitHubReader(SecretStr(token))


def persist_frozen_advisor_heads(
    manifest: dict[str, Any],
    results_dir: Path,
    github: GitHubReads,
) -> None:
    path = manifest_path(results_dir, manifest["run_id"])
    for target, trial in iter_trial_specs(manifest):
        frozen_head = trial.get("adjudication_frozen_head_sha")
        if frozen_head is not None:
            if not isinstance(frozen_head, str) or not frozen_head:
                raise ValueError(
                    f"{trial['trial_name']} frozen advisor head must be non-empty"
                )
            continue
        trial["adjudication_frozen_head_sha"] = freeze_advisor_head(
            target, trial, github
        )
        write_json(path, manifest)


DECISION_FIELDS = (
    "status",
    "selected_run_id",
    "reason",
    "pr_number",
    "result_commit_sha",
    "merge_commit_sha",
    "result_digest",
    "score",
)


def semantic_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {key: decision.get(key) for key in DECISION_FIELDS}


def record_adjudication(
    trial: dict[str, Any],
    trial_result: dict[str, Any],
    decision: Mapping[str, Any],
) -> None:
    evidence = dict(decision["evidence"])
    frozen_head = str(evidence["frozen_advisor_head"])
    previous_head = trial.get("adjudication_frozen_head_sha")
    if previous_head is not None and previous_head != frozen_head:
        raise ValueError(
            f"{trial['trial_name']} adjudication changed its frozen advisor head"
        )
    current_semantics = semantic_decision(decision)
    previous_adjudication = trial.get("adjudication")
    previous_evidence = (
        previous_adjudication.get("evidence")
        if isinstance(previous_adjudication, Mapping)
        else None
    )
    previous_semantics = (
        previous_evidence.get("decision")
        if isinstance(previous_evidence, Mapping)
        else None
    )
    if previous_semantics is not None:
        if not isinstance(previous_semantics, Mapping):
            raise ValueError(
                f"{trial['trial_name']} persisted decision must be a mapping"
            )
        persisted = dict(previous_semantics)
        if (
            persisted.get("status") != previous_adjudication.get("status")
            or persisted.get("selected_run_id")
            != previous_adjudication.get("selected_run_id")
        ):
            raise ValueError(
                f"{trial['trial_name']} persisted adjudication is inconsistent"
            )
        if persisted != current_semantics:
            raise RuntimeError(
                f"{trial['trial_name']} adjudication changed after it was persisted"
            )
        return

    evidence["decision"] = current_semantics
    trial["adjudication_frozen_head_sha"] = frozen_head
    trial["adjudication"] = {
        "status": decision["status"],
        "selected_run_id": decision["selected_run_id"],
        "evidence": evidence,
    }
    adjudication, selected = resolve_adjudication(
        trial, trial_result["candidate_runs"]
    )
    trial_result["adjudication_status"] = adjudication["status"]
    trial_result["adjudication_evidence"] = adjudication["evidence"]
    trial_result["selected"] = selected


def report_eval(
    manifest: dict[str, Any],
    results_dir: Path,
    *,
    log_wandb: bool,
    api: object | None = None,
    github: GitHubReads | None = None,
) -> tuple[dict[str, Any], str]:
    completed = manifest.get("status") == "completed"
    if completed:
        verify_report_sources(manifest)
    if log_wandb:
        if not completed:
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
    if completed:
        github = github or eval_github_reader(manifest)
        persist_frozen_advisor_heads(manifest, results_dir, github)
    api = api or wandb.Api()
    project = f"{manifest['wandb_entity']}/{manifest['wandb_project']}"
    target_results = []
    for target in manifest["targets"]:
        trial_results = []
        for trial in target["trials"]:
            runs = api.runs(project, filters={"group": trial["wandb_group"]})
            trial_result = score_trial_runs(manifest, target, trial, runs)
            if completed:
                decision = adjudicate_trial(
                    target,
                    trial,
                    github,
                    trial_result["candidate_runs"],
                    frozen_head_sha=trial["adjudication_frozen_head_sha"],
                )
                record_adjudication(trial, trial_result, decision)
                write_json(manifest_path(results_dir, manifest["run_id"]), manifest)
            trial_results.append(trial_result)
        target_results.append(aggregate_target_trials(target, trial_results))
    markdown = render_markdown(manifest, target_results)
    report = {
        "schema_version": 2,
        "run_id": manifest["run_id"],
        "n_trials": manifest["n_trials"],
        "generated_at": utc_now(),
        "eval_status": manifest.get("status", "unknown"),
        "provenance": {
            "senpai_repo_url": manifest["senpai_repo_url"],
            "senpai_repo_revision": manifest["senpai_repo_revision"],
            "evaluator_revision": manifest["evaluator_revision"],
            "evaluator_dirty": manifest["evaluator_dirty"],
            "evaluator_sha256": manifest["evaluator_sha256"],
            "adjudicator_sha256": manifest["adjudicator_sha256"],
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
            "trials": [
                {
                    "target": target["name"],
                    "trial_index": trial["trial_index"],
                    "trial_name": trial["trial_name"],
                    "trial_seed": trial["trial_seed"],
                    "research_tag": trial["research_tag"],
                    "wandb_group": trial["wandb_group"],
                    "advisor_branch": trial["advisor_branch"],
                    "student_name": trial["student_name"],
                }
                for target, trial in iter_trial_specs(manifest)
            ],
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
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="independent Senpai replications per target (default: 3)",
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
            n_trials=args.n_trials,
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
