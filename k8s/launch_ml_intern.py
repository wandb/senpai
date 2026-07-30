#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch ML Intern benchmark replicates for TandemFoilSet-Balanced."""

import base64
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

from launch_helpers import (
    kubectl_apply,
    preflight_check_anthropic_api_key,
    preflight_check_target_repo_access,
    render_configmap,
    render_template,
    resolve_anthropic_api_key,
    resolve_github_token,
    resolve_required_secret,
    target_repo_slug,
)

TEMPLATE = Path(__file__).parent / "ml-intern-deployment.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"
DNS_LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
ML_INTERN_PINNED_REF = "c4ac4e6292e82094d1aebcbffaae7202b35083ab"


@dataclass
class Args:
    """Launch Hugging Face ML Intern benchmark jobs on the pai2 cluster."""

    tag: str  # research tag; branch names default to <tag>-rN
    target_repo_url: str  # problem-package repo to clone, edit, train, commit, and push
    base_ref: str = "main"  # target repo base ref for replicate branches
    replicates: int = 5  # number of independent ML Intern launches
    gpus_per_replicate: int = 8  # GPUs exposed to each ML Intern job
    cpu_per_gpu: int = 15  # CPU requested per GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per GPU
    timeout_hours: float = 12.0  # hard pod wall-clock budget
    harvest_grace_minutes: int = 5  # time reserved for result commit/push before the pod deadline
    pvc_claim_name: str = "new-pvc"  # PVC name mounted into jobs
    pvc_mount_path: str = "/mnt/new-pvc"  # mount path for dataset PVC
    image: str = "ghcr.io/wandb/senpai:latest"  # CUDA-capable image used by Senpai students
    senpai_repo_url: str = "https://github.com/wandb/senpai.git"  # runner repo cloned inside the job
    senpai_repo_branch: str = "main"  # runner repo branch
    ml_intern_repo_url: str = "https://github.com/huggingface/ml-intern.git"  # ML Intern source repo
    ml_intern_repo_ref: str = ML_INTERN_PINNED_REF  # pinned ML Intern git ref; package version 0.1.0
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity
    wandb_project: str = "senpai-v1-ml-intern"  # W&B project
    model: str = "anthropic/claude-opus-4-7"  # ML Intern model
    senpai_timeout_minutes: float = 720.0  # target train.py safety timeout; not a per-experiment budget
    default_epochs: int = 999  # suggested no-epoch-limit train.py default
    max_iterations: int = 10_000  # ML Intern headless max LLM requests
    kube_context: str = "pai-2"  # kubectl context for the pai2 cluster
    smoke: bool = False  # render/launch one 1-GPU tiny debug replicate
    smoke_timeout_hours: float = 1.0  # hard wall-clock budget for --smoke
    dry_run: bool = False  # render and validate manifests only
    preflight_only: bool = False  # validate credentials/access only


def render_ml_intern_secret(secret_name: str, tag: str, github_token: str, anthropic_api_key: str, hf_token: str) -> str:
    """Per-launch Secret for ML Intern jobs."""
    return (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        f"  name: {secret_name}\n"
        "  labels:\n"
        "    app: ml-intern\n"
        f"    research-tag: {tag}\n"
        "type: Opaque\n"
        "data:\n"
        f"  github-token: {base64.b64encode(github_token.encode()).decode()}\n"
        f"  anthropic-api-key: {base64.b64encode(anthropic_api_key.encode()).decode()}\n"
        f"  hf-token: {base64.b64encode(hf_token.encode()).decode()}\n"
    )


def preflight_check_kube_context(expected: str) -> None:
    """Verify kubectl points at the pai2 context before launching GPU jobs."""
    print("Preflight: checking kubectl context")
    result = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True, check=True)
    current = result.stdout.strip()
    if current != expected:
        sys.exit(f"ERROR: kubectl current-context is {current!r}; expected {expected!r} for the pai2 cluster.")
    print(f"  OK - kubectl context is {current}")


def preflight_check_wandb_secret() -> None:
    """Verify the shared W&B Kubernetes Secret exists and has the expected key."""
    print("Preflight: checking Kubernetes Secret senpai-secrets/wandb-api-key")
    result = subprocess.run(
        ["kubectl", "get", "secret", "senpai-secrets", "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    if "wandb-api-key" not in payload.get("data", {}):
        sys.exit("ERROR: Kubernetes Secret senpai-secrets is missing key wandb-api-key.")
    print("  OK - W&B secret is present")


def preflight_check_hf_token(hf_token: str) -> None:
    """Verify the Hugging Face token can authenticate."""
    print("Preflight: checking Hugging Face token")
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {hf_token}", "User-Agent": "senpai-ml-intern-preflight"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace").replace(hf_token, "<redacted>")
        sys.exit(f"ERROR: Hugging Face token failed: HTTP {e.code}: {body[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: Hugging Face token failed: {e.reason}")
    print("  OK - Hugging Face token authenticated")


def github_api_json(
    target_repo_url: str,
    github_token: str,
    path: str,
    method: str = "GET",
    payload: dict[str, str] | None = None,
    allow_404: bool = False,
) -> dict | None:
    """Call the GitHub API for target-repo launch setup."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "senpai-ml-intern-launch",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body or b"{}")
    except urllib.error.HTTPError as e:
        if allow_404 and e.code == 404:
            return None
        body = e.read().decode(errors="replace").replace(github_token, "<redacted>")
        sys.exit(f"ERROR: GitHub API {e.code} while setting up {target_repo_url}: {body[:500]}")


def ensure_target_branches(args: Args, github_token: str, branches: list[str]) -> None:
    """Create missing target branches from the requested base ref before pods start."""
    slug = target_repo_slug(args.target_repo_url)
    encoded_base = urllib.parse.quote(args.base_ref, safe="")
    commit = github_api_json(args.target_repo_url, github_token, f"/repos/{slug}/commits/{encoded_base}")
    if commit is None:
        sys.exit(f"ERROR: unable to resolve base ref {args.base_ref!r} in {slug}.")
    base_sha = commit["sha"]
    print(f"Preflight: ensuring ML Intern target branches on {slug} from {args.base_ref} ({base_sha[:7]})")
    for branch in branches:
        ref_name = f"heads/{branch}"
        encoded_ref = urllib.parse.quote(ref_name, safe="/")
        existing = github_api_json(
            args.target_repo_url,
            github_token,
            f"/repos/{slug}/git/ref/{encoded_ref}",
            allow_404=True,
        )
        if existing is not None:
            print(f"  OK - {branch} already exists")
            continue
        github_api_json(
            args.target_repo_url,
            github_token,
            f"/repos/{slug}/git/refs",
            method="POST",
            payload={"ref": f"refs/{ref_name}", "sha": base_sha},
        )
        print(f"  created {branch}")


def effective_replicates(args: Args) -> int:
    return 1 if args.smoke else args.replicates


def effective_gpus(args: Args) -> int:
    return 1 if args.smoke else args.gpus_per_replicate


def effective_wall_seconds(args: Args) -> int:
    hours = args.smoke_timeout_hours if args.smoke else args.timeout_hours
    return int(hours * 3600)


def effective_agent_timeout_seconds(args: Args) -> int:
    wall = effective_wall_seconds(args)
    grace = min(args.harvest_grace_minutes * 60, max(60, wall // 4))
    return wall - grace


def branch_name(args: Args, replicate: int) -> str:
    branch_base = args.tag
    if args.smoke and not branch_base.endswith("-smoke"):
        branch_base = f"{branch_base}-smoke"
    return f"{branch_base}-r{replicate}"


def build_prompt(args: Args, replicate: int, branch: str) -> str:
    mode = "SMOKE DEBUG RUN" if args.smoke else "FULL BENCHMARK RUN"
    wall_hours = args.smoke_timeout_hours if args.smoke else args.timeout_hours
    smoke_instructions = ""
    if args.smoke:
        smoke_instructions = (
            "\nThis is a smoke run. Do not run a full search. Verify the local environment, "
            "confirm one GPU is visible, run one tiny debug training command such as "
            f"`python ./train.py --debug --epochs 1 --agent ml-intern-r{replicate} "
            f"--wandb_group {branch} --wandb_name {branch}/smoke`, then commit a short summary.\n"
        )

    return f"""# ML Intern TandemFoilSet-Balanced Benchmark ({mode})

You are Hugging Face ML Intern running headlessly inside a Kubernetes Job on the pai2 cluster.

## Benchmark contract
- Target repository: {args.target_repo_url}
- Base ref: {args.base_ref}
- Working branch: {branch}. The startup script creates and checks out this branch before ML Intern starts. Work only from this branch. Ignore all other branches and pull requests in the target repo: do not inspect, check out, merge, cherry-pick, or use them for ideas, results, or history. Keep all commits and final pushes on this branch.
- ML Intern model: {args.model}
- W&B project: {args.wandb_entity}/{args.wandb_project}
- Visible GPU budget for this launch: {effective_gpus(args)} GPU(s)
- Hard pod wall-clock budget: {wall_hours:g} hours
- The file `/workspace/ml-intern-benchmark/deadline.txt` contains exact epoch deadlines.

This is a fresh independent replicate. Do not inspect, query, or reuse any previous ML Intern or Senpai W&B groups/runs, old replicate branches, harvested result files, or prior PRs as evidence for experiment choices or metrics. Use this branch's own work, the target repo benchmark docs, public reference material, and the runs you launch inside this pod.

## Target repo context
Before planning experiments or editing code, read the target repo's own benchmark docs: `program.md` and `data/SPLITS.md` if present. Treat those files as the source of truth for the CFD task, input/target shapes, split design, metrics, file boundaries, masking/padding rules, and any physics context. You may read `README.md` for setup/background, but do not mine historical leaderboard or prior-agent result sections for experiment ideas unless they are explicitly part of the benchmark contract. Also inspect the training entrypoint's CLI help before your first training run so you use the exact flags this repo exposes.

Do not try to follow Senpai's advisor/student PR workflow. You are one autonomous ML Intern launch on the branch above; use the repo docs to understand the benchmark, then choose your own experiment strategy.

## Compute policy
Training compute must stay inside this local pai2 pod. Do not launch Hugging Face Jobs, Sandboxes, Spaces, or any other remote compute for training or evaluation. Hugging Face Hub session upload/logging is fine as long as training remains local.

You may decide how to use the visible GPUs: one experiment at a time, multiple parallel one-GPU jobs, or a mixed strategy. If you run jobs in parallel, explicitly pin each subprocess with `CUDA_VISIBLE_DEVICES` so two training jobs do not accidentally use the same GPU.

## Training budget
There is no Senpai per-experiment timeout for this comparison. The environment sets `SENPAI_TIMEOUT_MINUTES={args.senpai_timeout_minutes:g}` only to prevent the TandemFoil training script from using the previous 30-minute cap. The hard budget is the Kubernetes {wall_hours:g}-hour launch kill switch. A 30-minute per-experiment runtime was used elsewhere as an initial baseline; you may follow it, go shorter, or go longer if your strategy benefits.

When you run the target training entrypoint, treat this as the default full command shape unless you deliberately choose otherwise:

```bash
python ./train.py --epochs {args.default_epochs} --agent ml-intern-r{replicate} --wandb_group {branch} --wandb_name "{branch}/<short-description>"
```

Use `--epochs {args.default_epochs}` as the no-epoch-limit default. If you pick a different epoch count, document why.

When stopping your own background training jobs, track and kill the exact PIDs you launched. Do not use broad process-name matching to clean up training jobs. In particular, do not use `pkill`, `killall`, `pgrep -f`, `ps ... | grep ... | xargs kill`, or any command-name scan to find training processes. Record `$!` immediately after each background launch, keep those PIDs in a file or shell variable, and only terminate those exact PIDs.

## Objective and reporting
Optimize TandemFoilSet-Balanced under the target repo's own rules. Prioritize `val_avg/mae_surf_p` while preserving paper-facing `test_avg/mae_surf_p` reporting when final candidates are evaluated.

Before finishing, commit and push to `{branch}`:
- The code/config changes you want credited to this replicate.
- `research/MLINTERN_SUMMARY.md` with your strategy, commands, W&B run/group names, best validation metric, test metric if available, GPU usage strategy, and next recommendation.
- `research/MLINTERN_RESULTS.jsonl` with one JSON object per meaningful run when possible.

Do not delete ML Intern's local `session_logs/` directory or temporary command-output logs. The pod entrypoint will harvest those conversation/tool-call artifacts into `research/` before shutdown.
{smoke_instructions}
"""


def render_replicate(template: str, args: Args, secret_name: str, replicate: int) -> tuple[str, str]:
    branch = branch_name(args, replicate)
    configmap_name = f"ml-intern-config-{args.tag}-{replicate}"
    job_name = f"ml-intern-{args.tag}-{replicate}"
    gpus = effective_gpus(args)
    wall_seconds = effective_wall_seconds(args)
    agent_timeout_seconds = effective_agent_timeout_seconds(args)
    prompt_b64 = base64.b64encode(build_prompt(args, replicate, branch).encode()).decode()
    configmap = render_configmap(
        name=configmap_name,
        labels={
            "app": "ml-intern",
            "role": "agent",
            "replicate": f"r{replicate}",
            "research-tag": args.tag,
        },
        data={
            "SENPAI_REPO_URL": args.senpai_repo_url,
            "SENPAI_REPO_BRANCH": args.senpai_repo_branch,
            "TARGET_REPO_URL": args.target_repo_url,
            "BASE_REF": args.base_ref,
            "TARGET_BRANCH": branch,
            "REPLICATE": str(replicate),
            "RESEARCH_TAG": args.tag,
            "ML_INTERN_REPO_URL": args.ml_intern_repo_url,
            "ML_INTERN_REPO_REF": args.ml_intern_repo_ref,
            "ML_INTERN_MODEL": args.model,
            "ML_INTERN_PROMPT_B64": prompt_b64,
            "ML_INTERN_TIMEOUT_SECONDS": str(agent_timeout_seconds),
            "ML_INTERN_WALL_CLOCK_SECONDS": str(wall_seconds),
            "ML_INTERN_MAX_ITERATIONS": str(args.max_iterations),
            "ML_INTERN_DEFAULT_EPOCHS": str(args.default_epochs),
            "GPUS_PER_REPLICATE": str(gpus),
            "SENPAI_TIMEOUT_MINUTES": f"{args.senpai_timeout_minutes:g}",
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_MODE": "online",
            "PVC_MOUNT_PATH": args.pvc_mount_path,
        },
    )
    job = render_template(
        template,
        {
            "JOB_NAME": job_name,
            "CONFIGMAP_NAME": configmap_name,
            "REPLICATE": str(replicate),
            "RESEARCH_TAG": args.tag,
            "IMAGE": args.image,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "CPU": str(args.cpu_per_gpu * gpus),
            "MEMORY": f"{args.memory_gi_per_gpu * gpus}Gi",
            "GPUS_PER_REPLICATE": str(gpus),
            "ACTIVE_DEADLINE_SECONDS": str(wall_seconds),
        },
    )
    return branch, configmap + "\n---\n" + job


def validate_manifest(manifest: str, args: Args, branch: str, replicate: int) -> None:
    checks = {
        "ml-intern app label": "app: ml-intern",
        "agent role label": "role: agent",
        "research tag": f"research-tag: {args.tag}",
        "replicate label": f'replicate: "{replicate}"',
        "GPU request": f'nvidia.com/gpu: "{effective_gpus(args)}"',
        "PVC claim": f"claimName: {args.pvc_claim_name}",
        "PVC mount": f"mountPath: {args.pvc_mount_path}",
        "active deadline": f"activeDeadlineSeconds: {effective_wall_seconds(args)}",
        "branch": f'TARGET_BRANCH: "{branch}"',
        "model": f'ML_INTERN_MODEL: "{args.model}"',
        "W&B project": f'WANDB_PROJECT: "{args.wandb_project}"',
        "SENPAI timeout": f'SENPAI_TIMEOUT_MINUTES: "{args.senpai_timeout_minutes:g}"',
    }
    missing = [name for name, needle in checks.items() if needle not in manifest]
    if missing:
        sys.exit(f"ERROR: rendered manifest for {branch} failed validation: {', '.join(missing)}")


def validate_args(args: Args) -> None:
    if not DNS_LABEL_RE.fullmatch(args.tag):
        sys.exit("ERROR: --tag must be a lowercase Kubernetes DNS label fragment.")
    if args.replicates < 1:
        sys.exit("ERROR: --replicates must be at least 1.")
    if min(args.gpus_per_replicate, args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit("ERROR: --gpus_per_replicate, --cpu_per_gpu, and --memory_gi_per_gpu must all be at least 1.")
    if effective_wall_seconds(args) <= 0:
        sys.exit("ERROR: timeout must be positive.")
    if effective_agent_timeout_seconds(args) < 60:
        sys.exit("ERROR: timeout is too short after reserving harvest grace.")


def main() -> None:
    args = sp.parse(Args)
    validate_args(args)

    github_token = anthropic_api_key = hf_token = ""
    if not args.dry_run or args.preflight_only:
        github_token = resolve_github_token(DOTENV_PATH)
        anthropic_api_key = resolve_anthropic_api_key(DOTENV_PATH)
        hf_token = resolve_required_secret(DOTENV_PATH, "HF_TOKEN", "Hugging Face token")
        preflight_check_kube_context(args.kube_context)
        preflight_check_target_repo_access(args.target_repo_url, github_token)
        preflight_check_anthropic_api_key(anthropic_api_key)
        preflight_check_hf_token(hf_token)
        preflight_check_wandb_secret()
        if args.preflight_only:
            print("Preflight OK - credentials, target repo access, W&B secret, and pai2 context verified.")
            return

    template = TEMPLATE.read_text()
    secret_name = f"ml-intern-launch-secrets-{args.tag}"
    count = effective_replicates(args)
    branches = [branch_name(args, replicate) for replicate in range(1, count + 1)]

    if args.dry_run:
        print(f"--- Secret: {secret_name} ---")
        print(render_ml_intern_secret(secret_name, args.tag, "<REDACTED_GITHUB_TOKEN>", "<REDACTED_ANTHROPIC_API_KEY>", "<REDACTED_HF_TOKEN>"))
        print()
    else:
        ensure_target_branches(args, github_token, branches)
        kubectl_apply(
            render_ml_intern_secret(secret_name, args.tag, github_token, anthropic_api_key, hf_token),
            f"secret {secret_name}",
        )

    for replicate in range(1, count + 1):
        branch, manifest = render_replicate(template, args, secret_name, replicate)
        validate_manifest(manifest, args, branch, replicate)
        if args.dry_run:
            print(f"--- ML Intern replicate {replicate}: {branch} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, f"ml-intern replicate {replicate}")

    if args.dry_run:
        print(f"Dry-run validation OK for {count} ML Intern replicate manifest(s).")
        return

    print(f"\nLaunched {count} ML Intern job(s): {', '.join(branches)}")
    print("\nMonitor:")
    print(f"  kubectl get jobs,pods -l app=ml-intern,research-tag={args.tag}")
    print(f"  kubectl logs -f job/ml-intern-{args.tag}-1")
    print("\nStop:")
    print(f"  kubectl delete jobs,configmaps,secrets -l app=ml-intern,research-tag={args.tag}")


if __name__ == "__main__":
    main()
