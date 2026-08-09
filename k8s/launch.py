#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
import json
import posixpath
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai_agent.launch_context import render_launch_context

from launch_helpers import (
    ensure_advisor_branch,
    ensure_target_repo_labels,
    existing_advisor_deployments,
    existing_role_metadata,
    existing_student_names,
    expand_student_names,
    is_immutable_image_reference,
    kubectl_apply,
    kubectl_command,
    pod_template_hash,
    preflight_check_anthropic_api_key,
    preflight_check_exa_api_key,
    preflight_check_openai_api_key,
    preflight_check_student_name_availability,
    preflight_check_target_repo_access,
    preflight_check_target_repo_branch,
    preflight_check_wandb_api_key,
    preflight_check_wandb_inference,
    render_configmap,
    render_launch_secret,
    render_template,
    resolve_anthropic_api_key,
    resolve_exa_api_key,
    resolve_github_token,
    resolve_openai_api_key,
    resolve_wandb_api_key,
    routing_labels,
    source_revision_for_image,
    target_repo_slug,
)

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SUPERVISOR_TEMPLATE = Path(__file__).parent / "operational-supervisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"


@dataclass
class Args:
    """Launch senpai advisor and/or student agents on Kubernetes."""

    tag: str  # research tag (e.g. mar13)
    target_repo_url: str  # problem-package repo (entrypoint clones this into $PROBLEM_DIR; agent commits/PRs land here) — REQUIRED, no default
    target_repo_branch: str = ""  # target repo branch used as the base when creating advisor_branch; empty = target repo default branch
    problem_dir: str = "target/"  # active problem directory — entrypoint clones target_repo_url here (from senpai.yaml)
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students to launch (ignored if --names is provided)
    student_prefix: str = ""  # make assignment labels unique across parallel launches using the same base names
    gpus_per_student: int = 1  # GPUs requested by each student pod
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    repo_url: str = (
        "https://github.com/wandb/senpai.git"  # git repo URL (senpai runner)
    )
    repo_revision: str = (
        ""  # exact runner commit; derived from :sha-<commit> image tags
    )
    advisor_image: str = ""  # advisor source-SHA tag or image digest — REQUIRED
    student_image: str = ""  # student source-SHA tag or image digest — REQUIRED
    kube_context: str = ""  # kubectl context; empty uses the current context
    namespace: str = "default"  # Kubernetes namespace for all launch resources
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity (team or username)
    wandb_project: str = "senpai-v1"  # W&B project name
    advisor_model: str = "openai/gpt-5.6-sol"
    advisor_reasoning_effort: str = "xhigh"
    student_model: str = "openai/gpt-5.6-sol"
    student_reasoning_effort: str = "xhigh"
    smart_model: str = "openai/gpt-5.6-sol"
    smart_reasoning_effort: str = "xhigh"
    fast_model: str = "openai/gpt-5.6-luna"
    fast_reasoning_effort: str = "high"
    frontier_model: str = "openai/gpt-5.6-sol"
    frontier_reasoning_effort: str = "max"
    human_issues: bool = (
        True  # allow human GitHub issue triage; disable for isolated launches
    )
    advisor_branch: str = "schmidhuber"  # branch the advisor works on inside the problem-package repo (students PR into it; created from target_repo_branch if missing)
    gh_history_scope: str = "branch"  # branch=normal track memory, fresh=clean ablation, repo=whole-repo memory
    pvc_claim_name: str = "new-pvc"  # PVC name mounted into pods
    pvc_mount_path: str = (
        "/mnt/new-pvc"  # mount path for the dataset PVC inside the containers
    )
    advisor: bool = False  # also deploy the advisor pod (default: students only)
    operational_supervisor: bool = False  # deploy one independent operational supervisor for this campaign
    extra_instructions: str = (
        ""  # extra prompt text for the advisor: a .md file path or a literal string
    )
    timeout_minutes: float = (
        30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    )
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    poll_interval_s: int = (
        600  # default advisor/student outer-loop sleep between GitHub polls
    )
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
    supervisor_interval_s: int = 900  # operational supervisor wake cadence
    supervisor_research_interval_s: int = 21600  # research-philosophy review cadence
    supervisor_action_cooldown_s: int = 1800  # duplicate intervention cooldown
    start_gate_path: str = ""  # optional shared file path that must exist before advisor/student loops begin
    dry_run: bool = (
        False  # render manifests only: do not apply them or validate credentials
    )
    preflight_only: bool = (
        False  # validate credentials/access only: do not render or apply manifests
    )


MODEL_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic-api-key"),
    "openai": ("OPENAI_API_KEY", "openai-api-key"),
    "wandb": ("WANDB_API_KEY", "wandb-api-key"),
}
REASONING_EFFORTS = {
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "none",
}


def model_provider(model: str) -> str:
    provider, separator, model_name = model.partition("/")
    if not separator or not model_name or provider not in MODEL_PROVIDERS:
        supported = ", ".join(sorted(MODEL_PROVIDERS))
        sys.exit(
            "ERROR: model must be provider/name using a supported provider "
            f"({supported}): {model}"
        )
    return provider


def configured_models(args: Args, role: str | None = None) -> tuple[str, ...]:
    main_models = {
        "advisor": args.advisor_model,
        "student": args.student_model,
    }
    models = (
        args.smart_model,
        args.fast_model,
        args.frontier_model,
    )
    if role is not None:
        return (main_models[role], *models)
    return (*main_models.values(), *models)


def configured_model_providers(args: Args, role: str | None = None) -> set[str]:
    return {model_provider(model) for model in configured_models(args, role)}


def deployed_model_providers(args: Args) -> set[str]:
    providers = configured_model_providers(args, "student")
    if args.advisor or args.operational_supervisor:
        providers |= configured_model_providers(args, "advisor")
    return providers


def _supports_openai_pro(model: str) -> bool:
    normalized = model.lower()
    return normalized == "openai/gpt-5.6" or normalized.startswith("openai/gpt-5.6-")


def validate_model_config(args: Args) -> None:
    profiles = {
        "student": (args.student_model, args.student_reasoning_effort),
        "smart": (args.smart_model, args.smart_reasoning_effort),
        "fast": (args.fast_model, args.fast_reasoning_effort),
        "frontier": (args.frontier_model, args.frontier_reasoning_effort),
    }
    if args.advisor or args.operational_supervisor:
        profiles["advisor"] = (
            args.advisor_model,
            args.advisor_reasoning_effort,
        )
    for name, (model, effort) in profiles.items():
        model_provider(model)
        if effort not in REASONING_EFFORTS:
            choices = ", ".join(sorted(REASONING_EFFORTS))
            sys.exit(f"ERROR: --{name}_reasoning_effort must be one of: {choices}")
        normalized_model = model.lower()
        if normalized_model == "wandb/zai-org/glm-5.2":
            if effort not in {"high", "max"}:
                sys.exit(
                    f"ERROR: --{name}_reasoning_effort={effort} is "
                    f"unsupported for {model}"
                )
            continue
        if effort == "max" and not _supports_openai_pro(model):
            sys.exit(
                f"ERROR: --{name}_reasoning_effort={effort} is unsupported for {model}"
            )


def primary_model_config(args: Args, role: str) -> dict[str, str]:
    model = args.advisor_model if role == "advisor" else args.student_model
    reasoning_effort = (
        args.advisor_reasoning_effort
        if role == "advisor"
        else args.student_reasoning_effort
    )
    return {
        "SENPAI_OPENHANDS_MODEL": model,
        "SENPAI_OPENHANDS_REASONING_EFFORT": reasoning_effort,
    }


def role_model_config(args: Args, role: str) -> dict[str, str]:
    return {
        **primary_model_config(args, role),
        "SENPAI_OPENHANDS_SMART_MODEL": args.smart_model,
        "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": args.smart_reasoning_effort,
        "SENPAI_OPENHANDS_FAST_MODEL": args.fast_model,
        "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": args.fast_reasoning_effort,
        "SENPAI_OPENHANDS_FRONTIER_MODEL": args.frontier_model,
        "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": args.frontier_reasoning_effort,
    }


def model_provider_env(args: Args, role: str, secret_name: str) -> str:
    return provider_env(configured_model_providers(args, role), secret_name)


def provider_env(providers: set[str], secret_name: str) -> str:
    lines = []
    for index, provider in enumerate(sorted(providers - {"wandb"})):
        env_name, secret_key = MODEL_PROVIDERS[provider]
        lines.extend(
            (
                f"{'- name' if index == 0 else '        - name'}: {env_name}",
                "          valueFrom:",
                "            secretKeyRef:",
                f"              name: {secret_name}",
                f"              key: {secret_key}",
            )
        )
    return "\n".join(lines)


def validate_timing_args(args: Args) -> None:
    if args.timeout_minutes <= 0:
        sys.exit("ERROR: --timeout_minutes must be positive")
    if args.max_epochs < 1:
        sys.exit("ERROR: --max_epochs must be at least 1")
    positive = [
        "poll_interval_s",
        "supervisor_interval_s",
        "supervisor_research_interval_s",
        "supervisor_action_cooldown_s",
    ]
    non_negative = [
        "poll_jitter_s",
        "stale_wip_seconds",
    ]
    for name in positive:
        if getattr(args, name) < 1:
            sys.exit(f"ERROR: --{name} must be at least 1")
    for name in non_negative:
        if getattr(args, name) < 0:
            sys.exit(f"ERROR: --{name} must be non-negative")
    if args.start_gate_path:
        gate = posixpath.normpath(args.start_gate_path)
        mount = posixpath.normpath(args.pvc_mount_path)
        inside_mount = gate.startswith(f"{mount.rstrip('/')}/")
        if (
            not posixpath.isabs(gate)
            or gate != args.start_gate_path
            or not posixpath.isabs(mount)
            or not inside_mount
        ):
            sys.exit(
                "ERROR: --start_gate_path must be an absolute normalized file "
                "path beneath the shared PVC --pvc_mount_path"
            )


def build_extra_instructions(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
) -> str:
    return render_launch_context(
        backend=backend,
        gpus_per_student=args.gpus_per_student,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        students=student_list,
        extra_instructions=args.extra_instructions,
    )


def encoded_extra_instructions(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
) -> str:
    return base64.b64encode(
        build_extra_instructions(
            args,
            tag,
            student_list,
            backend=backend,
        ).encode()
    ).decode()


def render_student(
    template: str,
    student_name: str,
    tag: str,
    secret_name: str,
    launch_secret: str,
    args: Args,
) -> str:
    student_configmap_name = f"senpai-config-student-{tag}-{student_name}"
    student_deployment_name = f"senpai-{tag}-{student_name}"
    student_cpu = args.cpu_per_gpu * args.gpus_per_student
    student_memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    configmap = render_configmap(
        name=student_configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data={
            **role_model_config(args, "student"),
            "REPO_URL": args.repo_url,
            "REPO_REVISION": args.repo_revision,
            "TARGET_REPO_URL": args.target_repo_url,
            "TARGET_REPO_BRANCH": args.target_repo_branch,
            "GH_REPO": target_repo_slug(args.target_repo_url),
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_MODE": "online",
            "SENPAI_WANDB_SCOPE": tag,
            "WANDB_JOB_TYPE": student_name,
            "WANDB_TAGS": f"senpai,senpai:{tag},senpai-student:{student_name}",
            "ADVISOR_BRANCH": args.advisor_branch,
            "GH_HISTORY_SCOPE": args.gh_history_scope,
            "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
            "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
            "SENPAI_MAX_EPOCHS": str(args.max_epochs),
            "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
            "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
            "EXTRA_INSTRUCTIONS_B64": encoded_extra_instructions(
                args,
                tag,
                [student_name],
                backend="kubernetes",
            ),
            "PROBLEM_DIR": args.problem_dir,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "SENPAI_START_GATE_PATH": args.start_gate_path,
        },
    )
    deployment = render_template(
        template,
        {
            "STUDENT_DEPLOYMENT_NAME": student_deployment_name,
            "STUDENT_CONFIGMAP_NAME": student_configmap_name,
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "REPO_REVISION": args.repo_revision,
            "STUDENT_IMAGE": args.student_image,
            "ADVISOR_BRANCH": args.advisor_branch,
            "ADVISOR_BRANCH_JSON": json.dumps(args.advisor_branch),
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "STUDENT_CPU": str(student_cpu),
            "STUDENT_MEMORY": f"{student_memory_gi}Gi",
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": model_provider_env(args, "student", secret_name),
        },
    )
    return configmap + "\n---\n" + deployment


def render_advisor(
    template: str,
    tag: str,
    student_list: list[str],
    secret_name: str,
    launch_secret: str,
    args: Args,
) -> str:
    advisor_configmap_name = f"senpai-config-advisor-{tag}"
    advisor_deployment_name = f"senpai-advisor-{tag}"
    data = {
        **role_model_config(args, "advisor"),
        "REPO_URL": args.repo_url,
        "REPO_REVISION": args.repo_revision,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "RESEARCH_TAG": tag,
        "STUDENT_NAMES": ",".join(student_list),
        "GPUS_PER_STUDENT": str(args.gpus_per_student),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": "online",
        "ADVISOR_BRANCH": args.advisor_branch,
        "GH_HISTORY_SCOPE": args.gh_history_scope,
        "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "SENPAI_STALE_WIP_SECONDS": str(args.stale_wip_seconds),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }
    data["EXTRA_INSTRUCTIONS_B64"] = encoded_extra_instructions(
        args,
        tag,
        student_list,
        backend="kubernetes",
    )
    configmap = render_configmap(
        name=advisor_configmap_name,
        labels={"app": "senpai", "role": "advisor", "research-tag": tag},
        data=data,
    )
    deployment = render_template(
        template,
        {
            "ADVISOR_DEPLOYMENT_NAME": advisor_deployment_name,
            "ADVISOR_CONFIGMAP_NAME": advisor_configmap_name,
            "RESEARCH_TAG": tag,
            "REPO_REVISION": args.repo_revision,
            "ADVISOR_IMAGE": args.advisor_image,
            "ADVISOR_BRANCH": args.advisor_branch,
            "ADVISOR_BRANCH_JSON": json.dumps(args.advisor_branch),
            "STUDENT_NAMES": ",".join(student_list),
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": model_provider_env(args, "advisor", secret_name),
        },
    )
    return configmap + "\n---\n" + deployment


def render_operational_supervisor(
    template: str,
    tag: str,
    student_list: list[str],
    secret_name: str,
    launch_secret: str,
    args: Args,
) -> str:
    configmap_name = f"senpai-config-supervisor-{tag}"
    deployment_name = f"senpai-supervisor-{tag}"
    service_account_name = f"senpai-supervisor-{tag}"
    configmap = render_configmap(
        name=configmap_name,
        labels={"app": "senpai", "role": "supervisor", "research-tag": tag},
        data={
            **primary_model_config(args, "advisor"),
            "REPO_URL": args.repo_url,
            "REPO_REVISION": args.repo_revision,
            "GH_REPO": target_repo_slug(args.target_repo_url),
            "RESEARCH_TAG": tag,
            "STUDENT_NAMES": ",".join(student_list),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "SENPAI_WANDB_SCOPE": tag,
            "ADVISOR_BRANCH": args.advisor_branch,
            "SENPAI_SUPERVISOR_INTERVAL_SECONDS": str(args.supervisor_interval_s),
            "SENPAI_SUPERVISOR_RESEARCH_INTERVAL_SECONDS": str(
                args.supervisor_research_interval_s
            ),
            "SENPAI_SUPERVISOR_ACTION_COOLDOWN_SECONDS": str(
                args.supervisor_action_cooldown_s
            ),
            "SENPAI_KUBECTL_NAMESPACE": args.namespace,
        },
    )
    deployment = render_template(
        template,
        {
            "SUPERVISOR_DEPLOYMENT_NAME": deployment_name,
            "SUPERVISOR_CONFIGMAP_NAME": configmap_name,
            "SUPERVISOR_SERVICE_ACCOUNT_NAME": service_account_name,
            "RESEARCH_TAG": tag,
            "REPO_REVISION": args.repo_revision,
            "ADVISOR_IMAGE": args.advisor_image,
            "ADVISOR_BRANCH": args.advisor_branch,
            "ADVISOR_BRANCH_JSON": json.dumps(args.advisor_branch),
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "LAUNCH_SECRET_NAME": secret_name,
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": provider_env(
                {model_provider(args.advisor_model)},
                secret_name,
            ),
        },
    )
    return configmap + "\n---\n" + deployment


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    if min(args.gpus_per_student, args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit(
            "ERROR: --gpus_per_student, --cpu_per_gpu, and --memory_gi_per_gpu must all be at least 1"
        )
    validate_timing_args(args)
    validate_model_config(args)
    if not args.preflight_only:
        for role, image in (
            ("advisor", args.advisor_image),
            ("student", args.student_image),
        ):
            if not is_immutable_image_reference(image):
                sys.exit(
                    f"ERROR: --{role}_image must be an immutable digest or "
                    "a :sha-<40-character-commit> tag"
                )
        try:
            advisor_revision = source_revision_for_image(
                args.advisor_image, args.repo_revision
            )
            student_revision = source_revision_for_image(
                args.student_image, args.repo_revision
            )
        except ValueError as error:
            sys.exit(f"ERROR: {error}")
        if advisor_revision != student_revision:
            sys.exit(
                "ERROR: --advisor_image and --student_image must use the "
                "same source revision"
            )
        args.repo_revision = advisor_revision
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(args.repo_url):
        sys.exit("ERROR: --target_repo_url must be a different repo from --repo_url")

    # Resolve student list before backend-independent GitHub preflight.
    if args.names:
        student_list = [n.strip() for n in args.names.split(",")]
    else:
        student_list = expand_student_names(args.n_students)
    if args.student_prefix:
        student_list = [f"{args.student_prefix}-{name}" for name in student_list]

    model_providers = deployed_model_providers(args)
    github_token = exa_api_key = wandb_api_key = ""
    provider_api_keys: dict[str, str] = {}
    if not args.dry_run or args.preflight_only:
        github_token = resolve_github_token(DOTENV_PATH)
        if "anthropic" in model_providers:
            provider_api_keys["anthropic"] = resolve_anthropic_api_key(DOTENV_PATH)
        if "openai" in model_providers:
            provider_api_keys["openai"] = resolve_openai_api_key(DOTENV_PATH)
        exa_api_key = resolve_exa_api_key(DOTENV_PATH)
        wandb_api_key = resolve_wandb_api_key(DOTENV_PATH)
        preflight_check_target_repo_access(args.target_repo_url, github_token)
        args.target_repo_branch = preflight_check_target_repo_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
        )
        preflight_check_student_name_availability(
            args.target_repo_url,
            github_token,
            student_list,
            args.advisor_branch,
        )
        if anthropic_api_key := provider_api_keys.get("anthropic"):
            preflight_check_anthropic_api_key(anthropic_api_key)
        if openai_api_key := provider_api_keys.get("openai"):
            preflight_check_openai_api_key(openai_api_key)
        if "wandb" in model_providers:
            preflight_check_wandb_inference(
                wandb_api_key,
                args.wandb_entity,
                args.wandb_project,
            )
        preflight_check_exa_api_key(exa_api_key)
        preflight_check_wandb_api_key(wandb_api_key)
        if args.preflight_only:
            print("Preflight OK — credentials and target repo access verified.")
            return

    existing_supervised_students: list[str] | None = None
    if not args.dry_run:
        if args.operational_supervisor:
            advisors = existing_advisor_deployments(
                args.tag,
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
            expected_advisor = f"senpai-advisor-{args.tag}"
            if args.advisor and set(advisors) - {expected_advisor}:
                sys.exit(
                    "ERROR: existing exact-tag advisor Deployments would remain "
                    "alongside the managed advisor; remove or retag them first"
                )
            if not args.advisor and len(advisors) != 1:
                sys.exit(
                    "ERROR: --operational_supervisor without --advisor requires "
                    f"exactly one existing advisor Deployment for tag {args.tag!r}; "
                    f"found {len(advisors)}"
                )
            if not args.advisor:
                advisor_metadata = existing_role_metadata(
                    args.tag,
                    "advisor",
                    kube_context=args.kube_context,
                    namespace=args.namespace,
                )
                advisor_record = advisor_metadata.get(advisors[0], {})
                if set(advisor_metadata) != set(advisors) or any(
                    (
                        advisor_record.get("senpai.wandb.com/source-revision")
                        != args.repo_revision,
                        advisor_record.get("senpai.wandb.com/advisor-branch")
                        != args.advisor_branch,
                    )
                ):
                    sys.exit(
                        "ERROR: the existing advisor Deployment does not match the "
                        "requested Senpai source revision and advisor branch"
                    )
            student_metadata = existing_role_metadata(
                args.tag,
                "student",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
            incompatible_students = {
                student
                for student, record in student_metadata.items()
                if student not in student_list
                and (
                    record.get("senpai.wandb.com/source-revision")
                    != args.repo_revision
                    or record.get("senpai.wandb.com/advisor-branch")
                    != args.advisor_branch
                )
            }
            if incompatible_students:
                names = ", ".join(sorted(incompatible_students))
                sys.exit(
                    "ERROR: existing student Deployments not replaced by this "
                    "launch do not match the requested Senpai source revision "
                    "and advisor branch: "
                    f"{names}"
                )
            existing_supervised_students = list(student_metadata)
            if not args.advisor:
                configured_students = {
                    name
                    for name in advisor_record.get(
                        "senpai.wandb.com/student-names",
                        "",
                    ).split(",")
                    if name
                }
                launched_students = {*existing_supervised_students, *student_list}
                if configured_students != launched_students:
                    sys.exit(
                        "ERROR: incremental supervisor launch would change the "
                        "existing advisor's student inventory; relaunch with "
                        "--advisor"
                    )
        ensure_advisor_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
            args.advisor_branch,
        )
        ensure_target_repo_labels(
            args.target_repo_url,
            github_token,
            routing_labels(args.advisor_branch, student_list),
        )

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
    supervisor_template = SUPERVISOR_TEMPLATE.read_text()
    secret_name = f"senpai-launch-secrets-{args.tag}"
    if args.dry_run:
        provider_api_keys = {
            provider: f"<REDACTED_{MODEL_PROVIDERS[provider][0]}>"
            for provider in model_providers
        }
    launch_secret = render_launch_secret(
        args.tag,
        github_token if not args.dry_run else "<REDACTED_GITHUB_TOKEN>",
        exa_api_key if not args.dry_run else "<REDACTED_EXA_API_KEY>",
        wandb_api_key if not args.dry_run else "<REDACTED_WANDB_API_KEY>",
        anthropic_api_key=provider_api_keys.get("anthropic"),
        openai_api_key=provider_api_keys.get("openai"),
    )

    # --- Apply per-launch secret first (pods reference it on startup) ---
    if args.dry_run:
        print(f"--- Secret: {secret_name} ---")
        print(launch_secret)
        print()
    else:
        kubectl_apply(
            launch_secret,
            f"secret {secret_name}",
            kube_context=args.kube_context,
            namespace=args.namespace,
        )

    # --- Deploy students ---
    for name in student_list:
        manifest = render_student(
            student_template,
            name,
            args.tag,
            secret_name,
            launch_secret,
            args,
        )
        if args.dry_run:
            print(f"--- Student: {name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(
                manifest,
                f"student {name}",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )

    advisor_student_list = student_list
    if (args.advisor or args.operational_supervisor) and not args.dry_run:
        existing_students = (
            existing_supervised_students
            if existing_supervised_students is not None
            else existing_student_names(
                args.tag,
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        )
        advisor_student_list = list(
            dict.fromkeys(
                existing_students + student_list
            )
        )

    # --- Deploy advisor ---
    if args.advisor:
        manifest = render_advisor(
            advisor_template,
            args.tag,
            advisor_student_list,
            secret_name,
            launch_secret,
            args,
        )
        if args.dry_run:
            print("--- Advisor ---")
            print(manifest)
            print()
        else:
            kubectl_apply(
                manifest,
                "advisor",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )

    if args.operational_supervisor:
        manifest = render_operational_supervisor(
            supervisor_template,
            args.tag,
            advisor_student_list,
            secret_name,
            launch_secret,
            args,
        )
        if args.dry_run:
            print("--- Operational supervisor ---")
            print(manifest)
            print()
        else:
            kubectl_apply(
                manifest,
                "operational supervisor",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )

    if not args.dry_run:
        print(f"\nLaunched {len(student_list)} students: {', '.join(student_list)}")
        if args.advisor:
            print("Launched advisor pod")
        if args.operational_supervisor:
            print("Launched operational supervisor pod")
        kubectl = shlex.join(
            kubectl_command(
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        )
        print("\nMonitor:")
        print(f"  {kubectl} get deployments -l research-tag={args.tag}")
        if args.advisor:
            print(f"  {kubectl} get deployment senpai-advisor-{args.tag}")
        if student_list:
            print(
                f"  {kubectl} logs -f "
                f"deployment/senpai-{args.tag}-{student_list[0]}"
            )
        print("\nStop:")
        print(
            f"  {kubectl} delete deployments,configmaps,secrets,"
            "serviceaccounts,roles,rolebindings "
            f"-l research-tag={args.tag}"
        )


if __name__ == "__main__":
    main()
