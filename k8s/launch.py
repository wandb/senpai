#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
import os
import posixpath
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import simple_parsing as sp
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launch_helpers import (
    ensure_advisor_branch,
    ensure_target_repo_labels,
    existing_controller_wandb_viewers,
    existing_program_context_secret,
    existing_student_names,
    existing_wandb_viewer_owners,
    expand_student_names,
    is_immutable_image_reference,
    kubernetes_resource_name,
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
    read_program_context_secret,
    render_configmap,
    render_launch_secret,
    render_program_context_secret,
    render_student_wandb_secret,
    render_template,
    require_distinct_wandb_viewers,
    resolve_anthropic_api_key,
    resolve_custom_secrets,
    resolve_exa_api_key,
    resolve_github_token,
    resolve_openai_api_key,
    resolve_student_wandb_api_keys,
    resolve_wandb_api_key,
    resolve_wandb_inference_api_key,
    routing_labels,
    source_revision_for_image,
    student_wandb_api_key_env,
    target_repo_slug,
    validate_kubernetes_label,
)

from senpai_agent.git_transport import (
    PROXY_ENVIRONMENT,
    github_repository_url,
    run_git,
)
from senpai_agent.launch_context import (
    LAUNCH_CONTEXT_ENV,
    load_operator_instructions,
    render_launch_context,
)
from senpai_agent.program_context import (
    PROGRAM_CONTEXT_FILE_ENV,
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    ProgramSystemPrompt,
    decode_program_system_prompt,
    encode_program_system_prompt,
    load_program_system_prompt,
    normalize_program_path,
)
from senpai_agent.secrets import validate_custom_secret_env_names

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"
PROGRAM_CONTEXT_MOUNT_PATH = "/var/run/senpai-context/program-context.b64"


@dataclass
class Args:
    """Launch senpai advisor and/or student agents on Kubernetes."""

    tag: str  # research tag (e.g. mar13)
    target_repo_url: str  # problem-package repo (entrypoint clones this into $PROBLEM_DIR; agent commits/PRs land here) — REQUIRED, no default
    target_repo_branch: str = ""  # target repo branch used as the base when creating advisor_branch; empty = target repo default branch
    problem_dir: str = "target/"  # active problem directory — entrypoint clones target_repo_url here (from senpai.yaml)
    program_path: str = ""  # target-repo-relative program.md; blank requires exactly one root/one-level match
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # number of students to launch (ignored if --names is provided)
    student_prefix: str = ""  # make assignment labels unique across parallel launches using the same base names
    gpus_per_student: int = 1  # GPUs requested by each student pod
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    senpai_repo_url: str = (
        "https://github.com/wandb/senpai.git"  # public read-only runner source
    )
    senpai_repo_revision: str = (
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
    compaction_trigger_tokens: int = 200_000
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
    extra_instructions: str = (
        ""  # shared operator instructions: a .md file path or literal text
    )
    timeout_minutes: float = 30.0  # wall-clock policy in the launch context
    max_epochs: int = 50  # epoch policy in the launch context
    custom_secret_env_names: list[str] = field(
        default_factory=list
    )  # additional shell/.env credentials mounted into every role
    poll_interval_s: int = (
        600  # default advisor/student outer-loop sleep between GitHub polls
    )
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
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
    "wandb": ("WANDB_INFERENCE_API_KEY", "wandb-inference-api-key"),
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
    if args.advisor:
        providers |= configured_model_providers(args, "advisor")
    return providers


def _supports_openai_pro(model: str) -> bool:
    normalized = model.lower()
    return normalized == "openai/gpt-5.6" or normalized.startswith("openai/gpt-5.6-")


def validate_model_config(args: Args) -> None:
    if args.compaction_trigger_tokens < 50_000:
        sys.exit("ERROR: --compaction_trigger_tokens must be at least 50000")
    profiles = {
        "student": (args.student_model, args.student_reasoning_effort),
        "smart": (args.smart_model, args.smart_reasoning_effort),
        "fast": (args.fast_model, args.fast_reasoning_effort),
        "frontier": (args.frontier_model, args.frontier_reasoning_effort),
    }
    if args.advisor:
        profiles["advisor"] = (
            args.advisor_model,
            args.advisor_reasoning_effort,
        )
    for name, (model, effort) in profiles.items():
        provider = model_provider(model)
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
        if (
            effort == "max"
            and provider != "anthropic"
            and not _supports_openai_pro(model)
        ):
            sys.exit(
                f"ERROR: --{name}_reasoning_effort={effort} is unsupported for {model}"
            )


def role_model_config(args: Args, role: str) -> dict[str, str]:
    model = args.advisor_model if role == "advisor" else args.student_model
    reasoning_effort = (
        args.advisor_reasoning_effort
        if role == "advisor"
        else args.student_reasoning_effort
    )
    return {
        "SENPAI_OPENHANDS_MODEL": model,
        "SENPAI_OPENHANDS_REASONING_EFFORT": reasoning_effort,
        "SENPAI_OPENHANDS_SMART_MODEL": args.smart_model,
        "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": args.smart_reasoning_effort,
        "SENPAI_OPENHANDS_FAST_MODEL": args.fast_model,
        "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": args.fast_reasoning_effort,
        "SENPAI_OPENHANDS_FRONTIER_MODEL": args.frontier_model,
        "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": args.frontier_reasoning_effort,
        "SENPAI_COMPACTION_TRIGGER_TOKENS": str(args.compaction_trigger_tokens),
    }


def secret_env_refs(
    references: list[tuple[str, str]], secret_name: str
) -> str:
    lines = []
    for environment_name, secret_key in references:
        lines.extend(
            (
                f"        - name: {environment_name}",
                "          valueFrom:",
                "            secretKeyRef:",
                f"              name: {secret_name}",
                f"              key: {secret_key}",
            )
        )
    return "\n".join(lines)


def model_secret_env_refs(args: Args, role: str) -> list[tuple[str, str]]:
    providers = sorted(configured_model_providers(args, role))
    return [MODEL_PROVIDERS[provider] for provider in providers]


def validate_timing_args(args: Args) -> None:
    if args.timeout_minutes <= 0:
        sys.exit("ERROR: --timeout_minutes must be positive")
    if args.max_epochs < 1:
        sys.exit("ERROR: --max_epochs must be at least 1")
    if args.poll_interval_s < 1:
        sys.exit("ERROR: --poll_interval_s must be at least 1")
    non_negative = [
        "poll_jitter_s",
        "stale_wip_seconds",
    ]
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


def validate_program_path(args: Args) -> None:
    try:
        normalize_program_path(args.program_path)
    except ValueError as error:
        sys.exit(f"ERROR: --program_path: {error}")


def build_launch_context(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
    role: Literal["advisor", "student"],
) -> str:
    return render_launch_context(
        role=role,
        github_repo=target_repo_slug(args.target_repo_url),
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        backend=backend,
        gpus_per_student=args.gpus_per_student,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        students=student_list,
    )


def encoded_launch_context(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
    role: Literal["advisor", "student"],
) -> str:
    return base64.b64encode(
        build_launch_context(
            args,
            tag,
            student_list,
            backend=backend,
            role=role,
        ).encode()
    ).decode()


def encoded_operator_instructions(args: Args) -> str:
    return base64.b64encode(
        load_operator_instructions(args.extra_instructions).encode()
    ).decode()


def load_launch_program_snapshot(
    target_repo_url: str,
    advisor_branch: str,
    program_path: str,
    github_token: str,
) -> ProgramSystemPrompt:
    """Clone the advertised branch head and verify its committed program.md."""

    with TemporaryDirectory(prefix="senpai-program-") as directory:
        repository = Path(directory) / "target.git"
        run_git(
            Path(directory),
            "clone",
            "--bare",
            "--depth",
            "1",
            "--branch",
            advisor_branch,
            "--single-branch",
            "--no-tags",
            "--",
            github_repository_url(target_repo_slug(target_repo_url)),
            str(repository),
            token=SecretStr(github_token),
            # The operator's machine, not a pod: honor its proxy settings.
            extra_env={
                name: os.environ[name]
                for name in PROXY_ENVIRONMENT
                if name in os.environ
            },
        )
        return load_program_system_prompt(
            repository,
            program_path,
        )


def render_student(
    template: str,
    student_name: str,
    tag: str,
    secret_name: str,
    launch_secret: str,
    program_secret_name: str,
    program_secret: str,
    wandb_secret_name: str,
    wandb_secret: str,
    wandb_viewer: str,
    controller_wandb_viewer: str,
    inference_wandb_viewer: str | None,
    args: Args,
    program: ProgramSystemPrompt,
) -> str:
    student_configmap_name = kubernetes_resource_name(
        f"senpai-config-student-{tag}-{student_name}"
    )
    student_deployment_name = kubernetes_resource_name(
        f"senpai-{tag}-{student_name}"
    )
    student_cpu = args.cpu_per_gpu * args.gpus_per_student
    student_memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    configmap = render_configmap(
        name=student_configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data={
            **role_model_config(args, "student"),
            "SENPAI_REPO_URL": args.senpai_repo_url,
            "SENPAI_REPO_REVISION": args.senpai_repo_revision,
            "TARGET_REPO_URL": args.target_repo_url,
            "TARGET_REPO_BRANCH": args.target_repo_branch,
            PROGRAM_PATH_ENV: program.program_path,
            PROGRAM_SOURCE_COMMIT_ENV: program.source_commit,
            PROGRAM_CONTEXT_FILE_ENV: PROGRAM_CONTEXT_MOUNT_PATH,
            "GH_REPO": target_repo_slug(args.target_repo_url),
            "STUDENT_NAME": student_name,
            "RESEARCH_TAG": tag,
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_MODE": "online",
            "ADVISOR_BRANCH": args.advisor_branch,
            "GH_HISTORY_SCOPE": args.gh_history_scope,
            "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
            "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
            "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
            "SENPAI_CUSTOM_SECRET_ENV_NAMES": ",".join(
                args.custom_secret_env_names
            ),
            LAUNCH_CONTEXT_ENV: encoded_launch_context(
                args,
                tag,
                [student_name],
                backend="kubernetes",
                role="student",
            ),
            "EXTRA_INSTRUCTIONS_B64": encoded_operator_instructions(args),
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
            "STUDENT_IMAGE": args.student_image,
            "ADVISOR_BRANCH": args.advisor_branch,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "PROGRAM_CONTEXT_SECRET_NAME": program_secret_name,
            "WANDB_TRAINING_SECRET_NAME": wandb_secret_name,
            "WANDB_VIEWER": base64.b64encode(wandb_viewer.encode()).decode(),
            "CONTROLLER_WANDB_VIEWER": base64.b64encode(
                controller_wandb_viewer.encode()
            ).decode(),
            "INFERENCE_WANDB_VIEWER": base64.b64encode(
                (
                    inference_wandb_viewer
                    if "wandb" in configured_model_providers(args, "student")
                    else ""
                ).encode()
            ).decode(),
            "STUDENT_CPU": str(student_cpu),
            "STUDENT_MEMORY": f"{student_memory_gi}Gi",
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "POD_CONFIG_HASH": pod_template_hash(
                configmap,
                launch_secret,
                program_secret,
                wandb_secret,
            ),
            "MODEL_PROVIDER_ENV": secret_env_refs(
                model_secret_env_refs(args, "student"), secret_name
            ),
            "CUSTOM_SECRET_ENV_REFS": secret_env_refs(
                [(name, name) for name in args.custom_secret_env_names], secret_name
            ),
        },
    )
    return configmap + "\n---\n" + deployment


def render_advisor(
    template: str,
    tag: str,
    student_list: list[str],
    secret_name: str,
    launch_secret: str,
    program_secret_name: str,
    program_secret: str,
    wandb_viewer: str,
    inference_wandb_viewer: str | None,
    args: Args,
    program: ProgramSystemPrompt,
) -> str:
    advisor_configmap_name = kubernetes_resource_name(
        f"senpai-config-advisor-{tag}"
    )
    advisor_deployment_name = kubernetes_resource_name(f"senpai-advisor-{tag}")
    data = {
        **role_model_config(args, "advisor"),
        "SENPAI_REPO_URL": args.senpai_repo_url,
        "SENPAI_REPO_REVISION": args.senpai_repo_revision,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        PROGRAM_PATH_ENV: program.program_path,
        PROGRAM_SOURCE_COMMIT_ENV: program.source_commit,
        PROGRAM_CONTEXT_FILE_ENV: PROGRAM_CONTEXT_MOUNT_PATH,
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
        "SENPAI_CUSTOM_SECRET_ENV_NAMES": ",".join(args.custom_secret_env_names),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }
    data[LAUNCH_CONTEXT_ENV] = encoded_launch_context(
        args,
        tag,
        student_list,
        backend="kubernetes",
        role="advisor",
    )
    data["EXTRA_INSTRUCTIONS_B64"] = encoded_operator_instructions(args)
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
            "ADVISOR_IMAGE": args.advisor_image,
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "PVC_MOUNT_PATH": args.pvc_mount_path,
            "LAUNCH_SECRET_NAME": secret_name,
            "PROGRAM_CONTEXT_SECRET_NAME": program_secret_name,
            "WANDB_VIEWER": base64.b64encode(wandb_viewer.encode()).decode(),
            "INFERENCE_WANDB_VIEWER": base64.b64encode(
                (
                    inference_wandb_viewer
                    if "wandb" in configured_model_providers(args, "advisor")
                    else ""
                ).encode()
            ).decode(),
            "POD_CONFIG_HASH": pod_template_hash(
                configmap,
                launch_secret,
                program_secret,
            ),
            "MODEL_PROVIDER_ENV": secret_env_refs(
                model_secret_env_refs(args, "advisor"), secret_name
            ),
            "CUSTOM_SECRET_ENV_REFS": secret_env_refs(
                [(name, name) for name in args.custom_secret_env_names], secret_name
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
    validate_program_path(args)
    validate_model_config(args)
    try:
        validate_custom_secret_env_names(args.custom_secret_env_names)
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
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
                args.advisor_image, args.senpai_repo_revision
            )
            student_revision = source_revision_for_image(
                args.student_image, args.senpai_repo_revision
            )
        except ValueError as error:
            sys.exit(f"ERROR: {error}")
        if advisor_revision != student_revision:
            sys.exit(
                "ERROR: --advisor_image and --student_image must use the "
                "same source revision"
            )
        args.senpai_repo_revision = advisor_revision
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(
        args.senpai_repo_url
    ):
        sys.exit(
            "ERROR: --target_repo_url must be a different repo from "
            "--senpai_repo_url"
        )

    # Resolve student list before backend-independent GitHub preflight.
    if args.names:
        student_list = [n.strip() for n in args.names.split(",")]
    else:
        student_list = expand_student_names(args.n_students)
    if args.student_prefix:
        student_list = [f"{args.student_prefix}-{name}" for name in student_list]
    try:
        validate_kubernetes_label(args.tag, "--tag")
        for name in student_list:
            validate_kubernetes_label(name, "student name")
    except ValueError as error:
        sys.exit(f"ERROR: {error}")

    model_providers = deployed_model_providers(args)
    github_token = exa_api_key = wandb_api_key = ""
    student_wandb_api_keys: dict[str, str] = {}
    student_wandb_viewers: dict[str, str] = {}
    controller_wandb_viewer = ""
    inference_wandb_viewer: str | None = None
    provider_api_keys: dict[str, str] = {}
    custom_secrets: dict[str, str] = {}
    if not args.dry_run or args.preflight_only:
        custom_secrets = resolve_custom_secrets(
            DOTENV_PATH, args.custom_secret_env_names
        )
        github_token = resolve_github_token(DOTENV_PATH, args.custom_secret_env_names)
        if "anthropic" in model_providers:
            provider_api_keys["anthropic"] = resolve_anthropic_api_key(DOTENV_PATH)
        if "openai" in model_providers:
            provider_api_keys["openai"] = resolve_openai_api_key(DOTENV_PATH)
        if "wandb" in model_providers:
            provider_api_keys["wandb"] = resolve_wandb_inference_api_key(DOTENV_PATH)
        exa_api_key = resolve_exa_api_key(DOTENV_PATH)
        wandb_api_key = resolve_wandb_api_key(DOTENV_PATH)
        student_wandb_api_keys = resolve_student_wandb_api_keys(
            DOTENV_PATH, student_list
        )
        if wandb_api_key in student_wandb_api_keys.values():
            sys.exit(
                "ERROR: controller and student W&B API keys must be distinct"
            )
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
                provider_api_keys["wandb"],
                args.wandb_entity,
                args.wandb_project,
            )
        preflight_check_exa_api_key(exa_api_key)
        controller_wandb_viewer = preflight_check_wandb_api_key(wandb_api_key)
        inference_wandb_viewer = (
            preflight_check_wandb_api_key(provider_api_keys["wandb"])
            if "wandb" in model_providers
            else None
        )
        student_wandb_viewers = {
            name: preflight_check_wandb_api_key(key)
            for name, key in student_wandb_api_keys.items()
        }
        deployed_controller_viewers: dict[str, set[str]] = {}
        active_viewer_owners: dict[str, set[str]] = {}
        if not args.preflight_only:
            deployed_controller_viewers = existing_controller_wandb_viewers(
                args.tag,
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
            active_viewer_owners = existing_wandb_viewer_owners(
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
            replaced_roles = {f"student/{name}" for name in student_list}
            if args.advisor:
                replaced_roles.add("advisor")
            for identity, viewers in deployed_controller_viewers.items():
                if identity not in replaced_roles and viewers != {
                    controller_wandb_viewer
                }:
                    sys.exit(
                        "ERROR: the supplied controller W&B key does not match "
                        f"the deployed {identity.replace('/', ' ')} viewer; redeploy the complete "
                        "fleet to rotate the controller identity"
                    )
        require_distinct_wandb_viewers(
            controller_wandb_viewer,
            student_wandb_viewers,
            inference_viewers=(
                (inference_wandb_viewer,)
                if inference_wandb_viewer is not None
                else ()
            ),
            active_viewer_owners=active_viewer_owners,
            owner_scope=f"tag {args.tag!r}",
        )
        if args.preflight_only:
            print("Preflight OK — credentials and target repo access verified.")
            return

    program = ProgramSystemPrompt(
        program_path=args.program_path or "program.md",
        source_commit="0" * 40,
        content="Resolved and verified during a real launch.",
    )
    if not args.dry_run:
        ensure_advisor_branch(
            args.target_repo_url,
            github_token,
            args.target_repo_branch,
            args.advisor_branch,
        )
        try:
            program = load_launch_program_snapshot(
                args.target_repo_url,
                args.advisor_branch,
                args.program_path,
                github_token,
            )
        except RuntimeError as error:
            sys.exit(f"ERROR: {error}")
        if bound_secret := existing_program_context_secret(
            args.tag,
            kube_context=args.kube_context,
            namespace=args.namespace,
        ):
            bound_program = decode_program_system_prompt(
                read_program_context_secret(
                    bound_secret,
                    kube_context=args.kube_context,
                    namespace=args.namespace,
                )
            )
            if (
                bound_program.program_path != program.program_path
                or bound_program.content != program.content
            ):
                sys.exit(
                    "ERROR: program.md changed for an active launch tag; use a "
                    "new tag so every role receives one immutable policy snapshot"
                )
            program = bound_program
        ensure_target_repo_labels(
            args.target_repo_url,
            github_token,
            routing_labels(args.advisor_branch, student_list),
        )

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
    if args.dry_run:
        provider_api_keys = {
            provider: f"<REDACTED_{MODEL_PROVIDERS[provider][0]}>"
            for provider in model_providers
        }
        custom_secrets = {
            name: f"<REDACTED_{name}>" for name in args.custom_secret_env_names
        }
        student_wandb_api_keys = {
            name: f"<REDACTED_{student_wandb_api_key_env(name)}>"
            for name in student_list
        }
        student_wandb_viewers = {
            name: f"<REDACTED_WANDB_VIEWER_{name.upper()}>"
            for name in student_list
        }
        controller_wandb_viewer = "<REDACTED_WANDB_VIEWER_CONTROLLER>"
        inference_wandb_viewer = (
            "<REDACTED_WANDB_VIEWER_INFERENCE>"
            if "wandb" in model_providers
            else None
        )
    secret_name, launch_secret = render_launch_secret(
        args.tag,
        github_token if not args.dry_run else "<REDACTED_GITHUB_TOKEN>",
        exa_api_key if not args.dry_run else "<REDACTED_EXA_API_KEY>",
        wandb_api_key if not args.dry_run else "<REDACTED_WANDB_API_KEY>",
        anthropic_api_key=provider_api_keys.get("anthropic"),
        openai_api_key=provider_api_keys.get("openai"),
        wandb_inference_api_key=provider_api_keys.get("wandb"),
        custom_secrets=custom_secrets,
    )
    program_secret_name, program_secret = render_program_context_secret(
        args.tag,
        encode_program_system_prompt(program),
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

    if args.dry_run:
        print(f"--- Program context: {program_secret_name} ---")
        print(program_secret)
        print()
    else:
        kubectl_apply(
            program_secret,
            f"program context secret {program_secret_name}",
            kube_context=args.kube_context,
            namespace=args.namespace,
        )

    # --- Deploy students ---
    for name in student_list:
        wandb_secret_name, wandb_secret = render_student_wandb_secret(
            args.tag,
            name,
            student_wandb_api_keys[name],
            student_wandb_viewers[name],
        )
        if args.dry_run:
            print(f"--- W&B writer secret: {name} ---")
            print(wandb_secret)
            print()
        else:
            kubectl_apply(
                wandb_secret,
                f"W&B writer secret for student {name}",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        manifest = render_student(
            student_template,
            name,
            args.tag,
            secret_name,
            launch_secret,
            program_secret_name,
            program_secret,
            wandb_secret_name,
            wandb_secret,
            student_wandb_viewers[name],
            controller_wandb_viewer,
            inference_wandb_viewer,
            args,
            program,
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
    if args.advisor and not args.dry_run:
        advisor_student_list = list(
            dict.fromkeys(
                existing_student_names(
                    args.tag,
                    kube_context=args.kube_context,
                    namespace=args.namespace,
                )
                + student_list
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
            program_secret_name,
            program_secret,
            controller_wandb_viewer,
            inference_wandb_viewer,
            args,
            program,
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

    if not args.dry_run:
        print(f"\nLaunched {len(student_list)} students: {', '.join(student_list)}")
        if args.advisor:
            print("Launched advisor pod")
        kubectl = shlex.join(
            kubectl_command(
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        )
        print("\nMonitor:")
        print(f"  {kubectl} get deployments -l research-tag={args.tag}")
        if args.advisor:
            advisor_name = kubernetes_resource_name(
                f"senpai-advisor-{args.tag}"
            )
            print(f"  {kubectl} get deployment {advisor_name}")
        if student_list:
            student_name = kubernetes_resource_name(
                f"senpai-{args.tag}-{student_list[0]}"
            )
            print(
                f"  {kubectl} logs -f "
                f"deployment/{student_name}"
            )
        print("\nStop:")
        print(
            f"  {kubectl} delete deployments,configmaps,secrets "
            f"-l research-tag={args.tag}"
        )


if __name__ == "__main__":
    main()
