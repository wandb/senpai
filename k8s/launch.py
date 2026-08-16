#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch Senpai advisor and student agents on Kubernetes, Docker, or AWS."""

import posixpath
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k8s.launch_helpers import (  # noqa: E402
    ensure_advisor_branch,
    ensure_target_repo_labels,
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
    resolve_optional_secret,
    resolve_wandb_api_key,
    routing_labels,
    source_revision_for_image,
)
from senpai.launch.aws_backend import launch_aws, preflight_aws  # noqa: E402
from senpai.launch.aws_mac_backend import (  # noqa: E402
    launch_aws_mac,
    preflight_aws_mac,
)
from senpai.launch.docker_backend import (  # noqa: E402
    launch_docker,
    preflight_docker,
)
from senpai.launch.specs import (  # noqa: E402
    build_advisor_spec,
    build_student_spec,
    target_repo_slug,
)
from senpai_agent.model_compatibility import (  # noqa: E402
    REASONING_EFFORTS,
    supports_reasoning_effort,
)
from senpai_agent.program_context import normalize_program_path  # noqa: E402

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"


@dataclass
class Args:
    """Launch Senpai advisor and/or student agents."""

    tag: str  # research tag (e.g. mar13)
    target_repo_url: str  # problem-package repo (entrypoint clones this into $PROBLEM_DIR; agent commits/PRs land here) — REQUIRED, no default
    backend: str = "kubernetes"  # compute backend: kubernetes, docker, aws, or aws-mac
    target_repo_branch: str = ""  # target repo branch used as the base when creating advisor_branch; empty = target repo default branch
    problem_dir: str = "target/"  # active problem directory — entrypoint clones target_repo_url here (from senpai.yaml)
    program_path: str = ""  # target-repo-relative program.md; blank requires exactly one root/one-level match
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # students to launch on every backend; ignored when --names is set
    student_prefix: str = ""  # make assignment labels unique across parallel launches using the same base names
    gpus_per_student: int = 1  # GPUs allocated to each student on every backend
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    senpai_repo_url: str = "https://github.com/wandb/senpai.git"  # public read-only runner source
    senpai_repo_revision: str = ""  # exact runner commit; derived from :sha-<commit> image tags
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
    local_condenser_max_events: int = 0  # event fuse; 0 selects the model default
    local_condenser_max_tokens: int = 0  # token trigger; 0 selects the model default
    local_condenser_target_events: int = 0  # retained events; 0 selects model default
    human_issues: bool = True  # allow human GitHub issue triage; disable for isolated launches
    advisor_name: str = "advisor"  # neutral advisor identity used in Git, prompts, and traces
    advisor_branch: str = "schmidhuber"  # branch the advisor works on inside the problem-package repo (students PR into it; created from target_repo_branch if missing)
    gh_history_scope: str = "branch"  # branch=normal track memory, fresh=clean ablation, repo=whole-repo memory
    pvc_claim_name: str = "new-pvc"  # PVC name mounted into pods
    pvc_mount_path: str = "/mnt/new-pvc"  # mount path for the dataset PVC inside the containers
    advisor: bool = False  # also deploy the advisor pod (default: students only)
    extra_instructions: str = ""  # shared operator instructions: a .md file path or literal text
    timeout_minutes: float = 30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    poll_interval_s: int = 600  # default advisor/student outer-loop sleep between GitHub polls
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
    start_gate_path: str = ""  # optional shared file path that must exist before advisor/student loops begin
    docker_run_root: str = "~/.senpai/runs"  # host directory for Docker workdirs, state, and credentials
    docker_student_gpu_ids: str = ""  # explicit map such as fern:0,tanjiro:1 or fern:0+1
    data_dir: str = ""  # optional data root: mounted by Docker, uploaded once by AWS
    docker_shm_size: str = "32g"  # /dev/shm size for each Docker student
    docker_ready_timeout_s: int = 600  # wait for every role controller before opening the launch gate
    aws_region: str = ""  # AWS region; empty uses AWS_REGION or the selected CLI profile
    aws_profile: str = ""  # optional AWS CLI profile; empty uses the standard credential chain
    aws_instance_type: str = ""  # EC2 GPU type; empty chooses the smallest supported type for this launch
    aws_ami_id: str = ""  # x86_64 GPU AMI; empty uses the latest AWS Deep Learning Base AMI
    aws_subnet_id: str = ""  # public subnet; empty selects one that offers the instance type
    aws_volume_gib: int = 250  # bootstrap gp3 GiB; must fit the AMI and image-pull peak
    aws_runtime_reserve_gib: int = 80  # total free host disk retained after data upload
    aws_state_root: str = "~/.senpai/aws"  # local lifecycle state and ephemeral SSH keys
    aws_ssh_cidr: str = ""  # SSH source IPv4 /32; empty discovers the launcher's public IP
    aws_ready_timeout_s: int = 1800  # wait for EC2, cloud-init, Docker, and GPU readiness
    aws_data_timeout_s: int = 7200  # maximum time to stream data_dir to the host
    aws_ttl_hours: float = 24.0  # self-termination backstop; 0 disables it only on AWS Mac
    native_run_root: str = "~/.senpai/native"  # native macOS role state and logs
    native_ready_timeout_s: int = 600  # wait for native supervisor leases
    aws_mac_host_ids: str = ""  # existing Dedicated Host IDs, one per student
    aws_mac_subnet_ids: str = ""  # AZ=subnet-id map for the selected hosts
    aws_mac_security_group_id: str = ""  # existing SSH security group
    aws_mac_xcode_app: str = "/Applications/Xcode.app"  # full local Xcode copied to fresh Macs
    aws_mac_xcode_archive: str = ""  # optional prepared Xcode zip; avoids re-archiving
    aws_mac_metal_toolchain_archive: str = ""  # exported Metal toolchain bundle zip
    aws_mac_mlxfast_bundle: str = "~/.local/share/mlxfast/mlxfast.js"  # local CLI bundle installed on every Mac
    aws_mac_official_submit: bool = False  # give every active role the MLXFast submission token
    dry_run: bool = False  # render manifests only: do not apply them or validate credentials
    preflight_only: bool = False  # validate credentials/access only: do not render or apply manifests


MODEL_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic-api-key"),
    "openai": ("OPENAI_API_KEY", "openai-api-key"),
    "wandb": ("WANDB_API_KEY", "wandb-api-key"),
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


def deployed_model_providers(args: Args, *, has_students: bool = True) -> set[str]:
    providers = (
        configured_model_providers(args, "student") if has_students else set()
    )
    if args.advisor:
        providers |= configured_model_providers(args, "advisor")
    return providers


def validate_model_config(args: Args, *, has_students: bool = True) -> None:
    profiles = {
        "smart": (args.smart_model, args.smart_reasoning_effort),
        "fast": (args.fast_model, args.fast_reasoning_effort),
        "frontier": (args.frontier_model, args.frontier_reasoning_effort),
    }
    if has_students:
        profiles["student"] = (
            args.student_model,
            args.student_reasoning_effort,
        )
    if args.advisor:
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
        if not supports_reasoning_effort(model, effort):
            sys.exit(
                f"ERROR: --{name}_reasoning_effort={effort} is unsupported for "
                f"{model}"
            )


def model_provider_env(args: Args, role: str, secret_name: str) -> str:
    lines = []
    providers = sorted(configured_model_providers(args, role) - {"wandb"})
    for index, provider in enumerate(providers):
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
    positive = ["poll_interval_s"]
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


def validate_program_path(args: Args) -> None:
    try:
        normalize_program_path(args.program_path)
    except ValueError as error:
        sys.exit(f"ERROR: --program_path: {error}")


def render_student(
    template: str,
    student_name: str,
    tag: str,
    secret_name: str,
    launch_secret: str,
    args: Args,
) -> str:
    spec = build_student_spec(args, tag, student_name, secrets={})
    student_configmap_name = f"senpai-config-student-{tag}-{student_name}"
    student_deployment_name = f"senpai-{tag}-{student_name}"
    student_cpu = args.cpu_per_gpu * args.gpus_per_student
    student_memory_gi = args.memory_gi_per_gpu * args.gpus_per_student
    configmap = render_configmap(
        name=student_configmap_name,
        labels={"app": "senpai", "role": "student", "research-tag": tag},
        data=spec.env,
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
    spec = build_advisor_spec(args, tag, student_list, secrets={})
    advisor_configmap_name = f"senpai-config-advisor-{tag}"
    advisor_deployment_name = f"senpai-advisor-{tag}"
    configmap = render_configmap(
        name=advisor_configmap_name,
        labels={"app": "senpai", "role": "advisor", "research-tag": tag},
        data=spec.env,
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
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": model_provider_env(args, "advisor", secret_name),
        },
    )
    return configmap + "\n---\n" + deployment


def resolve_student_names(args: Args) -> list[str]:
    names = (
        [name.strip() for name in args.names.split(",") if name.strip()]
        if args.names
        else expand_student_names(args.n_students)
    )
    if args.student_prefix:
        names = [f"{args.student_prefix}-{name}" for name in names]
    if len(names) != len(set(names)):
        sys.exit("ERROR: student names must be unique")
    return names


def resolve_runner_revision(args: Args, *, has_students: bool) -> None:
    if not args.advisor and not has_students:
        sys.exit("ERROR: launch requires at least one advisor or student")
    images = []
    if args.advisor:
        images.append(("advisor", args.advisor_image))
    if has_students:
        images.append(("student", args.student_image))

    revisions = []
    for role, image in images:
        if not is_immutable_image_reference(image):
            sys.exit(
                f"ERROR: --{role}_image must be an immutable digest or "
                "a :sha-<40-character-commit> tag"
            )
        try:
            revisions.append(
                source_revision_for_image(image, args.senpai_repo_revision)
            )
        except ValueError as error:
            sys.exit(f"ERROR: {error}")
    if len(revisions) == 2 and revisions[0] != revisions[1]:
        sys.exit(
            "ERROR: --advisor_image and --student_image must use the same "
            "source revision"
        )
    args.senpai_repo_revision = revisions[0]


def resolve_checkout_revision(args: Args) -> None:
    """Bind a native launch to the exact local Senpai commit."""
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.senpai_repo_revision and args.senpai_repo_revision != head:
        sys.exit(
            "ERROR: --senpai_repo_revision is "
            f"{args.senpai_repo_revision}, but this checkout is {head}"
        )
    args.senpai_repo_revision = head


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    if args.backend not in {"kubernetes", "docker", "aws", "aws-mac"}:
        sys.exit(
            "ERROR: --backend must be one of: kubernetes, docker, aws, aws-mac"
        )
    if min(args.cpu_per_gpu, args.memory_gi_per_gpu) < 1:
        sys.exit("ERROR: --cpu_per_gpu and --memory_gi_per_gpu must be at least 1")
    if args.n_students < 0:
        sys.exit("ERROR: --n_students must be non-negative")
    if args.gpus_per_student < 0:
        sys.exit("ERROR: --gpus_per_student must be non-negative")
    if args.backend == "kubernetes" and args.gpus_per_student < 1:
        sys.exit("ERROR: Kubernetes launches require --gpus_per_student at least 1")
    if args.backend in {"docker", "aws", "aws-mac"} and args.docker_ready_timeout_s < 1:
        sys.exit("ERROR: --docker_ready_timeout_s must be at least 1")
    if args.backend == "aws-mac" and args.native_ready_timeout_s < 1:
        sys.exit("ERROR: --native_ready_timeout_s must be at least 1")
    if (
        args.local_condenser_max_events
        and args.local_condenser_max_events < 12
    ):
        sys.exit(
            "ERROR: --local_condenser_max_events must be 0 or at least 12"
        )
    if min(
        args.local_condenser_max_tokens,
        args.local_condenser_target_events,
    ) < 0:
        sys.exit("ERROR: local condenser token and target limits cannot be negative")
    if (
        args.local_condenser_max_events
        and args.local_condenser_target_events
        and args.local_condenser_target_events >= args.local_condenser_max_events
    ):
        sys.exit(
            "ERROR: --local_condenser_target_events must be less than "
            "--local_condenser_max_events"
        )
    validate_timing_args(args)
    validate_program_path(args)
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(
        args.senpai_repo_url
    ):
        sys.exit(
            "ERROR: --target_repo_url must be a different repo from "
            "--senpai_repo_url"
        )

    student_list = resolve_student_names(args)
    if args.backend == "aws-mac":
        resolve_checkout_revision(args)
    else:
        resolve_runner_revision(args, has_students=bool(student_list))
    validate_model_config(args, has_students=bool(student_list))

    model_providers = deployed_model_providers(
        args,
        has_students=bool(student_list),
    )
    github_token = exa_api_key = wandb_api_key = hf_token = ""
    provider_api_keys: dict[str, str] = {}
    mlxfast_api_token = ""
    if not args.dry_run or args.preflight_only:
        github_token = resolve_github_token(DOTENV_PATH)
        if "anthropic" in model_providers:
            provider_api_keys["anthropic"] = resolve_anthropic_api_key(DOTENV_PATH)
        if "openai" in model_providers:
            provider_api_keys["openai"] = resolve_openai_api_key(DOTENV_PATH)
        exa_api_key = resolve_exa_api_key(DOTENV_PATH)
        wandb_api_key = resolve_wandb_api_key(DOTENV_PATH)
        if "wandb" in model_providers:
            provider_api_keys["wandb"] = wandb_api_key
        hf_token = resolve_optional_secret(DOTENV_PATH, "HF_TOKEN")
        if args.backend == "aws-mac" and args.aws_mac_official_submit:
            mlxfast_api_token = resolve_optional_secret(
                DOTENV_PATH,
                "MLXFAST_API_TOKEN",
            )
            if not mlxfast_api_token:
                sys.exit(
                    "ERROR: --aws_mac_official_submit requires "
                    "MLXFAST_API_TOKEN"
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
                wandb_api_key,
                args.wandb_entity,
                args.wandb_project,
            )
        preflight_check_exa_api_key(exa_api_key)
        preflight_check_wandb_api_key(wandb_api_key)

    common_secrets = {
        "GITHUB_TOKEN": github_token,
        "EXA_API_KEY": exa_api_key,
        "WANDB_API_KEY": wandb_api_key,
        "HF_TOKEN": hf_token,
    }

    def role_secrets(role: str) -> dict[str, str]:
        secrets = {
            **common_secrets,
            **{
                MODEL_PROVIDERS[provider][0]: provider_api_keys.get(provider, "")
                for provider in configured_model_providers(args, role)
            },
        }
        if mlxfast_api_token:
            secrets["MLXFAST_API_TOKEN"] = mlxfast_api_token
        return secrets

    role_specs = [
        build_student_spec(
            args,
            args.tag,
            name,
            role_secrets("student"),
        )
        for name in student_list
    ]
    if args.advisor:
        role_specs.append(
            build_advisor_spec(
                args,
                args.tag,
                student_list,
                role_secrets("advisor"),
            )
        )

    docker_plan = aws_plan = aws_mac_plan = None
    if not args.dry_run or args.preflight_only:
        try:
            if args.backend == "docker":
                docker_plan = preflight_docker(args, role_specs)
            elif args.backend == "aws":
                aws_plan = preflight_aws(args, role_specs)
            elif args.backend == "aws-mac":
                aws_mac_plan = preflight_aws_mac(args, role_specs)
        except (ValueError, RuntimeError) as error:
            sys.exit(f"ERROR: {error}")

    if args.preflight_only:
        details = {
            "kubernetes": (
                "credentials and repository access are ready; Kubernetes access "
                "and capacity were not checked"
            ),
            "docker": "credentials, repository, images, Docker, and CUDA are ready",
            "aws": (
                "credentials, repository, source provenance, and the AWS launch "
                "target are ready"
            ),
            "aws-mac": (
                "credentials, repository, source provenance, and all existing "
                "EC2 Mac hosts are ready"
            ),
        }
        detail = details[args.backend]
        print(f"Preflight OK — {detail}.")
        return

    def prepare_github() -> None:
        if args.dry_run:
            return
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

    if args.backend == "aws":
        try:
            launch_aws(
                args,
                role_specs,
                aws_plan,
                before_start=prepare_github,
            )
        except (ValueError, RuntimeError) as error:
            sys.exit(f"ERROR: {error}")
        return

    if args.backend == "aws-mac":
        try:
            launch_aws_mac(
                args,
                role_specs,
                aws_mac_plan,
                before_start=prepare_github,
            )
        except (ValueError, RuntimeError) as error:
            sys.exit(f"ERROR: {error}")
        return

    prepare_github()

    if args.backend == "docker":
        try:
            launch_docker(args, role_specs, docker_plan)
        except (ValueError, RuntimeError) as error:
            sys.exit(f"ERROR: {error}")
        return

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
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
        hf_token=(
            hf_token
            if not args.dry_run
            else ("<REDACTED_HF_TOKEN>" if hf_token else "")
        ),
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
            print(f"  {kubectl} get deployment senpai-advisor-{args.tag}")
        if student_list:
            print(
                f"  {kubectl} logs -f "
                f"deployment/senpai-{args.tag}-{student_list[0]}"
            )
        print("\nStop:")
        print(
            f"  {kubectl} delete deployments,configmaps,secrets "
            f"-l research-tag={args.tag}"
        )


if __name__ == "__main__":
    main()
