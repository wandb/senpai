#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch Senpai advisor and student agents on Kubernetes, Docker, or AWS."""

import json
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
    content_addressed_name,
    ensure_advisor_branch,
    ensure_target_repo_labels,
    existing_advisor_deployments,
    existing_role_metadata,
    existing_student_names,
    expand_student_names,
    is_immutable_image_reference,
    kubectl_apply,
    kubectl_command,
    kubectl_rollout_status,
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
    render_supervisor_secret,
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

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SUPERVISOR_TEMPLATE = Path(__file__).parent / "operational-supervisor-deployment.yaml"
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
    names: str = ""  # comma-separated student names (e.g. "frieren,fern")
    n_students: int = 4  # students to launch on every backend; ignored when --names is set
    student_prefix: str = ""  # make assignment labels unique across parallel launches using the same base names
    gpus_per_student: int = 1  # GPUs allocated to each student on every backend
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    repo_url: str = "https://github.com/wandb/senpai.git"  # git repo URL (senpai runner)
    repo_revision: str = ""  # exact runner commit; derived from :sha-<commit> image tags
    advisor_image: str = ""  # immutable advisor image; required with --advisor
    student_image: str = ""  # immutable student image; required when students are launched
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
    operational_supervisor: bool = False  # deploy one independent operational supervisor for this campaign
    supervisor_dedicated_namespace: bool = False  # acknowledge that the Kubernetes namespace contains only this campaign
    extra_instructions: str = ""  # extra prompt text for the advisor: a .md file path or a literal string
    timeout_minutes: float = 30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    poll_interval_s: int = 600  # default advisor/student outer-loop sleep between GitHub polls
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
    supervisor_interval_s: int = 900  # operational supervisor wake cadence
    supervisor_research_interval_s: int = 21600  # research-philosophy review cadence
    supervisor_action_cooldown_s: int = 1800  # duplicate intervention cooldown
    supervisor_ready_timeout_s: int = 900  # wait for the supervisor Deployment rollout
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
    elif args.operational_supervisor:
        providers.add(model_provider(args.advisor_model))
    return providers


def validate_model_config(args: Args, *, has_students: bool = True) -> None:
    profiles = {}
    if args.advisor or has_students:
        profiles.update(
            {
                "smart": (args.smart_model, args.smart_reasoning_effort),
                "fast": (args.fast_model, args.fast_reasoning_effort),
                "frontier": (
                    args.frontier_model,
                    args.frontier_reasoning_effort,
                ),
            }
        )
    if has_students:
        profiles["student"] = (
            args.student_model,
            args.student_reasoning_effort,
        )
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
        if not supports_reasoning_effort(model, effort):
            sys.exit(
                f"ERROR: --{name}_reasoning_effort={effort} is unsupported for "
                f"{model}"
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
        "supervisor_ready_timeout_s",
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
    deployment_name = f"senpai-supervisor-{tag}"
    service_account_name = f"senpai-supervisor-{tag}"
    config_data = {
        **primary_model_config(args, "advisor"),
        "SENPAI_OPENHANDS_API_KEY_ENV": MODEL_PROVIDERS[
            model_provider(args.advisor_model)
        ][0],
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
        "SENPAI_SUPERVISOR_SECRET_HANDOFF": "1",
        "SENPAI_SUPERVISOR_TERMINAL_SOCKET": (
            "/run/senpai-terminal/terminal.sock"
        ),
        "SENPAI_SUPERVISOR_REPAIR_SOCKET": "/run/senpai-repair/repair.sock",
    }
    configmap_name = content_addressed_name(
        f"senpai-config-supervisor-{tag}",
        config_data,
    )
    configmap = render_configmap(
        name=configmap_name,
        labels={"app": "senpai", "role": "supervisor", "research-tag": tag},
        data=config_data,
        immutable=True,
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
            "SUPERVISOR_STATE_SUBPATH": f"{tag}/operational-supervisor",
            "SUPERVISOR_STATE_MOUNT_PATH": (
                f"/var/lib/senpai/{tag}/operational-supervisor"
            ),
            "LAUNCH_SECRET_NAME": secret_name,
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": provider_env(
                {model_provider(args.advisor_model)},
                secret_name,
            ),
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
    if not args.advisor and not has_students and not args.operational_supervisor:
        sys.exit(
            "ERROR: launch requires at least one advisor, student, or "
            "operational supervisor"
        )
    images = []
    if args.advisor or args.operational_supervisor:
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
            revisions.append(source_revision_for_image(image, args.repo_revision))
        except ValueError as error:
            sys.exit(f"ERROR: {error}")
    if len(revisions) == 2 and revisions[0] != revisions[1]:
        sys.exit(
            "ERROR: --advisor_image and --student_image must use the same "
            "source revision"
        )
    args.repo_revision = revisions[0]


def resolve_checkout_revision(args: Args) -> None:
    """Bind a native launch to the exact local Senpai commit."""
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.repo_revision and args.repo_revision != head:
        sys.exit(
            f"ERROR: --repo_revision is {args.repo_revision}, but this checkout "
            f"is {head}"
        )
    args.repo_revision = head


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG))
    if args.backend not in {"kubernetes", "docker", "aws", "aws-mac"}:
        sys.exit(
            "ERROR: --backend must be one of: kubernetes, docker, aws, aws-mac"
        )
    if args.operational_supervisor and args.backend != "kubernetes":
        sys.exit(
            "ERROR: --operational_supervisor currently requires "
            "--backend kubernetes"
        )
    if args.operational_supervisor and (
        args.namespace == "default" or not args.supervisor_dedicated_namespace
    ):
        sys.exit(
            "ERROR: --operational_supervisor requires a non-default, "
            "campaign-dedicated --namespace and "
            "--supervisor_dedicated_namespace"
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
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(args.repo_url):
        sys.exit("ERROR: --target_repo_url must be a different repo from --repo_url")

    student_list = resolve_student_names(args)
    launches_roles = args.advisor or bool(student_list)
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
        wandb_api_key = resolve_wandb_api_key(DOTENV_PATH)
        if "wandb" in model_providers:
            provider_api_keys["wandb"] = wandb_api_key
        if launches_roles:
            exa_api_key = resolve_exa_api_key(DOTENV_PATH)
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
        if launches_roles:
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
        if launches_roles:
            preflight_check_exa_api_key(exa_api_key)
        preflight_check_wandb_api_key(wandb_api_key)

    existing_supervised_students: list[str] | None = None
    if (
        args.backend == "kubernetes"
        and args.operational_supervisor
        and not args.dry_run
        and not args.preflight_only
    ):
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
                record.get("senpai.wandb.com/source-revision") != args.repo_revision
                or record.get("senpai.wandb.com/advisor-branch")
                != args.advisor_branch
            )
        }
        if incompatible_students:
            names = ", ".join(sorted(incompatible_students))
            sys.exit(
                "ERROR: existing student Deployments not replaced by this launch "
                "do not match the requested Senpai source revision and advisor "
                f"branch: {names}"
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
                    "existing advisor's student inventory; relaunch with --advisor"
                )

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
        if args.dry_run or not launches_roles:
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
    supervisor_template = SUPERVISOR_TEMPLATE.read_text()
    secret_name = f"senpai-launch-secrets-{args.tag}"
    if args.dry_run:
        provider_api_keys = {
            provider: f"<REDACTED_{MODEL_PROVIDERS[provider][0]}>"
            for provider in model_providers
        }
    launch_secret = ""
    if args.advisor or student_list:
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

        # Role launches own this fixed Secret. A supervisor-only launch must
        # never rewrite credentials referenced by existing role Deployments.
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
        supervisor_provider = model_provider(args.advisor_model)
        provider_secret_name = (
            None
            if supervisor_provider == "wandb"
            else MODEL_PROVIDERS[supervisor_provider][1]
        )
        provider_api_key = (
            None
            if provider_secret_name is None
            else provider_api_keys[supervisor_provider]
        )
        supervisor_secret_name, supervisor_secret = render_supervisor_secret(
            args.tag,
            github_token if not args.dry_run else "<REDACTED_GITHUB_TOKEN>",
            wandb_api_key if not args.dry_run else "<REDACTED_WANDB_API_KEY>",
            provider_secret_name=provider_secret_name,
            provider_api_key=provider_api_key,
        )
        if args.dry_run:
            print(f"--- Secret: {supervisor_secret_name} ---")
            print(supervisor_secret)
            print()
        else:
            kubectl_apply(
                supervisor_secret,
                f"secret {supervisor_secret_name}",
                kube_context=args.kube_context,
                namespace=args.namespace,
            )
        manifest = render_operational_supervisor(
            supervisor_template,
            args.tag,
            advisor_student_list,
            supervisor_secret_name,
            supervisor_secret,
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
            deployment_name = f"senpai-supervisor-{args.tag}"
            try:
                kubectl_rollout_status(
                    deployment_name,
                    timeout_seconds=args.supervisor_ready_timeout_s,
                    kube_context=args.kube_context,
                    namespace=args.namespace,
                )
            except RuntimeError as error:
                rollback = shlex.join(
                    kubectl_command(
                        "rollout",
                        "undo",
                        f"deployment/{deployment_name}",
                        kube_context=args.kube_context,
                        namespace=args.namespace,
                    )
                )
                print(
                    "ERROR: operational supervisor rollout failed: "
                    f"{error}\nRollback: {rollback}",
                    file=sys.stderr,
                )
                raise SystemExit(
                    "operational supervisor rollout failed"
                ) from error

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
