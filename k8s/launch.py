#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
import ipaddress
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import simple_parsing as sp
import yaml
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai_agent.launch_context import (
    LAUNCH_CONTEXT_ENV,
    load_operator_instructions,
    render_launch_context,
)
from senpai_agent.git_transport import (
    PROXY_ENVIRONMENT,
    github_repository_url,
    run_git,
)
from senpai_agent.program_context import (
    PROGRAM_CONTENT_SHA256_ENV,
    PROGRAM_CONTEXT_FILE_ENV,
    PROGRAM_PATH_ENV,
    PROGRAM_SOURCE_COMMIT_ENV,
    ProgramSystemPrompt,
    decode_program_system_prompt,
    encode_program_system_prompt,
    load_program_system_prompt,
    normalize_program_path,
)
from senpai_agent.compaction import (
    DEFAULT_COMPACTION_TRIGGER_EVENTS,
    DEFAULT_COMPACTION_TRIGGER_TOKENS,
    MIN_COMPACTION_TRIGGER_EVENTS,
)
from senpai_agent.custom_openai import (
    DEFAULT_MAX_OUTPUT_TOKENS as DEFAULT_CUSTOM_OPENAI_MAX_OUTPUT_TOKENS,
    CustomOpenAIEndpoint,
    served_model,
)
from senpai_agent.reasoning_effort import openhands_reasoning_effort
from senpai_agent.secrets import validate_custom_secret_env_names

from capacity_observer import (
    CAPACITY_SNAPSHOT_PATH,
    capacity_snapshot_mount,
    capacity_snapshot_volume,
    render_capacity_observer,
)
from launch_helpers import (
    ensure_advisor_branch,
    ensure_new_student_slot,
    ensure_target_repo_labels,
    existing_program_context_secret,
    existing_student_names,
    expand_student_names,
    is_digest_image_reference,
    is_immutable_image_reference,
    kubectl_apply,
    kubectl_command,
    kubectl_create,
    pod_template_hash,
    preflight_check_anthropic_api_key,
    preflight_check_custom_openai,
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
    render_template,
    resolve_anthropic_api_key,
    resolve_custom_openai_api_key,
    resolve_custom_secrets,
    resolve_exa_api_key,
    resolve_github_token,
    resolve_openai_api_key,
    resolve_wandb_api_key,
    routing_labels,
    source_revision_for_image,
    target_repo_slug,
    yaml_insertion,
)
from student_executor import (
    EXECUTOR_SOCKET_PATH,
    EXECUTOR_STATE_PATH,
    executor_container,
    executor_socket_mount,
    executor_volumes,
    student_training_access,
    training_access_name,
)

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
    nodes_per_student: int = 1  # worker nodes available to one supervised run
    gpus_per_student_node: int = 1  # GPUs on each remote training worker
    cpu_per_gpu: int = 15  # CPU requested per student GPU
    memory_gi_per_gpu: int = 120  # memory Gi requested per student GPU
    controller_node_selector: list[str] = field(
        default_factory=list
    )  # optional key=value placement selectors for advisor and multi-node controllers
    student_node_selector: list[str] = field(
        default_factory=list
    )  # optional key=value selectors for the GPU pods students train in (their own pod, or remote workers)
    student_scheduler_name: str = ""  # Kubernetes scheduler for those GPU pods; empty uses the cluster default
    agent_pod_labels: list[str] = field(
        default_factory=list
    )  # extra key=value labels on advisor and student pods, e.g. for NetworkPolicy access
    agent_host_aliases: list[str] = field(
        default_factory=list
    )  # ip=hostname entries added to advisor and student pods' /etc/hosts
    capacity_observer: bool = False  # install a credential-isolated cluster capacity observer
    capacity_node_selector: list[str] = field(
        default_factory=list
    )  # key=value selectors for the observed worker shape
    capacity_tolerations: list[str] | None = None  # JSON tolerations; inherit single-node student defaults when omitted
    capacity_hpc_verification: bool = False  # apply the operator-confirmed CoreWeave HPC eligibility policy
    senpai_repo_url: str = (
        "https://github.com/wandb/senpai.git"  # public read-only runner source
    )
    senpai_repo_revision: str = (
        ""  # exact runner commit; derived from :sha-<commit> image tags
    )
    advisor_image: str = ""  # advisor source-SHA tag or image digest — REQUIRED
    student_image: str = ""  # student source-SHA tag or image digest — REQUIRED
    training_image: str = ""  # optional digest-pinned image for training in separate GPU workers
    executor_image: str = ""  # immutable broker image; required for remote training or the capacity observer
    image_pull_secrets: list[str] = field(
        default_factory=list
    )  # existing registry credential secrets in the launch namespace
    kube_context: str = ""  # kubectl context; empty uses the current context
    namespace: str = "default"  # Kubernetes namespace for all launch resources
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity (team or username)
    wandb_project: str = "senpai-v1"  # W&B project name
    advisor_model: str = "anthropic/claude-opus-5-5"
    advisor_reasoning_effort: str = "xhigh"
    student_model: str = "anthropic/claude-opus-5-5"
    student_reasoning_effort: str = "high"
    smart_model: str = "anthropic/claude-opus-5-5"
    smart_reasoning_effort: str = "xhigh"
    fast_model: str = "anthropic/claude-sonnet-5"
    fast_reasoning_effort: str = "medium"
    frontier_model: str = "anthropic/claude-opus-5-5"
    frontier_reasoning_effort: str = "max"
    compaction_trigger_tokens: int = DEFAULT_COMPACTION_TRIGGER_TOKENS  # input tokens that start provider-native or OpenHands compaction
    compaction_trigger_events: int = DEFAULT_COMPACTION_TRIGGER_EVENTS  # model-visible events that start OpenHands compaction
    custom_openai_base_url: str = ""  # OpenAI-compatible chat-completions base URL for custom_openai/ models
    custom_openai_context_tokens: int = 0  # server window shared by input and output, e.g. vLLM --max-model-len
    custom_openai_max_output_tokens: int = DEFAULT_CUSTOM_OPENAI_MAX_OUTPUT_TOKENS  # per-request output budget, including reasoning
    custom_openai_canonical_model: str = ""  # optional OpenHands capability name, e.g. deepseek/deepseek-v4.1-flash
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

    @property
    def remote_training(self) -> bool:
        return self.nodes_per_student > 1 or bool(self.training_image)


MODEL_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic-api-key"),
    "openai": ("OPENAI_API_KEY", "openai-api-key"),
    "wandb": ("WANDB_API_KEY", "wandb-api-key"),
    "custom_openai": ("CUSTOM_OPENAI_API_KEY", "custom-openai-api-key"),
}
SHARED_MODEL_PROFILES = ("smart", "fast", "frontier")
SENPAI_POD_LABEL_KEYS = {"app", "role", "student", "research-tag"}
# Kubernetes label names and values, and DNS subdomains for prefixes and hostnames.
LABEL_NAME = re.compile(r"[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?")
DNS_SUBDOMAIN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*")


# --- Model configuration -----------------------------------------------------


def model_provider(model: str) -> str:
    provider, separator, model_name = model.partition("/")
    if not separator or not model_name or provider not in MODEL_PROVIDERS:
        supported = ", ".join(sorted(MODEL_PROVIDERS))
        sys.exit(
            "ERROR: model must be provider/name using a supported provider "
            f"({supported}): {model}"
        )
    return provider


def model_profiles(args: Args) -> dict[str, tuple[str, str]]:
    """Every model profile as ``name -> (model, reasoning_effort)``."""
    return {
        "advisor": (args.advisor_model, args.advisor_reasoning_effort),
        "student": (args.student_model, args.student_reasoning_effort),
        "smart": (args.smart_model, args.smart_reasoning_effort),
        "fast": (args.fast_model, args.fast_reasoning_effort),
        "frontier": (args.frontier_model, args.frontier_reasoning_effort),
    }


def role_model_providers(args: Args, role: str) -> set[str]:
    """Providers one role's pods reach: its own model plus the shared profiles."""
    profiles = model_profiles(args)
    return {
        model_provider(profiles[name][0]) for name in (role, *SHARED_MODEL_PROFILES)
    }


def deployed_model_providers(args: Args) -> set[str]:
    providers = role_model_providers(args, "student")
    if args.advisor:
        providers |= role_model_providers(args, "advisor")
    return providers


def deployed_model_profiles(args: Args) -> dict[str, tuple[str, str]]:
    """Profiles the launched roles use; the advisor's only when it is deployed."""
    profiles = model_profiles(args)
    if not args.advisor:
        del profiles["advisor"]
    return profiles


def custom_openai_served_model(args: Args) -> str | None:
    return served_model(model for model, _effort in deployed_model_profiles(args).values())


def custom_openai_efforts(args: Args) -> list[str]:
    return sorted(
        {
            effort
            for model, effort in deployed_model_profiles(args).values()
            if model_provider(model) == "custom_openai"
        }
    )


def validate_custom_openai_endpoint(args: Args) -> None:
    try:
        custom_openai_served_model(args)
        CustomOpenAIEndpoint(
            base_url=args.custom_openai_base_url,
            context_tokens=args.custom_openai_context_tokens,
            max_output_tokens=args.custom_openai_max_output_tokens,
        )
    except ValueError as error:
        sys.exit(f"ERROR: {error}")


def validate_model_config(args: Args) -> None:
    if args.compaction_trigger_tokens < 50_000:
        sys.exit("ERROR: --compaction_trigger_tokens must be at least 50000")
    if args.compaction_trigger_events < MIN_COMPACTION_TRIGGER_EVENTS:
        sys.exit(
            "ERROR: --compaction_trigger_events must be at least "
            f"{MIN_COMPACTION_TRIGGER_EVENTS}"
        )
    for name, (model, effort) in deployed_model_profiles(args).items():
        model_provider(model)
        try:
            openhands_reasoning_effort(effort, model)
        except ValueError as error:
            sys.exit(f"ERROR: --{name}_reasoning_effort: {error}")
    if "custom_openai" in deployed_model_providers(args):
        validate_custom_openai_endpoint(args)


def role_model_config(args: Args, role: str) -> dict[str, str]:
    model, reasoning_effort = model_profiles(args)[role]
    config = {
        "SENPAI_OPENHANDS_MODEL": model,
        "SENPAI_OPENHANDS_REASONING_EFFORT": reasoning_effort,
        "SENPAI_OPENHANDS_SMART_MODEL": args.smart_model,
        "SENPAI_OPENHANDS_SMART_REASONING_EFFORT": args.smart_reasoning_effort,
        "SENPAI_OPENHANDS_FAST_MODEL": args.fast_model,
        "SENPAI_OPENHANDS_FAST_REASONING_EFFORT": args.fast_reasoning_effort,
        "SENPAI_OPENHANDS_FRONTIER_MODEL": args.frontier_model,
        "SENPAI_OPENHANDS_FRONTIER_REASONING_EFFORT": args.frontier_reasoning_effort,
        "SENPAI_COMPACTION_TRIGGER_TOKENS": str(args.compaction_trigger_tokens),
        "SENPAI_COMPACTION_TRIGGER_EVENTS": str(args.compaction_trigger_events),
    }
    if "custom_openai" in role_model_providers(args, role):
        config.update(
            {
                "SENPAI_CUSTOM_OPENAI_BASE_URL": args.custom_openai_base_url,
                "SENPAI_CUSTOM_OPENAI_CONTEXT_TOKENS": str(
                    args.custom_openai_context_tokens
                ),
                "SENPAI_CUSTOM_OPENAI_MAX_OUTPUT_TOKENS": str(
                    args.custom_openai_max_output_tokens
                ),
                "SENPAI_CUSTOM_OPENAI_CANONICAL_MODEL": (
                    args.custom_openai_canonical_model
                ),
            }
        )
    return config


# --- Placement and capacity --------------------------------------------------


def key_value_mapping(values: list[str], *, kind: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"{kind}s must use key=value")
        if key in mapping:
            raise ValueError(f"duplicate {kind}: {key}")
        mapping[key] = value
    return mapping


def controller_node_selector(
    values: list[str], *, kind: str = "controller"
) -> dict[str, str]:
    return key_value_mapping(values, kind=f"{kind} node selector")


def student_gpu_node_selector(args: Args) -> dict[str, str]:
    return controller_node_selector(args.student_node_selector, kind="student")


def student_gpu_placement_config(args: Args) -> dict[str, str]:
    """Let the executor place remote training GPU pods like local student pods."""
    if not args.remote_training:
        return {}
    return {
        "SENPAI_STUDENT_NODE_SELECTOR": json.dumps(student_gpu_node_selector(args)),
        "SENPAI_STUDENT_SCHEDULER_NAME": args.student_scheduler_name,
    }


def is_dns_subdomain(value: str) -> bool:
    return len(value) <= 253 and DNS_SUBDOMAIN.fullmatch(value) is not None


def is_label_key(key: str) -> bool:
    prefix, separator, name = key.rpartition("/")
    return LABEL_NAME.fullmatch(name) is not None and (
        not separator or is_dns_subdomain(prefix)
    )


def agent_pod_labels(args: Args) -> dict[str, str]:
    labels = key_value_mapping(args.agent_pod_labels, kind="agent pod label")
    for key, value in labels.items():
        if not is_label_key(key) or LABEL_NAME.fullmatch(value) is None:
            raise ValueError(f"invalid Kubernetes agent pod label: {key}={value}")
    if reserved := sorted(SENPAI_POD_LABEL_KEYS & labels.keys()):
        raise ValueError(f"agent pod labels cannot replace: {', '.join(reserved)}")
    return labels


def agent_host_aliases(args: Args) -> list[dict]:
    hostnames_by_ip: dict[str, list[str]] = {}
    for item in args.agent_host_aliases:
        ip, separator, hostname = item.partition("=")
        if not separator or not ip or not hostname:
            raise ValueError("agent host aliases must use ip=hostname")
        ipaddress.ip_address(ip)
        if not is_dns_subdomain(hostname):
            raise ValueError(f"agent host alias hostname is invalid: {hostname}")
        hostnames_by_ip.setdefault(ip, []).append(hostname)
    return [
        {"ip": ip, "hostnames": hostnames}
        for ip, hostnames in hostnames_by_ip.items()
    ]


def capacity_config(args: Args) -> dict:
    # Imported here so plain launches skip the Kubernetes training client.
    from senpai_agent.cluster_capacity import CapacityConfig

    if args.capacity_tolerations is None:
        tolerations = _student_tolerations(args) if args.nodes_per_student == 1 else []
    else:
        tolerations = [json.loads(value) for value in args.capacity_tolerations]
    return CapacityConfig(
        nodes=args.nodes_per_student,
        gpus_per_node=args.gpus_per_student_node,
        cpu_per_node=args.cpu_per_gpu * args.gpus_per_student_node,
        memory_gib_per_node=args.memory_gi_per_gpu * args.gpus_per_student_node,
        node_selector=controller_node_selector(args.capacity_node_selector, kind="capacity"),
        tolerations=tolerations,
        hpc_verification=args.capacity_hpc_verification,
    ).model_dump(mode="json")


def _student_resources(args: Args) -> str:
    if args.remote_training:
        return json.dumps(
            {
                "requests": {"cpu": "2", "memory": "8Gi"},
                "limits": {"cpu": "4", "memory": "16Gi"},
            }
        )
    resources = {
        "cpu": str(args.cpu_per_gpu * args.gpus_per_student_node),
        "memory": f"{args.memory_gi_per_gpu * args.gpus_per_student_node}Gi",
        "nvidia.com/gpu": str(args.gpus_per_student_node),
    }
    return json.dumps({"requests": resources, "limits": resources})


def _student_tolerations(args: Args) -> list[dict]:
    if args.remote_training:
        return []
    return [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]


# --- Argument validation -----------------------------------------------------


def validate_timing_args(args: Args) -> None:
    if args.timeout_minutes <= 0:
        sys.exit("ERROR: --timeout_minutes must be positive")
    if args.max_epochs < 1:
        sys.exit("ERROR: --max_epochs must be at least 1")
    if args.poll_interval_s < 1:
        sys.exit("ERROR: --poll_interval_s must be at least 1")
    for name in ("poll_jitter_s", "stale_wip_seconds"):
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


def validate_mount_paths(args: Args) -> str:
    """Normalize ``problem_dir`` in place and return the target workspace mount."""
    problem_dir = args.problem_dir.removesuffix("/")
    if (
        not problem_dir
        or problem_dir in {".", ".."}
        or posixpath.isabs(problem_dir)
        or posixpath.normpath(problem_dir) != problem_dir
        or problem_dir.startswith("../")
    ):
        sys.exit("ERROR: --problem_dir must be a normalized relative child directory")
    tracked_paths = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    runner_roots = {path.partition("/")[0] for path in tracked_paths if path}
    if problem_dir.partition("/")[0] in runner_roots | {".git", ".agents", ".codex"}:
        sys.exit("ERROR: --problem_dir must not cover tracked runner assets or metadata")
    target_mount = f"/workspace/senpai/{problem_dir}"
    if not posixpath.isabs(args.pvc_mount_path):
        sys.exit("ERROR: --pvc_mount_path must be absolute")
    pvc_mount = "/" + posixpath.normpath(args.pvc_mount_path).lstrip("/")
    for private_mount in (target_mount, "/home/senpai/.venvs/senpai-target"):
        if posixpath.commonpath((pvc_mount, private_mount)) in {pvc_mount, private_mount}:
            sys.exit(
                "ERROR: --pvc_mount_path must not overlap the target checkout "
                "or target virtual environment"
            )
    args.problem_dir = problem_dir
    return target_mount


def validate_images(args: Args) -> None:
    """Require immutable role images that share one runner source revision."""
    if args.training_image and not is_digest_image_reference(args.training_image):
        sys.exit("ERROR: --training_image must use an immutable @sha256 digest")
    role_images = [("advisor", args.advisor_image), ("student", args.student_image)]
    if args.remote_training or args.capacity_observer:
        role_images.append(("executor", args.executor_image))
    for role, image in role_images:
        if role == "executor" and not is_digest_image_reference(image):
            sys.exit("ERROR: --executor_image must use an immutable @sha256 digest")
        if not is_immutable_image_reference(image):
            sys.exit(
                f"ERROR: --{role}_image must be an immutable digest or "
                "a :sha-<40-character-commit> tag"
            )
    try:
        revisions = {
            source_revision_for_image(image, args.senpai_repo_revision)
            for _role, image in role_images
        }
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
    if len(revisions) != 1:
        sys.exit("ERROR: role images must use the same source revision")
    args.senpai_repo_revision = revisions.pop()


def validate_args(args: Args) -> None:
    """Reject bad launch arguments before touching credentials or the cluster."""
    validate_mount_paths(args)
    if min(
        args.nodes_per_student,
        args.gpus_per_student_node,
        args.cpu_per_gpu,
        args.memory_gi_per_gpu,
    ) < 1:
        sys.exit(
            "ERROR: --nodes_per_student, --gpus_per_student_node, "
            "--cpu_per_gpu, and --memory_gi_per_gpu must all be at least 1"
        )
    validate_timing_args(args)
    validate_program_path(args)
    validate_model_config(args)
    try:
        controller_node_selector(args.controller_node_selector)
        student_gpu_node_selector(args)
        agent_pod_labels(args)
        agent_host_aliases(args)
        if args.capacity_observer:
            capacity_config(args)
        validate_custom_secret_env_names(args.custom_secret_env_names)
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
    if not args.preflight_only:
        validate_images(args)
    if args.gh_history_scope not in {"branch", "repo", "fresh"}:
        sys.exit("ERROR: --gh_history_scope must be one of: branch, repo, fresh")
    if target_repo_slug(args.target_repo_url) == target_repo_slug(
        args.senpai_repo_url
    ):
        sys.exit(
            "ERROR: --target_repo_url must be a different repo from "
            "--senpai_repo_url"
        )


def resolve_student_names(args: Args) -> list[str]:
    if args.names:
        names = [name.strip() for name in args.names.split(",")]
    else:
        names = expand_student_names(args.n_students)
    if args.student_prefix:
        names = [f"{args.student_prefix}-{name}" for name in names]
    return names


# --- Launch context and program snapshot ------------------------------------


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
        nodes_per_student=args.nodes_per_student,
        gpus_per_student_node=args.gpus_per_student_node,
        timeout_minutes=args.timeout_minutes,
        max_epochs=args.max_epochs,
        tag=tag,
        advisor_branch=args.advisor_branch,
        target_base=args.target_repo_branch,
        gh_history_scope=args.gh_history_scope,
        students=student_list,
        training_image=args.training_image,
    )


def encoded_launch_context(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    role: Literal["advisor", "student"],
) -> str:
    context = build_launch_context(
        args, tag, student_list, backend="kubernetes", role=role
    )
    return base64.b64encode(context.encode()).decode()


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
    """Capture and verify the committed program from one advertised branch head."""

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
            # Capture runs on the operator's machine, so honor its proxy settings.
            extra_env={
                name: os.environ[name]
                for name in PROXY_ENVIRONMENT
                if name in os.environ
            },
        )
        return load_program_system_prompt(repository, program_path)


def render_program_secret(tag: str, program: ProgramSystemPrompt) -> tuple[str, str]:
    try:
        return render_program_context_secret(tag, encode_program_system_prompt(program))
    except ValueError as error:
        sys.exit(f"ERROR: {error}")


# --- Manifest rendering ------------------------------------------------------


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


def role_labels(role: str, tag: str) -> dict[str, str]:
    return {"app": "senpai", "role": role, "research-tag": tag}


def _agent_pod_fields(args: Args) -> dict[str, str]:
    aliases = agent_host_aliases(args)
    return {
        "AGENT_POD_LABELS": yaml_insertion(agent_pod_labels(args), 8),
        "AGENT_HOST_ALIASES": yaml_insertion(
            {"hostAliases": aliases} if aliases else {}, 6
        ),
    }


def shared_configmap_data(
    args: Args, tag: str, role: str, program: ProgramSystemPrompt
) -> dict[str, str]:
    """ConfigMap entries every role receives; callers add role-specific keys."""
    data = {
        **role_model_config(args, role),
        "SENPAI_REPO_URL": args.senpai_repo_url,
        "SENPAI_REPO_REVISION": args.senpai_repo_revision,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        PROGRAM_PATH_ENV: program.program_path,
        PROGRAM_SOURCE_COMMIT_ENV: program.source_commit,
        PROGRAM_CONTENT_SHA256_ENV: program.content_sha256,
        PROGRAM_CONTEXT_FILE_ENV: PROGRAM_CONTEXT_MOUNT_PATH,
        "GH_REPO": target_repo_slug(args.target_repo_url),
        "RESEARCH_TAG": tag,
        "NODES_PER_STUDENT": str(args.nodes_per_student),
        "GPUS_PER_STUDENT_NODE": str(args.gpus_per_student_node),
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": "online",
        "ADVISOR_BRANCH": args.advisor_branch,
        "GH_HISTORY_SCOPE": args.gh_history_scope,
        "SENPAI_ENABLE_HUMAN_ISSUES": "true" if args.human_issues else "false",
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "SENPAI_CUSTOM_SECRET_ENV_NAMES": ",".join(args.custom_secret_env_names),
        "EXTRA_INSTRUCTIONS_B64": encoded_operator_instructions(args),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }
    if args.capacity_observer:
        data["SENPAI_CAPACITY_SNAPSHOT"] = CAPACITY_SNAPSHOT_PATH
    return data


def shared_template_values(
    args: Args,
    tag: str,
    role: str,
    *,
    secret_name: str,
    program_secret_name: str,
    target_workspace_mount: str,
    configmap: str,
    launch_secret: str,
    program_secret: str,
) -> dict[str, str]:
    """Template placeholders shared by the advisor and student deployments."""
    providers = sorted(role_model_providers(args, role) - {"wandb"})
    return {
        "RESEARCH_TAG": tag,
        "IMAGE_PULL_SECRETS": json.dumps(
            [{"name": name} for name in args.image_pull_secrets]
        ),
        "PVC_CLAIM_NAME": args.pvc_claim_name,
        "PVC_MOUNT_PATH": json.dumps(args.pvc_mount_path),
        "TARGET_WORKSPACE_MOUNT": json.dumps(target_workspace_mount),
        "LAUNCH_SECRET_NAME": secret_name,
        "PROGRAM_CONTEXT_SECRET_NAME": program_secret_name,
        "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret, program_secret),
        "MODEL_PROVIDER_ENV": secret_env_refs(
            [MODEL_PROVIDERS[provider] for provider in providers], secret_name
        ),
        "CUSTOM_SECRET_ENV_REFS": secret_env_refs(
            [(name, name) for name in args.custom_secret_env_names], secret_name
        ),
        "CAPACITY_SNAPSHOT_MOUNT": (
            capacity_snapshot_mount() if args.capacity_observer else ""
        ),
        "CAPACITY_SNAPSHOT_VOLUME": (
            capacity_snapshot_volume(tag) if args.capacity_observer else ""
        ),
        **_agent_pod_fields(args),
    }


def render_student(
    template: str,
    student_name: str,
    tag: str,
    secret_name: str,
    launch_secret: str,
    args: Args,
    *,
    program: ProgramSystemPrompt,
    program_secret_name: str,
    program_secret: str,
) -> str:
    target_workspace_mount = validate_mount_paths(args)
    configmap_name = f"senpai-config-student-{tag}-{student_name}"
    remote = args.remote_training
    configmap = render_configmap(
        name=configmap_name,
        labels=role_labels("student", tag),
        data={
            **shared_configmap_data(args, tag, "student", program),
            "STUDENT_NAME": student_name,
            "CPU_PER_STUDENT_GPU": str(args.cpu_per_gpu),
            "MEMORY_GI_PER_STUDENT_GPU": str(args.memory_gi_per_gpu),
            "WANDB_TAGS": f"senpai,{args.advisor_branch},{student_name}",
            LAUNCH_CONTEXT_ENV: encoded_launch_context(
                args, tag, [student_name], role="student"
            ),
            "SENPAI_TRAINING_SNAPSHOT_ROOT": (
                f"{args.pvc_mount_path.rstrip('/')}/.senpai/snapshots/"
                f"{tag}/{student_name}"
            ),
            "SENPAI_KUBERNETES_NAMESPACE": args.namespace,
            "SENPAI_KUBERNETES_EXECUTOR_SOCKET": EXECUTOR_SOCKET_PATH,
            "SENPAI_KUBERNETES_EXECUTOR_STATE": EXECUTOR_STATE_PATH,
            "SENPAI_EXECUTOR_IMAGE": args.executor_image,
            "SENPAI_TRAINING_IMAGE": args.training_image,
            "SENPAI_IMAGE_PULL_SECRETS": json.dumps(args.image_pull_secrets),
            "SENPAI_MAX_TRAINING_TIMEOUT_SECONDS": str(
                round(args.timeout_minutes * 60)
            ),
            "PVC_CLAIM_NAME": args.pvc_claim_name,
            "SENPAI_LAUNCH_SECRET_NAME": secret_name,
            **student_gpu_placement_config(args),
        },
    )
    deployment = render_template(
        template,
        {
            **shared_template_values(
                args,
                tag,
                "student",
                secret_name=secret_name,
                program_secret_name=program_secret_name,
                target_workspace_mount=target_workspace_mount,
                configmap=configmap,
                launch_secret=launch_secret,
                program_secret=program_secret,
            ),
            "STUDENT_DEPLOYMENT_NAME": f"senpai-{tag}-{student_name}",
            "STUDENT_CONFIGMAP_NAME": configmap_name,
            "STUDENT_NAME": student_name,
            "STUDENT_IMAGE": args.student_image,
            "STUDENT_SERVICE_ACCOUNT_NAME": (
                training_access_name(tag, student_name) if remote else "default"
            ),
            "STUDENT_RESOURCES": _student_resources(args),
            "STUDENT_NODE_SELECTOR": json.dumps(
                controller_node_selector(args.controller_node_selector)
                if remote
                else student_gpu_node_selector(args)
            ),
            "STUDENT_SCHEDULER_NAME": yaml_insertion(
                {"schedulerName": args.student_scheduler_name}
                if args.student_scheduler_name and not remote
                else {},
                6,
            ),
            "STUDENT_TOLERATIONS": json.dumps(_student_tolerations(args)),
            "EXECUTOR_SOCKET_MOUNT": executor_socket_mount() if remote else "",
            "KUBERNETES_EXECUTOR_CONTAINER": (
                executor_container(args.executor_image, secret_name, configmap_name)
                if remote
                else ""
            ),
            "KUBERNETES_EXECUTOR_VOLUMES": executor_volumes() if remote else "",
        },
    )
    documents = [configmap]
    if remote:
        documents.append(student_training_access(student_name, tag, args.namespace))
    documents.append(deployment)
    return "\n---\n".join(documents)


def render_advisor(
    template: str,
    tag: str,
    student_list: list[str],
    secret_name: str,
    launch_secret: str,
    args: Args,
    *,
    program: ProgramSystemPrompt,
    program_secret_name: str,
    program_secret: str,
) -> str:
    target_workspace_mount = validate_mount_paths(args)
    configmap_name = f"senpai-config-advisor-{tag}"
    configmap = render_configmap(
        name=configmap_name,
        labels=role_labels("advisor", tag),
        data={
            **shared_configmap_data(args, tag, "advisor", program),
            "STUDENT_NAMES": ",".join(student_list),
            "SENPAI_STALE_WIP_SECONDS": str(args.stale_wip_seconds),
            LAUNCH_CONTEXT_ENV: encoded_launch_context(
                args, tag, student_list, role="advisor"
            ),
        },
    )
    deployment = render_template(
        template,
        {
            **shared_template_values(
                args,
                tag,
                "advisor",
                secret_name=secret_name,
                program_secret_name=program_secret_name,
                target_workspace_mount=target_workspace_mount,
                configmap=configmap,
                launch_secret=launch_secret,
                program_secret=program_secret,
            ),
            "ADVISOR_DEPLOYMENT_NAME": f"senpai-advisor-{tag}",
            "ADVISOR_CONFIGMAP_NAME": configmap_name,
            "ADVISOR_IMAGE": args.advisor_image,
            "CONTROLLER_NODE_SELECTOR": json.dumps(
                controller_node_selector(args.controller_node_selector)
            ),
        },
    )
    return configmap + "\n---\n" + deployment


def validated_student_resources(manifest: str, student_name: str) -> list[dict]:
    """Parse one student manifest and return its Deployment last."""
    try:
        documents = list(yaml.safe_load_all(manifest))
    except yaml.YAMLError as error:
        raise RuntimeError(
            f"student {student_name} manifest contains invalid YAML"
        ) from error

    for index, document in enumerate(documents, start=1):
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("kind"), str)
            or not isinstance(document.get("metadata"), dict)
            or not isinstance(document["metadata"].get("name"), str)
        ):
            raise RuntimeError(
                f"student {student_name} manifest document {index} must have "
                "string kind and metadata.name fields"
            )

    deployments = [
        document for document in documents if document["kind"] == "Deployment"
    ]
    if len(deployments) != 1:
        raise RuntimeError(
            f"student {student_name} manifest must contain exactly one Deployment; "
            f"found {len(deployments)}"
        )
    return [
        document for document in documents if document["kind"] != "Deployment"
    ] + deployments


def render_student_manifests(
    args: Args,
    template: str,
    student_list: list[str],
    secret_name: str,
    launch_secret: str,
    program: ProgramSystemPrompt,
    program_secret_name: str,
    program_secret: str,
) -> list[tuple[str, str, list[dict]]]:
    """Render and structurally validate every student as (name, manifest, documents)."""
    manifests = []
    for name in student_list:
        manifest = render_student(
            template,
            name,
            args.tag,
            secret_name,
            launch_secret,
            args,
            program=program,
            program_secret_name=program_secret_name,
            program_secret=program_secret,
        )
        manifests.append((name, manifest, validated_student_resources(manifest, name)))
    return manifests


# --- Credentials and preflight -----------------------------------------------


@dataclass
class LaunchCredentials:
    github_token: str
    exa_api_key: str
    wandb_api_key: str
    provider_api_keys: dict[str, str]
    custom_secrets: dict[str, str]


def resolve_credentials(
    args: Args, student_list: list[str], model_providers: set[str]
) -> LaunchCredentials:
    """Load operator credentials, then verify GitHub, model, and W&B access."""
    custom_secrets = resolve_custom_secrets(DOTENV_PATH, args.custom_secret_env_names)
    github_token = resolve_github_token(DOTENV_PATH, args.custom_secret_env_names)
    provider_api_keys: dict[str, str] = {}
    if "anthropic" in model_providers:
        provider_api_keys["anthropic"] = resolve_anthropic_api_key(DOTENV_PATH)
    if "openai" in model_providers:
        provider_api_keys["openai"] = resolve_openai_api_key(DOTENV_PATH)
    if "custom_openai" in model_providers:
        provider_api_keys["custom_openai"] = resolve_custom_openai_api_key(DOTENV_PATH)
    credentials = LaunchCredentials(
        github_token=github_token,
        exa_api_key=resolve_exa_api_key(DOTENV_PATH),
        wandb_api_key=resolve_wandb_api_key(DOTENV_PATH),
        provider_api_keys=provider_api_keys,
        custom_secrets=custom_secrets,
    )
    preflight_check_target_repo_access(args.target_repo_url, github_token)
    args.target_repo_branch = preflight_check_target_repo_branch(
        args.target_repo_url, github_token, args.target_repo_branch
    )
    preflight_check_student_name_availability(
        args.target_repo_url, github_token, student_list, args.advisor_branch
    )
    if anthropic_api_key := provider_api_keys.get("anthropic"):
        preflight_check_anthropic_api_key(anthropic_api_key)
    if openai_api_key := provider_api_keys.get("openai"):
        preflight_check_openai_api_key(openai_api_key)
    if custom_openai_api_key := provider_api_keys.get("custom_openai"):
        preflight_check_custom_openai(
            custom_openai_api_key,
            args.custom_openai_base_url,
            custom_openai_served_model(args),
            custom_openai_efforts(args),
            args.custom_openai_context_tokens,
        )
    if "wandb" in model_providers:
        preflight_check_wandb_inference(
            credentials.wandb_api_key, args.wandb_entity, args.wandb_project
        )
    preflight_check_exa_api_key(credentials.exa_api_key)
    preflight_check_wandb_api_key(credentials.wandb_api_key)
    return credentials


def redacted_credentials(args: Args, model_providers: set[str]) -> LaunchCredentials:
    """Placeholders so a dry run renders the launch secret without reading any."""
    return LaunchCredentials(
        github_token="<REDACTED_GITHUB_TOKEN>",
        exa_api_key="<REDACTED_EXA_API_KEY>",
        wandb_api_key="<REDACTED_WANDB_API_KEY>",
        provider_api_keys={
            provider: f"<REDACTED_{MODEL_PROVIDERS[provider][0]}>"
            for provider in model_providers
        },
        custom_secrets={
            name: f"<REDACTED_{name}>" for name in args.custom_secret_env_names
        },
    )


# --- Cluster writes ----------------------------------------------------------


def kube_scope(args: Args) -> dict[str, str]:
    return {"kube_context": args.kube_context, "namespace": args.namespace}


def print_manifest(title: str, manifest: str) -> None:
    print(f"--- {title} ---")
    print(manifest)
    print()


def emit(args: Args, title: str, manifest: str, description: str) -> None:
    """Print a manifest during a dry run; otherwise apply it."""
    if args.dry_run:
        print_manifest(title, manifest)
    else:
        kubectl_apply(manifest, description, **kube_scope(args))


def reserve_launch(
    args: Args, student_list: list[str], secret_name: str, launch_secret: str
) -> None:
    """Claim each student slot and the tag's launch secret before any GitHub writes."""
    for name in student_list:
        ensure_new_student_slot(
            f"senpai-{args.tag}-{name}",
            tag=args.tag,
            student_name=name,
            require_mpijob=args.nodes_per_student > 1,
            **kube_scope(args),
        )
    try:
        kubectl_create(launch_secret, f"secret {secret_name}", **kube_scope(args))
    except RuntimeError as error:
        kubectl = shlex.join(kubectl_command(**kube_scope(args)))
        raise RuntimeError(
            f"{error}\nNo GitHub or controller writes were attempted. If this "
            f"is a stale reservation, inspect the tag before retrying, then "
            f"delete the reservation with: {kubectl} delete secret {secret_name}"
        ) from error


def resolve_launch_program(args: Args, github_token: str) -> ProgramSystemPrompt:
    """Capture the committed program, keeping an active tag's bound snapshot."""
    try:
        program = load_launch_program_snapshot(
            args.target_repo_url, args.advisor_branch, args.program_path, github_token
        )
        bound_secret = existing_program_context_secret(args.tag, **kube_scope(args))
        if not bound_secret:
            return program
        bound_program = decode_program_system_prompt(
            read_program_context_secret(bound_secret, args.tag, **kube_scope(args))
        )
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"ERROR: {error}")
    if (bound_program.program_path, bound_program.content) != (
        program.program_path,
        program.content,
    ):
        sys.exit(
            "ERROR: program.md changed for an active launch tag; use a new tag "
            "so every role receives one immutable policy snapshot"
        )
    return bound_program


def create_student_resources(
    args: Args, student_manifests: list[tuple[str, str, list[dict]]]
) -> None:
    for name, manifest, documents in student_manifests:
        if args.dry_run:
            print_manifest(f"Student: {name}", manifest)
            continue
        try:
            for document in documents:
                kubectl_create(
                    yaml.safe_dump(document, sort_keys=False),
                    f"student {name} {document['kind']} {document['metadata']['name']}",
                    **kube_scope(args),
                )
        except RuntimeError as error:
            raise RuntimeError(
                f"{error}\nLaunch tag {args.tag} remains reserved. Inspect and "
                "remove every stale resource for this tag before retrying."
            ) from error


def print_launch_summary(args: Args, student_list: list[str]) -> None:
    print(f"\nLaunched {len(student_list)} students: {', '.join(student_list)}")
    if args.advisor:
        print("Launched advisor pod")
    kubectl = shlex.join(kubectl_command(**kube_scope(args)))
    print("\nMonitor:")
    print(f"  {kubectl} get deployments -l research-tag={args.tag}")
    if args.advisor:
        print(f"  {kubectl} get deployment senpai-advisor-{args.tag}")
    if student_list:
        print(
            f"  {kubectl} logs -f "
            f"deployment/senpai-{args.tag}-{student_list[0]}"
            " -c student"
        )
    print("\nStop:")
    print(f"  {kubectl} delete deployments -l research-tag={args.tag}")
    workloads = "jobs,mpijobs" if args.nodes_per_student > 1 else "jobs"
    print(f"  {kubectl} delete {workloads} -l research-tag={args.tag}")
    print(
        f"  {kubectl} delete configmaps,secrets,serviceaccounts,roles,rolebindings "
        f"-l research-tag={args.tag}"
    )
    if args.capacity_observer:
        print(
            f"  {kubectl} delete clusterroles,clusterrolebindings "
            f"-l app=senpai-capacity-observer,research-tag={args.tag},"
            f"senpai.wandb.com/namespace={args.namespace}"
        )


def main():
    args = sp.parse(Args, config_path=str(SENPAI_CONFIG), allow_abbrev=False)
    validate_args(args)
    student_list = resolve_student_names(args)
    model_providers = deployed_model_providers(args)

    if args.dry_run and not args.preflight_only:
        credentials = redacted_credentials(args, model_providers)
    else:
        credentials = resolve_credentials(args, student_list, model_providers)
        if args.preflight_only:
            print("Preflight OK — credentials and target repo access verified.")
            return

    student_template = STUDENT_TEMPLATE.read_text()
    advisor_template = ADVISOR_TEMPLATE.read_text()
    secret_name = f"senpai-launch-secrets-{args.tag}"
    launch_secret = render_launch_secret(
        args.tag,
        credentials.github_token,
        credentials.exa_api_key,
        credentials.wandb_api_key,
        anthropic_api_key=credentials.provider_api_keys.get("anthropic"),
        openai_api_key=credentials.provider_api_keys.get("openai"),
        custom_openai_api_key=credentials.provider_api_keys.get("custom_openai"),
        custom_secrets=credentials.custom_secrets,
        immutable=bool(student_list),
    )

    # Validate template structure before reserving a launch or writing to GitHub.
    # The actual program is captured once its advisor branch has been ensured.
    program = ProgramSystemPrompt(
        program_path=args.program_path or "program.md",
        source_commit="0" * 40,
        content="Resolved and verified during a real launch.",
    )
    program_secret_name, program_secret = render_program_secret(args.tag, program)
    student_manifests = render_student_manifests(
        args,
        student_template,
        student_list,
        secret_name,
        launch_secret,
        program,
        program_secret_name,
        program_secret,
    )

    if not args.dry_run:
        if student_list:
            reserve_launch(args, student_list, secret_name, launch_secret)
        ensure_advisor_branch(
            args.target_repo_url,
            credentials.github_token,
            args.target_repo_branch,
            args.advisor_branch,
        )
        program = resolve_launch_program(args, credentials.github_token)
        ensure_target_repo_labels(
            args.target_repo_url,
            credentials.github_token,
            routing_labels(args.advisor_branch, student_list),
        )
        program_secret_name, program_secret = render_program_secret(args.tag, program)
        student_manifests = render_student_manifests(
            args,
            student_template,
            student_list,
            secret_name,
            launch_secret,
            program,
            program_secret_name,
            program_secret,
        )

    if args.dry_run:
        print_manifest(f"Secret: {secret_name}", launch_secret)
    elif not student_list:
        kubectl_apply(launch_secret, f"secret {secret_name}", **kube_scope(args))
    emit(
        args,
        f"Program context: {program_secret_name}",
        program_secret,
        f"program context secret {program_secret_name}",
    )

    if args.capacity_observer:
        manifest = render_capacity_observer(
            tag=args.tag,
            namespace=args.namespace,
            image=args.executor_image,
            image_pull_secrets=args.image_pull_secrets,
            revision=args.senpai_repo_revision,
            config=capacity_config(args),
            node_selector=controller_node_selector(args.controller_node_selector),
        )
        emit(args, "Capacity observer", manifest, "capacity observer")

    create_student_resources(args, student_manifests)

    advisor_student_list = student_list
    if args.advisor and not args.dry_run:
        advisor_student_list = list(
            dict.fromkeys(
                existing_student_names(args.tag, **kube_scope(args)) + student_list
            )
        )
    if args.advisor:
        manifest = render_advisor(
            advisor_template,
            args.tag,
            advisor_student_list,
            secret_name,
            launch_secret,
            args,
            program=program,
            program_secret_name=program_secret_name,
            program_secret=program_secret,
        )
        emit(args, "Advisor", manifest, "advisor")

    if not args.dry_run:
        print_launch_summary(args, student_list)


if __name__ == "__main__":
    main()
