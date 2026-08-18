#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Launch senpai advisor and student agents as K8s resources."""

import base64
import json
import posixpath
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senpai_agent.launch_context import (
    LAUNCH_CONTEXT_ENV,
    load_operator_instructions,
    render_launch_context,
)
from senpai_agent.program_context import PROGRAM_PATH_ENV, normalize_program_path

from launch_helpers import (
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
    preflight_check_hivemind_token,
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
    resolve_hivemind_token,
    resolve_openai_api_key,
    resolve_wandb_api_key,
    routing_labels,
    source_revision_for_image,
    target_repo_slug,
)

STUDENT_TEMPLATE = Path(__file__).parent / "student-deployment.yaml"
ADVISOR_TEMPLATE = Path(__file__).parent / "advisor-deployment.yaml"
SENPAI_CONFIG = Path(__file__).parent.parent / "senpai.yaml"
DOTENV_PATH = Path(__file__).parent.parent / ".env"
AGENTSTREAM_REVISION = "fdb6b22a7ba45a0645ff1b535e62d98fc974f593"
DEFAULT_HIVEMIND_DAEMON_SOURCE = (
    "wandb-hivemind @ git+https://github.com/wandb/agentstream.git@"
    f"{AGENTSTREAM_REVISION}#subdirectory=daemon"
)
DEFAULT_HIVEMIND_AGENTSTREAM_SOURCE = (
    "wandb-agentstream @ git+https://github.com/wandb/agentstream.git@"
    f"{AGENTSTREAM_REVISION}"
)
KUBERNETES_DNS_LABEL = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")


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
    hivemind_enabled: bool = False  # record OpenHands conversations in Hivemind
    hivemind_daemon_source: str = DEFAULT_HIVEMIND_DAEMON_SOURCE
    hivemind_agentstream_source: str = DEFAULT_HIVEMIND_AGENTSTREAM_SOURCE
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
    max_parallel_agents: int = 8  # active subagent limit; supported range is 1-8
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
    timeout_minutes: float = (
        30.0  # training run wall-clock limit (SENPAI_TIMEOUT_MINUTES)
    )
    max_epochs: int = 50  # maximum training epochs (SENPAI_MAX_EPOCHS)
    poll_interval_s: int = (
        600  # default advisor/student outer-loop sleep between GitHub polls
    )
    poll_jitter_s: int = 120  # max random jitter added to outer-loop sleeps
    stale_wip_seconds: int = 7200  # advisor-action threshold for stale WIP PRs
    start_gate_path: str = ""  # optional shared file path that must exist before advisor/student loops begin
    dry_run: bool = (
        False  # render manifests only: do not apply them or validate credentials
    )
    colocate_students: bool = (
        False  # require students from this launch to share one Kubernetes node
    )
    student_node_name: str = ""  # exact kubernetes.io/hostname label for students
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
    if args.advisor:
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
    }


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


def hivemind_config(args: Args) -> dict[str, str]:
    return {
        "SENPAI_HIVEMIND_ENABLED": str(args.hivemind_enabled).lower(),
    }


def hivemind_sidecar(
    args: Args,
    *,
    role: str,
    image: str,
    secret_name: str,
    tag: str,
) -> str:
    if not args.hivemind_enabled:
        return ""
    if role == "student":
        openhands_state_dir = "/watch/openhands_state"
        hivemind_state_dir = "/var/lib/hivemind/student"
    else:
        openhands_state_dir = f"/watch/{tag}/advisor/openhands_state"
        hivemind_state_dir = "/var/lib/hivemind/advisor"
    daemon_source = json.dumps(args.hivemind_daemon_source)
    agentstream_source = json.dumps(args.hivemind_agentstream_source)
    default_repo = json.dumps(args.target_repo_url)
    return f"""      initContainers:
      - name: hivemind
        image: {image}
        restartPolicy: Always
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: [ALL]
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e
          mkdir -p "$SENPAI_HIVEMIND_STATE_DIR"
          exec uvx \\
            --from "$SENPAI_HIVEMIND_DAEMON_SOURCE" \\
            --with "$SENPAI_HIVEMIND_AGENTSTREAM_SOURCE" \\
            hivemind run --state-dir "$SENPAI_HIVEMIND_STATE_DIR"
        env:
        - name: SENPAI_HIVEMIND_DAEMON_SOURCE
          value: {daemon_source}
        - name: SENPAI_HIVEMIND_AGENTSTREAM_SOURCE
          value: {agentstream_source}
        - name: SENPAI_OPENHANDS_STATE_DIR
          value: {openhands_state_dir}
        - name: HIVEMIND_WATCH_PATHS
          value: {openhands_state_dir}
        - name: SENPAI_HIVEMIND_STATE_DIR
          value: {hivemind_state_dir}
        - name: HIVEMIND_DEFAULT_REPO
          value: {default_repo}
        - name: HIVEMIND_UPGRADE_WATCHER_DISABLED
          value: "1"
        - name: HIVEMIND_TOKEN
          valueFrom:
            secretKeyRef:
              name: {secret_name}
              key: hivemind-token
        resources:
          requests:
            cpu: "250m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
        volumeMounts:
        - name: state
          mountPath: /watch
          readOnly: true
        - name: hivemind-state
          mountPath: /var/lib/hivemind
"""


def hivemind_state_volume(args: Args) -> str:
    if not args.hivemind_enabled:
        return ""
    return """      - name: hivemind-state
        emptyDir: {}
"""


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


def validate_delegation_args(args: Args) -> None:
    if not 1 <= args.max_parallel_agents <= 8:
        sys.exit("ERROR: --max_parallel_agents must be between 1 and 8")


def validate_student_node_name(node_name: str) -> None:
    if node_name and KUBERNETES_DNS_LABEL.fullmatch(node_name) is None:
        raise ValueError(
            "must be a Kubernetes DNS label: at most 63 lowercase alphanumeric "
            "or '-' characters, starting and ending with an alphanumeric character"
        )


def validate_student_scheduling_args(args: Args) -> None:
    try:
        validate_student_node_name(args.student_node_name)
    except ValueError as error:
        sys.exit(f"ERROR: --student_node_name {error}")


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
    )


def encoded_launch_context(
    args: Args,
    tag: str,
    student_list: list[str],
    *,
    backend: str,
) -> str:
    return base64.b64encode(
        build_launch_context(
            args,
            tag,
            student_list,
            backend=backend,
        ).encode()
    ).decode()


def encoded_operator_instructions(args: Args) -> str:
    return base64.b64encode(
        load_operator_instructions(args.extra_instructions).encode()
    ).decode()


def student_affinity(tag: str) -> str:
    return f"""      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: senpai
                role: student
                research-tag: {tag}
            topologyKey: kubernetes.io/hostname
"""


def student_node_selector(node_name: str) -> str:
    validate_student_node_name(node_name)
    if not node_name:
        return ""
    return f"""      nodeSelector:
        kubernetes.io/hostname: {node_name}
"""


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
            **hivemind_config(args),
            "SENPAI_REPO_URL": args.senpai_repo_url,
            "SENPAI_REPO_REVISION": args.senpai_repo_revision,
            "TARGET_REPO_URL": args.target_repo_url,
            "TARGET_REPO_BRANCH": args.target_repo_branch,
            PROGRAM_PATH_ENV: args.program_path,
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
            "SENPAI_TIMEOUT_MINUTES": str(args.timeout_minutes),
            "SENPAI_MAX_EPOCHS": str(args.max_epochs),
            "SENPAI_MAX_PARALLEL_AGENTS": str(args.max_parallel_agents),
            "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
            "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
            LAUNCH_CONTEXT_ENV: encoded_launch_context(
                args,
                tag,
                [student_name],
                backend="kubernetes",
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
            "STUDENT_CPU": str(student_cpu),
            "STUDENT_MEMORY": f"{student_memory_gi}Gi",
            "GPUS_PER_STUDENT": str(args.gpus_per_student),
            "STUDENT_AFFINITY": (
                student_affinity(tag) if args.colocate_students else ""
            ),
            "STUDENT_NODE_SELECTOR": student_node_selector(args.student_node_name),
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": model_provider_env(args, "student", secret_name),
            "HIVEMIND_SIDECAR": hivemind_sidecar(
                args,
                role="student",
                image=args.student_image,
                secret_name=secret_name,
                tag=tag,
            ),
            "HIVEMIND_STATE_VOLUME": hivemind_state_volume(args),
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
        **hivemind_config(args),
        "SENPAI_REPO_URL": args.senpai_repo_url,
        "SENPAI_REPO_REVISION": args.senpai_repo_revision,
        "TARGET_REPO_URL": args.target_repo_url,
        "TARGET_REPO_BRANCH": args.target_repo_branch,
        PROGRAM_PATH_ENV: args.program_path,
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
        "SENPAI_MAX_PARALLEL_AGENTS": str(args.max_parallel_agents),
        "SENPAI_POLL_INTERVAL_S": str(args.poll_interval_s),
        "SENPAI_POLL_JITTER_S": str(args.poll_jitter_s),
        "SENPAI_STALE_WIP_SECONDS": str(args.stale_wip_seconds),
        "PROBLEM_DIR": args.problem_dir,
        "PVC_MOUNT_PATH": args.pvc_mount_path,
        "SENPAI_START_GATE_PATH": args.start_gate_path,
    }
    data[LAUNCH_CONTEXT_ENV] = encoded_launch_context(
        args,
        tag,
        student_list,
        backend="kubernetes",
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
            "POD_CONFIG_HASH": pod_template_hash(configmap, launch_secret),
            "MODEL_PROVIDER_ENV": model_provider_env(args, "advisor", secret_name),
            "HIVEMIND_SIDECAR": hivemind_sidecar(
                args,
                role="advisor",
                image=args.advisor_image,
                secret_name=secret_name,
                tag=tag,
            ),
            "HIVEMIND_STATE_VOLUME": hivemind_state_volume(args),
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
    validate_delegation_args(args)
    validate_student_scheduling_args(args)
    validate_program_path(args)
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

    model_providers = deployed_model_providers(args)
    github_token = exa_api_key = hivemind_token = wandb_api_key = ""
    provider_api_keys: dict[str, str] = {}
    if not args.dry_run or args.preflight_only:
        github_token = resolve_github_token(DOTENV_PATH)
        if "anthropic" in model_providers:
            provider_api_keys["anthropic"] = resolve_anthropic_api_key(DOTENV_PATH)
        if "openai" in model_providers:
            provider_api_keys["openai"] = resolve_openai_api_key(DOTENV_PATH)
        exa_api_key = resolve_exa_api_key(DOTENV_PATH)
        if args.hivemind_enabled:
            hivemind_token = resolve_hivemind_token(DOTENV_PATH)
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
        if hivemind_token:
            preflight_check_hivemind_token(hivemind_token)
        if args.preflight_only:
            print("Preflight OK — credentials and target repo access verified.")
            return

    if not args.dry_run:
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
        hivemind_token=(
            hivemind_token
            if not args.dry_run
            else "<REDACTED_HIVEMIND_TOKEN>"
        )
        if args.hivemind_enabled
        else None,
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
