# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Pure helpers shared by launch.py.

Keep this file free of CLI/argparse coupling so helpers can be reused by
future scripts (teardown, status, etc.) and unit-tested in isolation.
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import dotenv_values

_KUBERNETES_DNS_PART = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
_KUBERNETES_DNS_SUBDOMAIN = re.compile(
    rf"{_KUBERNETES_DNS_PART}(?:[.]{_KUBERNETES_DNS_PART})*"
)
_KUBERNETES_NAME_LIMIT = 63

STUDENT_NAMES = [
    "frieren",
    "fern",
    "tanjiro",
    "nezuko",
    "alphonse",
    "edward",
    "thorfinn",
    "askeladd",
    "violet",
    "gilbert",
    "senku",
    "kohaku",
    "emma",
    "norman",
    "chihiro",
    "haku",
    "shoya",
    "shouko",
    "mitsuha",
    "taki",
    "shinji",
    "rei",
    "kaneda",
    "tetsuo",
    "naruto",
    "sasuke",
    "sakura",
    "kakashi",
    "hinata",
    "itachi",
    "roy",
    "winry",
    "eren",
    "mikasa",
    "armin",
    "levi",
    "historia",
    "ymir",
    "zenitsu",
    "inosuke",
    "giyu",
    "shinobu",
    "chrome",
    "gen",
    "ray",
    "asuka",
    "kaworu",
    "luffy",
    "zoro",
    "nami",
    "sanji",
    "robin",
    "chopper",
    "usopp",
    "franky",
    "brook",
    "yuji",
    "megumi",
    "nobara",
    "gojo",
    "sukuna",
    "spike",
    "jet",
    "faye",
    "vash",
    "wolfwood",
    "guts",
    "casca",
    "griffith",
    "einar",
    "canute",
    "stark",
    "himmel",
    "mugen",
    "jin",
]

LABEL_COLOR_ADVISOR_BRANCH = "0075ca"
LABEL_COLOR_STATUS_WIP = "fbca04"
LABEL_COLOR_STATUS_REVIEW = "0e8a16"
LABEL_COLOR_STUDENT = "f9d0c4"
FULL_SHA_IMAGE = re.compile(r"[^\s]+:sha-([0-9a-f]{40})")
DIGEST_IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")


def expand_student_names(n: int, names: list[str] = STUDENT_NAMES) -> list[str]:
    """Return n student names, cycling through `names` with a numeric suffix
    once the list is exhausted.

    First pass uses bare names; each subsequent pass appends an incrementing
    suffix (e.g. for `n=3*len(names)`: frieren, ..., jin, frieren2, ..., jin2,
    frieren3, ..., jin3).
    """
    out = []
    for i in range(n):
        base = names[i % len(names)]
        round_num = i // len(names)
        out.append(base if round_num == 0 else f"{base}{round_num + 1}")
    return out


def validate_kubernetes_label(value: str, option: str) -> None:
    """Require one value that is safe in Kubernetes names and labels."""

    if (
        len(value) > _KUBERNETES_NAME_LIMIT
        or _KUBERNETES_DNS_SUBDOMAIN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{option} must be a lowercase Kubernetes DNS name of at most "
            f"{_KUBERNETES_NAME_LIMIT} characters"
        )


def kubernetes_resource_name(value: str) -> str:
    """Keep a readable name and hash only an overlong constructed tail."""

    if len(value) <= _KUBERNETES_NAME_LIMIT:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    prefix = value[: _KUBERNETES_NAME_LIMIT - len(digest) - 1].rstrip("-.")
    return f"{prefix}-{digest}"


def routing_labels(
    advisor_branch: str, student_names: list[str]
) -> dict[str, tuple[str, str]]:
    """Labels required for advisor/student PR routing."""
    return {
        advisor_branch: (
            LABEL_COLOR_ADVISOR_BRANCH,
            f"Advisor branch: {advisor_branch}",
        ),
        "status:wip": (LABEL_COLOR_STATUS_WIP, "Work in progress"),
        "status:review": (LABEL_COLOR_STATUS_REVIEW, "Ready for advisor review"),
        **{
            f"student:{name}": (LABEL_COLOR_STUDENT, f"Assigned to student {name}")
            for name in student_names
        },
    }


def is_immutable_image_reference(image: str) -> bool:
    """Return whether an image is pinned by full source SHA or registry digest."""
    return bool(FULL_SHA_IMAGE.fullmatch(image) or DIGEST_IMAGE.fullmatch(image))


def source_revision_for_image(image: str, senpai_repo_revision: str = "") -> str:
    """Resolve the exact runner commit that must match the image metadata."""
    tagged = FULL_SHA_IMAGE.fullmatch(image)
    tagged_revision = tagged.group(1) if tagged else ""
    if senpai_repo_revision and not re.fullmatch(
        r"[0-9a-f]{40}", senpai_repo_revision
    ):
        raise ValueError(
            "senpai_repo_revision must be a full lowercase commit SHA"
        )
    if (
        tagged_revision
        and senpai_repo_revision
        and tagged_revision != senpai_repo_revision
    ):
        raise ValueError(
            "senpai_repo_revision does not match the image source-SHA tag"
        )
    if tagged_revision:
        return tagged_revision
    if DIGEST_IMAGE.fullmatch(image) and senpai_repo_revision:
        return senpai_repo_revision
    raise ValueError(
        "digest-pinned images require an explicit senpai_repo_revision"
    )


def kubectl_command(
    *arguments: str,
    kube_context: str = "",
    namespace: str = "default",
) -> list[str]:
    command = ["kubectl"]
    if kube_context:
        command.extend(("--context", kube_context))
    command.extend(("--namespace", namespace, *arguments))
    return command


def existing_student_names(
    tag: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> list[str]:
    result = subprocess.run(
        kubectl_command(
            "get",
            "deployments",
            "-l",
            f"app=senpai,role=student,research-tag={tag}",
            "-o",
            'jsonpath={range .items[*]}{.metadata.labels.student}{"\\n"}{end}',
            kube_context=kube_context,
            namespace=namespace,
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _live_role_resources(items: list[dict]) -> list[dict]:
    """Drop Pods that no longer run: terminal phases and pending deletions."""

    return [
        resource
        for resource in items
        if resource.get("kind") != "Pod"
        or (
            resource.get("status", {}).get("phase") not in {"Succeeded", "Failed"}
            and "deletionTimestamp" not in resource.get("metadata", {})
        )
    ]


def _role_annotation_values(
    tag: str,
    annotation: str,
    description: str,
    *,
    students_only: bool = False,
    base64_encoded: bool = False,
    allow_empty: bool = False,
    kube_context: str = "",
    namespace: str = "default",
) -> dict[str, set[str]]:
    """Read one binding from desired Deployments and every live Pod."""

    result = subprocess.run(
        kubectl_command(
            "get",
            "deployments,pods",
            "-l",
            f"app=senpai,research-tag={tag}",
            "-o",
            "json",
            kube_context=kube_context,
            namespace=namespace,
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    bindings: dict[str, set[str]] = {}
    for resource in _live_role_resources(json.loads(result.stdout).get("items", [])):
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})
        role = labels.get("role")
        student = labels.get("student")
        if students_only and role != "student":
            continue
        if role == "advisor":
            identity = "advisor"
        elif role == "student" and isinstance(student, str) and student:
            identity = f"student/{student}"
        else:
            sys.exit(
                "ERROR: an existing Senpai role resource lacks a valid role "
                "identity; use a new launch tag"
            )
        annotations = metadata.get("annotations", {})
        if resource.get("kind") == "Deployment":
            annotations = (
                resource.get("spec", {})
                .get("template", {})
                .get("metadata", {})
                .get("annotations", {})
            )
        value = annotations.get(annotation)
        try:
            if base64_encoded:
                value = base64.b64decode(value, validate=True).decode()
            elif not isinstance(value, str):
                raise TypeError
        except (TypeError, ValueError, UnicodeDecodeError):
            sys.exit(
                f"ERROR: existing {identity.replace('/', ' ')} lacks a valid "
                f"{description} binding; use a new launch tag"
            )
        if not value and not allow_empty:
            sys.exit(
                f"ERROR: existing {identity.replace('/', ' ')} lacks a valid "
                f"{description} binding; use a new launch tag"
            )
        bindings.setdefault(identity, set()).add(value)
    return bindings


def existing_student_wandb_viewers(
    tag: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> dict[str, set[str]]:
    """Return every active or desired W&B writer identity by student."""

    bindings = _role_annotation_values(
        tag,
        "senpai.wandb.com/wandb-viewer",
        "W&B writer viewer",
        students_only=True,
        base64_encoded=True,
        kube_context=kube_context,
        namespace=namespace,
    )
    return {
        identity.removeprefix("student/"): viewers
        for identity, viewers in bindings.items()
    }


def existing_controller_wandb_viewers(
    tag: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> dict[str, set[str]]:
    """Return every active or desired controller W&B identity by role."""

    return _role_annotation_values(
        tag,
        "senpai.wandb.com/controller-wandb-viewer",
        "controller W&B viewer",
        base64_encoded=True,
        kube_context=kube_context,
        namespace=namespace,
    )


def existing_inference_wandb_viewers(
    tag: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> dict[str, set[str]]:
    """Return every active or desired W&B Inference identity by role."""

    return _role_annotation_values(
        tag,
        "senpai.wandb.com/inference-wandb-viewer",
        "W&B Inference viewer",
        base64_encoded=True,
        allow_empty=True,
        kube_context=kube_context,
        namespace=namespace,
    )


def existing_wandb_viewer_owners(
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> dict[str, set[str]]:
    """Return every desired or live W&B viewer owner in the namespace."""

    result = subprocess.run(
        kubectl_command(
            "get",
            "deployments,pods",
            "-l",
            "app=senpai",
            "-o",
            "json",
            kube_context=kube_context,
            namespace=namespace,
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    owners: dict[str, set[str]] = {}
    for resource in _live_role_resources(json.loads(result.stdout).get("items", [])):
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})
        tag = labels.get("research-tag")
        role = labels.get("role")
        student = labels.get("student")
        if not isinstance(tag, str) or not tag:
            sys.exit(
                "ERROR: an existing Senpai role resource lacks a research-tag; "
                "remove or upgrade every active legacy role before launching"
            )
        if role == "advisor":
            identity = "advisor"
        elif role == "student" and isinstance(student, str) and student:
            identity = f"student {student!r}"
        else:
            sys.exit(
                "ERROR: an existing Senpai role resource lacks a valid role "
                "identity; remove or upgrade every active legacy role before "
                "launching"
            )
        annotations = metadata.get("annotations", {})
        if resource.get("kind") == "Deployment":
            annotations = (
                resource.get("spec", {})
                .get("template", {})
                .get("metadata", {})
                .get("annotations", {})
            )
        bindings = [
            (
                f"tag {tag!r} controller",
                "senpai.wandb.com/controller-wandb-viewer",
                False,
            ),
            (
                f"tag {tag!r} W&B Inference",
                "senpai.wandb.com/inference-wandb-viewer",
                True,
            ),
        ]
        if role == "student":
            bindings.append(
                (
                    f"tag {tag!r} {identity}",
                    "senpai.wandb.com/wandb-viewer",
                    False,
                )
            )
        for owner, annotation, allow_empty in bindings:
            try:
                viewer = base64.b64decode(
                    annotations.get(annotation), validate=True
                ).decode()
            except (TypeError, ValueError, UnicodeDecodeError):
                sys.exit(
                    f"ERROR: existing {identity} in tag {tag!r} lacks a valid "
                    "W&B viewer binding; remove or upgrade every active legacy "
                    "role before launching"
                )
            if not viewer:
                if allow_empty:
                    continue
                sys.exit(
                    f"ERROR: existing {identity} in tag {tag!r} lacks a valid "
                    "W&B viewer binding; remove or upgrade every active legacy "
                    "role before launching"
                )
            owners.setdefault(owner, set()).add(viewer)
    return owners


def existing_program_context_secret(
    tag: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> str | None:
    """Return the one immutable program snapshot bound to a live launch tag."""

    bindings = _role_annotation_values(
        tag,
        "senpai.program.com/context-secret",
        "program context",
        kube_context=kube_context,
        namespace=namespace,
    )
    names = {name for values in bindings.values() for name in values}
    if len(names) > 1:
        sys.exit(
            "ERROR: existing roles use different program snapshots; use a new "
            "launch tag"
        )
    return next(iter(names), None)


def read_program_context_secret(
    name: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> str:
    """Read the encoded program snapshot from a launch-owned Secret."""

    result = subprocess.run(
        kubectl_command(
            "get",
            "secret",
            name,
            "-o",
            "json",
            kube_context=kube_context,
            namespace=namespace,
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    labels = payload.get("metadata", {}).get("labels", {})
    value = payload.get("data", {}).get("program-context")
    if labels.get("senpai.wandb.com/secret-role") != "program-context":
        sys.exit("ERROR: the bound program context Secret has an invalid role")
    try:
        encoded = base64.b64decode(value, validate=True).decode()
    except (TypeError, ValueError, UnicodeDecodeError):
        sys.exit("ERROR: the bound program context Secret is invalid")
    if not encoded:
        sys.exit("ERROR: the bound program context Secret is empty")
    return encoded


def render_template(template: str, replacements: dict[str, str]) -> str:
    """Replace {{PLACEHOLDER}} tokens in a K8s manifest template."""
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def render_configmap(name: str, labels: dict[str, str], data: dict[str, str]) -> str:
    """Generate a ConfigMap YAML document."""
    lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        f"  name: {name}",
        "  labels:",
    ]
    for k, v in labels.items():
        lines.append(f"    {k}: {v}")
    lines.append("data:")
    for k, v in data.items():
        lines.append(f'  {k}: "{v}"')
    return "\n".join(lines)


def pod_template_hash(*resources: str) -> str:
    """Hash the complete pod configuration that must trigger a rollout."""
    return hashlib.sha256("\0".join(resources).encode()).hexdigest()


def target_repo_slug(url: str) -> str:
    """Extract owner/repo slug from a GitHub URL (for `gh --repo`)."""
    return url.split("github.com", 1)[-1].lstrip(":/").removesuffix(".git")


def _github_api(
    path: str,
    token: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    timeout: int = 10,
) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "senpai-launch-preflight",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body or b"{}")


def _branch_api_path(slug: str, branch: str) -> str:
    return f"/repos/{slug}/branches/{urllib.parse.quote(branch, safe='')}"


def resolve_target_repo_branch(
    target_repo_url: str, token: str, target_repo_branch: str
) -> str:
    """Return the target repo branch used as the advisor-branch base."""
    if target_repo_branch:
        return target_repo_branch
    slug = target_repo_slug(target_repo_url)
    return _github_api(f"/repos/{slug}", token).get("default_branch", "")


def preflight_check_target_repo_branch(
    target_repo_url: str, token: str, target_repo_branch: str
) -> str:
    """Verify the base branch exists and return the resolved branch name."""
    slug = target_repo_slug(target_repo_url)
    branch = resolve_target_repo_branch(target_repo_url, token, target_repo_branch)
    if not branch:
        sys.exit(f"ERROR: could not resolve default branch for {slug}")

    print(f"Preflight: checking target repo base branch {slug}@{branch}")
    try:
        _github_api(_branch_api_path(slug, branch), token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit(f"ERROR: target repo branch {slug}@{branch} does not exist")
        raise
    print(f"  OK — target repo branch {branch} exists")
    return branch


def preflight_check_student_name_availability(
    target_repo_url: str,
    token: str,
    student_names: list[str],
    advisor_branch: str,
) -> None:
    """Reject student names routing an active PR from another advisor branch."""
    slug = target_repo_slug(target_repo_url)
    print(f"Preflight: checking student assignment labels on {slug}")
    conflicts: dict[str, list[tuple[int, str]]] = {}
    active_statuses = {"status:wip", "status:review"}

    for student in dict.fromkeys(student_names):
        page = 1
        while True:
            query = urllib.parse.urlencode(
                {
                    "state": "open",
                    "labels": f"student:{student}",
                    "per_page": 100,
                    "page": page,
                }
            )
            issues = _github_api(f"/repos/{slug}/issues?{query}", token)
            if not isinstance(issues, list):
                sys.exit(
                    "ERROR: GitHub returned invalid active assignments for "
                    f"student:{student}"
                )
            for issue in issues:
                labels = {label["name"] for label in issue["labels"]}
                if issue.get("pull_request") is None or not active_statuses & labels:
                    continue
                number = int(issue["number"])
                pull = _github_api(f"/repos/{slug}/pulls/{number}", token)
                base_branch = pull["base"]["ref"]
                if base_branch != advisor_branch:
                    conflicts.setdefault(student, []).append((number, base_branch))
            if len(issues) < 100:
                break
            page += 1

    if conflicts:
        conflict_lines = "\n".join(
            f"  student:{student}: "
            + ", ".join(
                f"#{number} (base {base_branch})"
                for number, base_branch in student_conflicts
            )
            for student, student_conflicts in conflicts.items()
        )
        sys.exit(
            "ERROR: target repo has active assignment PRs on other advisor "
            f"branches for requested students:\n{conflict_lines}\n"
            "  Launching would make student routing ambiguous. Use "
            "--student_prefix <prefix> for unique student labels, or finish/relabel "
            "the existing assignments."
        )
    print("  OK — requested student labels are available for this advisor branch")


def ensure_advisor_branch(
    target_repo_url: str,
    token: str,
    target_repo_branch: str,
    advisor_branch: str,
) -> str:
    """Ensure the advisor branch exists and return its exact head commit."""
    slug = target_repo_slug(target_repo_url)
    base_branch = preflight_check_target_repo_branch(
        target_repo_url, token, target_repo_branch
    )

    if advisor_branch == base_branch:
        branch = _github_api(_branch_api_path(slug, advisor_branch), token)
        print(
            f"Preflight: advisor branch is target base branch {slug}@{advisor_branch}"
        )
        return str(branch["commit"]["sha"])

    print(f"Preflight: ensuring advisor branch {slug}@{advisor_branch} exists")
    try:
        branch = _github_api(_branch_api_path(slug, advisor_branch), token)
        print(f"  OK — advisor branch {advisor_branch} already exists")
        return str(branch["commit"]["sha"])
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    base_info = _github_api(_branch_api_path(slug, base_branch), token)
    base_sha = base_info["commit"]["sha"]
    payload = json.dumps(
        {
            "ref": f"refs/heads/{advisor_branch}",
            "sha": base_sha,
        }
    ).encode()
    _github_api(f"/repos/{slug}/git/refs", token, method="POST", data=payload)
    print(f"  created {advisor_branch} from {base_branch} at {base_sha[:7]}")
    return str(base_sha)


LAUNCH_CREDENTIAL_ENV_NAMES = (
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "EXA_API_KEY",
    "WANDB_API_KEY",
)
def _dotenv_values(path: Path) -> dict[str, str | None]:
    """Read literal values from a dotenv file without mutating the environment."""
    if not path.exists():
        return {}
    return dict(dotenv_values(path, interpolate=False))


def resolve_custom_secrets(
    dotenv_path: Path, names: Sequence[str]
) -> dict[str, str]:
    """Resolve explicitly allowlisted secrets from the shell, then .env."""
    dotenv = _dotenv_values(dotenv_path)
    resolved = {}
    missing = []
    for name in names:
        value = os.environ.get(name)
        if value is None or not value.strip():
            value = dotenv.get(name)
        if value is None or not value.strip():
            missing.append(name)
        else:
            resolved[name] = value
    if missing:
        joined = ", ".join(missing)
        sys.exit(
            "ERROR: missing or blank custom secrets: "
            f"{joined}. Set them in your shell or repository-root .env."
        )
    return resolved


def _subprocess_env_without_launch_credentials(
    custom_secret_env_names: Sequence[str],
) -> dict[str, str]:
    excluded = {*LAUNCH_CREDENTIAL_ENV_NAMES, *custom_secret_env_names}
    return {k: v for k, v in os.environ.items() if k not in excluded}


def resolve_required_secret(dotenv_path: Path, env_name: str, label: str) -> str:
    """Resolve a required secret from the shell env, then .env."""
    value = os.environ.get(env_name, "").strip()
    if not value:
        value = (_dotenv_values(dotenv_path).get(env_name) or "").strip()
    if value:
        return value
    sys.exit(
        f"ERROR: no {label}. Set ${env_name} in your shell or add {env_name}=<key> to .env."
    )


def resolve_github_token(
    dotenv_path: Path, custom_secret_env_names: Sequence[str]
) -> str:
    """Resolve the GitHub token: $GITHUB_TOKEN → .env → `gh auth token` → hard error."""
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if not tok:
        tok = (_dotenv_values(dotenv_path).get("GITHUB_TOKEN") or "").strip()
    if tok:
        return tok
    try:
        res = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            env=_subprocess_env_without_launch_credentials(custom_secret_env_names),
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except FileNotFoundError:
        pass
    sys.exit(
        "ERROR: no github token. Set $GITHUB_TOKEN in your shell, add it to .env "
        "at the senpai repo root, or run `gh auth login`."
    )


def resolve_anthropic_api_key(dotenv_path: Path) -> str:
    """Resolve the Anthropic API key: $ANTHROPIC_API_KEY → .env → hard error."""
    return resolve_required_secret(
        dotenv_path, "ANTHROPIC_API_KEY", "Anthropic API key"
    )


def resolve_openai_api_key(dotenv_path: Path) -> str:
    """Resolve the OpenAI API key: $OPENAI_API_KEY → .env → hard error."""
    return resolve_required_secret(dotenv_path, "OPENAI_API_KEY", "OpenAI API key")


def resolve_exa_api_key(dotenv_path: Path) -> str:
    """Resolve the Exa API key: $EXA_API_KEY → .env → hard error."""
    return resolve_required_secret(dotenv_path, "EXA_API_KEY", "Exa API key")


def resolve_wandb_api_key(dotenv_path: Path) -> str:
    """Resolve the W&B API key: $WANDB_API_KEY → .env → hard error."""
    return resolve_required_secret(dotenv_path, "WANDB_API_KEY", "W&B API key")


def resolve_wandb_inference_api_key(dotenv_path: Path) -> str:
    """Resolve the dedicated W&B Inference model credential."""

    return resolve_required_secret(
        dotenv_path,
        "WANDB_INFERENCE_API_KEY",
        "W&B Inference API key",
    )


def student_wandb_api_key_env(student_name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]", "_", student_name).upper()
    return f"WANDB_API_KEY_{suffix}"


def resolve_student_wandb_api_keys(
    dotenv_path: Path,
    student_names: Sequence[str],
) -> dict[str, str]:
    """Resolve one distinct W&B writer key for each student pod."""

    env_names: dict[str, str] = {}
    for name in student_names:
        env_name = student_wandb_api_key_env(name)
        if env_name in env_names:
            other = env_names[env_name]
            sys.exit(
                f"ERROR: student names {other!r} and {name!r} both map to "
                f"{env_name}; choose names with distinct credential suffixes"
            )
        env_names[env_name] = name

    resolved = {
        name: resolve_required_secret(
            dotenv_path,
            student_wandb_api_key_env(name),
            f"W&B training API key for student {name}",
        )
        for name in student_names
    }
    if len(set(resolved.values())) != len(resolved):
        sys.exit("ERROR: every student requires a distinct W&B training API key")
    return resolved


def render_launch_secret(
    tag: str,
    github_token: str,
    exa_api_key: str,
    wandb_api_key: str,
    *,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    wandb_inference_api_key: str | None = None,
    custom_secrets: dict[str, str],
) -> tuple[str, str]:
    """Render content-addressed shared credentials for one launch update."""
    credentials = {
        "github-token": github_token,
        "exa-api-key": exa_api_key,
        "wandb-api-key": wandb_api_key,
    }
    if anthropic_api_key is not None:
        credentials["anthropic-api-key"] = anthropic_api_key
    if openai_api_key is not None:
        credentials["openai-api-key"] = openai_api_key
    if wandb_inference_api_key is not None:
        credentials["wandb-inference-api-key"] = wandb_inference_api_key
    credentials.update(custom_secrets)
    encoded = {
        name: base64.b64encode(value.encode()).decode()
        for name, value in credentials.items()
    }
    digest = hashlib.sha256(
        json.dumps(credentials, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()[:16]
    name = kubernetes_resource_name(f"senpai-launch-secrets-{tag}-{digest}")
    lines = [
        "apiVersion: v1",
        "kind: Secret",
        "metadata:",
        f"  name: {name}",
        "  labels:",
        "    app: senpai",
        f"    research-tag: {tag}",
        "type: Opaque",
        "immutable: true",
        "data:",
    ]
    lines.extend(f"  {name}: {value}" for name, value in encoded.items())
    return name, "\n".join(lines) + "\n"


def render_program_context_secret(
    tag: str,
    program_context: str,
) -> tuple[str, str]:
    """Render an immutable, content-addressed program snapshot Secret."""

    digest = hashlib.sha256(program_context.encode()).hexdigest()[:16]
    name = kubernetes_resource_name(f"senpai-program-context-{tag}-{digest}")
    encoded = base64.b64encode(program_context.encode()).decode()
    manifest = "\n".join(
        [
            "apiVersion: v1",
            "kind: Secret",
            "metadata:",
            f"  name: {name}",
            "  labels:",
            "    app: senpai",
            f"    research-tag: {tag}",
            "    senpai.wandb.com/secret-role: program-context",
            "type: Opaque",
            "immutable: true",
            "data:",
            f"  program-context: {encoded}",
        ]
    )
    return name, manifest + "\n"


def render_student_wandb_secret(
    tag: str,
    student_name: str,
    api_key: str,
    viewer_id: str,
) -> tuple[str, str]:
    """Render one student's isolated W&B writer credential."""

    digest = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    name = kubernetes_resource_name(
        f"senpai-wandb-student-{tag}-{student_name}-{digest}"
    )
    encoded = base64.b64encode(api_key.encode()).decode()
    viewer = base64.b64encode(viewer_id.encode()).decode()
    manifest = "\n".join(
        [
            "apiVersion: v1",
            "kind: Secret",
            "metadata:",
            f"  name: {name}",
            "  labels:",
            "    app: senpai",
            "    role: student",
            f"    student: {student_name}",
            f"    research-tag: {tag}",
            "    senpai.wandb.com/secret-role: training-writer",
            "  annotations:",
            f"    senpai.wandb.com/wandb-viewer: {viewer}",
            "type: Opaque",
            "immutable: true",
            "data:",
            f"  wandb-api-key: {encoded}",
        ]
    )
    return name, manifest + "\n"


def _redact_secrets(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def _api_error_summary(error: urllib.error.HTTPError, *secrets: str) -> str:
    body = error.read().decode(errors="replace").strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        summary = body
    else:
        err = payload.get("error", payload)
        if isinstance(err, dict):
            parts = [str(err[k]) for k in ("type", "tag", "message") if err.get(k)]
            summary = ": ".join(parts) if parts else json.dumps(err, sort_keys=True)
        else:
            summary = str(err)
        request_id = payload.get("request_id") or payload.get("requestId")
        if request_id:
            summary = f"{summary} (request id: {request_id})"
    summary = _redact_secrets(summary, *secrets)
    return (summary or "<empty response>")[:1000]


def _preflight_http(
    name: str,
    req: urllib.request.Request,
    secret: str,
    timeout: int,
) -> object:
    print(f"Preflight: checking {name}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(
            f"ERROR: {name} failed: HTTP {e.code}: {_api_error_summary(e, secret)}"
        )
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: {name} failed: {e.reason}")
    print(f"  OK — {name} authenticated")
    return payload


def preflight_check_anthropic_api_key(api_key: str) -> None:
    """Verify the supplied Anthropic API key can authenticate to the API."""
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "senpai-launch-preflight",
        },
    )
    _preflight_http("Anthropic API key", req, api_key, timeout=10)


def preflight_check_openai_api_key(api_key: str) -> None:
    """Verify the supplied OpenAI API key can authenticate to the API."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "senpai-launch-preflight",
        },
    )
    _preflight_http("OpenAI API key", req, api_key, timeout=10)


def preflight_check_exa_api_key(api_key: str) -> None:
    """Verify the supplied Exa API key can authenticate and run a minimal search."""
    payload = json.dumps(
        {
            "query": "api credential preflight",
            "type": "instant",
            "category": "publication",
            "numResults": 1,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.exa.ai/search",
        data=payload,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "senpai-launch-preflight",
        },
        method="POST",
    )
    response = _preflight_http("Exa API key", req, api_key, timeout=15)
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        sys.exit("ERROR: Exa API key check returned an invalid search response")


def preflight_check_wandb_api_key(api_key: str) -> str:
    """Verify the supplied W&B API key and return its viewer identity."""
    basic_auth = base64.b64encode(f"api:{api_key}".encode()).decode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps(
            {
                "query": "query SenpaiPreflight { viewer { id } }",
            }
        ).encode(),
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/json",
            "User-Agent": "senpai-launch-preflight",
        },
        method="POST",
    )
    print("Preflight: checking W&B API key", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as error:
        sys.exit(
            "ERROR: W&B API key failed: "
            f"HTTP {error.code}: {_api_error_summary(error, api_key, basic_auth)}"
        )
    except urllib.error.URLError as error:
        sys.exit(f"ERROR: W&B API key failed: {error.reason}")
    viewer = (
        payload.get("data", {}).get("viewer")
        if isinstance(payload, dict)
        else None
    )
    viewer_id = viewer.get("id") if isinstance(viewer, dict) else None
    if not isinstance(viewer_id, str) or not viewer_id.strip():
        errors = json.dumps(
            payload.get("errors", []) if isinstance(payload, dict) else [],
            sort_keys=True,
        )
        errors = _redact_secrets(errors, api_key, basic_auth)
        sys.exit(f"ERROR: W&B API key failed to resolve a viewer: {errors[:1000]}")
    print("  OK — W&B API key authenticated")
    return viewer_id.strip()


def require_distinct_wandb_viewers(
    controller_viewer: str,
    student_viewers: Mapping[str, str],
    *,
    inference_viewers: Sequence[str] = (),
    active_controller_viewers: Sequence[str] = (),
    active_inference_viewers: Sequence[str] = (),
    active_student_viewers: Mapping[str, Sequence[str]] | None = None,
    active_viewer_owners: Mapping[str, Sequence[str]] | None = None,
    owner_scope: str = "",
) -> None:
    """Reject viewer reuse across controller, inference, and writer owners."""

    owners: dict[str, str] = {}

    def bind(viewer: str, owner: str) -> None:
        if not viewer:
            return
        if previous := owners.get(viewer):
            if previous != owner:
                sys.exit(
                    "ERROR: W&B API keys for "
                    f"{previous} and {owner} authenticate as the same viewer"
                )
            return
        owners[viewer] = owner

    def owner(name: str) -> str:
        return f"{owner_scope} {name}".strip()

    for active_owner, viewers in (active_viewer_owners or {}).items():
        for viewer in viewers:
            bind(viewer, active_owner)
    for viewer in (*active_controller_viewers, controller_viewer):
        bind(viewer, owner("controller"))
    for viewer in (*active_inference_viewers, *inference_viewers):
        bind(viewer, owner("W&B Inference"))
    for student, viewers in (active_student_viewers or {}).items():
        for viewer in viewers:
            bind(viewer, owner(f"student {student!r}"))
    for student, viewer in student_viewers.items():
        bind(viewer, owner(f"student {student!r}"))


def preflight_check_wandb_inference(
    api_key: str,
    entity: str,
    project: str,
) -> None:
    """Verify W&B Inference auth and project-pool routing."""
    req = urllib.request.Request(
        "https://api.inference.wandb.ai/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Project": f"{entity}/{project}",
            "User-Agent": "senpai-launch-preflight",
        },
    )
    response = _preflight_http("W&B Inference API key", req, api_key, timeout=10)
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        sys.exit("ERROR: W&B Inference check returned an invalid models response")


def preflight_check_target_repo_access(target_repo_url: str, token: str) -> None:
    """Verify the token can write repository contents without changing a ref."""
    slug = target_repo_slug(target_repo_url)
    print(f"Preflight: checking github token against {slug}")
    payload = json.dumps(
        {
            "ref": "refs/heads/senpai-write-preflight",
            "sha": "0" * 40,
        }
    ).encode()
    try:
        _github_api(f"/repos/{slug}/git/refs", token, method="POST", data=payload)
    except urllib.error.HTTPError as error:
        if error.code == 422:
            print(f"  OK — token has Contents write access to {slug}")
            return
        summary = _api_error_summary(error, token)
        sys.exit(
            f"ERROR: github token cannot write repository contents to {slug}: "
            f"HTTP {error.code}: {summary}\n"
            "  Fix: grant a fine-grained token 'Contents: Read and write' for this "
            "repository, or supply a classic token with repo scope."
        )
    else:
        sys.exit(
            f"ERROR: GitHub unexpectedly accepted the impossible write probe for {slug}"
        )


def ensure_target_repo_labels(
    target_repo_url: str, token: str, labels: dict[str, tuple[str, str]]
) -> None:
    """Create missing GitHub labels used for Senpai assignment routing."""
    slug = target_repo_slug(target_repo_url)
    print(f"Preflight: ensuring routing labels on {slug}")

    for name, (color, description) in labels.items():
        encoded = urllib.parse.quote(name, safe="")
        try:
            _github_api(f"/repos/{slug}/labels/{encoded}", token)
            continue
        except urllib.error.HTTPError as e:
            if e.code == 404:
                payload = json.dumps(
                    {
                        "name": name,
                        "color": color,
                        "description": description,
                    }
                ).encode()
                _github_api(
                    f"/repos/{slug}/labels",
                    token,
                    method="POST",
                    data=payload,
                )
                print(f"  created label {name}")
                continue

            print(f"GitHub label check failed for {name!r}", file=sys.stderr)
            raise


def kubectl_apply(
    manifest: str,
    name: str,
    *,
    kube_context: str = "",
    namespace: str = "default",
) -> None:
    """Apply a manifest via kubectl."""
    print(f"Launching: {name}")
    result = subprocess.run(
        kubectl_command(
            "apply",
            "-f",
            "-",
            kube_context=kube_context,
            namespace=namespace,
        ),
        input=manifest,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "kubectl returned no error text"
        raise RuntimeError(f"kubectl apply failed for {name}: {detail}")
    print(f"  {result.stdout.strip()}")
