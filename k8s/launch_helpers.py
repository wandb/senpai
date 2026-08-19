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
from collections.abc import Sequence
from pathlib import Path

from dotenv import dotenv_values
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


def pod_template_hash(configmap: str, launch_secret: str) -> str:
    """Hash the complete pod configuration that must trigger a rollout."""
    return hashlib.sha256(f"{configmap}\0{launch_secret}".encode()).hexdigest()


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
    target_repo_url: str,
    token: str,
    target_repo_branch: str,
    target_repo_revision: str = "",
) -> str:
    """Verify the base branch and optional exact revision."""
    slug = target_repo_slug(target_repo_url)
    branch = resolve_target_repo_branch(target_repo_url, token, target_repo_branch)
    if not branch:
        sys.exit(f"ERROR: could not resolve default branch for {slug}")
    if target_repo_revision and not re.fullmatch(
        r"[0-9a-f]{40}", target_repo_revision
    ):
        sys.exit("ERROR: --target_repo_revision must be a full lowercase commit SHA")

    print(f"Preflight: checking target repo base branch {slug}@{branch}")
    try:
        branch_info = _github_api(_branch_api_path(slug, branch), token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit(f"ERROR: target repo branch {slug}@{branch} does not exist")
        raise
    actual_revision = branch_info["commit"]["sha"]
    if target_repo_revision and actual_revision != target_repo_revision:
        sys.exit(
            f"ERROR: target repo branch {slug}@{branch} is at {actual_revision}, "
            f"expected {target_repo_revision}"
        )
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
    target_repo_revision: str = "",
) -> None:
    """Create or verify the advisor branch at the requested target revision."""
    slug = target_repo_slug(target_repo_url)
    base_branch = preflight_check_target_repo_branch(
        target_repo_url,
        token,
        target_repo_branch,
        target_repo_revision,
    )

    if advisor_branch == base_branch:
        print(
            f"Preflight: advisor branch is target base branch {slug}@{advisor_branch}"
        )
        return

    print(f"Preflight: ensuring advisor branch {slug}@{advisor_branch} exists")
    try:
        advisor_info = _github_api(_branch_api_path(slug, advisor_branch), token)
        advisor_revision = advisor_info["commit"]["sha"]
        if target_repo_revision and advisor_revision != target_repo_revision:
            sys.exit(
                f"ERROR: existing advisor branch {slug}@{advisor_branch} is at "
                f"{advisor_revision}, expected {target_repo_revision}"
            )
        print(f"  OK — advisor branch {advisor_branch} already exists")
        return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    base_info = _github_api(_branch_api_path(slug, base_branch), token)
    base_sha = base_info["commit"]["sha"]
    if target_repo_revision and base_sha != target_repo_revision:
        sys.exit(
            f"ERROR: target repo branch {slug}@{base_branch} moved to "
            f"{base_sha}; expected {target_repo_revision}"
        )
    payload = json.dumps(
        {
            "ref": f"refs/heads/{advisor_branch}",
            "sha": base_sha,
        }
    ).encode()
    _github_api(f"/repos/{slug}/git/refs", token, method="POST", data=payload)
    print(f"  created {advisor_branch} from {base_branch} at {base_sha[:7]}")


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


def render_launch_secret(
    tag: str,
    github_token: str,
    exa_api_key: str | None,
    wandb_api_key: str,
    *,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    custom_secrets: dict[str, str],
) -> str:
    """Per-launch k8s Secret holding API credentials used by advisor/student pods."""
    credentials = {
        "github-token": github_token,
        "wandb-api-key": wandb_api_key,
    }
    if exa_api_key is not None:
        credentials["exa-api-key"] = exa_api_key
    if anthropic_api_key is not None:
        credentials["anthropic-api-key"] = anthropic_api_key
    if openai_api_key is not None:
        credentials["openai-api-key"] = openai_api_key
    credentials.update(custom_secrets)
    encoded = {
        name: base64.b64encode(value.encode()).decode()
        for name, value in credentials.items()
    }
    lines = [
        "apiVersion: v1",
        "kind: Secret",
        "metadata:",
        f"  name: senpai-launch-secrets-{tag}",
        "  labels:",
        "    app: senpai",
        f"    research-tag: {tag}",
        "type: Opaque",
        "data:",
    ]
    lines.extend(f"  {name}: {value}" for name, value in encoded.items())
    return "\n".join(lines) + "\n"


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


def preflight_check_wandb_api_key(api_key: str) -> None:
    """Verify the supplied W&B API key with the smallest viewer query."""
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
    if not payload.get("data", {}).get("viewer"):
        errors = json.dumps(payload.get("errors", []), sort_keys=True)
        errors = _redact_secrets(errors, api_key, basic_auth)
        sys.exit(f"ERROR: W&B API key failed to resolve a viewer: {errors[:1000]}")
    print("  OK — W&B API key authenticated")


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
