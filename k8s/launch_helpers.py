# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Pure helpers shared by launch.py.

Keep this file free of CLI/argparse coupling so helpers can be reused by
future scripts (teardown, status, etc.) and unit-tested in isolation.
"""

import base64
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from senpai.launch.specs import target_repo_slug

STUDENT_NAMES = [
    "frieren", "fern", "tanjiro", "nezuko", "alphonse", "edward",
    "thorfinn", "askeladd", "violet", "gilbert", "senku", "kohaku",
    "emma", "norman", "chihiro", "haku", "shoya", "shouko",
    "mitsuha", "taki", "shinji", "rei", "kaneda", "tetsuo",
    "naruto", "sasuke", "sakura", "kakashi", "hinata", "itachi",
    "roy", "winry", "eren", "mikasa", "armin", "levi",
    "historia", "ymir", "zenitsu", "inosuke", "giyu", "shinobu",
    "chrome", "gen", "ray", "asuka", "kaworu", "luffy",
    "zoro", "nami", "sanji", "robin", "chopper", "usopp",
    "franky", "brook", "yuji", "megumi", "nobara", "gojo",
    "sukuna", "spike", "jet", "faye", "vash", "wolfwood",
    "guts", "casca", "griffith", "einar", "canute", "stark",
    "himmel", "mugen", "jin",
]

LABEL_COLOR_ADVISOR_BRANCH = "0075ca"
LABEL_COLOR_STATUS_WIP = "fbca04"
LABEL_COLOR_STATUS_REVIEW = "0e8a16"
LABEL_COLOR_STUDENT = "f9d0c4"


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


def routing_labels(advisor_branch: str, student_names: list[str]) -> dict[str, tuple[str, str]]:
    """Labels required for advisor/student PR routing."""
    return {
        advisor_branch: (LABEL_COLOR_ADVISOR_BRANCH, f"Advisor branch: {advisor_branch}"),
        "status:wip": (LABEL_COLOR_STATUS_WIP, "Work in progress"),
        "status:review": (LABEL_COLOR_STATUS_REVIEW, "Ready for advisor review"),
        **{f"student:{name}": (LABEL_COLOR_STUDENT, f"Assigned to student {name}") for name in student_names},
    }


def _kubectl_get_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["kubectl", "get", *args],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def existing_student_names(tag: str) -> list[str]:
    return _kubectl_get_lines(
        "deployments",
        "-l",
        f"app=senpai,role=student,research-tag={tag}",
        "-o",
        'jsonpath={range .items[*]}{.metadata.labels.student}{"\\n"}{end}',
    )


def existing_deployment_names(tag: str) -> list[str]:
    return _kubectl_get_lines(
        "deployments",
        "-l",
        f"app=senpai,research-tag={tag}",
        "-o",
        "name",
    )


def render_template(template: str, replacements: dict[str, str]) -> str:
    """Replace {{PLACEHOLDER}} tokens in a K8s manifest template."""
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def render_configmap(name: str, labels: dict[str, str], data: dict[str, str]) -> str:
    """Generate a ConfigMap YAML document."""
    lines = ["apiVersion: v1", "kind: ConfigMap", "metadata:", f"  name: {name}", "  labels:"]
    for k, v in labels.items():
        lines.append(f"    {k}: {v}")
    lines.append("data:")
    for k, v in data.items():
        lines.append(f"  {k}: \"{v}\"")
    return "\n".join(lines)


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


def resolve_target_repo_branch(target_repo_url: str, token: str, target_repo_branch: str) -> str:
    """Return the target repo branch used as the advisor-branch base."""
    if target_repo_branch:
        return target_repo_branch
    slug = target_repo_slug(target_repo_url)
    return _github_api(f"/repos/{slug}", token).get("default_branch", "")


def preflight_check_target_repo_branch(target_repo_url: str, token: str, target_repo_branch: str) -> str:
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


def ensure_advisor_branch(
    target_repo_url: str,
    token: str,
    target_repo_branch: str,
    advisor_branch: str,
) -> None:
    """Create advisor_branch from target_repo_branch when it does not exist."""
    slug = target_repo_slug(target_repo_url)
    base_branch = preflight_check_target_repo_branch(target_repo_url, token, target_repo_branch)

    if advisor_branch == base_branch:
        print(f"Preflight: advisor branch is target base branch {slug}@{advisor_branch}")
        return

    print(f"Preflight: ensuring advisor branch {slug}@{advisor_branch} exists")
    try:
        _github_api(_branch_api_path(slug, advisor_branch), token)
        print(f"  OK — advisor branch {advisor_branch} already exists")
        return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise

    base_info = _github_api(_branch_api_path(slug, base_branch), token)
    base_sha = base_info["commit"]["sha"]
    payload = json.dumps({
        "ref": f"refs/heads/{advisor_branch}",
        "sha": base_sha,
    }).encode()
    _github_api(f"/repos/{slug}/git/refs", token, method="POST", data=payload)
    print(f"  created {advisor_branch} from {base_branch} at {base_sha[:7]}")


def render_launch_secret(tag: str, secrets: dict[str, str]) -> str:
    """Per-launch k8s Secret holding API credentials used by advisor/student pods."""
    manifest = f"apiVersion: v1\nkind: Secret\nmetadata:\n  name: senpai-launch-secrets-{tag}\n  labels:\n    app: senpai\n    research-tag: {tag}\ntype: Opaque\ndata:\n"
    for name, value in sorted(secrets.items()):
        encoded = base64.b64encode(value.encode()).decode()
        manifest += f"  {name}: {encoded}\n"
    return manifest


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
    for secret in secrets:
        if secret:
            summary = summary.replace(secret, "<redacted>")
    return (summary or "<empty response>")[:1000]


def _preflight_http(name: str, req: urllib.request.Request, secret: str, timeout: int) -> dict:
    print(f"Preflight: checking {name}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: {name} failed: HTTP {e.code}: {_api_error_summary(e, secret)}")
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


def preflight_check_exa_api_key(api_key: str) -> None:
    """Verify the supplied Exa API key can authenticate and run a minimal search."""
    payload = json.dumps({
        "query": "api credential preflight",
        "type": "fast",
        "numResults": 1,
    }).encode()
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
    _preflight_http("Exa API key", req, api_key, timeout=15)


def preflight_check_wandb_api_key(api_key: str) -> None:
    """Verify the supplied W&B API key can authenticate."""
    import wandb

    print("Preflight: checking W&B API key", flush=True)
    try:
        wandb.Api(
            overrides={"base_url": "https://api.wandb.ai"},
            timeout=10,
            api_key=api_key,
        )
    except wandb.errors.AuthenticationError as error:
        sys.exit(f"ERROR: W&B API key failed: {error}")
    print("  OK — W&B API key authenticated")


def _oauth_scopes(header_value: str | None) -> set[str]:
    return {scope.strip() for scope in (header_value or "").split(",") if scope.strip()}


def preflight_check_target_repo_access(target_repo_url: str, token: str) -> None:
    """Verify the supplied github token has push access to target_repo_url.

    Catches the 403-on-push scenario before pods spin up.
    """
    slug = target_repo_slug(target_repo_url)
    print(f"Preflight: checking github token against {slug}")
    seen_scopes: set[str] = set()

    def gh_api(path: str) -> dict:
        req = urllib.request.Request(
            f"https://api.github.com{path}",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "senpai-launch-preflight"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                seen_scopes.update(_oauth_scopes(resp.headers.get("X-OAuth-Scopes")))
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            hint = ""
            if e.code == 401:
                hint = "\n  Your token is invalid or expired. Replace GITHUB_TOKEN in .env at the senpai repo root."
            sys.exit(f"ERROR: GitHub API {e.code} for {path}: {_api_error_summary(e, token)}{hint}")

    perms = gh_api(f"/repos/{slug}").get("permissions", {})
    if seen_scopes and not ({"read:org", "admin:org"} & seen_scopes):
        sys.exit("ERROR: github token is missing read:org scope.\n  Fix: create a PAT with repo + read:org and put it in .env as GITHUB_TOKEN.")
    if perms.get("push"):
        print(f"  OK — token has push access to {slug}")
        return

    user = gh_api("/user").get("login", "<unknown>")
    sys.exit(f"ERROR: github token (user '{user}') cannot push to {slug}\n  permissions: {perms}\n  Fix: put a token with write on {slug} in .env, or grant '{user}' collaborator write on {slug}.")


def ensure_target_repo_labels(target_repo_url: str, token: str, labels: dict[str, tuple[str, str]]) -> None:
    """Create missing GitHub labels used for Senpai assignment routing."""
    slug = target_repo_slug(target_repo_url)
    print(f"Preflight: ensuring routing labels on {slug}")

    def gh_api(path: str, method: str = "GET", data: bytes | None = None) -> dict:
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read() or b"{}")

    for name, (color, description) in labels.items():
        encoded = urllib.parse.quote(name, safe="")
        try:
            gh_api(f"/repos/{slug}/labels/{encoded}")
            continue
        except urllib.error.HTTPError as e:
            if e.code == 404:
                payload = json.dumps({
                    "name": name,
                    "color": color,
                    "description": description,
                }).encode()
                gh_api(f"/repos/{slug}/labels", method="POST", data=payload)
                print(f"  created label {name}")
                continue

            print(f"GitHub label check failed for {name!r}", file=sys.stderr)
            raise


def kubectl_apply(manifest: str, name: str) -> None:
    """Apply a manifest via kubectl."""
    print(f"Launching: {name}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: could not apply {name}: {result.stderr.strip()}")
    print(f"  {result.stdout.strip()}")
