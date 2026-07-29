# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Credential and target-repository checks performed before launch."""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import wandb

from .specs import target_repo_slug


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


def preflight_check_wandb_api_key(
    api_key: str,
    entity: str | None,
) -> str:
    """Verify the W&B key and resolve its configured or default entity."""
    print("Preflight: checking W&B API key", flush=True)
    try:
        api = wandb.Api(
            overrides={"base_url": "https://api.wandb.ai"},
            timeout=10,
            api_key=api_key,
        )
        default_entity = api.default_entity
    except wandb.errors.AuthenticationError as error:
        sys.exit(f"ERROR: W&B API key failed: {error}")
    resolved_entity = entity or default_entity
    if not resolved_entity:
        sys.exit(
            "ERROR: W&B API key has no default entity. "
            "Set wandb_entity in senpai.yaml or pass --wandb_entity."
        )
    print(f"  OK — W&B API key authenticated as {resolved_entity}")
    return resolved_entity


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
