# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

"""Pure helpers shared by launch.py.

Keep this file free of CLI/argparse coupling so helpers can be reused by
future scripts (teardown, status, etc.) and unit-tested in isolation.
"""

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

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


def target_repo_slug(url: str) -> str:
    """Extract owner/repo slug from a GitHub URL (for `gh --repo`)."""
    return url.split("github.com", 1)[-1].lstrip(":/").removesuffix(".git")


def _load_dotenv(path: Path) -> None:
    """Read KEY=VAL lines from path into os.environ (doesn't override existing)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def resolve_github_token(dotenv_path: Path) -> str:
    """Resolve the GitHub token: $GITHUB_TOKEN → .env → `gh auth token` → hard error."""
    # Treat empty $GITHUB_TOKEN as unset so .env fallback still kicks in.
    if not os.environ.get("GITHUB_TOKEN", "").strip():
        os.environ.pop("GITHUB_TOKEN", None)
    _load_dotenv(dotenv_path)
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    try:
        res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except FileNotFoundError:
        pass
    sys.exit("ERROR: no github token. Set $GITHUB_TOKEN in your shell, add it to .env "
             "at the senpai repo root, or run `gh auth login`.")


def render_token_secret(tag: str, token: str) -> str:
    """Per-launch k8s Secret holding only the github-token key."""
    enc = base64.b64encode(token.encode()).decode()
    return (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        f"  name: senpai-github-token-{tag}\n"
        "  labels:\n"
        "    app: senpai\n"
        f"    research-tag: {tag}\n"
        "type: Opaque\n"
        "data:\n"
        f"  github-token: {enc}\n"
    )


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
                hint = (
                    "\n  Your token is invalid or expired. Grab a fresh one with:\n"
                    "    gh auth token\n"
                    "  then paste it into .env at the senpai repo root as GITHUB_TOKEN=<token>\n"
                    "  (or `export GITHUB_TOKEN=$(gh auth token)` for this shell)."
                )
            sys.exit(
                f"ERROR: GitHub API {e.code} for {path}: "
                f"{e.read().decode(errors='replace')}{hint}"
            )

    perms = gh_api(f"/repos/{slug}").get("permissions", {})
    if seen_scopes and not ({"read:org", "admin:org"} & seen_scopes):
        sys.exit(
            "ERROR: github token is missing read:org scope.\n"
            "  Fix: create a PAT with repo + read:org and put it in .env as GITHUB_TOKEN."
        )
    if perms.get("push"):
        print(f"  OK — token has push access to {slug}")
        return

    user = gh_api("/user").get("login", "<unknown>")
    sys.exit(f"ERROR: github token (user '{user}') cannot push to {slug}\n"
             f"  permissions: {perms}\n"
             f"  Fix: supply a token with write on {slug} via $GITHUB_TOKEN / .env / gh auth, "
             f"or grant '{user}' collaborator write on {slug}.")


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
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  {result.stdout.strip()}")
