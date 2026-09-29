"""Resolve the published Senpai image set once, before launching any Pods."""

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from launch_helpers import source_revision_for_image

_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def published_image(role: str, reference: str) -> tuple[str, str]:
    """Read an official public image's digest and embedded source revision."""
    repository = f"wandb/senpai-{role}"
    query = urllib.parse.urlencode({
        "service": "ghcr.io", "scope": f"repository:{repository}:pull",
    })
    with urllib.request.urlopen(f"https://ghcr.io/token?{query}", timeout=30) as response:
        token = json.load(response)["token"]

    def read(kind: str, name: str) -> tuple[dict, str]:
        request = urllib.request.Request(
            f"https://ghcr.io/v2/{repository}/{kind}/{name}",
            headers={"Authorization": f"Bearer {token}", "Accept": _ACCEPT},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if name.startswith("sha256:") and digest != name:
            raise ValueError(f"registry digest mismatch for {repository}/{name}")
        return json.loads(content), digest

    manifest, image_digest = read("manifests", reference)
    if "manifests" in manifest:
        candidates = [item for item in manifest["manifests"]
                      if item.get("platform", {}).get("os") == "linux"
                      and item.get("platform", {}).get("architecture") == "amd64"]
        if len(candidates) != 1:
            raise ValueError(f"{repository}:{reference} needs one linux/amd64 image")
        manifest, _ = read("manifests", candidates[0]["digest"])
    config, _ = read("blobs", manifest["config"]["digest"])
    revision = config.get("config", {}).get("Labels", {}).get(
        "org.opencontainers.image.revision", ""
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{repository}:{reference} has no full source revision label")
    return f"ghcr.io/{repository}@{image_digest}", revision


def resolve_launch_images(images: dict[str, str], revision: str) -> tuple[dict[str, str], str]:
    """Fill omitted images from one published source revision; preserve overrides."""
    resolved = images.copy()
    if all(resolved.values()):
        return resolved, revision
    if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("senpai_repo_revision must be a full lowercase commit SHA")
    supplied_revisions = {
        source_revision_for_image(image, revision)
        for image in resolved.values() if image
    }
    if len(supplied_revisions) > 1:
        raise ValueError("role images must use the same source revision")
    revision = next(iter(supplied_revisions), revision)
    try:
        if not revision:
            resolved["student"], revision = published_image("student", "latest")
        for role, image in resolved.items():
            if image:
                continue
            resolved[role], image_revision = published_image(role, f"sha-{revision}")
            if image_revision != revision:
                raise ValueError(f"published {role} image does not match {revision}")
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        raise ValueError(
            "could not resolve a complete matching Senpai image set; "
            "check registry access and that the image build has finished"
        ) from error
    return resolved, revision
