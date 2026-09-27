import hashlib
import io
import json
import urllib.error

import pytest

from launch_test_support import launch
from images import published_image, resolve_launch_images


REVISION = "d" * 40


def registry(monkeypatch, *, wrong_role_revision=False):
    """A small OCI index/manifest/blob response set at the HTTP boundary."""
    documents = {}
    pins = {}
    requests = []
    for role in ("advisor", "student", "executor"):
        prefix = f"https://ghcr.io/v2/wandb/senpai-{role}"
        source = "e" * 40 if wrong_role_revision and role == "executor" else REVISION
        config = json.dumps({"config": {"Labels": {
            "org.opencontainers.image.revision": source,
        }}}).encode()
        config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
        manifest = json.dumps({"config": {"digest": config_digest}}).encode()
        manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
        index = json.dumps({"manifests": [{
            "platform": {"os": "linux", "architecture": "amd64"},
            "digest": manifest_digest,
        }]}).encode()
        index_digest = "sha256:" + hashlib.sha256(index).hexdigest()
        documents[f"{prefix}/manifests/latest"] = index
        documents[f"{prefix}/manifests/sha-{REVISION}"] = index
        documents[f"{prefix}/manifests/{manifest_digest}"] = manifest
        documents[f"{prefix}/blobs/{config_digest}"] = config
        pins[role] = f"ghcr.io/wandb/senpai-{role}@{index_digest}"

    def urlopen(request, *, timeout):
        assert timeout > 0
        url = request if isinstance(request, str) else request.full_url
        requests.append(url)
        if url.startswith("https://ghcr.io/token?"):
            return io.BytesIO(b'{"token":"public-pull-token"}')
        assert request.headers["Authorization"] == "Bearer public-pull-token"
        return io.BytesIO(documents[url])

    monkeypatch.setattr("images.urllib.request.urlopen", urlopen)
    return documents, pins, requests


def test_default_images_pin_one_published_release_and_expose_it_to_the_student(monkeypatch):
    _, pins, requests = registry(monkeypatch)

    images, revision = resolve_launch_images(dict.fromkeys(pins, ""), "")

    assert images == pins
    assert revision == REVISION
    assert sum(url.endswith("/latest") for url in requests) == 1
    assert any(f"/senpai-executor/manifests/sha-{REVISION}" in url for url in requests)
    args = launch.Args(tag="image-proof", target_repo_url="https://github.com/example/problem")
    args.student_image = images["student"]
    context = launch.build_launch_context(args, args.tag, ["fern"], backend="kubernetes", role="student")
    assert pins["student"] in context


def test_defaults_reject_mislabelled_matching_build(monkeypatch):
    registry(monkeypatch, wrong_role_revision=True)

    with pytest.raises(ValueError, match="executor image does not match"):
        resolve_launch_images(dict.fromkeys(("advisor", "student", "executor"), ""), "")


def test_registry_bytes_must_match_the_referenced_config_digest(monkeypatch):
    documents, _, _ = registry(monkeypatch)
    path = next(path for path in documents if "/senpai-student/blobs/" in path)
    documents[path] = b'{}'

    with pytest.raises(ValueError, match="digest mismatch"):
        published_image("student", "latest")


def test_explicit_image_set_is_kept_without_registry_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit image set must not contact the public registry")
    monkeypatch.setattr("images.urllib.request.urlopen", forbidden)
    images = {"student": "private.example/student@sha256:" + "a" * 64}

    assert resolve_launch_images(images, REVISION) == (images, REVISION)


def test_unpublished_matching_image_stops_resolution(monkeypatch):
    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("image not published")
    monkeypatch.setattr("images.urllib.request.urlopen", unavailable)

    with pytest.raises(ValueError, match="complete matching Senpai image set"):
        resolve_launch_images({"student": "", "executor": ""}, "")
