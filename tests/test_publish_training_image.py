import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish-training-image.py"
DIGEST = "sha256:" + "a" * 64
CONFIG = json.dumps({
    "os": "linux", "architecture": "arm64", "variant": "v8",
    "rootfs": {"type": "layers", "diff_ids": []},
}).encode()
CONFIG_DIGEST = "sha256:" + hashlib.sha256(CONFIG).hexdigest()
ARCHIVE_MANIFEST = [{
    "RepoTags": ["local/training:v1"], "Config": "config.json", "Layers": [],
}]


def save_archive(path, manifest):
    with tarfile.open(path, "w:gz") as saved:
        for name, contents in (
            ("manifest.json", json.dumps(manifest).encode()), ("config.json", CONFIG),
        ):
            entry = tarfile.TarInfo(name)
            entry.size = len(contents)
            saved.addfile(entry, io.BytesIO(contents))


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        f"#!{sys.executable}\n" + """
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as calls:
    calls.write(json.dumps(args) + "\\n")
operation = args[1] if args[0] == "buildx" else args[0]
if operation == os.environ.get("FAIL_OPERATION"):
    print("Docker operation failed", file=sys.stderr)
    sys.exit(23)
digest = os.environ["IMAGE_DIGEST"]
if operation == "build":
    Path(args[args.index("--metadata-file") + 1]).write_text(
        json.dumps({"containerimage.digest": digest})
    )
if operation == "imagetools":
    if "--raw" in args:
        print(json.dumps({"config": {"digest": os.environ["PUBLISHED_CONFIG_DIGEST"]}}))
    else:
        print(json.dumps({"digest": digest}))
else:
    print("Docker progress")
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("IMAGE_DIGEST", DIGEST)
    monkeypatch.setenv("PUBLISHED_CONFIG_DIGEST", CONFIG_DIGEST)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    archive = tmp_path / "image.tar.gz"
    save_archive(archive, ARCHIVE_MANIFEST)

    def run(*args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            capture_output=True,
            text=True,
            check=False,
        )
        calls_file = tmp_path / "calls.jsonl"
        calls = (
            [json.loads(line) for line in calls_file.read_text().splitlines()]
            if calls_file.exists() else []
        )
        return result, calls

    return run, dockerfile, archive


@pytest.mark.parametrize("explicit_context", [False, True])
def test_build_publishes_selected_context_and_prints_only_immutable_reference(
    publisher, tmp_path, explicit_context
):
    run, dockerfile, _ = publisher
    context = tmp_path / "source" if explicit_context else tmp_path
    extra = []
    if explicit_context:
        context.mkdir()
        extra = ["--context", context]
    tag = "registry.example:5000/team/training:v1"

    result, calls = run("--dockerfile", dockerfile, "--tag", tag, *extra)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"registry.example:5000/team/training@{DIGEST}\n"
    assert "Docker progress" in result.stderr
    (build,) = calls
    assert "--push" in build
    assert build[build.index("--file") + 1] == str(dockerfile)
    assert build[build.index("--tag") + 1] == tag
    assert build[build.index("--platform") + 1] == "linux/amd64"
    assert build[-1] == str(context)


@pytest.mark.parametrize("platform", ["linux/arm64", "linux/arm64/v8"])
def test_archive_loads_selected_platform_then_publishes_selected_image(publisher, platform):
    run, _, archive = publisher
    tag = "docker.io/team/training:v1"

    result, calls = run(
        "--archive", archive, "--archive-image", "local/training:v1",
        "--platform", platform, "--tag", tag,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"docker.io/team/training@{DIGEST}\n"
    load, retag, push, inspect, verify = calls
    assert load[0] == "load"
    assert load[load.index("--input") + 1] == str(archive)
    assert load[load.index("--platform") + 1] == platform
    assert retag == ["tag", "local/training:v1", tag]
    assert push == ["push", "--platform", platform, tag]
    assert inspect[:4] == ["buildx", "imagetools", "inspect", tag]
    assert verify == [
        "buildx", "imagetools", "inspect", f"docker.io/team/training@{DIGEST}", "--raw",
    ]


@pytest.mark.parametrize("operation", ["build", "load", "push", "imagetools"])
def test_docker_failure_stops_publication_without_printing_a_reference(
    publisher, monkeypatch, operation
):
    run, dockerfile, archive = publisher
    monkeypatch.setenv("FAIL_OPERATION", operation)
    source = ["--dockerfile", dockerfile] if operation == "build" else [
        "--archive", archive, "--archive-image", "local/training:v1",
        "--platform", "linux/arm64",
    ]

    result, calls = run(*source, "--tag", "ghcr.io/team/training:v1")

    assert result.returncode == 23
    assert result.stdout == ""
    assert "Docker operation failed" in result.stderr
    last_operation = calls[-1][1] if calls[-1][0] == "buildx" else calls[-1][0]
    assert last_operation == operation


@pytest.mark.parametrize(
    "source, extra, message",
    [
        ("archive", ["--archive-image", "unrelated:v1"], "is not saved"),
        ("archive", [], "requires --archive-image"),
        ("archive", ["--archive-image", "local/training:v1"], "exactly one linux/amd64"),
        ("dockerfile", ["--archive-image", "local:v1"], "requires --archive"),
        ("dockerfile", ["--tag", "ghcr.io/team/training"], "explicit destination tag"),
    ],
)
def test_invalid_input_fails_before_docker_mutations(publisher, source, extra, message):
    run, dockerfile, archive = publisher
    path = archive if source == "archive" else dockerfile

    result, calls = run(f"--{source}", path, "--tag", "ghcr.io/team/train:v1", *extra)

    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""
    assert calls == []


@pytest.mark.parametrize("problem", ["missing_manifest", "ambiguous_platform"])
def test_invalid_archive_fails_before_docker_mutations(publisher, problem):
    run, _, archive = publisher
    if problem == "missing_manifest":
        with tarfile.open(archive, "w:gz"):
            pass
        message = "archive must contain manifest.json"
    else:
        save_archive(archive, ARCHIVE_MANIFEST * 2)
        message = "found 2"

    result, calls = run(
        "--archive", archive, "--archive-image", "local/training:v1",
        "--platform", "linux/arm64", "--tag", "ghcr.io/team/training:v1",
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""
    assert calls == []


def test_concurrent_registry_tag_change_cannot_report_an_unrelated_image(
    publisher, monkeypatch
):
    run, _, archive = publisher
    monkeypatch.setenv("PUBLISHED_CONFIG_DIGEST", "sha256:" + "b" * 64)

    result, calls = run(
        "--archive", archive, "--archive-image", "local/training:v1",
        "--platform", "linux/arm64", "--tag", "ghcr.io/team/training:v1",
    )

    assert result.returncode == 2
    assert "published image does not match the selected archive image" in result.stderr
    assert result.stdout == ""
    assert calls[-1][3] == f"ghcr.io/team/training@{DIGEST}"
