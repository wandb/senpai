import pytest
import yaml

from git_workflow_support import git
from launch_test_support import launch, launch_args, render_role


@pytest.mark.parametrize("role", ["advisor", "student"])
@pytest.mark.parametrize(
    "problem_dir,pvc_mount_path",
    [
        ("target/", "/mnt/data"),
        ('research/target: #1 "quoted" \\path', '/mnt/data: #1 "quoted"'),
        ("target/{{LAUNCH_SECRET_NAME}}", "/mnt/{{TARGET_WORKSPACE_MOUNT}}"),
        ("k8s-experiments", "/workspace/senpai/target-sibling"),
    ],
)
def test_render_preserves_distinct_mount_paths(role, problem_dir, pvc_mount_path):
    configmap, deployment, _ = render_role(
        role,
        launch_args(problem_dir=problem_dir, pvc_mount_path=pvc_mount_path),
    )

    data = yaml.safe_load(configmap)["data"]
    container = yaml.safe_load(deployment)["spec"]["template"]["spec"]["containers"][0]
    paths = [mount["mountPath"] for mount in container["volumeMounts"]]
    assert paths.count(f"/workspace/senpai/{problem_dir.rstrip('/')}") == 1
    assert paths.count(pvc_mount_path) == 1
    assert data["PROBLEM_DIR"] == problem_dir.rstrip("/")
    assert data["PVC_MOUNT_PATH"] == pvc_mount_path


@pytest.mark.parametrize("role", ["advisor", "student"])
@pytest.mark.parametrize(
    "problem_dir",
    [
        "",
        ".",
        "../target",
        "/target",
        "research/../target",
        "research//target",
        "target//",
        ".git/checkout",
        ".codex/checkout",
        ".agents/checkout",
        "k8s",
        "system_instructions/target",
        "senpai_agent/target",
        "plugins/target",
        "README.md",
        "pyproject.toml/target",
    ],
)
def test_render_rejects_target_mounts_outside_a_distinct_runner_child(role, problem_dir):
    with pytest.raises(SystemExit, match="ERROR: --problem_dir"):
        render_role(role, launch_args(problem_dir=problem_dir))


@pytest.mark.parametrize("role", ["advisor", "student"])
@pytest.mark.parametrize(
    "pvc_mount_path",
    [
        "/",
        "/workspace/senpai",
        "/workspace/senpai/target",
        "/workspace/senpai/target/data",
        "/workspace/senpai/other/../target",
        "//workspace/senpai",
        "/home/senpai/.venvs",
        "/home/senpai/.venvs/senpai-target",
        "/home/senpai/.venvs/senpai-target/data",
        "relative/data",
    ],
)
def test_render_rejects_pvc_mounts_that_cover_or_enter_private_workspaces(
    role, pvc_mount_path
):
    with pytest.raises(SystemExit, match="ERROR: --pvc_mount_path"):
        render_role(role, launch_args(pvc_mount_path=pvc_mount_path))


@pytest.mark.parametrize(
    "overrides,option",
    [
        ({"problem_dir": "../target"}, "--problem_dir"),
        ({"pvc_mount_path": "/workspace"}, "--pvc_mount_path"),
    ],
)
def test_launch_rejects_invalid_mounts_before_resolving_credentials(
    monkeypatch, overrides, option
):
    args = launch_args(**overrides)
    monkeypatch.setattr(launch.sp, "parse", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(
        launch,
        "resolve_custom_secrets",
        lambda *_args: pytest.fail("invalid mounts must fail before credential access"),
    )

    with pytest.raises(SystemExit, match=f"ERROR: {option}"):
        launch.main()


def test_render_protects_tracked_assets_but_allows_existing_untracked_target(
    tmp_path, monkeypatch
):
    git(tmp_path, "init")
    (tmp_path / "runner-code").mkdir()
    (tmp_path / "runner-code" / "entry.py").write_text("print('runner')\n")
    git(tmp_path, "add", "runner-code")
    (tmp_path / "target").mkdir()
    monkeypatch.setattr(launch, "ROOT", tmp_path)

    with pytest.raises(SystemExit, match="ERROR: --problem_dir"):
        render_role("student", launch_args(problem_dir="runner-code/checkout"))
    configmap, _, _ = render_role("student", launch_args(problem_dir="target/"))
    assert yaml.safe_load(configmap)["data"]["PROBLEM_DIR"] == "target"
