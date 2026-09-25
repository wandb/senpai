import subprocess
from pathlib import Path


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def commit_file(
    workspace: Path,
    path: str,
    contents: str,
    message: str,
) -> str:
    (workspace / path).write_text(contents)
    git(workspace, "add", path)
    git(workspace, "commit", "-m", message)
    return git(workspace, "rev-parse", "HEAD")


def commit_workspace(workspace: Path, message: str = "program snapshot") -> str:
    if not (workspace / ".git").is_dir():
        git(workspace.parent, "init", str(workspace))
        git(workspace, "config", "user.name", "Operator")
        git(workspace, "config", "user.email", "operator@example.com")
    if git(workspace, "status", "--porcelain"):
        git(workspace, "add", "-A")
        git(workspace, "commit", "-m", message)
    return git(workspace, "rev-parse", "HEAD")


def detached_commit(workspace: Path, parent: str, message: str) -> str:
    tree = git(workspace, "rev-parse", f"{parent}^{{tree}}")
    return git(workspace, "commit-tree", tree, "-p", parent, "-m", message)


def repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    workspace = tmp_path / "workspace"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", str(workspace))
    git(workspace, "config", "user.name", "Student")
    git(workspace, "config", "user.email", "student@example.com")
    commit_file(workspace, "model.py", "baseline = 1\n", "baseline")
    git(workspace, "branch", "-M", "experiment-7")
    git(workspace, "remote", "add", "origin", str(remote))
    git(workspace, "push", "-u", "origin", "experiment-7")
    return workspace, remote, git(workspace, "rev-parse", "HEAD")
