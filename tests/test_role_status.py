import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_SCRIPT = ROOT / "plugins/senpai/scripts/senpai-gh.sh"
GATE_SCRIPT = ROOT / "k8s/wait-senpai-start-gate.sh"


def run_shell(body: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **environment}
    return subprocess.run(
        ["bash", "-c", f'set -eo pipefail\nsource "{STATUS_SCRIPT}"\n{body}'],
        capture_output=True,
        text=True,
        env=env,
    )


def write_status(
    directory: Path,
    *,
    epoch: int,
    branch: str = "experiment",
    training: int = 1,
    gpu: int = 0,
    claude: int = 0,
    dirty: int = 0,
) -> None:
    student_dir = directory / "fern"
    student_dir.mkdir()
    (student_dir / "status").write_text(
        "\n".join(
            [
                f"epoch={epoch}",
                f"branch={branch}",
                f"training={training}",
                f"gpu={gpu}",
                f"claude={claude}",
                f"dirty={dirty}",
                "",
            ]
        )
    )


class StudentStatusTests(unittest.TestCase):
    def test_matching_fresh_status_keeps_active_pr_live_without_kubectl(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_dir = Path(tmp)
            write_status(status_dir, epoch=int(time.time()))

            result = run_shell(
                'student_pr_looks_live "fern" "experiment"',
                SENPAI_STATUS_DIR=str(status_dir),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("kubectl", STATUS_SCRIPT.read_text())

    def test_stale_status_does_not_keep_pr_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_dir = Path(tmp)
            write_status(status_dir, epoch=int(time.time()) - 300)

            result = run_shell(
                'student_pr_looks_live "fern" "experiment"',
                SENPAI_STATUS_DIR=str(status_dir),
            )

            self.assertNotEqual(result.returncode, 0)

    def test_mismatched_branch_does_not_keep_pr_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_dir = Path(tmp)
            write_status(status_dir, epoch=int(time.time()), branch="other-experiment")

            result = run_shell(
                'student_pr_looks_live "fern" "experiment"',
                SENPAI_STATUS_DIR=str(status_dir),
            )

            self.assertNotEqual(result.returncode, 0)

    def test_wrong_branch_training_is_reported_from_shared_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_dir = Path(tmp)
            write_status(status_dir, epoch=int(time.time()), branch="wrong-branch")
            result = run_shell(
                """
rest_labeled_pull_details() {
    printf '%s\\n' '[{"headRefName":"expected-branch","labels":[{"name":"student:fern"}]}]'
}
list_student_pod_anomalies "fern" "research"
""",
                SENPAI_STATUS_DIR=str(status_dir),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            anomalies = json.loads(result.stdout)
            self.assertEqual(len(anomalies), 1)
            self.assertIn("wrong-branch", anomalies[0])
            self.assertIn("expected-branch", anomalies[0])

    def test_stale_worker_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_dir = Path(tmp)
            write_status(status_dir, epoch=int(time.time()) - 300)
            result = run_shell(
                """
rest_labeled_pull_details() { printf '[]\\n'; }
list_student_pod_anomalies "fern" "research"
""",
                SENPAI_STATUS_DIR=str(status_dir),
                SENPAI_STATUS_STALE_SECONDS="120",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("status heartbeat is stale", json.loads(result.stdout)[0])

    def test_writer_publishes_a_complete_atomic_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            status_dir = root / "status"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            for command in ("ps", "nvidia-smi"):
                executable = fake_bin / command
                executable.write_text("#!/bin/sh\nexit 0\n")
                executable.chmod(0o755)
            subprocess.run(
                ["git", "init", "-q", "-b", "experiment", str(target)],
                check=True,
            )

            result = run_shell(
                "write_student_status",
                SENPAI_STATUS_DIR=str(status_dir),
                SENPAI_STATUS_INTERVAL_SECONDS="30",
                STUDENT_NAME="fern",
                TARGET_WORKDIR=str(target),
                PATH=f"{fake_bin}:{os.environ['PATH']}",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            fields = dict(
                line.split("=", 1)
                for line in (status_dir / "fern" / "status").read_text().splitlines()
            )
            self.assertEqual(
                set(fields),
                {"epoch", "branch", "training", "gpu", "claude", "dirty"},
            )
            self.assertEqual(fields["branch"], "experiment")
            self.assertFalse(list(status_dir.rglob("*.tmp.*")))


class RoleGateTests(unittest.TestCase):
    def test_role_marks_ready_then_accepts_launcher_and_user_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / "role" / "ready"
            launch_gate = root / "shared" / ".launch"
            user_gate = root / "shared" / ".user"
            launch_gate.parent.mkdir()
            launch_gate.touch()
            user_gate.touch()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        f'set -e\nsource "{GATE_SCRIPT}"\n'
                        "mark_senpai_ready\n"
                        "wait_for_senpai_start_gate\n"
                    ),
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SENPAI_READY_FILE": str(ready),
                    "SENPAI_LAUNCH_GATE_PATH": str(launch_gate),
                    "SENPAI_START_GATE_PATH": str(user_gate),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(ready.is_file())

    def test_entrypoints_wait_only_at_the_role_loop_boundary(self):
        for name in ("entrypoint-advisor.sh", "entrypoint-student.sh"):
            text = (ROOT / "k8s" / name).read_text()
            boundary = "mark_senpai_ready\nwait_for_senpai_start_gate"
            self.assertEqual(text.count("wait_for_senpai_start_gate"), 1)
            self.assertLess(text.index(boundary), text.index("ITERATION=0\nwhile true"))


if __name__ == "__main__":
    unittest.main()
