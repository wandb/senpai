import unittest
from unittest.mock import patch

from k8s import launch

REVISION = "a" * 40


class BackendDefaultTests(unittest.TestCase):
    def test_defaults_preserve_kubernetes_shape(self):
        args = launch.Args("tag", "https://github.com/example/target.git")

        launch.resolve_backend_defaults(args)

        self.assertEqual((args.n_students, args.gpus_per_student), (4, 8))

    def test_defaults_make_single_host_launch_small(self):
        for backend in ("docker", "aws"):
            args = launch.Args(
                "tag",
                "https://github.com/example/target.git",
                backend=backend,
            )

            launch.resolve_backend_defaults(args)

            self.assertEqual((args.n_students, args.gpus_per_student), (1, 1))


class LaunchOrderingTests(unittest.TestCase):
    def test_docker_compute_preflight_precedes_github_mutations(self):
        args = launch.Args(
            "safe-tag",
            "https://github.com/example/target.git",
            backend="docker",
            names="fern",
            gpus_per_student=1,
            advisor_image=f"ghcr.io/wandb/senpai-advisor:sha-{REVISION}",
            student_image=f"ghcr.io/wandb/senpai-student:sha-{REVISION}",
        )
        events = []

        with (
            patch.object(launch.sp, "parse", return_value=args),
            patch.object(launch, "resolve_github_token", return_value="github"),
            patch.object(launch, "resolve_anthropic_api_key", return_value="anthropic"),
            patch.object(launch, "resolve_exa_api_key", return_value="exa"),
            patch.object(launch, "resolve_wandb_api_key", return_value="wandb"),
            patch.object(launch, "resolve_optional_secret", return_value=""),
            patch.object(launch, "preflight_check_target_repo_access"),
            patch.object(
                launch,
                "preflight_check_target_repo_branch",
                return_value="main",
            ),
            patch.object(launch, "preflight_check_anthropic_api_key"),
            patch.object(launch, "preflight_check_exa_api_key"),
            patch.object(launch, "preflight_check_wandb_api_key"),
            patch.object(
                launch,
                "preflight_docker",
                side_effect=lambda *_: events.append("compute") or "plan",
            ),
            patch.object(
                launch,
                "ensure_advisor_branch",
                side_effect=lambda *_: events.append("branch"),
            ),
            patch.object(
                launch,
                "ensure_target_repo_labels",
                side_effect=lambda *_: events.append("labels"),
            ),
            patch.object(
                launch,
                "launch_docker",
                side_effect=lambda *values: events.append(("launch", values[2])),
            ),
        ):
            launch.main()

        self.assertEqual(
            events,
            ["compute", "branch", "labels", ("launch", "plan")],
        )

    def test_aws_mutates_github_only_after_remote_preflight(self):
        args = launch.Args(
            "safe-tag",
            "https://github.com/example/target.git",
            backend="aws",
            names="fern",
            gpus_per_student=1,
            student_image=f"ghcr.io/wandb/senpai-student:sha-{REVISION}",
        )
        events = []

        def launch_after_preflight(*values, before_start):
            events.append(("remote-ready", values[2]))
            before_start()
            events.append("roles")

        with (
            patch.object(launch.sp, "parse", return_value=args),
            patch.object(launch, "resolve_github_token", return_value="github"),
            patch.object(launch, "resolve_anthropic_api_key", return_value="anthropic"),
            patch.object(launch, "resolve_exa_api_key", return_value="exa"),
            patch.object(launch, "resolve_wandb_api_key", return_value="wandb"),
            patch.object(launch, "resolve_optional_secret", return_value=""),
            patch.object(launch, "preflight_check_target_repo_access"),
            patch.object(
                launch,
                "preflight_check_target_repo_branch",
                return_value="main",
            ),
            patch.object(launch, "preflight_check_anthropic_api_key"),
            patch.object(launch, "preflight_check_exa_api_key"),
            patch.object(launch, "preflight_check_wandb_api_key"),
            patch.object(
                launch,
                "preflight_aws",
                side_effect=lambda *_: events.append("account") or "plan",
            ),
            patch.object(
                launch,
                "ensure_advisor_branch",
                side_effect=lambda *_: events.append("branch"),
            ),
            patch.object(
                launch,
                "ensure_target_repo_labels",
                side_effect=lambda *_: events.append("labels"),
            ),
            patch.object(
                launch,
                "launch_aws",
                side_effect=launch_after_preflight,
            ),
        ):
            launch.main()

        self.assertEqual(
            events,
            ["account", ("remote-ready", "plan"), "branch", "labels", "roles"],
        )

    def test_aws_remote_preflight_failure_leaves_github_unchanged(self):
        args = launch.Args(
            "safe-tag",
            "https://github.com/example/target.git",
            backend="aws",
            names="fern",
            gpus_per_student=1,
            student_image=f"ghcr.io/wandb/senpai-student:sha-{REVISION}",
        )

        with (
            patch.object(launch.sp, "parse", return_value=args),
            patch.object(launch, "resolve_github_token", return_value="github"),
            patch.object(launch, "resolve_anthropic_api_key", return_value="anthropic"),
            patch.object(launch, "resolve_exa_api_key", return_value="exa"),
            patch.object(launch, "resolve_wandb_api_key", return_value="wandb"),
            patch.object(launch, "resolve_optional_secret", return_value=""),
            patch.object(launch, "preflight_check_target_repo_access"),
            patch.object(
                launch,
                "preflight_check_target_repo_branch",
                return_value="main",
            ),
            patch.object(launch, "preflight_check_anthropic_api_key"),
            patch.object(launch, "preflight_check_exa_api_key"),
            patch.object(launch, "preflight_check_wandb_api_key"),
            patch.object(launch, "preflight_aws", return_value="plan"),
            patch.object(launch, "ensure_advisor_branch") as branch,
            patch.object(launch, "ensure_target_repo_labels") as labels,
            patch.object(
                launch,
                "launch_aws",
                side_effect=RuntimeError("remote preflight failed"),
            ),
        ):
            with self.assertRaisesRegex(SystemExit, "remote preflight failed"):
                launch.main()

        branch.assert_not_called()
        labels.assert_not_called()

    def test_preflight_only_never_mutates_github_or_launches(self):
        args = launch.Args(
            "safe-tag",
            "https://github.com/example/target.git",
            backend="docker",
            names="fern",
            gpus_per_student=1,
            advisor_image=f"ghcr.io/wandb/senpai-advisor:sha-{REVISION}",
            student_image=f"ghcr.io/wandb/senpai-student:sha-{REVISION}",
            preflight_only=True,
        )

        with (
            patch.object(launch.sp, "parse", return_value=args),
            patch.object(launch, "resolve_github_token", return_value="github"),
            patch.object(launch, "resolve_anthropic_api_key", return_value="anthropic"),
            patch.object(launch, "resolve_exa_api_key", return_value="exa"),
            patch.object(launch, "resolve_wandb_api_key", return_value="wandb"),
            patch.object(launch, "resolve_optional_secret", return_value=""),
            patch.object(launch, "preflight_check_target_repo_access"),
            patch.object(
                launch,
                "preflight_check_target_repo_branch",
                return_value="main",
            ),
            patch.object(launch, "preflight_check_anthropic_api_key"),
            patch.object(launch, "preflight_check_exa_api_key"),
            patch.object(launch, "preflight_check_wandb_api_key"),
            patch.object(launch, "preflight_docker", return_value="plan"),
            patch.object(launch, "ensure_advisor_branch") as branch,
            patch.object(launch, "ensure_target_repo_labels") as labels,
            patch.object(launch, "launch_docker") as start,
        ):
            launch.main()

        branch.assert_not_called()
        labels.assert_not_called()
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
