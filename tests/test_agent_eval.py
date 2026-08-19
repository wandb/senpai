# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import json
import subprocess
from pathlib import Path

import pytest
import yaml

import eval.run as agent_eval
from eval.run import (
    DEFAULT_TRAINING_TIMEOUT_MINUTES,
    MODEL,
    NANOGPT_BENCHMARK,
    REASONING_EFFORT,
    TANDEM_SPLITS,
    build_manifest,
    launch_eval,
    report_eval,
    score_nanogpt_run,
    score_tandem_run,
    target_launch_config,
    validate_run_id,
)


class FakeRun:
    def __init__(
        self,
        *,
        config=None,
        summary=None,
        validation=(),
        markers=(),
        state="finished",
    ):
        self.id = "run-123"
        self.name = "candidate"
        self.url = "https://wandb.ai/acme/evals/runs/run-123"
        self.state = state
        self.config = config or {}
        self.summary = summary or {}
        self.validation = list(validation)
        self.markers = list(markers)

    def scan_history(self, *, keys):
        if "val/loss" in keys:
            return iter(self.validation)
        return iter(self.markers)


def local_config():
    return {
        "wandb_entity": "acme",
        "wandb_project": "senpai-evals",
        "pvc_claim_name": "datasets",
        "pvc_mount_path": "/mnt/data",
        "advisor_image": "ghcr.io/wandb/senpai-advisor:sha-" + "a" * 40,
        "student_image": "ghcr.io/wandb/senpai-student:sha-" + "a" * 40,
    }


def test_eval_launch_config_is_bounded_isolated_and_all_luna():
    base = local_config()
    manifest = build_manifest(
        base,
        "eval-20260819-120000-abc123",
        web_search=False,
        dry_run=False,
    )
    target = manifest["targets"][0]

    config = target_launch_config(base, manifest, target)

    assert config["target_repo_url"].endswith("modded-nanogpt-senpai")
    assert config["target_repo_branch"] == "master"
    assert len(target["base_revision"]) == 40
    assert config["advisor"] is True
    assert config["n_students"] == 1
    assert config["timeout_minutes"] == DEFAULT_TRAINING_TIMEOUT_MINUTES
    assert config["web_search"] is False
    assert config["human_issues"] is False
    assert config["gh_history_scope"] == "fresh"
    assert config["wandb_run_group"] == target["wandb_group"]
    assert config["start_gate_path"] == manifest["start_gate_path"]
    assert {
        config["advisor_model"],
        config["student_model"],
        config["smart_model"],
        config["fast_model"],
        config["frontier_model"],
    } == {MODEL}
    assert {
        config["advisor_reasoning_effort"],
        config["student_reasoning_effort"],
        config["smart_reasoning_effort"],
        config["fast_reasoning_effort"],
        config["frontier_reasoning_effort"],
    } == {REASONING_EFFORT}
    assert config["target_repo_revision"] == target["base_revision"]


def valid_nanogpt_run(final_loss=3.275, first_step=3350):
    return FakeRun(
        config={"benchmark": NANOGPT_BENCHMARK, "num_trials": 1},
        validation=[
            {"trial": 0, "val/step": 3000, "val/loss": 3.29},
            {"trial": 0, "val/step": 3350, "val/loss": final_loss},
        ],
        markers=[
            {
                "trial": 0,
                "speedrun/final_first_step_to_target": first_step,
                "speedrun/final_reached_target": int(first_step >= 0),
            }
        ],
    )


def test_nanogpt_scores_first_target_step_only_after_final_loss_gate():
    score, reason = score_nanogpt_run(valid_nanogpt_run())

    assert reason is None
    assert score["score"] == 3350
    assert score["diagnostics"]["final_val_loss"] == 3.275
    assert score["diagnostics"]["final_val_step"] == 3350
    assert score["diagnostics"]["significance_gate_margin"] > 0


def test_nanogpt_rejects_minus_one_and_best_only_results():
    score, reason = score_nanogpt_run(valid_nanogpt_run(first_step=-1))
    assert score is None
    assert reason == "missing successful final target marker"

    score, reason = score_nanogpt_run(valid_nanogpt_run(final_loss=3.279))
    assert score is None
    assert "significance" in reason


def test_nanogpt_rejects_multi_trial_runs_for_comparable_scoring():
    run = valid_nanogpt_run()
    run.config["num_trials"] = 2

    score, reason = score_nanogpt_run(run)

    assert score is None
    assert reason == "num_trials must equal 1"


def test_nanogpt_rejects_a_marker_that_disagrees_with_validation_history():
    score, reason = score_nanogpt_run(valid_nanogpt_run(first_step=3200))

    assert score is None
    assert reason == "final target marker disagrees with validation history"


@pytest.mark.parametrize("trial", [-0.5, 0.5, 1])
def test_nanogpt_requires_exact_integer_trial_zero_in_validation(trial):
    run = valid_nanogpt_run()
    for row in run.validation:
        row["trial"] = trial

    score, reason = score_nanogpt_run(run)

    assert score is None
    assert reason == "validation trial must equal integer 0"


@pytest.mark.parametrize("trial", [-0.5, 0.5, 1])
def test_nanogpt_requires_exact_integer_trial_zero_in_final_marker(trial):
    run = valid_nanogpt_run()
    run.markers[-1]["trial"] = trial

    score, reason = score_nanogpt_run(run)

    assert score is None
    assert reason == "final marker trial must equal integer 0"


@pytest.mark.parametrize("step", [-1, 3350.5])
def test_nanogpt_requires_nonnegative_integer_validation_steps(step):
    run = valid_nanogpt_run(first_step=step)
    run.validation[-1]["val/step"] = step

    score, reason = score_nanogpt_run(run)

    assert score is None
    assert reason == "validation steps must be nonnegative integers"


def test_nanogpt_requires_a_nonnegative_integer_final_marker():
    run = valid_nanogpt_run(first_step=3350.5)

    score, reason = score_nanogpt_run(run)

    assert score is None
    assert reason == "missing successful final target marker"


def valid_tandem_summary():
    values = [12.0, 14.0, 16.0, 18.0]
    summary = {
        f"test/{split}/mae_surf_p": value
        for split, value in zip(TANDEM_SPLITS, values, strict=True)
    }
    summary.update(
        {
            "test_avg/mae_surf_p": 15.0,
            "best_val_avg/mae_surf_p": 13.5,
            "best_epoch": 7,
            "total_train_minutes": 17.2,
        }
    )
    return summary


def test_tandem_scores_recomputed_complete_test_average_even_if_run_crashed():
    run = FakeRun(
        config={"debug": False, "skip_test": False},
        summary=valid_tandem_summary(),
        state="crashed",
    )

    score, reason = score_tandem_run(run)

    assert reason is None
    assert score["score"] == 15.0
    assert score["state"] == "crashed"
    assert score["diagnostics"]["recomputed_test_avg"] == 15.0


def test_tandem_rejects_debug_missing_and_inconsistent_test_metrics():
    debug = FakeRun(config={"debug": True}, summary=valid_tandem_summary())
    assert score_tandem_run(debug) == (None, "debug run")

    missing_summary = valid_tandem_summary()
    missing_summary.pop(f"test/{TANDEM_SPLITS[-1]}/mae_surf_p")
    score, reason = score_tandem_run(FakeRun(summary=missing_summary))
    assert score is None
    assert "missing finite" in reason

    inconsistent = valid_tandem_summary()
    inconsistent["test_avg/mae_surf_p"] = 1.0
    score, reason = score_tandem_run(FakeRun(summary=inconsistent))
    assert score is None
    assert "inconsistent" in reason


@pytest.mark.parametrize("split", TANDEM_SPLITS)
def test_tandem_rejects_negative_split_metrics(split):
    negative_split = valid_tandem_summary()
    split_key = f"test/{split}/mae_surf_p"
    negative_split[split_key] = -1.0
    negative_split["test_avg/mae_surf_p"] = sum(
        negative_split[f"test/{split}/mae_surf_p"] for split in TANDEM_SPLITS
    ) / len(TANDEM_SPLITS)

    score, reason = score_tandem_run(FakeRun(summary=negative_split))
    assert score is None
    assert reason == f"{split_key} must be nonnegative"


def test_tandem_rejects_negative_average_metric():
    negative_average = valid_tandem_summary()
    negative_average["test_avg/mae_surf_p"] = -1.0

    score, reason = score_tandem_run(FakeRun(summary=negative_average))
    assert score is None
    assert reason == "test_avg/mae_surf_p must be nonnegative"


def test_eval_timeouts_are_configurable_and_share_one_absolute_deadline():
    manifest = build_manifest(
        local_config(),
        "eval-20260819-120000-abc123",
        web_search=False,
        dry_run=False,
        training_timeout_minutes=12.5,
        total_timeout_hours=1.25,
    )

    assert manifest["training_timeout_minutes"] == 12.5
    assert manifest["total_timeout_hours"] == 1.25
    assert manifest["deadline_epoch"] - manifest["started_at_epoch"] == 4500
    config = target_launch_config(local_config(), manifest, manifest["targets"][0])
    assert config["timeout_minutes"] == 12.5
    assert "12.5-minute wall-clock ceiling" in config["extra_instructions"]

    args = agent_eval.parse_args(
        [
            "launch",
            "--training-timeout-minutes",
            "12.5",
            "--total-timeout-hours",
            "1.25",
        ]
    )
    assert args.training_timeout_minutes == 12.5
    assert args.total_timeout_hours == 1.25
    with pytest.raises(ValueError, match="total timeout must be a positive"):
        build_manifest(
            local_config(),
            "eval-invalid-timeout",
            web_search=False,
            dry_run=False,
            total_timeout_hours=0,
        )
    with pytest.raises(ValueError, match="training timeout must be at least one second"):
        build_manifest(
            local_config(),
            "eval-subsecond-timeout",
            web_search=False,
            dry_run=False,
            training_timeout_minutes=0.001,
        )


def test_eval_run_id_keeps_every_github_routing_label_within_50_characters():
    run_id = "a" * 27
    manifest = build_manifest(
        local_config(), run_id, web_search=False, dry_run=False
    )

    assert all(len(target["advisor_branch"]) <= 50 for target in manifest["targets"])
    with pytest.raises(ValueError, match="at most 27"):
        validate_run_id("a" * 28)

    mutable = {**local_config(), "advisor_image": "senpai-advisor:latest"}
    with pytest.raises(ValueError, match="immutable digest"):
        build_manifest(mutable, "eval-mutable", web_search=False, dry_run=False)


def test_launch_preflights_both_targets_then_arms_and_rolls_back_on_failure(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "senpai.yaml"
    config_path.write_text(yaml.safe_dump(local_config()))
    events = []

    def run(argv, *, env=None):
        config = yaml.safe_load(Path(argv[-1]).read_text(encoding="utf-8"))
        kind = "preflight" if config["preflight_only"] else "launch"
        events.append((kind, config["tag"]))
        if kind == "launch" and config["tag"].endswith("tandemfoil"):
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(agent_eval, "run_checked", run)
    monkeypatch.setattr(
        agent_eval,
        "arm_cutoff",
        lambda _config, _manifest: events.append(("arm", "cutoff")),
    )
    monkeypatch.setattr(
        agent_eval,
        "wait_for_cutoff_ready",
        lambda _manifest: events.append(("ready", "cutoff")),
    )
    monkeypatch.setattr(
        agent_eval,
        "cleanup_eval_resources",
        lambda _manifest: events.append(("cleanup", "all")) or True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        launch_eval(
            config_path,
            tmp_path,
            run_id="eval-rollback",
            web_search=False,
            cutoff_image=None,
            dry_run=False,
        )

    assert events == [
        ("preflight", "eval-rollback-nanogpt"),
        ("preflight", "eval-rollback-tandemfoil"),
        ("arm", "cutoff"),
        ("ready", "cutoff"),
        ("launch", "eval-rollback-nanogpt"),
        ("launch", "eval-rollback-tandemfoil"),
        ("cleanup", "all"),
    ]
    saved = json.loads((tmp_path / "eval-rollback.json").read_text())
    assert saved["status"] == "launch_failed"
    assert saved["cleanup_status"] == "complete"


def test_cleanup_deletes_tagged_resources_before_the_cutoff(monkeypatch):
    manifest = build_manifest(
        local_config(), "eval-cleanup", web_search=False, dry_run=False
    )
    commands = []

    def run(argv):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(agent_eval, "run_best_effort", run)

    assert agent_eval.cleanup_eval_resources(manifest) is True
    assert commands[0][commands[0].index("delete") + 1] == (
        "deployments,configmaps,secrets"
    )
    assert (
        "app=senpai,research-tag in "
        "(eval-cleanup-nanogpt,eval-cleanup-tandemfoil)"
    ) in commands[0]
    assert commands[1][commands[1].index("delete") + 1] == "pods"
    assert commands[2][commands[2].index("delete") + 1] == "job"
    assert commands[3][commands[3].index("delete") + 1] == "configmap"


def test_cleanup_leaves_cutoff_armed_when_target_deletion_fails(monkeypatch):
    manifest = build_manifest(
        local_config(), "eval-cleanup-fail", web_search=False, dry_run=False
    )
    commands = []

    def fail(argv):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(agent_eval, "run_best_effort", fail)

    assert agent_eval.cleanup_eval_resources(manifest) is False
    assert len(commands) == 1
    assert "job" not in commands[0]


def test_cutoff_worker_must_exist_and_be_ready_before_launch(monkeypatch):
    manifest = build_manifest(
        local_config(), "eval-cutoff-ready", web_search=False, dry_run=False
    )
    commands = []
    monkeypatch.setattr(
        agent_eval, "run_checked", lambda argv: commands.append(argv)
    )

    agent_eval.wait_for_cutoff_ready(manifest)

    assert "--for=create" in commands[0]
    assert "--for=condition=Ready" in commands[1]
    assert all(
        "app=senpai-cutoff,run-slug=eval-cutoff-ready" in command
        for command in commands
    )


def test_cutoff_status_collects_logs_from_every_job_retry_pod(monkeypatch):
    manifest = build_manifest(
        local_config(), "eval-cutoff-retry", web_search=False, dry_run=False
    )
    commands = []

    def run(argv, **_kwargs):
        commands.append(argv)
        if "get" in argv and "job" in argv:
            stdout = json.dumps(
                {
                    "status": {
                        "conditions": [
                            {
                                "type": "Complete",
                                "status": "True",
                                "lastTransitionTime": "2026-08-19T18:00:00Z",
                            }
                        ]
                    }
                }
            )
            return subprocess.CompletedProcess(argv, 0, stdout=stdout)
        if "get" in argv and "pods" in argv:
            stdout = json.dumps(
                {
                    "items": [
                        {
                            "metadata": {
                                "name": "cutoff-success",
                                "creationTimestamp": "2026-08-19T12:01:00Z",
                            }
                        },
                        {
                            "metadata": {
                                "name": "cutoff-failed",
                                "creationTimestamp": "2026-08-19T12:00:00Z",
                            }
                        },
                    ]
                }
            )
            return subprocess.CompletedProcess(argv, 0, stdout=stdout)
        pod = argv[argv.index("logs") + 1]
        if pod == "pod/cutoff-failed":
            return subprocess.CompletedProcess(argv, 1, stdout="")
        stdout = (
            "Cutoff armed: ARM_REASON=all_ready "
            "KILL_AT_UTC=2026-08-19T18:00:00Z\n"
            "Ready gate poll: ready=4/4, pods=4/4, deployments=4/4\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout)

    monkeypatch.setattr(agent_eval.subprocess, "run", run)

    agent_eval.refresh_cutoff_status(manifest)

    assert manifest["status"] == "completed"
    assert manifest["cutoff_arm_reason"] == "all_ready"
    assert manifest["cutoff_last_ready_counts"] == {
        "ready_pods": 4,
        "expected_ready_pods": 4,
        "pods": 4,
        "expected_pods": 4,
        "deployments": 4,
        "expected_deployments": 4,
    }
    log_targets = [
        command[command.index("logs") + 1]
        for command in commands
        if "logs" in command
    ]
    assert log_targets == ["pod/cutoff-failed", "pod/cutoff-success"]


class EmptyApi:
    def __init__(self):
        self.filters = []

    def runs(self, _project, *, filters):
        self.filters.append(filters)
        return []


def test_report_keeps_provenance_and_queries_only_exact_groups(tmp_path):
    manifest = build_manifest(
        local_config(), "eval-report", web_search=False, dry_run=False
    )
    manifest["status"] = "launched"
    api = EmptyApi()

    report, _ = report_eval(manifest, tmp_path, log_wandb=False, api=api)

    assert api.filters == [
        {"group": "eval-report/nanogpt"},
        {"group": "eval-report/tandemfoil"},
    ]
    assert report["provenance"]["senpai_repo_revision"] == "a" * 40
    assert report["provenance"]["target_revisions"] == {
        target["name"]: target["base_revision"] for target in manifest["targets"]
    }


def test_report_refuses_to_publish_before_cutoff_completion(tmp_path):
    manifest = build_manifest(
        local_config(), "eval-partial", web_search=False, dry_run=False
    )
    manifest["status"] = "launched"

    with pytest.raises(RuntimeError, match="before the cutoff job completes"):
        report_eval(manifest, tmp_path, log_wandb=True, api=EmptyApi())


def test_wandb_aggregate_records_provenance_primary_and_diagnostics(monkeypatch):
    manifest = build_manifest(
        local_config(), "eval-aggregate", web_search=False, dry_run=False
    )
    manifest.update(status="completed", cutoff_arm_reason="all_ready")

    class AggregateRun:
        url = "https://wandb.ai/acme/senpai-evals/runs/report"

        def __init__(self):
            self.summary = {}
            self.logged = None

        def log(self, metrics):
            self.logged = metrics

        def finish(self):
            pass

    aggregate = AggregateRun()
    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return aggregate

    monkeypatch.setattr(agent_eval.wandb, "init", init)
    target_results = [
        {
            "name": "nanogpt",
            "total_runs": 1,
            "eligible_runs": 1,
            "best": {
                "score": 3350,
                "diagnostics": {"final_val_loss": 3.275},
            },
        }
    ]

    url = agent_eval.log_report_to_wandb(manifest, target_results, "# Report\n")

    assert url == aggregate.url
    assert captured["config"]["senpai_repo_revision"] == "a" * 40
    assert captured["config"]["targets"][0]["base_revision"] == (
        manifest["targets"][0]["base_revision"]
    )
    assert aggregate.logged["eval/nanogpt/primary"] == 3350
    assert aggregate.logged["eval/nanogpt/diagnostic/final_val_loss"] == 3.275
