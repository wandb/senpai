# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

import json
import subprocess
import threading
from pathlib import Path

import pytest
import yaml

import eval.run as agent_eval
from eval.run import (
    DEFAULT_N_TRIALS,
    DEFAULT_TRAINING_TIMEOUT_MINUTES,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
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
        run_id="run-123",
        group="",
    ):
        self.id = run_id
        self.name = "candidate"
        self.url = f"https://wandb.ai/acme/evals/runs/{run_id}"
        self.state = state
        self.group = group
        self.config = config or {}
        self.summary = summary or {}
        self.validation = list(validation)
        self.markers = list(markers)

    def scan_history(self, *, keys):
        if "val/loss" in keys:
            return iter(self.validation)
        return iter(self.markers)


def test_eval_script_runs_directly_from_the_documented_entrypoint():
    result = subprocess.run(
        [agent_eval.sys.executable, str(agent_eval.EVALUATOR_PATH), "--help"],
        cwd=agent_eval.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "{launch,report}" in result.stdout


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
    trial = target["trials"][0]

    config = target_launch_config(base, manifest, target, trial)

    assert config["target_repo_url"].endswith("modded-nanogpt-senpai")
    assert config["target_repo_branch"] == target["base_branch"]
    assert len(target["base_revision"]) == 40
    assert config["advisor"] is True
    assert config["n_students"] == 1
    assert config["timeout_minutes"] == DEFAULT_TRAINING_TIMEOUT_MINUTES
    assert config["web_search"] is False
    assert config["human_issues"] is False
    assert config["gh_history_scope"] == "fresh"
    assert config["wandb_run_group"] == trial["wandb_group"]
    assert config["wandb_entity"] == DEFAULT_WANDB_ENTITY
    assert config["wandb_project"] == DEFAULT_WANDB_PROJECT
    assert config["trial_index"] == 0
    assert config["trial_seed"] == trial["trial_seed"]
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
    assert config["extra_instructions"] == ""


def test_eval_defaults_to_three_isolated_trials_and_standard_wandb_project():
    base = local_config()

    manifest = build_manifest(
        base,
        "eval-defaults",
        web_search=False,
        dry_run=False,
    )
    trials = [
        (target, trial)
        for target in manifest["targets"]
        for trial in target["trials"]
    ]

    assert manifest["n_trials"] == DEFAULT_N_TRIALS == 3
    assert manifest["wandb_entity"] == DEFAULT_WANDB_ENTITY
    assert manifest["wandb_project"] == DEFAULT_WANDB_PROJECT
    assert len(trials) == 6
    for key in (
        "research_tag",
        "wandb_group",
        "advisor_branch",
        "student_name",
        "trial_seed",
    ):
        assert len({trial[key] for _target, trial in trials}) == 6
    assert all(len(trial["advisor_branch"]) <= 50 for _target, trial in trials)
    assert all(
        trial["adjudication"]
        == {"status": "pending", "selected_run_id": None, "evidence": {}}
        for _target, trial in trials
    )


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


def tandem_run(score, run_id):
    summary = {
        f"test/{split}/mae_surf_p": score for split in TANDEM_SPLITS
    }
    summary["test_avg/mae_surf_p"] = score
    return FakeRun(summary=summary, run_id=run_id)


def eval_contract_run(run, manifest, target, trial):
    run.group = trial["wandb_group"]
    run.config.update(
        {
            "wandb_entity": manifest["wandb_entity"],
            "wandb_project": manifest["wandb_project"],
            "wandb_group": trial["wandb_group"],
            "wandb_run_group": trial["wandb_group"],
            "senpai_trial_index": trial["trial_index"],
            "senpai_trial_seed": trial["trial_seed"],
            "senpai_timeout_minutes": manifest["training_timeout_minutes"],
            "git_commit": "c" * 40,
            "git_dirty": False,
        }
    )
    if target["name"] == "nanogpt":
        run.config.update(
            {
                "benchmark": agent_eval.NANOGPT_BENCHMARK,
                "run_kind": "full-training",
                "num_trials": 1,
                "val_tokens": agent_eval.NANOGPT_VAL_TOKENS,
                "target_val_loss": agent_eval.NANOGPT_TARGET_LOSS,
                "stat_sig_delta": agent_eval.NANOGPT_SIGNIFICANCE_DELTA,
                "data_contract": agent_eval.NANOGPT_DATA_CONTRACT,
                "metric_contract": agent_eval.NANOGPT_METRIC_CONTRACT,
                "source_sha256": "d" * 64,
                "seed": trial["trial_seed"],
                "model_config": {"layers": 12},
                "optimizer_groups": [{"optimizer": "AdamW"}],
                "train_shards": [
                    {
                        "name": f"fineweb_train_{index:06d}.bin",
                        "bytes": agent_eval.NANOGPT_SHARD_BYTES,
                    }
                    for index in range(1, 21)
                ],
                "val_shards": [
                    {
                        "name": "fineweb_val_000000.bin",
                        "bytes": agent_eval.NANOGPT_SHARD_BYTES,
                    }
                ],
            }
        )
        run.summary.update(
            {
                "eval/completed": True,
                "eval/data_contract_satisfied": True,
                "eval/all_trials_reached_target": True,
                "eval/ranking_eligible": True,
                "eval/train_shard_count": 20,
                "eval/val_shard_count": 1,
                "eval/primary_metric_name": (
                    "speedrun/final_first_step_to_target"
                ),
                "eval/primary_metric_direction": "minimize",
                "speedrun/statistically_valid": True,
            }
        )
    else:
        run.config.update(
            {
                "debug": False,
                "skip_test": False,
                "splits_dir": "/mnt/new-pvc/datasets/tandemfoil/splits_v2",
                "seed": trial["trial_seed"],
                "metric_contract": agent_eval.TANDEM_METRIC_CONTRACT,
                "train_samples": 1499,
                "val_samples": {
                    split: 100 for split in agent_eval.TANDEM_VAL_SPLITS
                },
                "training_source_sha256": "e" * 64,
                "materialized_split_manifest_sha256": (
                    agent_eval.TANDEM_PROTECTED_HASHES[
                        "split_manifest_sha256"
                    ]
                ),
                "data_contract_satisfied": True,
                **agent_eval.TANDEM_PROTECTED_HASHES,
                "model_config": {"layers": 5},
                "optimizer_config": {"name": "AdamW"},
                "scheduler_config": {"name": "CosineAnnealingLR"},
            }
        )
        run.summary.update(
            {
                "eval/completed": True,
                "eval/ranking_eligible": True,
                "eval/data_contract_satisfied": True,
                "eval/full_test_splits": 4,
                "eval/primary_metric_name": "test_avg/mae_surf_p",
                "eval/primary_metric_direction": "minimize",
            }
        )
    return run


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
    target = manifest["targets"][0]
    config = target_launch_config(
        local_config(), manifest, target, target["trials"][0]
    )
    assert config["timeout_minutes"] == 12.5
    assert config["extra_instructions"] == ""

    args = agent_eval.parse_args(
        [
            "launch",
            "--training-timeout-minutes",
            "12.5",
            "--total-timeout-hours",
            "1.25",
            "--n-trials",
            "5",
        ]
    )
    assert args.training_timeout_minutes == 12.5
    assert args.total_timeout_hours == 1.25
    assert args.n_trials == 5
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
    with pytest.raises(ValueError, match="n_trials must be a positive integer"):
        build_manifest(
            local_config(),
            "eval-zero-trials",
            web_search=False,
            dry_run=False,
            n_trials=0,
        )


def test_eval_run_id_keeps_every_github_routing_label_within_50_characters():
    run_id = "a" * 27
    manifest = build_manifest(
        local_config(), run_id, web_search=False, dry_run=False
    )

    assert all(
        len(trial["advisor_branch"]) <= 50
        for target in manifest["targets"]
        for trial in target["trials"]
    )
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
    launch_barrier = threading.Barrier(4)
    settled = []

    def run(argv, *, env=None):
        config = yaml.safe_load(Path(argv[-1]).read_text(encoding="utf-8"))
        kind = "preflight" if config["preflight_only"] else "launch"
        events.append((kind, config["tag"]))
        if kind == "launch":
            launch_barrier.wait(timeout=2)
            settled.append(config["tag"])
        if kind == "launch" and config["tag"].endswith("foil-t02"):
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
        lambda _manifest: events.append(("cleanup", str(len(settled)))) or True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        launch_eval(
            config_path,
            tmp_path,
            run_id="eval-rollback",
            web_search=False,
            cutoff_image=None,
            dry_run=False,
            n_trials=2,
        )

    assert events[:6] == [
        ("preflight", "eval-rollback-nano-t01"),
        ("preflight", "eval-rollback-nano-t02"),
        ("preflight", "eval-rollback-foil-t01"),
        ("preflight", "eval-rollback-foil-t02"),
        ("arm", "cutoff"),
        ("ready", "cutoff"),
    ]
    assert {
        tag for kind, tag in events[6:-1] if kind == "launch"
    } == {
        "eval-rollback-nano-t01",
        "eval-rollback-nano-t02",
        "eval-rollback-foil-t01",
        "eval-rollback-foil-t02",
    }
    assert events[-1] == ("cleanup", "4")
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
    tags = ",".join(
        trial["research_tag"]
        for target in manifest["targets"]
        for trial in target["trials"]
    )
    assert f"app=senpai,research-tag in ({tags})" in commands[0]
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


def test_one_cutoff_accounts_for_every_target_trial(monkeypatch):
    manifest = build_manifest(
        local_config(),
        "eval-cutoff-size",
        web_search=False,
        dry_run=True,
        n_trials=4,
    )
    commands = []
    monkeypatch.setattr(
        agent_eval,
        "run_checked",
        lambda argv, *, env=None: commands.append(argv),
    )

    agent_eval.arm_cutoff(local_config(), manifest)

    command = commands[0]
    assert command[command.index("--expected-pods") + 1] == "16"
    assert command[command.index("--expected-deployments") + 1] == "16"
    assert command[command.index("--deadline-epoch") + 1] == str(
        manifest["deadline_epoch"]
    )
    assert command.count("--deadline-epoch") == 1


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


class RunsApi(EmptyApi):
    def __init__(self, runs_by_group):
        super().__init__()
        self.runs_by_group = runs_by_group

    def runs(self, _project, *, filters):
        self.filters.append(filters)
        return self.runs_by_group.get(filters["group"], [])


def test_report_keeps_provenance_and_queries_only_exact_groups(tmp_path):
    manifest = build_manifest(
        local_config(), "eval-report", web_search=False, dry_run=False
    )
    manifest["status"] = "launched"
    api = EmptyApi()

    report, _ = report_eval(manifest, tmp_path, log_wandb=False, api=api)

    assert api.filters == [
        {"group": f"eval-report/{target}/{trial}"}
        for target in ("nanogpt", "tandemfoil")
        for trial in ("trial-01", "trial-02", "trial-03")
    ]
    assert report["provenance"]["senpai_repo_revision"] == "a" * 40
    assert report["provenance"]["target_revisions"] == {
        target["name"]: target["base_revision"] for target in manifest["targets"]
    }
    assert len(report["provenance"]["trials"]) == 6


def test_trial_candidates_require_finished_full_contract_runs():
    manifest = build_manifest(
        local_config(), "eval-contract", web_search=False, dry_run=False
    )
    target = manifest["targets"][0]
    trial = target["trials"][0]
    valid = eval_contract_run(valid_nanogpt_run(), manifest, target, trial)
    wrong_group = eval_contract_run(
        valid_nanogpt_run(), manifest, target, trial
    )
    wrong_group.id = "wrong-group"
    wrong_group.group = "manual-test-group"
    incomplete = eval_contract_run(
        valid_nanogpt_run(), manifest, target, trial
    )
    incomplete.id = "incomplete"
    incomplete.state = "crashed"

    result = agent_eval.score_trial_runs(
        manifest, target, trial, [valid, wrong_group, incomplete]
    )

    assert [run["run_id"] for run in result["candidate_runs"]] == ["run-123"]
    assert {run["reason"] for run in result["rejected_runs"]} == {
        "W&B run object has the wrong group",
        "W&B run did not finish",
    }


def test_nanogpt_candidate_requires_exact_full_shard_manifest():
    manifest = build_manifest(
        local_config(), "eval-nanogpt-data", web_search=False, dry_run=False
    )
    target = manifest["targets"][0]
    trial = target["trials"][0]
    run = eval_contract_run(
        valid_nanogpt_run(), manifest, target, trial
    )
    run.config["train_shards"][0]["bytes"] -= 2

    result = agent_eval.score_trial_runs(manifest, target, trial, [run])

    assert result["candidate_runs"] == []
    assert result["rejected_runs"][0]["reason"] == (
        "config train_shards does not match the full data contract"
    )


def test_tandem_candidate_requires_protected_data_and_scorer_hashes():
    manifest = build_manifest(
        local_config(), "eval-data-contract", web_search=False, dry_run=False
    )
    target = manifest["targets"][1]
    trial = target["trials"][0]
    run = eval_contract_run(
        tandem_run(12.0, "tampered-data"), manifest, target, trial
    )
    run.config["split_manifest_sha256"] = "0" * 64

    result = agent_eval.score_trial_runs(manifest, target, trial, [run])

    assert result["candidate_runs"] == []
    assert result["rejected_runs"][0]["reason"] == (
        "config split_manifest_sha256 does not match the TandemFoil contract"
    )


def test_tandem_candidate_requires_materialized_data_manifest_binding():
    manifest = build_manifest(
        local_config(), "eval-materialized-data", web_search=False, dry_run=False
    )
    target = manifest["targets"][1]
    trial = target["trials"][0]
    run = eval_contract_run(
        tandem_run(12.0, "stale-data"), manifest, target, trial
    )
    run.config["materialized_split_manifest_sha256"] = "0" * 64

    result = agent_eval.score_trial_runs(manifest, target, trial, [run])

    assert result["candidate_runs"] == []
    assert result["rejected_runs"][0]["reason"] == (
        "config materialized_split_manifest_sha256 does not match the "
        "TandemFoil contract"
    )


def test_report_aggregates_only_explicitly_adjudicated_trial_results(tmp_path):
    manifest = build_manifest(
        local_config(), "eval-adjudicated", web_search=False, dry_run=False
    )
    tandem = manifest["targets"][1]
    first, second, pending = tandem["trials"]
    first["adjudication"] = {
        "status": "accepted",
        "selected_run_id": "selected-14",
        "evidence": {"kind": "senpai-result", "pr": 14},
    }
    second["adjudication"] = {
        "status": "accepted",
        "selected_run_id": "selected-12",
        "evidence": {"kind": "senpai-result", "pr": 12},
    }
    api = RunsApi(
        {
            first["wandb_group"]: [
                eval_contract_run(
                    tandem_run(10.0, "raw-minimum"), manifest, tandem, first
                ),
                eval_contract_run(
                    tandem_run(14.0, "selected-14"), manifest, tandem, first
                ),
            ],
            second["wandb_group"]: [
                eval_contract_run(
                    tandem_run(12.0, "selected-12"), manifest, tandem, second
                )
            ],
            pending["wandb_group"]: [
                eval_contract_run(
                    tandem_run(1.0, "pending-minimum"), manifest, tandem, pending
                )
            ],
        }
    )

    report, markdown = report_eval(
        manifest, tmp_path, log_wandb=False, api=api
    )

    result = next(
        target for target in report["targets"] if target["name"] == "tandemfoil"
    )
    assert [row["run_id"] for row in result["final_results"]] == [
        "selected-14",
        "selected-12",
    ]
    assert result["trials"][0]["raw_candidate"]["run_id"] == "raw-minimum"
    assert result["trials"][2]["selected"] is None
    assert result["accepted_trials"] == 2
    assert result["adjudicated_trials"] == 2
    assert result["distribution"] == {
        "count": 2,
        "scores": [14.0, 12.0],
        "mean": 13.0,
        "median": 13.0,
        "minimum": 12.0,
        "maximum": 14.0,
        "population_variance": 1.0,
        "population_stddev": 1.0,
        "coefficient_of_variation": 1.0 / 13.0,
    }
    assert "Accepted / adjudicated / trials" in markdown
    assert "| TandemFoilSet Balanced | 2 / 2 / 3 |" in markdown
    assert "Raw metric minima remain candidates only" in markdown


def test_report_rejects_an_adjudication_that_selects_an_ineligible_run(tmp_path):
    manifest = build_manifest(
        local_config(), "eval-bad-selection", web_search=False, dry_run=False
    )
    manifest["targets"][0]["trials"][0]["adjudication"] = {
        "status": "accepted",
        "selected_run_id": "not-eligible",
        "evidence": {},
    }

    with pytest.raises(ValueError, match="must select an eligible run"):
        report_eval(manifest, tmp_path, log_wandb=False, api=EmptyApi())


@pytest.mark.parametrize("field", ["evaluator_sha256", "adjudicator_sha256"])
def test_completed_report_rejects_source_drift_before_external_reads(
    monkeypatch, tmp_path, field
):
    manifest = build_manifest(
        local_config(), "eval-source-drift", web_search=False, dry_run=False
    )
    manifest["status"] = "completed"
    manifest[field] = "0" * 64
    monkeypatch.setattr(
        agent_eval.wandb,
        "Api",
        lambda: pytest.fail("W&B client created before source verification"),
    )
    monkeypatch.setattr(
        agent_eval,
        "eval_github_reader",
        lambda _manifest: pytest.fail(
            "GitHub client created before source verification"
        ),
    )

    with pytest.raises(RuntimeError, match=field):
        report_eval(manifest, tmp_path, log_wandb=False)

    assert not (tmp_path / "eval-source-drift.report.json").exists()
    assert not (tmp_path / "eval-source-drift.report.md").exists()


def test_completed_report_uses_and_persists_github_adjudication(monkeypatch, tmp_path):
    manifest = build_manifest(
        local_config(), "eval-github-ledger", web_search=False, dry_run=False
    )
    manifest["status"] = "completed"
    target = manifest["targets"][0]
    winning_trial = target["trials"][0]
    winner = eval_contract_run(
        valid_nanogpt_run(), manifest, target, winning_trial
    )
    calls = []
    freeze_calls = []

    def freeze(target, trial, github):
        freeze_calls.append((target["name"], trial["trial_index"], github))
        ordinal = (
            (0 if target["name"] == "nanogpt" else 3)
            + trial["trial_index"]
            + 1
        )
        return f"{ordinal:040x}"

    def adjudicate(target, trial, github, candidates, *, frozen_head_sha):
        calls.append((target["name"], trial["trial_index"], frozen_head_sha, github))
        saved = json.loads((tmp_path / "eval-github-ledger.json").read_text())
        saved_trial = next(
            candidate
            for saved_target in saved["targets"]
            if saved_target["name"] == target["name"]
            for candidate in saved_target["trials"]
            if candidate["trial_index"] == trial["trial_index"]
        )
        assert saved_trial["adjudication_frozen_head_sha"] == frozen_head_sha
        selected = candidates[0]["run_id"] if candidates else None
        return {
            "status": "accepted" if selected else "rejected",
            "reason": "test ledger decision",
            "selected_run_id": selected,
            "score": candidates[0]["score"] if candidates else None,
            "evidence": {"frozen_advisor_head": frozen_head_sha},
        }

    monkeypatch.setattr(agent_eval, "freeze_advisor_head", freeze)
    monkeypatch.setattr(agent_eval, "adjudicate_trial", adjudicate)
    api = RunsApi({winning_trial["wandb_group"]: [winner]})

    report, _ = report_eval(
        manifest,
        tmp_path,
        log_wandb=False,
        api=api,
        github=object(),
    )

    nano = next(result for result in report["targets"] if result["name"] == "nanogpt")
    assert nano["final_results"][0]["run_id"] == "run-123"
    assert len(calls) == 6
    assert len(freeze_calls) == 6
    assert all(frozen is not None for _target, _trial, frozen, _github in calls)
    saved = json.loads((tmp_path / "eval-github-ledger.json").read_text())
    assert all(
        trial["adjudication"]["status"] in {"accepted", "rejected"}
        and len(trial["adjudication_frozen_head_sha"]) == 40
        for target in saved["targets"]
        for trial in target["trials"]
    )

    calls.clear()
    report_eval(
        manifest,
        tmp_path,
        log_wandb=False,
        api=api,
        github=object(),
    )

    assert len(freeze_calls) == 6
    assert len(calls) == 6
    assert all(frozen is not None for _target, _trial, frozen, _github in calls)


def test_persisted_adjudication_rejects_semantic_replay_drift():
    manifest = build_manifest(
        local_config(), "eval-decision-drift", web_search=False, dry_run=False
    )
    trial = manifest["targets"][0]["trials"][0]
    head = "f" * 40
    trial["adjudication_frozen_head_sha"] = head
    trial_result = {
        "candidate_runs": [
            {"run_id": "first", "score": 10},
            {"run_id": "second", "score": 9},
        ]
    }
    first = {
        "status": "accepted",
        "reason": "first decision",
        "selected_run_id": "first",
        "score": 10,
        "pr_number": 1,
        "result_digest": "a" * 64,
        "evidence": {"frozen_advisor_head": head},
    }
    agent_eval.record_adjudication(trial, trial_result, first)
    persisted = json.loads(json.dumps(trial))
    changed = {
        **first,
        "selected_run_id": "second",
        "score": 9,
        "pr_number": 2,
        "result_digest": "b" * 64,
    }

    with pytest.raises(RuntimeError, match="changed after it was persisted"):
        agent_eval.record_adjudication(trial, trial_result, changed)

    assert trial == persisted


def test_report_refuses_to_publish_before_cutoff_completion(tmp_path):
    manifest = build_manifest(
        local_config(), "eval-partial", web_search=False, dry_run=False
    )
    manifest["status"] = "launched"

    with pytest.raises(RuntimeError, match="before the cutoff job completes"):
        report_eval(manifest, tmp_path, log_wandb=True, api=EmptyApi())


def test_wandb_aggregate_logs_provenance_distribution_table_and_scatter(monkeypatch):
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
    tables = []
    scatters = []

    def init(**kwargs):
        captured.update(kwargs)
        return aggregate

    def table(**kwargs):
        tables.append(kwargs)
        return kwargs

    def scatter(table_value, x, y, *, title):
        value = {"table": table_value, "x": x, "y": y, "title": title}
        scatters.append(value)
        return value

    monkeypatch.setattr(agent_eval.wandb, "init", init)
    monkeypatch.setattr(agent_eval.wandb, "Table", table)
    monkeypatch.setattr(agent_eval.wandb.plot, "scatter", scatter)
    target = manifest["targets"][0]
    target["trials"][0]["adjudication"] = {
        "status": "accepted",
        "selected_run_id": "run-123",
        "evidence": {"kind": "senpai-result", "pr": 7},
    }
    target["trials"][1]["adjudication"] = {
        "status": "rejected",
        "selected_run_id": None,
        "evidence": {"kind": "senpai-result", "reason": "no winner"},
    }
    trial_results = [
        agent_eval.score_trial_runs(
            manifest,
            target,
            trial,
            [eval_contract_run(valid_nanogpt_run(), manifest, target, trial)]
            if trial["trial_index"] == 0
            else [],
        )
        for trial in target["trials"]
    ]
    target_results = [agent_eval.aggregate_target_trials(target, trial_results)]

    url = agent_eval.log_report_to_wandb(manifest, target_results, "# Report\n")

    assert url == aggregate.url
    assert captured["entity"] == DEFAULT_WANDB_ENTITY
    assert captured["project"] == DEFAULT_WANDB_PROJECT
    assert captured["config"]["senpai_repo_revision"] == "a" * 40
    assert captured["config"]["n_trials"] == 3
    assert captured["config"]["targets"][0]["base_revision"] == (
        manifest["targets"][0]["base_revision"]
    )
    assert aggregate.logged["eval/nanogpt/trials_accepted"] == 1
    assert aggregate.logged["eval/nanogpt/trials_accepted_fraction"] == pytest.approx(
        1 / 3
    )
    assert aggregate.logged["eval/nanogpt/trials_adjudicated"] == 2
    assert aggregate.logged[
        "eval/nanogpt/trials_adjudicated_fraction"
    ] == pytest.approx(2 / 3)
    assert aggregate.logged["eval/nanogpt/distribution/mean"] == 3350
    assert aggregate.logged["eval/nanogpt/distribution/population_variance"] == 0
    assert aggregate.logged["eval/nanogpt/trial/0/final_primary"] == 3350
    assert "eval/nanogpt/trial_results" in aggregate.logged
    assert "eval/nanogpt/score_scatter" in aggregate.logged
    assert aggregate.logged["eval/targets_with_accepted_results"] == 1
    assert aggregate.logged["eval/targets_fully_adjudicated"] == 0
    assert aggregate.logged["eval/trials_accepted"] == 1
    assert aggregate.logged["eval/trials_adjudicated"] == 2
    assert "eval/adjudicated_targets" not in aggregate.logged
    assert scatters[0]["title"] == "Modded NanoGPT accepted final scores"
    assert len(tables) == 2
    assert len(scatters) == 1
