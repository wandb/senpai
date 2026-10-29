from pathlib import Path

import pytest
from git_workflow_support import commit_file, repository

from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    ResultStatus,
    WandbRunRef,
)
from senpai_agent.provenance import IdentifierProvenance, ProvenanceError
from senpai_agent.training import TrainingResult, TrainingState

TRAINING_ID = "b81440b1-b803-471e-9fe0-6dcabd756b83"


class FakeLookup:
    def __init__(self, states: dict[str, str] | None = None):
        self.states = states or {}
        self.calls: list[tuple[str, str, str]] = []

    def run_state(self, entity: str, project: str, run_id: str) -> str | None:
        self.calls.append((entity, project, run_id))
        return self.states.get(run_id)


def write_record(training_dir: Path, *, state: TrainingState, run_ids=()) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / f"{TRAINING_ID}.json").write_text(
        TrainingResult(
            training_id=TRAINING_ID,
            state=state,
            exit_code=None if state is TrainingState.RUNNING else 1,
            elapsed_seconds=12,
            log_path=str(training_dir / f"{TRAINING_ID}.log"),
            wandb_run_ids=tuple(run_ids),
        ).model_dump_json()
    )


@pytest.fixture
def case(tmp_path: Path):
    workspace, _remote, head = repository(tmp_path)
    lookup = FakeLookup({"ab12cd34": "finished", "cd34ef56": "crashed"})
    provenance = IdentifierProvenance(
        workspace=workspace,
        training_dir=tmp_path / "state" / "training",
        wandb_entity="acme",
        wandb_project="cfd",
        lookup=lookup,
    )
    return provenance, lookup, head


def test_unknown_wandb_run_id_in_run_context_is_refused(case):
    provenance, lookup, _ = case

    with pytest.raises(ProvenanceError, match="'58vh6sck'.*acme/cfd"):
        provenance.check_text("W&B run `58vh6sck` finished in 12 s.", source="comment")

    assert lookup.calls == [("acme", "cfd", "58vh6sck")]


@pytest.mark.parametrize(
    "text",
    [
        "Run `ab12cd34` finished in 12 s.",
        "wandb run ab12cd34: 118.2 µs",
        "run_id=ab12cd34 (finished)",
        "The W&B run ab12cd34 failed, then cd34ef56 finished the sweep.",
    ],
)
def test_known_wandb_run_ids_pass_when_their_state_is_not_contradicted(case, text):
    provenance, lookup, _ = case

    provenance.check_text(text, source="comment")

    assert ("acme", "cfd", "ab12cd34") in lookup.calls


def test_quoted_wandb_state_must_match_the_api_state(case):
    provenance, _, _ = case

    with pytest.raises(ProvenanceError, match="ab12cd34 is finished in W&B.*crashed"):
        provenance.check_text("run `ab12cd34` crashed after 3 steps", source="comment")


def test_run_id_from_a_training_record_needs_no_wandb_lookup(case):
    provenance, lookup, _ = case
    write_record(
        provenance.training_dir,
        state=TrainingState.FAILED,
        run_ids=("0123456789abcdef",),
    )

    provenance.check_text("run id `0123456789abcdef` failed at step 3", source="comment")
    assert lookup.calls == []

    with pytest.raises(ProvenanceError, match="recorded as failed.*finished"):
        provenance.check_text("W&B run 0123456789abcdef finished", source="comment")


def test_training_id_state_must_match_the_record(case):
    provenance, _, _ = case
    write_record(provenance.training_dir, state=TrainingState.TIMED_OUT)

    provenance.check_text(f"training {TRAINING_ID} timed out", source="comment")
    provenance.check_text(
        "conversation 00000000-0000-0000-0000-000000000001 finished", source="comment"
    )

    with pytest.raises(ProvenanceError, match=f"training {TRAINING_ID} is recorded as timed_out"):
        provenance.check_text(f"training {TRAINING_ID} finished cleanly", source="comment")


def test_full_shas_and_contextual_short_shas_must_resolve(case):
    provenance, lookup, head = case

    provenance.check_text(f"commit {head[:7]} is the candidate", source="comment")
    provenance.check_text(f"pushed {head} to the branch", source="comment")
    # Hexadecimal words without commit or run context are left alone.
    provenance.check_text("stride x4 kept 0747eaa0 and deadbeef bytes", source="comment")

    with pytest.raises(ProvenanceError, match="'0747eaa'.*does not resolve"):
        provenance.check_text("commit 0747eaa adds the twin", source="comment")
    with pytest.raises(ProvenanceError, match=f"'{'0' * 40}'.*does not resolve"):
        provenance.check_text(f"see {'0' * 40}", source="comment")
    assert lookup.calls == []


def test_sixteen_hex_tokens_follow_the_nearest_context_word(case):
    provenance, lookup, head = case
    sixteen = "0123456789abcdef"

    with pytest.raises(ProvenanceError, match=f"'{sixteen}'.*acme/cfd"):
        provenance.check_text(f"commit {head[:7]}, run id {sixteen}", source="comment")
    with pytest.raises(ProvenanceError, match=f"'{sixteen}'.*does not resolve"):
        provenance.check_text(f"run ab12cd34 at commit {sixteen}", source="comment")


def test_wandb_urls_are_checked_in_their_own_project(case):
    provenance, lookup, _ = case

    provenance.check_text(
        "https://forge.example.com/wandb/other/project/runs/ab12cd34 "
        "and https://github.com/acme/cfd/actions/runs/123456",
        source="comment",
    )
    assert lookup.calls == [("other", "project", "ab12cd34")]

    with pytest.raises(ProvenanceError, match="'zz99yy88'.*other/project"):
        provenance.check_text(
            "[run](https://wandb.ai/other/project/runs/zz99yy88)", source="comment"
        )


def test_table_column_header_gives_a_run_id_its_context(case):
    provenance, lookup, _ = case
    table = (
        "| W&B run | d32 µs |\n"
        "|---|---|\n"
        "| `ab12cd34` | 118.2 |\n"
        "| `58vh6sck` | 117.9 |\n"
    )

    with pytest.raises(ProvenanceError, match="'58vh6sck'"):
        provenance.check_text(table, source="comment")

    assert [call[2] for call in lookup.calls] == ["ab12cd34", "58vh6sck"]


def test_words_and_numbers_without_identifier_context_are_left_alone(case):
    provenance, lookup, _ = case

    provenance.check_text(
        "baseline finished; config lr3e4bs8 at 20261028; the kernel ran 12345678 "
        "cases and the run finished with x5 −0.53% (best 2abc3def1).",
        source="comment",
    )

    assert lookup.calls == []


def test_without_a_configured_project_bare_run_ids_are_not_looked_up(tmp_path):
    workspace, _remote, _head = repository(tmp_path)
    lookup = FakeLookup()
    provenance = IdentifierProvenance(workspace=workspace, lookup=lookup)

    provenance.check_text("run `58vh6sck` finished", source="comment")

    assert lookup.calls == []


def test_check_result_resolves_short_shas_and_accounts_for_its_runs(case):
    provenance, lookup, head = case
    result = ExperimentResult(
        assignment=AssignmentKey(
            repo="acme/widgets",
            pr_number=17,
            assignment_id="assignment-17",
            revision_id="revision-1",
            expected_head_sha=head[:8],
            student="student-one",
        ),
        status=ResultStatus.SUCCEEDED,
        hypothesis="A wider tile is faster.",
        summary=f"Run ab12cd34 finished 1.2% faster than the base at commit {head[:7]}.",
        runs=(
            WandbRunRef(
                run_id="ab12cd34",
                url="https://wandb.ai/acme/cfd/runs/ab12cd34",
                state="finished",
            ),
        ),
        primary_metric=None,
        commit_sha=head[:8],
    )

    checked = provenance.check_result(result)

    assert checked.commit_sha == head
    assert checked.assignment.expected_head_sha == head
    assert checked.model_dump(exclude={"commit_sha", "assignment"}) == result.model_dump(
        exclude={"commit_sha", "assignment"}
    )

    mismatched = result.model_copy(
        update={
            "runs": (
                WandbRunRef(
                    run_id="cd34ef56",
                    url="https://wandb.ai/acme/cfd/runs/cd34ef56",
                    state="finished",
                ),
            )
        }
    )
    with pytest.raises(ProvenanceError, match="cd34ef56 is crashed in W&B.*finished"):
        provenance.check_result(mismatched)

    foreign_url = result.model_copy(
        update={
            "runs": (
                WandbRunRef(
                    run_id="ab12cd34",
                    url="https://wandb.ai/acme/cfd/runs/cd34ef56",
                    state="finished",
                ),
            )
        }
    )
    with pytest.raises(ProvenanceError, match="links to a different run"):
        provenance.check_result(foreign_url)


def test_resolve_commit_accepts_only_hexadecimal_shas(case):
    provenance, _, head = case
    other = commit_file(provenance.workspace, "model.py", "baseline = 3\n", "other")

    assert provenance.resolve_commit(head[:6], field="local_commit_sha") == head
    assert provenance.resolve_commit(other, field="local_commit_sha") == other

    with pytest.raises(ProvenanceError, match="local_commit_sha must be a commit SHA"):
        provenance.resolve_commit("HEAD", field="local_commit_sha")
