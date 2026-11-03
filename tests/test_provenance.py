import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from git_workflow_support import commit_file, git, repository

from senpai_agent.models import (
    AssignmentKey,
    ExperimentResult,
    ResultStatus,
    WandbRunRef,
)
from senpai_agent.github.http import GitHubReadError
from senpai_agent.provenance import (
    GitHubCommitLookup,
    IdentifierProvenance,
    ObservedIdentifiers,
    ProvenanceError,
)
from senpai_agent.training import TrainingResult, TrainingState

TRAINING_ID = "b81440b1-b803-471e-9fe0-6dcabd756b83"


class FakeLookup:
    def __init__(self, states: dict[str, str] | None = None):
        self.states = states or {}
        self.calls: list[tuple[str, str, str]] = []

    def run_state(self, entity: str, project: str, run_id: str) -> str | None:
        self.calls.append((entity, project, run_id))
        return self.states.get(run_id)


class FakeCommitLookup:
    def __init__(self, commits: dict[str, str] | None = None, error: Exception | None = None):
        self.commits = commits or {}
        self.error = error
        self.calls: list[str] = []

    def commit_sha(self, sha: str) -> str | None:
        self.calls.append(sha)
        if self.error is not None:
            raise self.error
        return next(
            (full for full in self.commits.values() if full.startswith(sha)),
            None,
        )


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


def test_numbers_after_commit_words_are_not_shas(case):
    provenance, lookup, _ = case

    provenance.check_text(
        "Base batch 1048576 tokens; merged 20261028 after the push of 3145728 rows.",
        source="comment",
    )

    assert lookup.calls == []


@pytest.mark.parametrize(
    "text",
    [
        "Each run took about twelve minutes with bf16gemm tiles.",
        "That run finished. bf16gemm is next.",
        "The commit message explains why; deadbeef0 stays a constant.",
    ],
)
def test_a_context_word_outside_the_tokens_clause_or_reach_is_ignored(case, text):
    provenance, lookup, _ = case

    provenance.check_text(text, source="comment")

    assert lookup.calls == []


def test_every_id_in_a_list_after_a_run_word_is_checked(case):
    provenance, lookup, _ = case

    with pytest.raises(ProvenanceError, match="'58vh6sck'.*full run URL.*reword"):
        provenance.check_text("runs ab12cd34, cd34ef56 and 58vh6sck", source="comment")

    assert [call[2] for call in lookup.calls] == ["ab12cd34", "cd34ef56", "58vh6sck"]


def test_unresolved_commit_names_the_url_escape(case):
    provenance, _, _ = case

    with pytest.raises(ProvenanceError, match="'0747eaa'.*by its URL"):
        provenance.check_text("commit 0747eaa adds the twin", source="comment")


def test_recorded_states_compare_by_class(case):
    provenance, lookup, _ = case
    write_record(
        provenance.training_dir,
        state=TrainingState.TIMED_OUT,
        run_ids=("0123456789abcdef",),
    )

    provenance.check_text(f"training {TRAINING_ID} was killed at the limit", source="comment")
    provenance.check_text("W&B run 0123456789abcdef crashed at step 3", source="comment")

    assert lookup.calls == []


def test_a_state_in_another_clause_is_not_the_runs_state(case):
    provenance, _, _ = case

    provenance.check_text(
        "W&B run ab12cd34 is the new baseline. An earlier attempt failed.",
        source="comment",
    )

    with pytest.raises(ProvenanceError, match="ab12cd34 is finished in W&B.*own sentence"):
        provenance.check_text("W&B run ab12cd34 failed at step 3.", source="comment")


@pytest.mark.parametrize(
    ("state", "run_ids"),
    [
        (TrainingState.RUNNING, ("ab12cd34",)),
        (TrainingState.FAILED, ("ab12cd34", "cd34ef56")),
    ],
)
def test_state_of_a_run_the_record_cannot_vouch_for_comes_from_wandb(
    case, state, run_ids
):
    provenance, lookup, _ = case
    write_record(provenance.training_dir, state=state, run_ids=run_ids)

    provenance.check_text("run ab12cd34 finished", source="comment")
    provenance.check_text("run ab12cd34 is in the sweep", source="comment")

    assert lookup.calls == [("acme", "cfd", "ab12cd34")]


def test_recorded_run_missing_from_wandb_is_still_accounted_for(case):
    provenance, lookup, _ = case
    write_record(provenance.training_dir, state=TrainingState.RUNNING, run_ids=("zz11yy22",))

    provenance.check_text("offline run zz11yy22 is running", source="comment")

    assert lookup.calls == [("acme", "cfd", "zz11yy22")]


def test_failed_wandb_lookup_is_reported_as_unchecked(case):
    provenance, _, _ = case

    class Unreachable:
        def run_state(self, entity: str, project: str, run_id: str) -> str | None:
            raise ConnectionError("api.wandb.ai timed out")

    unreachable = IdentifierProvenance(
        workspace=provenance.workspace,
        wandb_entity="acme",
        wandb_project="cfd",
        lookup=Unreachable(),
    )

    with pytest.raises(
        ProvenanceError,
        match="Could not check W&B run 'ab12cd34'.*ConnectionError.*Nothing was published",
    ):
        unreachable.check_text("run ab12cd34 finished", source="comment")


PEER_HEAD = "e162d688f945e668a759bbe6c02b73fa24cfe505"
DIGEST = "391730cf" + "0a1b2c3d" * 7


@pytest.mark.parametrize(
    "text",
    [
        "The file SHA-256 `391730cf…` matches the judge's copy.",
        f"file SHA-256 {DIGEST} (full)",
        f"sha256 digest {DIGEST[:12]} of baseline.py",
        "checksum 391730cf unchanged after the rebase",
        "the digest 391730cf1a2b differs from HEAD's file",
    ],
)
def test_file_digests_and_their_prefixes_are_neither_commits_nor_runs(case, text):
    provenance, lookup, _ = case

    provenance.check_text(text, source="comment")

    assert lookup.calls == []


def test_a_commit_word_nearer_than_the_digest_word_still_names_a_commit(case):
    provenance, _, head = case

    provenance.check_text(f"the file at commit {head[:7]} is fine", source="comment")
    with pytest.raises(ProvenanceError, match="'0747eaa'.*does not resolve"):
        provenance.check_text("the file at commit 0747eaa is fine", source="comment")


@pytest.mark.parametrize(
    "text",
    [
        "official run `8b7803d1a6e04f78` scored 118.2 µs",
        "the scored run `4e0029a22c034d2a` finished at 09:41",
        "check `4e0029a22c034d2a` passed every shape",
        "check run 4e0029a22c034d2a: correctness ok",
        "job 8b7803d1a6e04f78 is the official submission",
        "official run 58vh6sck, submission 3",
    ],
)
def test_judge_job_ids_near_a_job_word_are_not_looked_up_in_wandb(case, text):
    provenance, lookup, _ = case

    provenance.check_text(text, source="comment")

    assert lookup.calls == []


def test_a_run_word_alone_still_sends_a_sixteen_hex_id_to_wandb(case):
    provenance, lookup, _ = case

    with pytest.raises(ProvenanceError, match="'8b7803d1a6e04f78'.*acme/cfd"):
        provenance.check_text("W&B run `8b7803d1a6e04f78` finished", source="comment")

    assert lookup.calls == [("acme", "cfd", "8b7803d1a6e04f78")]


def test_commits_the_worktree_has_not_fetched_resolve_at_the_repository(case):
    provenance, _, head = case
    commits = FakeCommitLookup({"peer": PEER_HEAD, "old": "739f3ed" + "c" * 33})
    checked = IdentifierProvenance(
        workspace=provenance.workspace,
        wandb_entity="acme",
        wandb_project="cfd",
        lookup=provenance.lookup,
        commit_lookup=commits,
    )

    checked.check_text(f"peer PR head {PEER_HEAD} beats commit 739f3ed", source="comment")
    assert checked.resolve_commit("739f3ed", field="x") == "739f3ed" + "c" * 33
    # The worktree answers first; the repository is asked only for what it lacks.
    assert checked.resolve_commit(head[:7], field="x") == head
    assert commits.calls == [PEER_HEAD, "739f3ed", "739f3ed"]

    with pytest.raises(ProvenanceError, match="'0747eaa'.*does not resolve"):
        checked.check_text("commit 0747eaa adds the twin", source="comment")


def test_a_failed_repository_lookup_refuses_as_unchecked(case):
    provenance, _, _ = case
    checked = IdentifierProvenance(
        workspace=provenance.workspace,
        commit_lookup=FakeCommitLookup(error=TimeoutError("api.github.com")),
    )

    with pytest.raises(ProvenanceError, match="Could not check.*TimeoutError.*retry"):
        checked.check_text(f"commit {PEER_HEAD} from the peer", source="comment")


class FakeReader:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.paths: list[str] = []

    def get(self, path: str) -> object:
        self.paths.append(path)
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response


def test_github_commit_lookup_reads_the_commit_endpoint():
    reader = FakeReader(
        {
            f"/repos/acme/cfd/commits/{PEER_HEAD[:7]}": {"sha": PEER_HEAD},
            "/repos/acme/cfd/commits/0747eaa": GitHubReadError("HTTP 422", status_code=422),
            "/repos/acme/cfd/commits/deadbeef": GitHubReadError("HTTP 404", status_code=404),
            "/repos/acme/cfd/commits/abc1234": {"sha": "f" * 40},
            "/repos/acme/cfd/commits/cafe": GitHubReadError("HTTP 500", status_code=500),
        }
    )
    lookup = GitHubCommitLookup("acme/cfd", reader)

    assert lookup.commit_sha(PEER_HEAD[:7]) == PEER_HEAD
    assert lookup.commit_sha("0747eaa") is None
    assert lookup.commit_sha("deadbeef") is None
    # A branch named like a SHA is not that commit.
    assert lookup.commit_sha("abc1234") is None
    with pytest.raises(GitHubReadError, match="500"):
        lookup.commit_sha("cafe")
    with pytest.raises(ValueError):
        lookup.commit_sha("main")
    with pytest.raises(ValueError):
        GitHubCommitLookup("acme", reader)


# --- Classes the 2026-11-03 campaign review found refused or let through ---

SHA256 = "391730cf" + "0a1b2c3d" * 7
CORRUPTED = SHA256[:-1] + "e"
BLOB = "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"


class Event(SimpleNamespace):
    """Duck-typed stand-in for one OpenHands event."""


def observation(text: str, *, action_id: str = "a0", tool_name: str = "terminal") -> Event:
    return Event(
        id=f"o-{action_id}",
        action_id=action_id,
        tool_name=tool_name,
        observation=SimpleNamespace(model_dump_json=lambda: json.dumps({"content": text})),
    )


def action(text: str, *, event_id: str = "a0") -> Event:
    return Event(
        id=event_id,
        tool_name="terminal",
        action=SimpleNamespace(model_dump_json=lambda: json.dumps({"command": text})),
        thought=[SimpleNamespace(text="")],
    )


def message(text: str, *, source: str = "user") -> Event:
    return Event(
        source=source,
        llm_message=SimpleNamespace(content=[SimpleNamespace(text=text)]),
    )


def conversation(*events: Event) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), state=SimpleNamespace(events=list(events)))


def observed(*events: Event) -> ObservedIdentifiers:
    return ObservedIdentifiers(list(events))


@pytest.mark.parametrize(
    "text",
    [
        f"SHA-256 {SHA256} of submission.py",
        f"file SHA-256 `{SHA256[:8]}…` matches the judge's copy",
        f"sha256sum: {SHA256[:12]}",
        f"`{SHA256}` is the shipped digest",
        f"the digest {SHA256[:16]} is unchanged",
    ],
)
def test_digests_a_tool_printed_pass_as_digests_and_as_prefixes(case, text):
    provenance, lookup, _ = case
    index = observed(action("sha256sum submission.py"), observation(f"{SHA256}  submission.py"))

    provenance.check_text(text, source="comment", observed=index)

    assert lookup.calls == []


@pytest.mark.parametrize(
    "text",
    [
        f"SHA-256 {CORRUPTED} of submission.py",
        f"`{CORRUPTED}` is the shipped digest",
        "digest 391730cf1a2b differs from HEAD's file",
    ],
)
def test_digests_no_tool_printed_are_refused_by_class(case, text):
    provenance, _, _ = case
    index = observed(observation(f"{SHA256}  submission.py"))

    with pytest.raises(ProvenanceError, match="^digest '.*not printed by any tool.*sha256sum"):
        provenance.check_text(text, source="comment", observed=index)


def test_digests_are_left_alone_without_the_conversations_tool_output(case):
    provenance, lookup, _ = case

    provenance.check_text(f"SHA-256 {CORRUPTED} of submission.py", source="comment")
    provenance.check_text(f"`{CORRUPTED}` is the shipped digest", source="comment")

    assert lookup.calls == []


def test_an_echo_of_the_agents_own_command_does_not_vouch_for_a_digest(case):
    provenance, _, _ = case
    index = observed(
        action(f"echo 'digest {CORRUPTED}' >> RESULTS.md", event_id="a1"),
        observation(f"$ echo 'digest {CORRUPTED}' >> RESULTS.md", action_id="a1"),
        action("sha256sum submission.py", event_id="a2"),
        observation(f"{SHA256}  submission.py", action_id="a2"),
    )

    with pytest.raises(ProvenanceError, match="^digest"):
        provenance.check_text(f"digest {CORRUPTED}", source="comment", observed=index)
    provenance.check_text(f"digest {SHA256}", source="comment", observed=index)


def test_delivered_messages_count_and_agent_messages_do_not(case):
    provenance, _, _ = case
    from_advisor = observed(message(f"The judge's copy has digest {SHA256}."))
    from_self = observed(message(f"I believe the digest is {SHA256}.", source="agent"))

    provenance.check_text(f"digest {SHA256[:10]}", source="comment", observed=from_advisor)
    with pytest.raises(ProvenanceError, match="^digest"):
        provenance.check_text(f"digest {SHA256[:10]}", source="comment", observed=from_self)


def test_the_index_scans_new_events_incrementally(case):
    provenance, _, _ = case
    events = [observation("nothing yet")]
    index = ObservedIdentifiers(events)

    with pytest.raises(ProvenanceError, match="^digest"):
        provenance.check_text(f"digest {SHA256}", source="comment", observed=index)
    events.append(observation(f"{SHA256}  submission.py", action_id="a9"))
    provenance.check_text(f"digest {SHA256}", source="comment", observed=index)


def test_git_object_ids_pass_from_the_worktree_or_tool_output(case):
    provenance, lookup, head = case
    blob = git(provenance.workspace, "rev-parse", f"{head}:model.py")
    printed = observed(observation(f"{BLOB}  tmp/scratch.py", tool_name="terminal"))

    provenance.check_text(f"blob {blob} is model.py at HEAD", source="comment")
    provenance.check_text(f"git hash-object gave {BLOB[:12]}", source="comment", observed=printed)
    provenance.check_text(f"ls-tree row: 100644 blob {blob[:10]}", source="comment", observed=printed)
    # Without tool output a hash-object id cannot be checked and is left alone.
    provenance.check_text(f"git hash-object gave {BLOB}", source="comment")

    with pytest.raises(ProvenanceError, match="^blob id '.*neither in this conversation.*hash-object"):
        provenance.check_text(f"blob {BLOB}", source="comment", observed=observed())
    assert lookup.calls == []


@pytest.mark.parametrize(
    "text",
    [
        "okc job `a1b2c3d4` scored 2.447",
        "job id a1b2c3d4 passed",
        "the official run 8b7803d1a6e04f78 is the submission",
    ],
)
def test_job_system_ids_are_not_looked_up_in_wandb(case, text):
    provenance, lookup, _ = case

    provenance.check_text(text, source="comment", observed=observed())

    assert lookup.calls == []


def test_training_id_prefixes_are_checked_against_the_run_records(case):
    provenance, lookup, _ = case
    write_record(provenance.training_dir, state=TrainingState.FINISHED, run_ids=("ab12cd34",))
    prefix = TRAINING_ID[:8]

    provenance.check_text(f"training {prefix} finished in 40 s", source="comment")
    provenance.check_text(f"training run {prefix} finished", source="comment")
    provenance.check_text(f"run_training id {prefix}", source="comment")
    # "training run <W&B id>" still names the W&B run of a training; the
    # record vouches for this one, so W&B is not asked.
    provenance.check_text("the training run ab12cd34 finished", source="comment")
    with pytest.raises(ProvenanceError, match="^W&B run id '58vh6sck'"):
        provenance.check_text("the training run 58vh6sck finished", source="comment")
    assert lookup.calls == [("acme", "cfd", "58vh6sck")]

    with pytest.raises(ProvenanceError, match=f"^training {TRAINING_ID} is recorded as finished"):
        provenance.check_text(f"training {prefix} failed", source="comment")
    with pytest.raises(ProvenanceError, match="^training id '0123abcd'.*matches no run record.*training_id"):
        provenance.check_text("training 0123abcd finished", source="comment")


@pytest.mark.parametrize(
    "text",
    [
        "run ab12cd34: 12 passed, 0 failed",
        "run ab12cd34 reports failed=0 and finished=12",
        "run ab12cd34 (3 failed: 0)",
    ],
)
def test_counted_outcomes_are_not_the_runs_state(case, text):
    provenance, _, _ = case

    provenance.check_text(text, source="comment")

    with pytest.raises(ProvenanceError, match="ab12cd34 is finished in W&B.*failed"):
        provenance.check_text("run ab12cd34 failed after 3 steps", source="comment")


def test_unlabelled_hex_tokens_must_match_a_source_when_tool_output_is_known(case):
    provenance, lookup, head = case
    write_record(provenance.training_dir, state=TrainingState.FINISHED, run_ids=("0123456789abcdef",))
    index = observed(observation(f"commit {head}\nAuthor: Student\n\n    baseline"))

    # A tool printed it, a record starts with it, or it resolves as a commit.
    provenance.check_text(f"{head[:7]} is the candidate", source="comment", observed=index)
    provenance.check_text(f"see {head}", source="comment", observed=index)
    provenance.check_text(f"{TRAINING_ID[:8]} and 0123456789ab", source="comment", observed=index)
    # Words spelt in hex letters and numbers are not identifiers.
    provenance.check_text("defaced and deadbeef bytes at 20261103", source="comment", observed=index)

    # A commit prefix with an invented eighth character matches nothing.
    invented = head[:7] + ("f" if head[7] != "f" else "e")
    with pytest.raises(ProvenanceError, match=f"^'{invented}'.*has no label and matches no commit.*label them"):
        provenance.check_text(f"{invented} is the candidate", source="comment", observed=index)
    assert lookup.calls == []


def test_without_tool_output_only_a_full_unlabelled_sha_is_checked(case):
    provenance, lookup, head = case

    provenance.check_text(f"{head[:7] + 'f'} is the candidate", source="comment")
    with pytest.raises(ProvenanceError, match=f"^commit '{'0' * 40}'.*does not resolve"):
        provenance.check_text(f"see {'0' * 40}", source="comment")
    assert lookup.calls == []


def test_a_sentence_final_full_stop_does_not_hide_a_token(case):
    provenance, _, head = case

    provenance.check_text(f"The candidate is commit {head[:7]}.", source="comment")
    provenance.check_text("Rebuilt from model.abc1234.py and run.0747eaa0 files.", source="comment")
    with pytest.raises(ProvenanceError, match="^commit '0747eaa'"):
        provenance.check_text("The twin is commit 0747eaa.", source="comment")


def test_the_observed_index_is_built_once_per_conversation(case):
    provenance, _, _ = case
    talk = conversation(observation(f"{SHA256}  submission.py"))

    first = provenance.observed(talk)
    assert first is provenance.observed(talk)
    assert first is not None and first.contains(SHA256)
    assert provenance.observed(SimpleNamespace()) is None

    provenance.check_text(f"digest {SHA256}", source="comment", observed=first)


def test_every_refusal_names_its_class_and_fix(case):
    provenance, _, head = case
    index = observed()
    cases = {
        "W&B run `58vh6sck` finished": ("W&B run id '58vh6sck'", "full run URL"),
        "commit 0747eaa adds the twin": ("commit '0747eaa'", "by its URL"),
        f"digest {CORRUPTED}": ("digest '", "sha256sum"),
        f"blob {BLOB}": ("blob id '", "hash-object"),
        "training 0123abcd finished": ("training id '0123abcd'", "training_id"),
        f"{head[:7] + 'f'} is the candidate": ("has no label", "label them"),
    }
    for text, (klass, fix) in cases.items():
        with pytest.raises(ProvenanceError) as error:
            provenance.check_text(text, source="comment", observed=index)
        assert klass in str(error.value), text
        assert fix in str(error.value), text
