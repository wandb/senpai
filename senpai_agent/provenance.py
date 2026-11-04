"""Check identifiers a student quotes against the records that own them.

Students type W&B run ids, commit SHAs and run states into comments and
results. Typed from memory, a few percent of them name runs that never
existed, commits that do not resolve, or a state the record contradicts. Each
identifier found in student text is checked against its own source: W&B for
run ids, the assignment repository for commits and git objects, the training
records for a training id and the state of a supervised run, and the tool
output of the conversation for a file digest. The words before a token say
which class it belongs to; a hexadecimal token with no label must match one of
those sources or a token a tool printed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from senpai_agent.git_transport import GitWorkflowPreconditionError, run_git
from senpai_agent.github.http import GitHubReader, GitHubReadError
from senpai_agent.models import ExperimentResult, WandbRunRef
from senpai_agent.training import TrainingResult, TrainingState, training_result_paths


class ProvenanceError(ValueError):
    """An identifier in student text that no record accounts for."""


class WandbRunLookup(Protocol):
    def run_state(self, entity: str, project: str, run_id: str) -> str | None: ...


class WandbApiRunLookup:
    """Read one run's state from W&B; None when the project has no such run."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def run_state(self, entity: str, project: str, run_id: str) -> str | None:
        import wandb

        runs = wandb.Api(timeout=self.timeout_seconds).runs(
            f"{entity}/{project}",
            filters={"name": run_id},
            per_page=1,
        )
        run = next(iter(runs), None)
        return None if run is None else str(run.state)


class CommitLookup(Protocol):
    def commit_sha(self, sha: str) -> str | None: ...


class GitHubCommitLookup:
    """Resolve a commit in the repository on GitHub; None when it has none.

    The student's worktree holds only the history it has fetched, so a commit
    on another branch, such as a peer's PR head, resolves only at the remote.
    """

    def __init__(self, repo: str, reader: GitHubReader):
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("repo must use owner/name form")
        self.repo = repo
        self.reader = reader

    def commit_sha(self, sha: str) -> str | None:
        if _SHA.fullmatch(sha) is None:
            raise ValueError("sha must be hexadecimal")
        try:
            payload = self.reader.get(f"/repos/{self.repo}/commits/{sha}")
        except GitHubReadError as error:
            # 404 is an unknown ref; 422 is a SHA no commit has.
            if error.status_code in {404, 422}:
                return None
            raise
        full = payload.get("sha") if isinstance(payload, dict) else None
        if not isinstance(full, str) or not full.lower().startswith(sha.lower()):
            # A ref that happens to be named like a SHA is not that commit.
            return None
        return full


# A W&B run page is <host>/<entity>/<project>/runs/<id>, on wandb.ai or a
# self-hosted instance. GitHub Actions pages also end in /runs/<number>.
_WANDB_RUN_URL = re.compile(
    r"https?://[^\s/]+(?:/[^\s/]+)*?/([^\s/]+)/(?!actions/)([^\s/]+)/runs/"
    r"([A-Za-z0-9_-]+)"
)
_TRAINING_ID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
# Bare candidates: W&B's eight-character ids, sixteen-hex ids such as those
# Senpai gives Kubernetes runs, abbreviated or full commit SHAs, and file
# digests of up to 64 hex characters. The words before a token decide which.
# A token inside a path or a file name ("x.abc1234", "abc1234.py") is not a
# candidate; a sentence-final full stop does not make one a file name.
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_/.-])(?:[0-9a-f]{7,64}|[a-z0-9]{8})(?![A-Za-z0-9_/-])(?!\.[A-Za-z0-9])"
)
# Hex tokens a tool may have printed; the same shape as the hex candidates.
_HEX_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_/.-])[0-9a-f]{7,64}(?![A-Za-z0-9_/-])(?!\.[A-Za-z0-9])"
)
_WANDB_CONTEXT = re.compile(r"(?i)\bw&b\b|\bwandb\b|\bruns?\b|\brun[ _-]?ids?\b")
# "SHA" names a commit, but the "SHA" of "SHA-256" names a digest algorithm.
_COMMIT_CONTEXT = re.compile(
    r"(?i)\bcommits?\b|\bshas?\b(?!-\d)|\bhead\b|\bbase\b|\bgit\b|\bcheckout\b"
    r"|\bmerged?\b|\bpush(?:ed)?\b|\brev-parse\b|@$"
)
# A file digest ("SHA-256 `391730cf…`", "sha256sum", "checksum") is checked
# against the digests tools printed in the conversation. "file" is ordinary
# prose around commits too, so only an algorithm word names a digest.
_DIGEST_CONTEXT = re.compile(
    r"(?i)\bsha-?\d{1,3}\b|\b(?:sha\d*|md5|b2)sum\b|\bmd5\b|\bblake\d\w*\b"
    r"|\bdigests?\b|\bchecksums?\b"
)
# A git object id ("blob 3917…", "git hash-object", an ls-tree row) is an
# object of the worktree, or output of a git command, not a commit.
_BLOB_CONTEXT = re.compile(
    r"(?i)\bblobs?\b|\btrees?\b|\bobjects?\b|\bhash-object\b|\bls-tree\b"
    r"|\bls-files\b"
)
# A judge or job system issues run-shaped ids too ("official run", "scored
# run", "check", "job", "submission"); a job word within reach names one.
_JOB_CONTEXT = re.compile(
    r"(?i)\bjobs?\b|\bchecks?\b|\bscored\b|\bofficial\b|\bsubmissions?\b"
    r"|\bjudge\b"
)
# A Senpai training id ("training b81440b1", "run_training id") is a prefix
# of a run record's UUID, not a W&B run.
_TRAINING_CONTEXT = re.compile(r"(?i)\btraining(?:[ _-]?ids?)?\b|\brun_training\b")
# The nearest of these words labels a token; wandb applies to run-shaped
# tokens only, the hexadecimal classes to hexadecimal tokens only.
_KIND_CONTEXT = {
    "digest": _DIGEST_CONTEXT,
    "blob": _BLOB_CONTEXT,
    "job": _JOB_CONTEXT,
    "training": _TRAINING_CONTEXT,
    "wandb": _WANDB_CONTEXT,
    "commit": _COMMIT_CONTEXT,
}
_HEX_KINDS = {"digest", "blob", "training", "commit"}
_STATE_WORDS = re.compile(
    r"(?i)\b(running|finished|failed|timed[ _-]?out|cancell?ed|crashed|killed)\b"
)
# "12 passed, 0 failed" and "failed=0" count outcomes; they are not the state
# of the run named in the same clause.
_COUNT_BEFORE = re.compile(r"(?<![A-Za-z0-9_])\d+\s*$")
# "failed=0" is a count; "failed: 0" is one only when the number ends the
# phrase, so "failed: 3 of 5 seeds diverged" still claims the state.
_COUNT_AFTER = re.compile(r"\s*(?:=\s*\d|:\s*\d+\s*(?:[,;.)(]|$))")
# A context word names the identifier only within a few words of it, in the
# same clause; other identifiers in a list between them do not count.
_CONTEXT_WORDS = 3
_WORD = re.compile(r"[A-Za-z0-9_'-]+")
_CLAUSE_END = re.compile(r"[.;!?](?=\s|$)")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)*\s*:?-{3,}:?\s*\|?\s*$")
_SHA = re.compile(r"[0-9a-fA-F]{4,40}")
_CONTEXT_CHARS = 48
_STATE_CHARS = 64
_FROM_TOOL_OUTPUT = (
    "Copy identifiers from tool output (run_training, get_training_status, "
    "git) or from a page you opened; do not type them from memory."
)
_NOT_A_RUN_ID = (
    "Cite a run from another W&B project by its full run URL. If {token!r} is "
    "not a W&B run id, reword so it does not directly follow 'run', 'W&B' or "
    "'wandb': a judge's or job system's id is recognised after 'job', 'check', "
    "'official', 'scored' or 'submission', and a Senpai training id after "
    "'training'."
)
_NOT_A_COMMIT = (
    "Cite a commit from another repository by its URL. If {token!r} is not a "
    "commit, reword so it does not directly follow 'commit', 'SHA', 'HEAD' or "
    "a similar word: a file digest is recognised after 'SHA-256', 'sha256sum', "
    "'digest' or 'checksum', and a git object id after 'blob', "
    "'hash-object' or 'ls-tree'."
)
_NOT_A_DIGEST = (
    "Re-run the command that computes it, such as `sha256sum <path>`, and "
    "paste its output exactly; an abbreviated digest must be the prefix of a "
    "digest a tool printed."
)
_NOT_A_BLOB = "Paste it from `git hash-object` or `git ls-tree` output."
_NOT_A_TRAINING_ID = (
    "Copy the training_id from run_training or get_training_status output; a "
    "W&B run id is recognised after 'run', 'W&B' or 'wandb'."
)
_UNLABELLED = (
    "Paste identifiers from tool output and label them (commit, run, digest, "
    "blob, job or training), or cite them by URL."
)
_STATE_HINT = (
    "A state word in the same clause as a run id is read as that run's state; "
    "correct the state, or move the word to its own sentence if it describes "
    "something else."
)


class ObservedIdentifiers:
    """Hexadecimal tokens that tools printed or messages delivered in a conversation.

    An observation often echoes the agent's own input, such as the terminal
    command it answers or the text of an edited file, so the tokens of the
    action an observation answers are not counted for it. A token the agent
    typed becomes observed once a tool prints it on its own. Agent messages
    and condensation summaries are model-written, and a tool error is the
    harness quoting the agent's call, so none of them count.
    """

    def __init__(self, events: Sequence[object]):
        self._events = events
        self._scanned = 0
        self._typed: dict[str, frozenset[str]] = {}
        self._tokens: set[str] = set()

    def refresh(self) -> None:
        total = len(self._events)
        for index in range(self._scanned, total):
            event = self._events[index]
            if hasattr(event, "action") and hasattr(event, "tool_name"):
                self._typed[str(getattr(event, "id", index))] = _hex_tokens(
                    _action_text(event)
                )
                continue
            text = _observed_text(event)
            if text is None:
                continue
            tokens = _hex_tokens(text)
            action_id = getattr(event, "action_id", None)
            if action_id is not None:
                tokens -= self._typed.get(str(action_id), frozenset())
            self._tokens |= tokens
        self._scanned = total

    def contains(self, token: str) -> bool:
        self.refresh()
        return token in self._tokens

    def has_prefix(self, token: str) -> bool:
        """Return whether a token a tool printed starts with this abbreviation."""

        self.refresh()
        return any(seen.startswith(token) for seen in self._tokens)


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _hex_tokens(text: str) -> frozenset[str]:
    return frozenset(_HEX_TOKEN.findall(_ANSI_ESCAPE.sub("", text)))


def _text_of(value: object) -> str:
    """Join the strings inside a model or its dump, one per line.

    Scanning the strings themselves, not their JSON encoding, keeps a token
    that follows a newline or a tab in tool output a token of its own.
    """

    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        return _text_of(value.model_dump())
    if isinstance(value, dict):
        return "\n".join(_text_of(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_text_of(item) for item in value)
    return ""


def _action_text(event: object) -> str:
    parts = [_text_of(getattr(event, "action", None))]
    parts.extend(getattr(part, "text", "") for part in getattr(event, "thought", ()))
    return "\n".join(parts)


def _observed_text(event: object) -> str | None:
    observation = getattr(event, "observation", None)
    if observation is not None:
        return _text_of(observation) if hasattr(observation, "model_dump") else str(observation)
    if hasattr(event, "rejection_reason"):
        return str(event.rejection_reason)
    message = getattr(event, "llm_message", None)
    if message is not None and getattr(event, "source", None) != "agent":
        return " ".join(
            getattr(part, "text", "") for part in getattr(message, "content", ())
        )
    return None


@dataclass(frozen=True)
class IdentifierProvenance:
    """Account for every identifier and run state in student text."""

    workspace: Path
    training_dir: Path | None = None
    wandb_entity: str | None = None
    wandb_project: str | None = None
    lookup: WandbRunLookup = field(default_factory=WandbApiRunLookup)
    # Resolves commits the worktree has not fetched; None asks only the worktree.
    commit_lookup: CommitLookup | None = None
    _observed: dict[str, ObservedIdentifiers] = field(
        default_factory=dict, repr=False, compare=False
    )

    def observed(self, conversation: object) -> ObservedIdentifiers | None:
        """Return the tool-output index of a conversation, or None without one."""

        events = getattr(getattr(conversation, "state", None), "events", None)
        if events is None:
            return None
        key = str(getattr(conversation, "id", id(conversation)))
        index = self._observed.get(key)
        if index is None:
            index = self._observed[key] = ObservedIdentifiers(events)
        return index

    def check_text(
        self,
        text: str,
        *,
        source: str,
        observed: ObservedIdentifiers | None = None,
    ) -> None:
        """Refuse text that quotes an identifier no record accounts for.

        ``observed`` is the conversation's tool output; without it, digests,
        git object ids and unlabelled tokens have nothing to be checked
        against and are left alone.
        """

        records = self._training_records()
        states: dict[tuple[str, str, str], str | None] = {}
        spans: list[tuple[int, int]] = []
        for match in _WANDB_RUN_URL.finditer(text):
            entity, project, run_id = match.groups()
            self._check_run(
                run_id,
                quoted=_quoted_states(text, *match.span(3)),
                source=source,
                records=records,
                states=states,
                project=(entity, project),
            )
            spans.append(match.span())
        for match in _TRAINING_ID.finditer(text):
            record = records.get(match.group())
            if record is not None:
                _require_state(
                    f"training {record.training_id}",
                    record.state.value,
                    _quoted_states(text, *match.span()),
                    source=source,
                )
            spans.append(match.span())
        for match in _TOKEN.finditer(text):
            start, end = match.span()
            if any(start < span_end and span_start < end for span_start, span_end in spans):
                continue
            token = match.group()
            context = _context_before(text, start)
            kind = _identifier_kind(token, context)
            quoted = _quoted_states(text, start, end)
            if kind == "wandb":
                self._check_run(
                    token,
                    quoted=quoted,
                    source=source,
                    records=records,
                    states=states,
                    hint=_NOT_A_RUN_ID.format(token=token),
                )
            elif kind == "commit":
                self.resolve_commit(
                    token,
                    field=f"commit {token!r} in the {source}",
                    hint=_NOT_A_COMMIT.format(token=token),
                )
            elif kind == "digest":
                self._check_digest(token, observed, source=source)
            elif kind == "blob":
                self._check_blob(token, observed, source=source)
            elif kind == "training":
                self._check_training_id(
                    token,
                    context,
                    quoted=quoted,
                    source=source,
                    records=records,
                    states=states,
                )
            elif kind is None:
                self._check_unlabelled(token, observed, source=source, records=records)

    def check_result(
        self,
        result: ExperimentResult,
        *,
        observed: ObservedIdentifiers | None = None,
    ) -> ExperimentResult:
        """Resolve a result's commit SHAs and account for its runs and text."""

        commit_sha = self.resolve_commit(result.commit_sha, field="result.commit_sha")
        expected_head_sha = self.resolve_commit(
            result.assignment.expected_head_sha,
            field="result.assignment.expected_head_sha",
        )
        records = self._training_records()
        states: dict[tuple[str, str, str], str | None] = {}
        for run in result.runs:
            self._check_run_ref(run, records, states)
        self.check_text(result.hypothesis, source="result hypothesis", observed=observed)
        self.check_text(result.summary, source="result summary", observed=observed)
        if (commit_sha, expected_head_sha) == (
            result.commit_sha,
            result.assignment.expected_head_sha,
        ):
            return result
        return result.model_copy(
            update={
                "commit_sha": commit_sha,
                "assignment": result.assignment.model_copy(
                    update={"expected_head_sha": expected_head_sha}
                ),
            }
        )

    def resolve_commit(self, sha: str, *, field: str, hint: str = "") -> str:
        """Return the full SHA of an exact or abbreviated commit, or refuse.

        The worktree answers first. A commit it has not fetched, such as a
        peer's PR head or an older branch tip, is then looked up in the
        repository itself when a commit lookup is configured.
        """

        if _SHA.fullmatch(sha) is None:
            raise ProvenanceError(
                f"{field} must be a commit SHA of 4 to 40 hexadecimal characters, "
                f"not {sha!r}. {_FROM_TOOL_OUTPUT}"
            )
        resolved, local_error = self._resolved_commit(sha, field=field)
        if resolved is not None:
            return resolved
        raise ProvenanceError(
            f"{field} does not resolve to a commit in the assignment "
            f"repository ({local_error}). {_FROM_TOOL_OUTPUT} {hint}".rstrip()
        )

    def _resolved_commit(self, sha: str, *, field: str) -> tuple[str | None, str]:
        """Return the full commit SHA, or None and why the worktree lacks it."""

        try:
            return run_git(self.workspace, "rev-parse", "--verify", f"{sha}^{{commit}}"), ""
        except GitWorkflowPreconditionError as error:
            local_error = str(error)
        if self.commit_lookup is not None:
            try:
                resolved = self.commit_lookup.commit_sha(sha)
            except Exception as error:
                raise ProvenanceError(
                    f"Could not check {field}: it is not in the worktree and the "
                    f"repository lookup failed ({type(error).__name__}: {error}). "
                    "Nothing was published; retry the call once GitHub answers."
                ) from error
            if resolved is not None:
                return resolved, ""
        return None, local_error

    def _check_digest(
        self, token: str, observed: ObservedIdentifiers | None, *, source: str
    ) -> None:
        if observed is None or observed.has_prefix(token):
            return
        raise ProvenanceError(
            f"digest {token!r} in the {source} was not printed by any tool in this "
            f"conversation. {_NOT_A_DIGEST}"
        )

    def _check_blob(
        self, token: str, observed: ObservedIdentifiers | None, *, source: str
    ) -> None:
        if observed is not None and observed.has_prefix(token):
            return
        try:
            run_git(self.workspace, "cat-file", "-e", token)
        except GitWorkflowPreconditionError:
            pass
        else:
            return
        if observed is None:
            # git hash-object prints an id without storing the object; only the
            # conversation's tool output could vouch for it.
            return
        raise ProvenanceError(
            f"blob id {token!r} in the {source} is neither in this conversation's "
            f"tool output nor an object in the assignment worktree. {_NOT_A_BLOB}"
        )

    def _check_training_id(
        self,
        token: str,
        context: str,
        *,
        quoted: set[str],
        source: str,
        records: dict[str, TrainingResult],
        states: dict[tuple[str, str, str], str | None],
    ) -> None:
        matching = {
            record.training_id: record
            for record in records.values()
            if record.training_id.startswith(token)
        }
        if len(matching) == 1:
            record = next(iter(matching.values()))
            _require_state(
                f"training {record.training_id}", record.state.value, quoted, source=source
            )
            return
        if matching:
            # Two records share the prefix; the id is real but names neither.
            return
        if _run_shape(token) and _last_match_end(_WANDB_CONTEXT, context) >= 0:
            # "training run <id>" may name the W&B run of a training.
            self._check_run(
                token,
                quoted=quoted,
                source=source,
                records=records,
                states=states,
                hint=_NOT_A_RUN_ID.format(token=token),
            )
            return
        raise ProvenanceError(
            f"training id {token!r} in the {source} matches no run record of this "
            f"pod. {_NOT_A_TRAINING_ID}"
        )

    def _check_unlabelled(
        self,
        token: str,
        observed: ObservedIdentifiers | None,
        *,
        source: str,
        records: dict[str, TrainingResult],
    ) -> None:
        """Account for a hexadecimal token no context word labels.

        It passes when a tool printed it, a run record or recorded run id
        starts with it, or it resolves as a commit. Without the conversation's
        tool output only a full 40-hex SHA is checked, as a commit.
        """

        if not all(char in "0123456789abcdef" for char in token):
            return
        full_sha = len(token) == 40
        field = f"commit {token!r} in the {source}"
        if observed is None:
            if full_sha:
                self.resolve_commit(token, field=field, hint=_NOT_A_COMMIT.format(token=token))
            return
        if not full_sha and (token.isdigit() or not any(char.isdigit() for char in token)):
            # A number, or a word spelt in hex letters such as "defaced", is not an id.
            return
        if observed.has_prefix(token):
            return
        if any(key.startswith(token) for key in records):
            return
        if len(token) <= 40 and self._resolved_commit(token, field=field)[0] is not None:
            return
        raise ProvenanceError(
            f"{token!r} in the {source} has no label and matches no commit, run "
            f"record or tool output in this conversation. {_UNLABELLED}"
        )

    def _check_run_ref(
        self,
        run: WandbRunRef,
        records: dict[str, TrainingResult],
        states: dict[tuple[str, str, str], str | None],
    ) -> None:
        linked = _WANDB_RUN_URL.search(run.url)
        if linked is not None and linked.group(3) != run.run_id:
            raise ProvenanceError(
                f"result run {run.run_id!r} links to a different run's page "
                f"({linked.group(3)!r}). {_FROM_TOOL_OUTPUT}"
            )
        self._check_run(
            run.run_id,
            quoted={run.state},
            source="result runs",
            records=records,
            states=states,
            project=linked.group(1, 2) if linked is not None else None,
        )

    def _check_run(
        self,
        run_id: str,
        *,
        quoted: set[str],
        source: str,
        records: dict[str, TrainingResult],
        states: dict[tuple[str, str, str], str | None],
        project: tuple[str, str] | None = None,
        hint: str = "",
    ) -> None:
        record = records.get(run_id)
        if record is not None:
            # The harness read this id from the run's own output, so it exists.
            # The process state is the run's state only for the last run of a
            # finished process: an earlier run of the same process, or the run
            # of a process still running, may already have ended in W&B.
            if not quoted:
                return
            # A finished record with a declared non-zero exit code says only
            # that the process exited as designed; W&B still owns its run's state.
            declared_exit = (
                record.state is TrainingState.FINISHED and record.exit_code not in (None, 0)
            )
            if (
                record.state is not TrainingState.RUNNING
                and record.wandb_run_ids[-1:] == (run_id,)
                and not declared_exit
            ):
                _require_state(
                    f"W&B run {run_id} (training {record.training_id})",
                    record.state.value,
                    quoted,
                    source=source,
                )
                return
        if project is None:
            if self.wandb_entity is None or self.wandb_project is None:
                return
            project = (self.wandb_entity, self.wandb_project)
        entity, project_name = project
        key = (entity, project_name, run_id)
        if key not in states:
            try:
                states[key] = self.lookup.run_state(entity, project_name, run_id)
            except Exception as error:
                raise ProvenanceError(
                    f"Could not check W&B run {run_id!r} in the {source} against "
                    f"W&B project {entity}/{project_name}: the W&B lookup failed "
                    f"({type(error).__name__}: {error}). Nothing was published; "
                    "retry the call once W&B answers."
                ) from error
        state = states[key]
        if state is None:
            if record is not None:
                # Recorded from the run's output but not in this project, such
                # as an offline run: its existence is accounted for.
                return
            raise ProvenanceError(
                f"W&B run id {run_id!r} in the {source} is not in W&B project "
                f"{entity}/{project_name}. {_FROM_TOOL_OUTPUT} {hint}".rstrip()
            )
        quoted_classes = {_state_class(value) for value in quoted}
        if quoted_classes and _state_class(state) not in quoted_classes:
            raise ProvenanceError(
                f"W&B run {run_id} is {state} in W&B, but the {source} calls it "
                f"{', '.join(sorted(quoted))}. {_FROM_TOOL_OUTPUT} {_STATE_HINT}"
            )

    def _training_records(self) -> dict[str, TrainingResult]:
        records: dict[str, TrainingResult] = {}
        if self.training_dir is None:
            return records
        for path in training_result_paths(self.training_dir):
            record = TrainingResult.model_validate_json(path.read_text())
            records[record.training_id] = record
            for run_id in record.wandb_run_ids:
                records[run_id] = record
        return records


def _require_state(
    label: str,
    recorded: str,
    quoted: set[str],
    *,
    source: str,
) -> None:
    # Compare finished, running and stopped: "crashed" or "killed" is a fair
    # description of a failed, timed-out or cancelled run.
    if quoted and _state_class(recorded) not in {_state_class(value) for value in quoted}:
        raise ProvenanceError(
            f"{label} is recorded as {recorded}, but the {source} calls it "
            f"{', '.join(sorted(quoted))}. {_FROM_TOOL_OUTPUT} {_STATE_HINT}"
        )


def _run_shape(token: str) -> bool:
    hexadecimal = all(char in "0123456789abcdef" for char in token)
    return (
        any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
        and (len(token) == 8 or (len(token) == 16 and hexadecimal))
    )


def _identifier_kind(token: str, context: str) -> str | None:
    """Return the class the nearest context word gives a token.

    One of "wandb", "commit", "digest", "blob", "job" or "training", or None
    when no word within reach labels the token. A 64-hex token is a digest
    whatever precedes it. A job or training word within reach of a run word
    wins over it: "official run" and "training run" do not name W&B runs.
    """

    hexadecimal = all(char in "0123456789abcdef" for char in token)
    if token.isdigit():
        # A count, size or date ("base batch 1048576"), not an abbreviated SHA.
        return None
    if hexadecimal and len(token) == 64:
        return "digest"
    reach: dict[str, int] = {}
    for kind, pattern in _KIND_CONTEXT.items():
        if kind == "wandb" and not _run_shape(token):
            continue
        if kind in _HEX_KINDS and not hexadecimal:
            continue
        at = _last_match_end(pattern, context)
        if at >= 0:
            reach[kind] = at
    if not reach:
        return None
    nearest = max(reach, key=reach.__getitem__)
    if nearest in {"wandb", "job"} and "job" in reach:
        return "job"
    if nearest in {"wandb", "training"} and "training" in reach:
        return "training"
    return nearest


def _last_match_end(pattern: re.Pattern[str], text: str) -> int:
    """Return where the last context word ends, if it is close to the token."""

    end = -1
    for match in pattern.finditer(text):
        end = match.end()
    if end < 0:
        return end
    words = [word for word in _WORD.findall(text[end:]) if _TOKEN.fullmatch(word) is None]
    return end if len(words) <= _CONTEXT_WORDS else -1


def _line_bounds(text: str, position: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return start, len(text) if end < 0 else end


def _context_before(text: str, start: int) -> str:
    """Return the words before a token in its clause, or its table cell and header."""

    line_start, _ = _line_bounds(text, start)
    before = text[max(line_start, start - _CONTEXT_CHARS) : start]
    header = _table_header_cell(text, start)
    if header is not None:
        return f"{header} | {before.rsplit('|', 1)[-1]}"
    return _CLAUSE_END.split(before)[-1]


def _table_header_cell(text: str, position: int) -> str | None:
    line_start, line_end = _line_bounds(text, position)
    line = text[line_start:line_end]
    if not line.lstrip().startswith("|"):
        return None
    column = line.count("|", 0, position - line_start) - 1
    block: list[str] = []
    for row in reversed(text[:line_start].splitlines()):
        if not row.lstrip().startswith("|"):
            break
        block.insert(0, row)
    if len(block) < 2 or _TABLE_SEPARATOR.match(block[1]) is None:
        return None
    cells = block[0].strip().strip("|").split("|")
    return cells[column] if 0 <= column < len(cells) else None


def _quoted_states(text: str, start: int, end: int) -> set[str]:
    line_start, line_end = _line_bounds(text, start)
    before = _CLAUSE_END.split(text[max(line_start, start - _STATE_CHARS) : start])[-1]
    after = _CLAUSE_END.split(text[end : min(line_end, end + _STATE_CHARS)])[0]
    window = f"{before} {after}"
    return {
        _normalize_state(match.group(1))
        for match in _STATE_WORDS.finditer(window)
        if _COUNT_BEFORE.search(window[: match.start()]) is None
        and _COUNT_AFTER.match(window[match.end() :]) is None
    }


def _normalize_state(state: str) -> str:
    state = re.sub(r"[ -]", "_", state.lower())
    return "cancelled" if state == "canceled" else state


def _state_class(state: str) -> str:
    state = _normalize_state(state)
    return state if state in {"running", "finished"} else "stopped"
