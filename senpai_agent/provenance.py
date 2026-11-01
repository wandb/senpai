"""Check identifiers a student quotes against the records that own them.

Students type W&B run ids, commit SHAs and run states into comments and
results. Typed from memory, a few percent of them name runs that never
existed, commits that do not resolve, or a state the record contradicts. Each
identifier found in student text is checked against its own source: W&B for
run ids, the assignment repository for commits, and the training records for
the state of a supervised run. File digests and the ids of a judge or job
system look like those identifiers, so the words before a token also say what
it is not.
"""

from __future__ import annotations

import re
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
# Senpai gives Kubernetes runs, and abbreviated or full commit SHAs. The words
# before a token decide which one it is.
_TOKEN = re.compile(r"(?<![A-Za-z0-9_/.-])(?:[0-9a-f]{7,40}|[a-z0-9]{8})(?![A-Za-z0-9_/.-])")
_WANDB_CONTEXT = re.compile(r"(?i)\bw&b\b|\bwandb\b|\bruns?\b|\brun[ _-]?ids?\b")
# "SHA" names a commit, but the "SHA" of "SHA-256" names a digest algorithm.
_COMMIT_CONTEXT = re.compile(
    r"(?i)\bcommits?\b|\bshas?\b(?!-\d)|\bhead\b|\bbase\b|\bgit\b|\bcheckout\b"
    r"|\bmerged?\b|\bpush(?:ed)?\b|\brev-parse\b|@$"
)
# A file digest ("file SHA-256 `391730cf…`", "sha256 digest", "checksum") is
# neither a commit nor a run; no record checks it, and its prefixes are hex.
_DIGEST_CONTEXT = re.compile(
    r"(?i)\bsha-?\d{1,3}\b|\bmd5\b|\bblake\d\w*\b|\bdigests?\b|\bchecksums?\b"
    r"|\bfiles?\b"
)
# A judge or job system issues run-shaped ids too ("official run", "scored
# run", "check", "job", "submission"); a job word within reach names one.
_JOB_CONTEXT = re.compile(
    r"(?i)\bjobs?\b|\bchecks?\b|\bscored\b|\bofficial\b|\bsubmissions?\b"
    r"|\bjudge\b"
)
_STATE_WORDS = re.compile(
    r"(?i)\b(running|finished|failed|timed[ _-]?out|cancell?ed|crashed|killed)\b"
)
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
    "'wandb'; a judge's or job system's id is recognised after 'job', 'check', "
    "'official', 'scored' or 'submission'."
)
_NOT_A_COMMIT = (
    "Cite a commit from another repository by its URL. If {token!r} is not a "
    "commit, reword so it does not directly follow 'commit', 'SHA', 'HEAD' or "
    "a similar word; a file digest is recognised after 'SHA-256', 'sha256', "
    "'digest', 'checksum' or 'file'."
)
_STATE_HINT = (
    "A state word in the same clause as a run id is read as that run's state; "
    "correct the state, or move the word to its own sentence if it describes "
    "something else."
)


@dataclass(frozen=True)
class IdentifierProvenance:
    """Account for every run id, commit and run state in student text."""

    workspace: Path
    training_dir: Path | None = None
    wandb_entity: str | None = None
    wandb_project: str | None = None
    lookup: WandbRunLookup = field(default_factory=WandbApiRunLookup)
    # Resolves commits the worktree has not fetched; None asks only the worktree.
    commit_lookup: CommitLookup | None = None

    def check_text(self, text: str, *, source: str) -> None:
        """Refuse text that quotes an identifier no record accounts for."""

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
            kind = _identifier_kind(token, _context_before(text, start))
            if kind == "wandb":
                self._check_run(
                    token,
                    quoted=_quoted_states(text, start, end),
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

    def check_result(self, result: ExperimentResult) -> ExperimentResult:
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
        self.check_text(result.hypothesis, source="result hypothesis")
        self.check_text(result.summary, source="result summary")
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
        try:
            return run_git(self.workspace, "rev-parse", "--verify", f"{sha}^{{commit}}")
        except GitWorkflowPreconditionError as error:
            local_error = error
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
                return resolved
        raise ProvenanceError(
            f"{field} does not resolve to a commit in the assignment "
            f"repository ({local_error}). {_FROM_TOOL_OUTPUT} {hint}".rstrip()
        ) from local_error

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
            if (
                record.state is not TrainingState.RUNNING
                and record.wandb_run_ids[-1:] == (run_id,)
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


def _identifier_kind(token: str, context: str) -> str | None:
    hexadecimal = all(char in "0123456789abcdef" for char in token)
    digest_at = _last_match_end(_DIGEST_CONTEXT, context)
    if hexadecimal and len(token) == 40:
        # A full SHA needs no context word, unless the nearest one names a digest.
        return None if digest_at > _last_match_end(_COMMIT_CONTEXT, context) else "commit"
    if token.isdigit():
        # A count, size or date ("base batch 1048576"), not an abbreviated SHA.
        return None
    run_shape = (
        any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
        and (len(token) == 8 or (len(token) == 16 and hexadecimal))
    )
    wandb_at = _last_match_end(_WANDB_CONTEXT, context) if run_shape else -1
    commit_at = _last_match_end(_COMMIT_CONTEXT, context) if hexadecimal else -1
    if digest_at > max(wandb_at, commit_at):
        # The nearest context word names a file digest, which no record checks.
        return None
    if wandb_at > commit_at:
        # "official run", "scored run", "check run": the run belongs to a judge
        # or job system, not to W&B, when a job word is also within reach.
        return None if _last_match_end(_JOB_CONTEXT, context) >= 0 else "wandb"
    if commit_at >= 0:
        return "commit"
    return None


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
    return {
        _normalize_state(match.group(1))
        for match in _STATE_WORDS.finditer(f"{before} {after}")
    }


def _normalize_state(state: str) -> str:
    state = re.sub(r"[ -]", "_", state.lower())
    return "cancelled" if state == "canceled" else state


def _state_class(state: str) -> str:
    state = _normalize_state(state)
    return state if state in {"running", "finished"} else "stopped"
