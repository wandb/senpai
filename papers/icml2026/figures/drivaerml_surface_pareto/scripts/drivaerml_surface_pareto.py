#!/usr/bin/env python3
"""Build DrivAerML surface-pressure test Pareto frontier chart.

The GitHub side is used only to order experiments, attach W&B runs to PRs, and
locate human interventions. W&B summaries are the source of truth for metrics.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator


REPO = "morganmcg1/DrivAerML"
ENTITY = "wandb-applied-ai-team"
LABELS = ("yi", "tay", "drivaerml-long-20260504")
PLOT_LABELS = ("yi", "tay")
LABEL_PROJECTS = {
    "yi": ("senpai-v1-drivaerml",),
    "tay": ("senpai-v1-drivaerml-ddp8",),
    "drivaerml-long-20260504": ("senpai-v1-drivaerml-ddp8",),
}
METRIC_KEY = "test_primary/surface_pressure_rel_l2_pct"
ABUPT_KEY = "test_primary/abupt_axis_mean_rel_l2_pct"

OUT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = OUT_DIR / "data"
POINTS_CSV = DATA_DIR / "drivaerml_surface_pressure_pareto_points.csv"
INTERVENTIONS_CSV = DATA_DIR / "drivaerml_surface_pressure_pareto_interventions.csv"
SUMMARY_JSON = DATA_DIR / "drivaerml_surface_pressure_pareto_summary.json"
WANDB_CACHE = DATA_DIR / "drivaerml_surface_pressure_pareto_wandb_cache.json"
PR_CACHE = DATA_DIR / "drivaerml_surface_pressure_pareto_prs.json"
COMMENT_CACHE = DATA_DIR / "drivaerml_surface_pressure_pareto_comments.json"
PNG_OUT = OUT_DIR / "drivaerml_surface_pressure_pareto.png"
PDF_OUT = OUT_DIR / "drivaerml_surface_pressure_pareto.pdf"

TMP_PR_CACHE = Path("/tmp/drivaer_prs.json")
TMP_COMMENT_CACHE = Path("/tmp/drivaer_issue_comments.json")

RUN_URL_RE = re.compile(r"https?://wandb\.ai/[^\s)\]>'\"]+")
BACKTICK_ID_RE = re.compile(r"`([a-z0-9]{8})`")
JSON_RUN_IDS_RE = re.compile(r'"wandb_run_ids"\s*:\s*\[([^\]]*)\]')
RUN_ID_IN_JSON_RE = re.compile(r'"([a-z0-9]{8})"')
RUN_CONTEXT_RE = re.compile(
    r"wandb|w&b|run|runs|rank0|test|val|metric|summary|result|senpai-result",
    re.IGNORECASE,
)
HUMAN_PREFIX_RE = re.compile(r"^(HUMAN RESEARCHER|OPERATOR|OPERATOR UPDATE)\b", re.IGNORECASE)
AGENT_PREFIX_RE = re.compile(r"^(ADVISOR|STUDENT|SENPAI|SENPAI-RESULT|Closing:)\b", re.IGNORECASE)
NUDGE_RE = re.compile(
    r"\b(advisor nudge|live triage|human|operator|please|ensure|progress|"
    r"moving|kill|restart|stale|hung|why wait|push for|consider|inspired)\b",
    re.IGNORECASE,
)

COLORS = {
    "yi": os.environ.get("DRIVAERML_PARETO_YI_COLOR", "#0072B2"),
    "tay": os.environ.get("DRIVAERML_PARETO_TAY_COLOR", "#CC79A7"),
    "drivaerml-long-20260504": os.environ.get("DRIVAERML_PARETO_LONG_COLOR", "#6B7280"),
}
INK_COLOR = "#1F2933"
MUTED_COLOR = "#4B5563"
GRID_COLOR = "#D9DEE6"
MINOR_GRID_COLOR = "#ECEFF4"
DISPLAY = {
    "yi": "yi",
    "tay": "tay",
    "drivaerml-long-20260504": "drivaerml-long-20260504",
}


HUMAN_ISSUES_FALLBACK = [
    {
        "number": 18,
        "label": "yi",
        "createdAt": "2026-04-28T20:24:10Z",
        "title": "Ensure you're really pushing hard on new ideas",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/18",
    },
    {
        "number": 19,
        "label": "yi",
        "createdAt": "2026-04-28T21:20:00Z",
        "title": "Don't log gradients too frequently",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/19",
    },
    {
        "number": 48,
        "label": "tay",
        "createdAt": "2026-04-29T20:50:04Z",
        "title": "Hows it going? we making progress?",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/48",
    },
    {
        "number": 53,
        "label": "tay",
        "createdAt": "2026-04-29T23:08:53Z",
        "title": "Infra: stale assignment state",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/53",
    },
    {
        "number": 252,
        "label": "tay",
        "createdAt": "2026-05-01T19:41:55Z",
        "title": "Get inspired by Modded-NanoGPT",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/252",
    },
    {
        "number": 285,
        "label": "tay",
        "createdAt": "2026-05-01T22:14:35Z",
        "title": "Consider Group Representational Position Encoding",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/285",
    },
    {
        "number": 606,
        "label": "tay",
        "createdAt": "2026-05-04T12:24:37Z",
        "title": "Empty assignment PR merged before student ran",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/606",
    },
    {
        "number": 618,
        "label": "tay",
        "createdAt": "2026-05-04T15:39:25Z",
        "title": "Explore true STRING/RoPE beyond input STRING-sep",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/618",
    },
    {
        "number": 644,
        "label": "tay",
        "createdAt": "2026-05-04T23:15:00Z",
        "title": "Askeladd pod hung",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/644",
    },
    {
        "number": 717,
        "label": "tay",
        "createdAt": "2026-05-05T15:57:56Z",
        "title": "Push for Volume improvements",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/717",
    },
    {
        "number": 759,
        "label": "tay",
        "createdAt": "2026-05-06T06:38:52Z",
        "title": "Optional Bengio PR ideas for surface-error work",
        "url": "https://github.com/morganmcg1/DrivAerML/issues/759",
    },
]


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    project: str
    name: str
    group: str | None
    tags: tuple[str, ...]
    created_at: str | None
    url: str
    surface_pressure_rel_l2_pct: float | None
    abupt_axis_mean_rel_l2_pct: float | None
    state: str | None


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def slug_time(value: datetime | None) -> str:
    return value.isoformat().replace("+00:00", "Z") if value else ""


def run_json_command(args: list[str]) -> Any:
    result = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(result.stdout)


def load_json_cache(path: Path, tmp_path: Path | None, command: list[str]) -> Any:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    if tmp_path and tmp_path.exists():
        return json.loads(tmp_path.read_text())
    data = run_json_command(command)
    path.write_text(json.dumps(data, indent=2))
    return data


def load_comment_cache() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if COMMENT_CACHE.exists():
        return json.loads(COMMENT_CACHE.read_text())
    if TMP_COMMENT_CACHE.exists():
        return json.loads(TMP_COMMENT_CACHE.read_text())
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/comments", "--paginate", "--slurp"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    pages = json.loads(result.stdout)
    comments = [comment for page in pages for comment in page]
    return comments


def load_github_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prs = load_json_cache(
        PR_CACHE,
        TMP_PR_CACHE,
        [
            "gh",
            "pr",
            "list",
            "--repo",
            REPO,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,title,headRefName,baseRefName,createdAt,closedAt,mergedAt,labels,url,author,body",
        ],
    )
    comments = load_comment_cache()
    return prs, comments


def labels_for_pr(pr: dict[str, Any]) -> set[str]:
    return {label["name"] for label in pr.get("labels", [])}


def relevant_labels(pr: dict[str, Any]) -> set[str]:
    names = labels_for_pr(pr)
    head = (pr.get("headRefName") or "").lower()
    out = {label for label in LABELS if label in names}
    if "yi" in head:
        out.add("yi")
    if "tay" in head:
        out.add("tay")
    return out


def comments_by_issue_number(comments: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for comment in comments:
        issue_url = comment.get("issue_url") or ""
        if not issue_url:
            continue
        number = int(issue_url.rsplit("/", 1)[-1])
        grouped.setdefault(number, []).append(comment)
    return grouped


def extract_run_ids(text: str) -> set[str]:
    run_ids: set[str] = set()
    for url in RUN_URL_RE.findall(text):
        match = re.search(r"/runs/([A-Za-z0-9_-]+)", url)
        if match:
            run_id = match.group(1).strip(".,;:")
            if run_id and run_id != "runs":
                run_ids.add(run_id)
    for match in JSON_RUN_IDS_RE.finditer(text):
        for run_match in RUN_ID_IN_JSON_RE.finditer(match.group(1)):
            run_ids.add(run_match.group(1))
    for match in BACKTICK_ID_RE.finditer(text):
        run_id = match.group(1)
        if run_id.isdigit():
            continue
        window = text[max(0, match.start() - 90) : min(len(text), match.end() + 140)]
        if RUN_CONTEXT_RE.search(window):
            run_ids.add(run_id)
    return run_ids


def pr_student_tokens(pr: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for label in labels_for_pr(pr):
        if not label.startswith("student:"):
            continue
        value = label.split(":", 1)[1].lower()
        tokens.add(value)
        if value.startswith("dl24-"):
            tokens.add(value.removeprefix("dl24-"))
    head_prefix = (pr.get("headRefName") or "").split("/", 1)[0].lower()
    if head_prefix:
        tokens.add(head_prefix)
        if head_prefix.startswith("dl24-"):
            tokens.add(head_prefix.removeprefix("dl24-"))
    return tokens


def run_matches_pr(run: RunInfo, pr: dict[str, Any]) -> bool:
    student_labels = [
        label.split(":", 1)[1].lower()
        for label in labels_for_pr(pr)
        if label.startswith("student:")
    ]
    tokens = pr_student_tokens(pr)
    if not tokens:
        return True
    run_text = " ".join([run.name, run.group or "", *run.tags]).lower()
    for student in student_labels:
        if student.startswith("dl24-"):
            continue
        if f"dl24-{student}" in run_text:
            return False
    if any(token in run_text for token in tokens):
        return True
    # Ensemble evals can have neutral names but still carry useful ensemble tags.
    title_head = f"{pr.get('title', '')} {pr.get('headRefName', '')}".lower()
    if "ensemble" in run_text and ("ensemble" in title_head or "greedy" in title_head):
        return True
    return False


def run_in_pr_window(run: RunInfo, pr: dict[str, Any]) -> bool:
    run_time = parse_time(run.created_at)
    created = parse_time(pr.get("createdAt"))
    if not run_time or not created:
        return True
    end = parse_time(pr.get("closedAt")) or parse_time(pr.get("mergedAt"))
    if end is None:
        end = datetime.now(timezone.utc) + timedelta(days=1)
    return created - timedelta(hours=1) <= run_time <= end + timedelta(hours=36)


_thread_local = threading.local()


def wandb_api():
    api = getattr(_thread_local, "api", None)
    if api is None:
        import wandb

        api = wandb.Api()
        _thread_local.api = api
    return api


def fetch_run(run_id: str, projects: tuple[str, ...]) -> RunInfo | None:
    api = wandb_api()
    for project in projects:
        try:
            run = api.run(f"{ENTITY}/{project}/{run_id}")
        except Exception:
            continue
        summary = run.summary
        return RunInfo(
            run_id=run_id,
            project=project,
            name=run.name,
            group=run.group,
            tags=tuple(run.tags or ()),
            created_at=run.created_at,
            url=run.url,
            surface_pressure_rel_l2_pct=_as_float(summary.get(METRIC_KEY)),
            abupt_axis_mean_rel_l2_pct=_as_float(summary.get(ABUPT_KEY)),
            state=getattr(run, "state", None),
        )
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_wandb_cache() -> dict[str, dict[str, Any] | None]:
    if WANDB_CACHE.exists():
        return json.loads(WANDB_CACHE.read_text())
    return {}


def save_wandb_cache(cache: dict[str, dict[str, Any] | None]) -> None:
    WANDB_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolve_wandb_runs(candidate_ids_by_label: dict[str, set[str]]) -> dict[str, RunInfo]:
    cache = load_wandb_cache()
    run_to_projects: dict[str, set[str]] = {}
    for label, run_ids in candidate_ids_by_label.items():
        for run_id in run_ids:
            run_to_projects.setdefault(run_id, set()).update(LABEL_PROJECTS[label])

    missing = sorted(run_id for run_id in run_to_projects if run_id not in cache)
    if missing:
        print(f"Fetching {len(missing)} W&B run summaries...")
        completed = 0
        with ThreadPoolExecutor(max_workers=18) as executor:
            futures = {
                executor.submit(fetch_run, run_id, tuple(sorted(run_to_projects[run_id]))): run_id
                for run_id in missing
            }
            for future in as_completed(futures):
                run_id = futures[future]
                info = future.result()
                cache[run_id] = info.__dict__ if info else None
                completed += 1
                if completed % 100 == 0:
                    save_wandb_cache(cache)
                    print(f"  cached {completed}/{len(missing)}")
        save_wandb_cache(cache)

    resolved: dict[str, RunInfo] = {}
    for run_id, payload in cache.items():
        if not payload:
            continue
        resolved[run_id] = RunInfo(
            run_id=payload["run_id"],
            project=payload["project"],
            name=payload["name"],
            group=payload.get("group"),
            tags=tuple(payload.get("tags") or ()),
            created_at=payload.get("created_at"),
            url=payload["url"],
            surface_pressure_rel_l2_pct=payload.get("surface_pressure_rel_l2_pct"),
            abupt_axis_mean_rel_l2_pct=payload.get("abupt_axis_mean_rel_l2_pct"),
            state=payload.get("state"),
        )
    return resolved


def build_points(
    prs: list[dict[str, Any]], comments_by_num: dict[int, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    relevant_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for pr in prs:
        for label in relevant_labels(pr):
            relevant_by_label[label].append(pr)
    for label in LABELS:
        relevant_by_label[label].sort(key=lambda pr: (parse_time(pr.get("createdAt")) or datetime.max.replace(tzinfo=timezone.utc), pr["number"]))

    ordinal: dict[str, dict[int, int]] = {}
    for label, label_prs in relevant_by_label.items():
        ordinal[label] = {pr["number"]: i + 1 for i, pr in enumerate(label_prs)}

    candidate_ids_by_label: dict[str, set[str]] = {label: set() for label in LABELS}
    pr_run_candidates: dict[tuple[str, int], set[str]] = {}
    for label, label_prs in relevant_by_label.items():
        for pr in label_prs:
            corpus = "\n".join(
                [pr.get("body") or "", *[comment.get("body") or "" for comment in comments_by_num.get(pr["number"], [])]]
            )
            run_ids = extract_run_ids(corpus)
            pr_run_candidates[(label, pr["number"])] = run_ids
            candidate_ids_by_label[label].update(run_ids)

    resolved_runs = resolve_wandb_runs(candidate_ids_by_label)

    points: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for label, label_prs in relevant_by_label.items():
        allowed_projects = set(LABEL_PROJECTS[label])
        for pr in label_prs:
            for run_id in sorted(pr_run_candidates.get((label, pr["number"]), set())):
                run = resolved_runs.get(run_id)
                if not run or run.project not in allowed_projects:
                    continue
                value = run.surface_pressure_rel_l2_pct
                if value is None:
                    continue
                if not run_in_pr_window(run, pr):
                    continue
                if not run_matches_pr(run, pr):
                    continue
                key = (label, pr["number"], run_id)
                if key in seen:
                    continue
                seen.add(key)
                points.append(
                    {
                        "label": label,
                        "experiment_index": ordinal[label][pr["number"]],
                        "pr_number": pr["number"],
                        "pr_title": pr["title"],
                        "pr_created_at": pr.get("createdAt") or "",
                        "pr_closed_at": pr.get("closedAt") or "",
                        "pr_merged_at": pr.get("mergedAt") or "",
                        "head_ref": pr.get("headRefName") or "",
                        "run_id": run.run_id,
                        "wandb_project": run.project,
                        "wandb_name": run.name,
                        "wandb_group": run.group or "",
                        "wandb_tags": "|".join(run.tags),
                        "wandb_created_at": run.created_at or "",
                        "wandb_url": run.url,
                        "surface_pressure_rel_l2_pct": value,
                        "abupt_axis_mean_rel_l2_pct": run.abupt_axis_mean_rel_l2_pct,
                        "state": run.state or "",
                    }
                )
    points.sort(key=lambda row: (row["label"], row["experiment_index"], row["wandb_created_at"], row["run_id"]))
    return points, relevant_by_label


def is_human_comment(comment: dict[str, Any], issue_context: bool = False) -> bool:
    user = (comment.get("user") or {}).get("login")
    if user != "morganmcg1":
        return False
    body = (comment.get("body") or "").strip()
    if not body:
        return False
    if HUMAN_PREFIX_RE.search(body):
        return True
    if AGENT_PREFIX_RE.search(body):
        return False
    if issue_context:
        return True
    return bool(NUDGE_RE.search(body))


def build_interventions(
    prs_by_number: dict[int, dict[str, Any]],
    relevant_by_label: dict[str, list[dict[str, Any]]],
    comments_by_num: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordinal = {
        label: {pr["number"]: i + 1 for i, pr in enumerate(label_prs)}
        for label, label_prs in relevant_by_label.items()
    }
    pr_times_by_label = {
        label: [parse_time(pr.get("createdAt")) for pr in label_prs]
        for label, label_prs in relevant_by_label.items()
    }
    interventions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()

    def issue_x(label: str, timestamp: datetime | None) -> int:
        times = [t for t in pr_times_by_label[label] if t is not None]
        if not timestamp or not times:
            return 1
        return max(1, sum(t <= timestamp for t in times))

    for issue in HUMAN_ISSUES_FALLBACK:
        timestamp = parse_time(issue["createdAt"])
        row = {
            "label": issue["label"],
            "experiment_index": issue_x(issue["label"], timestamp),
            "kind": "issue",
            "number": issue["number"],
            "created_at": issue["createdAt"],
            "url": issue["url"],
            "title": issue["title"],
            "snippet": issue["title"],
            "classification": "human-labeled issue",
        }
        key = (row["label"], row["kind"], row["number"], row["created_at"])
        if key not in seen:
            seen.add(key)
            interventions.append(row)

        for comment in comments_by_num.get(issue["number"], []):
            if not is_human_comment(comment, issue_context=True):
                continue
            timestamp = parse_time(comment.get("created_at"))
            row = {
                "label": issue["label"],
                "experiment_index": issue_x(issue["label"], timestamp),
                "kind": "issue_comment",
                "number": issue["number"],
                "created_at": comment.get("created_at") or "",
                "url": comment.get("html_url") or issue["url"],
                "title": issue["title"],
                "snippet": compact(comment.get("body") or ""),
                "classification": "human-labeled issue comment",
            }
            key = (row["label"], row["kind"], row["number"], row["created_at"])
            if key not in seen:
                seen.add(key)
                interventions.append(row)

    for number, pr in prs_by_number.items():
        labels = relevant_labels(pr)
        if not labels:
            continue
        for comment in comments_by_num.get(number, []):
            if not is_human_comment(comment):
                continue
            body = compact(comment.get("body") or "")
            # PR-level intervention markers are kept strict: explicit
            # HUMAN/OPERATOR comments only. Human-labeled issues are handled
            # above, including their discussion comments.
            keep = HUMAN_PREFIX_RE.search((comment.get("body") or "").strip())
            if not keep:
                continue
            for label in labels:
                if number not in ordinal[label]:
                    continue
                row = {
                    "label": label,
                    "experiment_index": ordinal[label][number],
                    "kind": "pr_comment",
                    "number": number,
                    "created_at": comment.get("created_at") or "",
                    "url": comment.get("html_url") or pr.get("url") or "",
                    "title": pr.get("title") or "",
                    "snippet": body,
                    "classification": "explicit human/operator prefix",
                }
                key = (row["label"], row["kind"], row["number"], row["created_at"])
                if key not in seen:
                    seen.add(key)
                    interventions.append(row)

    interventions.sort(key=lambda row: (row["label"], row["experiment_index"], row["created_at"], row["number"]))
    return interventions


def compact(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pareto_rows(points: list[dict[str, Any]], relevant_by_label: dict[str, list[dict[str, Any]]]) -> dict[str, list[tuple[int, float]]]:
    by_label_x: dict[str, dict[int, list[float]]] = {label: {} for label in LABELS}
    for row in points:
        by_label_x[row["label"]].setdefault(int(row["experiment_index"]), []).append(float(row["surface_pressure_rel_l2_pct"]))

    out: dict[str, list[tuple[int, float]]] = {}
    for label in LABELS:
        best: float | None = None
        series: list[tuple[int, float]] = []
        for x in range(1, len(relevant_by_label[label]) + 1):
            vals = by_label_x[label].get(x)
            if vals:
                candidate = min(vals)
                best = candidate if best is None else min(best, candidate)
            if best is not None:
                series.append((x, best))
        out[label] = series
    return out


def render_chart(
    points: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    pareto: dict[str, list[tuple[int, float]]],
    relevant_by_label: dict[str, list[dict[str, Any]]],
) -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "font.family": [
                "Times New Roman",
                "Times",
                "Latin Modern Roman",
                "CMU Serif",
                "Computer Modern Serif",
                "serif",
            ],
            "font.serif": [
                "Times New Roman",
                "Times",
                "Latin Modern Roman",
                "CMU Serif",
                "Computer Modern Serif",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED_COLOR,
            "axes.labelcolor": INK_COLOR,
            "xtick.color": INK_COLOR,
            "ytick.color": INK_COLOR,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for label in PLOT_LABELS:
        label_points = [row for row in points if row["label"] == label]
        color = COLORS[label]
        ax.scatter(
            [row["experiment_index"] for row in label_points],
            [row["surface_pressure_rel_l2_pct"] for row in label_points],
            s=14 if label != "drivaerml-long-20260504" else 20,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=2,
        )
        series = pareto[label]
        if series:
            ax.plot(
                [x for x, _ in series],
                [y for _, y in series],
                color=color,
                linewidth=1.8,
                drawstyle="steps-post",
                zorder=4,
            )

    for row in interventions:
        label = row["label"]
        if label not in PLOT_LABELS:
            continue
        color = COLORS[label]
        x = int(row["experiment_index"])
        ax.axvline(x, color=color, linewidth=0.65, linestyle="--", alpha=0.14, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Experiment count", fontsize=11.5)
    ax.set_ylabel("Surface pressure rel. L2 (%)", fontsize=11.5, labelpad=8)
    ax.set_title("DrivAerML surface-pressure Pareto frontier", fontsize=14, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=9.5, length=3)
    ax.grid(True, which="major", axis="both", color=GRID_COLOR, linewidth=0.6, alpha=0.85)
    ax.grid(True, which="minor", axis="y", color=MINOR_GRID_COLOR, linewidth=0.45, alpha=0.75)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 4, 5, 6, 8)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    max_x = max(len(relevant_by_label[label]) for label in PLOT_LABELS)
    ax.set_xlim(0, max_x + 5)
    all_y = [
        float(row["surface_pressure_rel_l2_pct"])
        for row in points
        if row["label"] in PLOT_LABELS
    ]
    ax.set_ylim(max(min(all_y) * 0.8, 0.1), max(all_y) * 1.35)
    legend_handles = []
    legend_labels = []
    for label in PLOT_LABELS:
        color = COLORS[label]
        legend_handles.append(
            (
                Line2D([], [], color=color, linewidth=1.8),
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markersize=5.2,
                    markerfacecolor=color,
                    markeredgecolor="none",
                    alpha=0.28,
                ),
            )
        )
        legend_labels.append(DISPLAY[label])
    ax.legend(
        legend_handles,
        legend_labels,
        title="Best so far + runs",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        title_fontsize=9.5,
        borderaxespad=0,
        columnspacing=1.8,
        handlelength=3.2,
        handletextpad=0.7,
        markerfirst=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.875, bottom=0.13)

    fig.savefig(PNG_OUT, dpi=600, facecolor="#FFFFFF")
    fig.savefig(PDF_OUT)
    plt.close(fig)


def build_summary(
    points: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    pareto: dict[str, list[tuple[int, float]]],
    relevant_by_label: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metric_source": "W&B summary",
        "metric_key": METRIC_KEY,
        "repo": REPO,
        "outputs": {
            "points_csv": str(POINTS_CSV),
            "interventions_csv": str(INTERVENTIONS_CSV),
            "png": str(PNG_OUT),
            "pdf": str(PDF_OUT),
        },
        "labels": {},
    }
    for label in LABELS:
        label_points = [row for row in points if row["label"] == label]
        best_row = min(label_points, key=lambda row: row["surface_pressure_rel_l2_pct"]) if label_points else None
        summary["labels"][label] = {
            "pr_count": len(relevant_by_label[label]),
            "test_point_count": len(label_points),
            "intervention_count": sum(1 for row in interventions if row["label"] == label),
            "best_surface_pressure_rel_l2_pct": best_row["surface_pressure_rel_l2_pct"] if best_row else None,
            "best_pr": best_row["pr_number"] if best_row else None,
            "best_run_id": best_row["run_id"] if best_row else None,
            "best_wandb_url": best_row["wandb_url"] if best_row else None,
            "frontier_points": [{"experiment_index": x, "value": y} for x, y in pareto[label]],
        }
    return summary


def main() -> int:
    os.environ.setdefault("WANDB_SILENT", "true")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prs, comments = load_github_data()
    comments_by_num = comments_by_issue_number(comments)
    prs_by_number = {pr["number"]: pr for pr in prs}

    points, relevant_by_label = build_points(prs, comments_by_num)
    interventions = build_interventions(prs_by_number, relevant_by_label, comments_by_num)
    pareto = pareto_rows(points, relevant_by_label)

    write_csv(POINTS_CSV, points)
    write_csv(INTERVENTIONS_CSV, interventions)
    summary = build_summary(points, interventions, pareto, relevant_by_label)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    render_chart(points, interventions, pareto, relevant_by_label)

    compact_labels = {}
    for label, payload in summary["labels"].items():
        compact_labels[label] = {
            key: payload[key]
            for key in (
                "pr_count",
                "test_point_count",
                "intervention_count",
                "best_surface_pressure_rel_l2_pct",
                "best_pr",
                "best_run_id",
                "best_wandb_url",
            )
        }
    print(
        json.dumps(
            {
                "metric_source": summary["metric_source"],
                "metric_key": summary["metric_key"],
                "repo": summary["repo"],
                "outputs": summary["outputs"],
                "labels": compact_labels,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
