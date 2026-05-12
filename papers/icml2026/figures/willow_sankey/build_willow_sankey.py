#!/usr/bin/env python3
"""Build a compact Sankey-style SVG for the Willow hosted-W&B PR ledger."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = "morganmcg1/TandemFoilSet-Balanced"
BRANCHES = [f"icml-appendix-willow-pai2e-r{i}" for i in range(1, 6)]
SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        result_path = candidate / "experiment_log/tandemfoil_balanced_pr_results_2026-05-06_07-27-38.md"
        if result_path.exists():
            return candidate
    raise FileNotFoundError("Could not find experiment_log/tandemfoil_balanced_pr_results_2026-05-06_07-27-38.md")


REPO_ROOT = find_repo_root()
RESULTS_MD = REPO_ROOT / "experiment_log/tandemfoil_balanced_pr_results_2026-05-06_07-27-38.md"
OUT_DIR = SCRIPT_DIR
BACKGROUND_COLOR = os.environ.get("WILLOW_SANKEY_BG", "#FFFFFF")
OUTPUT_SUFFIX = os.environ.get("WILLOW_SANKEY_SUFFIX", "")
PHYSICS_COLOR = os.environ.get("WILLOW_SANKEY_PHYSICS_COLOR", "#56B4E9")
INK_COLOR = "#1F2933"
MUTED_COLOR = "#4B5563"
BRANCH_COLOR = "#293241"
FAMILY_COLORS = {
    "LR / schedule": "#0072B2",
    "Training Efficiency": "#785EF0",
    "EMA / stability": "#CC79A7",
    "Loss / weighting": "#E69F00",
    "Physics / features": PHYSICS_COLOR,
    "Model capacity": "#D55E00",
    "Tooling / audit": "#666666",
}
OUTCOME_COLORS = {
    "Merged": "#009E73",
    "Closed": "#B0B0B0",
    "Open at cutoff": "#6B7280",
}


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    state: str
    base: str
    url: str
    category: str
    family: str


def run_gh(branch: str) -> list[dict]:
    fields = ",".join(
        [
            "number",
            "title",
            "state",
            "mergedAt",
            "baseRefName",
            "headRefName",
            "url",
            "labels",
            "createdAt",
            "updatedAt",
        ]
    )
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        REPO,
        "--base",
        branch,
        "--state",
        "all",
        "--limit",
        "500",
        "--json",
        fields,
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def parse_result_categories(path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    row_re = re.compile(
        r"^\| \[#(?P<num>\d+)\]\([^)]*\) \| "
        r"(?P<state>MERGED|CLOSED) \| "
        r"(?P<date>[^|]*) \| "
        r"(?P<cat>[^|]*) \| "
    )
    for line in path.read_text().splitlines():
        match = row_re.match(line)
        if match:
            categories[int(match.group("num"))] = match.group("cat").strip()
    return categories


def compact_family(title: str, category: str) -> str:
    text = f"{title} {category}".lower()

    if any(
        key in text
        for key in [
            "nan",
            "bugfix",
            "bug fix",
            "evaluate",
            "scoring",
            "seed determinism",
            "3-seed",
            "baseline",
            "anchor",
            "tooling",
        ]
    ):
        return "Tooling / audit"

    if any(
        key in text
        for key in [
            "loss",
            "huber",
            "l1",
            "mae",
            "surf_weight",
            "surf weight",
            "channel-weight",
            "channel weight",
            "pressure 3x",
            "relative",
            "focal",
            "sigma normalization",
            "hard-negative",
            "per-sample norm",
        ]
    ) or category in {"Loss function", "Loss weighting"}:
        return "Loss / weighting"

    if any(
        key in text
        for key in [
            "bf16",
            "amp",
            "batch",
            "throughput",
            "compile",
            "vectorize",
            "epochs",
            "longer training",
            "training efficiency",
        ]
    ) or category == "Training efficiency":
        return "Training Efficiency"

    if any(
        key in text
        for key in [
            "t_max",
            "tmax",
            "onecycle",
            "one-cycle",
            "cosine",
            "warmup",
            "lr",
            "learning rate",
            "adamw",
            "lion",
            "weight_decay",
            "beta",
            "sgdr",
            "pct_start",
        ]
    ) or category == "LR / optimizer":
        return "LR / schedule"

    if any(
        key in text
        for key in [
            "grad-clip",
            "gradient norm",
            "clip norm",
            "ema",
            "polyak",
            "droppath",
            "drop path",
            "dropout",
            "stochastic depth",
            "regularization",
        ]
    ) or category == "Regularization":
        return "EMA / stability"

    if any(
        key in text
        for key in [
            "fourier",
            "rff",
            "film",
            "re-",
            "re ",
            "naca",
            "boundary",
            "distance",
            "domain",
            "stratified",
            "sampling",
            "normalization",
            "coordinate",
            "canonical",
            "jitter",
            "augmentation",
            "physics",
            "derived features",
            "slice_num",
            "slice num",
            "slices",
        ]
    ) or category in {"Feature engineering", "Physics / normalization"}:
        return "Physics / features"

    return "Model capacity"


def collect_prs() -> list[PullRequest]:
    categories = parse_result_categories(RESULTS_MD)
    prs: list[PullRequest] = []
    for branch in BRANCHES:
        for item in run_gh(branch):
            number = int(item["number"])
            title = item["title"]
            category = categories.get(number, "Open / uncategorized")
            family = compact_family(title, category)
            prs.append(
                PullRequest(
                    number=number,
                    title=title,
                    state=item["state"],
                    base=item["baseRefName"],
                    url=item["url"],
                    category=category,
                    family=family,
                )
            )
    return sorted(prs, key=lambda pr: (pr.base, pr.number))


def branch_label(branch: str) -> str:
    return "A" + branch.rsplit("-r", 1)[1]


def write_csv(prs: Iterable[PullRequest]) -> None:
    with (OUT_DIR / "willow_prs_classified.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pr", "base", "state", "family", "category", "title", "url"],
        )
        writer.writeheader()
        for pr in prs:
            writer.writerow(
                {
                    "pr": pr.number,
                    "base": pr.base,
                    "state": pr.state,
                    "family": pr.family,
                    "category": pr.category,
                    "title": pr.title,
                    "url": pr.url,
                }
            )


def flow_counts(prs: list[PullRequest]) -> tuple[Counter, Counter, Counter]:
    branch_family: Counter[tuple[str, str]] = Counter()
    family_outcome: Counter[tuple[str, str]] = Counter()
    state_counts: Counter[str] = Counter()
    for pr in prs:
        b = branch_label(pr.base)
        branch_family[(b, pr.family)] += 1
        outcome = {
            "MERGED": "Merged",
            "CLOSED": "Closed",
            "OPEN": "Open at cutoff",
        }[pr.state]
        family_outcome[(pr.family, outcome)] += 1
        state_counts[outcome] += 1
    return branch_family, family_outcome, state_counts


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def draw_ribbon(
    x0: float,
    y0a: float,
    y0b: float,
    x1: float,
    y1a: float,
    y1b: float,
    color: str,
    opacity: float = 0.42,
) -> str:
    c0 = x0 + (x1 - x0) * 0.52
    c1 = x0 + (x1 - x0) * 0.48
    return (
        f'<path d="M {x0:.1f} {y0a:.1f} '
        f"C {c0:.1f} {y0a:.1f}, {c1:.1f} {y1a:.1f}, {x1:.1f} {y1a:.1f} "
        f"L {x1:.1f} {y1b:.1f} "
        f"C {c1:.1f} {y1b:.1f}, {c0:.1f} {y0b:.1f}, {x0:.1f} {y0b:.1f} Z"
        f'" fill="{color}" opacity="{opacity}"/>'
    )


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_color(color: str) -> str:
    r, g, b = hex_to_rgb(color)
    return f"{r:.4f} {g:.4f} {b:.4f} rg"


def pdf_y(y: float, height: float) -> float:
    return height - y


TIMES_ROMAN_WIDTHS = {
    " ": 250,
    "(": 333,
    ")": 333,
    "/": 278,
    "0": 500,
    "1": 500,
    "2": 500,
    "3": 500,
    "4": 500,
    "5": 500,
    "6": 500,
    "7": 500,
    "8": 500,
    "9": 500,
    "A": 722,
}


def pdf_text_width(text: str, font_size: float) -> float:
    return sum(TIMES_ROMAN_WIDTHS.get(char, 500) for char in text) * font_size / 1000



def pdf_text(
    text: str,
    x: float,
    y: float,
    font_size: float,
    color: str,
    page_height: float,
    anchor: str = "start",
) -> str:
    if anchor == "end":
        x -= pdf_text_width(text, font_size)
    elif anchor == "middle":
        x -= pdf_text_width(text, font_size) / 2
    return (
        "BT "
        f"/F1 {font_size:.1f} Tf "
        f"{pdf_color(color)} "
        f"{x:.2f} {pdf_y(y, page_height):.2f} Td "
        f"({pdf_escape(text)}) Tj ET"
    )


def pdf_ribbon(
    x0: float,
    y0a: float,
    y0b: float,
    x1: float,
    y1a: float,
    y1b: float,
    color: str,
    opacity_name: str,
    page_height: float,
) -> str:
    c0 = x0 + (x1 - x0) * 0.52
    c1 = x0 + (x1 - x0) * 0.48
    return "\n".join(
        [
            "q",
            f"/{opacity_name} gs",
            pdf_color(color),
            f"{x0:.2f} {pdf_y(y0a, page_height):.2f} m",
            (
                f"{c0:.2f} {pdf_y(y0a, page_height):.2f} "
                f"{c1:.2f} {pdf_y(y1a, page_height):.2f} "
                f"{x1:.2f} {pdf_y(y1a, page_height):.2f} c"
            ),
            f"{x1:.2f} {pdf_y(y1b, page_height):.2f} l",
            (
                f"{c1:.2f} {pdf_y(y1b, page_height):.2f} "
                f"{c0:.2f} {pdf_y(y0b, page_height):.2f} "
                f"{x0:.2f} {pdf_y(y0b, page_height):.2f} c"
            ),
            "h f",
            "Q",
        ]
    )


def pdf_rounded_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
    color: str,
    page_height: float,
) -> str:
    r = min(radius, width / 2, height / 2)
    k = 0.55228475
    x0, x1 = x, x + width
    y0, y1 = y, y + height
    return "\n".join(
        [
            pdf_color(color),
            f"{x0 + r:.2f} {pdf_y(y0, page_height):.2f} m",
            f"{x1 - r:.2f} {pdf_y(y0, page_height):.2f} l",
            (
                f"{x1 - r + r * k:.2f} {pdf_y(y0, page_height):.2f} "
                f"{x1:.2f} {pdf_y(y0 + r - r * k, page_height):.2f} "
                f"{x1:.2f} {pdf_y(y0 + r, page_height):.2f} c"
            ),
            f"{x1:.2f} {pdf_y(y1 - r, page_height):.2f} l",
            (
                f"{x1:.2f} {pdf_y(y1 - r + r * k, page_height):.2f} "
                f"{x1 - r + r * k:.2f} {pdf_y(y1, page_height):.2f} "
                f"{x1 - r:.2f} {pdf_y(y1, page_height):.2f} c"
            ),
            f"{x0 + r:.2f} {pdf_y(y1, page_height):.2f} l",
            (
                f"{x0 + r - r * k:.2f} {pdf_y(y1, page_height):.2f} "
                f"{x0:.2f} {pdf_y(y1 - r + r * k, page_height):.2f} "
                f"{x0:.2f} {pdf_y(y1 - r, page_height):.2f} c"
            ),
            f"{x0:.2f} {pdf_y(y0 + r, page_height):.2f} l",
            (
                f"{x0:.2f} {pdf_y(y0 + r - r * k, page_height):.2f} "
                f"{x0 + r - r * k:.2f} {pdf_y(y0, page_height):.2f} "
                f"{x0 + r:.2f} {pdf_y(y0, page_height):.2f} c"
            ),
            "h f",
        ]
    )


def write_pdf(path: Path, content: str, width: int, height: int) -> None:
    stream = content.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R >> "
            "/ExtGState << /GS32 5 0 R /GS38 6 0 R >> >> "
            "/Contents 7 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
        b"<< /Type /ExtGState /ca 0.32 /CA 0.32 >>",
        b"<< /Type /ExtGState /ca 0.38 /CA 0.38 >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{i} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(output)


def stack_positions(labels: list[str], totals: dict[str, int], scale: float, y_top: float, gap: float) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    y = y_top
    for label in labels:
        h = max(totals[label] * scale, 3.0)
        positions[label] = (y, y + h)
        y += h + gap
    return positions


def draw_svg(prs: list[PullRequest]) -> None:
    branch_family, family_outcome, state_counts = flow_counts(prs)

    branches = [branch_label(b) for b in BRANCHES]
    families = [
        "LR / schedule",
        "Training Efficiency",
        "EMA / stability",
        "Loss / weighting",
        "Physics / features",
        "Model capacity",
        "Tooling / audit",
    ]
    outcomes = ["Merged", "Closed", "Open at cutoff"]

    branch_totals = {b: sum(v for (bb, _), v in branch_family.items() if bb == b) for b in branches}
    family_totals = {f: sum(v for (_, ff), v in branch_family.items() if ff == f) for f in families}
    outcome_totals = {o: state_counts[o] for o in outcomes}

    width, height = 650, 410
    x_branch, x_family, x_outcome = 72, 315, 520
    node_w = 24
    scale = 1.68
    y_top = 28

    branch_pos = stack_positions(branches, branch_totals, scale, y_top, 18)
    family_pos = stack_positions(families, family_totals, scale, y_top - 6, 10)
    outcome_pos = stack_positions(outcomes, outcome_totals, scale, y_top + 22, 22)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND_COLOR}"/>',
        f'<style>text{{font-family:"Times New Roman",Times,"Latin Modern Roman","CMU Serif","Computer Modern Serif",serif;fill:{INK_COLOR}}}.label{{font-size:10.5px;font-weight:400}}.small{{font-size:8.8px;fill:{MUTED_COLOR};font-weight:400}}</style>',
    ]

    # Branch to family ribbons.
    branch_offsets = {b: branch_pos[b][0] for b in branches}
    family_in_offsets = {f: family_pos[f][0] for f in families}
    for b in branches:
        for f in families:
            v = branch_family.get((b, f), 0)
            if not v:
                continue
            h = v * scale
            y0a = branch_offsets[b]
            y0b = y0a + h
            y1a = family_in_offsets[f]
            y1b = y1a + h
            branch_offsets[b] = y0b
            family_in_offsets[f] = y1b
            parts.append(draw_ribbon(x_branch + node_w, y0a, y0b, x_family, y1a, y1b, FAMILY_COLORS[f], 0.32))

    # Family to outcome ribbons.
    family_out_offsets = {f: family_pos[f][0] for f in families}
    outcome_offsets = {o: outcome_pos[o][0] for o in outcomes}
    for f in families:
        for o in outcomes:
            v = family_outcome.get((f, o), 0)
            if not v:
                continue
            h = v * scale
            y0a = family_out_offsets[f]
            y0b = y0a + h
            y1a = outcome_offsets[o]
            y1b = y1a + h
            family_out_offsets[f] = y0b
            outcome_offsets[o] = y1b
            parts.append(draw_ribbon(x_family + node_w, y0a, y0b, x_outcome, y1a, y1b, OUTCOME_COLORS[o], 0.38))

    def add_nodes(labels: list[str], positions: dict[str, tuple[float, float]], totals: dict[str, int], x: float, colors: dict[str, str] | None = None) -> None:
        for label in labels:
            y0, y1 = positions[label]
            label_nudge = 6 if label == "Tooling / audit" else 0
            color = colors.get(label, BRANCH_COLOR) if colors else BRANCH_COLOR
            parts.append(f'<rect x="{x}" y="{y0:.1f}" width="{node_w}" height="{y1-y0:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text class="label" x="{x + node_w + 10}" y="{(y0+y1)/2 - 2 + label_nudge:.1f}">{svg_escape(label)}</text>')
            parts.append(f'<text class="small" x="{x + node_w + 10}" y="{(y0+y1)/2 + 5 + label_nudge:.1f}">({totals[label]})</text>')

    branch_colors = {b: BRANCH_COLOR for b in branches}
    branch_label_x = x_branch - 23
    for b in branches:
        y0, y1 = branch_pos[b]
        color = branch_colors[b]
        parts.append(f'<rect x="{x_branch}" y="{y0:.1f}" width="{node_w}" height="{y1-y0:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text class="label" x="{branch_label_x}" text-anchor="middle" y="{(y0+y1)/2 - 2:.1f}">{b}</text>')
        parts.append(f'<text class="small" x="{branch_label_x}" text-anchor="middle" y="{(y0+y1)/2 + 5:.1f}">({branch_totals[b]})</text>')

    add_nodes(families, family_pos, family_totals, x_family, FAMILY_COLORS)
    add_nodes(outcomes, outcome_pos, outcome_totals, x_outcome, OUTCOME_COLORS)

    parts.append("</svg>")
    (OUT_DIR / f"willow_simplified_sankey{OUTPUT_SUFFIX}.svg").write_text("\n".join(parts))


def draw_pdf(prs: list[PullRequest]) -> None:
    branch_family, family_outcome, state_counts = flow_counts(prs)

    branches = [branch_label(b) for b in BRANCHES]
    families = [
        "LR / schedule",
        "Training Efficiency",
        "EMA / stability",
        "Loss / weighting",
        "Physics / features",
        "Model capacity",
        "Tooling / audit",
    ]
    outcomes = ["Merged", "Closed", "Open at cutoff"]

    branch_totals = {b: sum(v for (bb, _), v in branch_family.items() if bb == b) for b in branches}
    family_totals = {f: sum(v for (_, ff), v in branch_family.items() if ff == f) for f in families}
    outcome_totals = {o: state_counts[o] for o in outcomes}

    width, height = 650, 410
    x_branch, x_family, x_outcome = 72, 315, 520
    node_w = 24
    scale = 1.68
    y_top = 28

    branch_pos = stack_positions(branches, branch_totals, scale, y_top, 18)
    family_pos = stack_positions(families, family_totals, scale, y_top - 6, 10)
    outcome_pos = stack_positions(outcomes, outcome_totals, scale, y_top + 22, 22)

    parts: list[str] = [
        pdf_color(BACKGROUND_COLOR),
        f"0 0 {width} {height} re f",
    ]

    branch_offsets = {b: branch_pos[b][0] for b in branches}
    family_in_offsets = {f: family_pos[f][0] for f in families}
    for b in branches:
        for f in families:
            v = branch_family.get((b, f), 0)
            if not v:
                continue
            h = v * scale
            y0a = branch_offsets[b]
            y0b = y0a + h
            y1a = family_in_offsets[f]
            y1b = y1a + h
            branch_offsets[b] = y0b
            family_in_offsets[f] = y1b
            parts.append(pdf_ribbon(x_branch + node_w, y0a, y0b, x_family, y1a, y1b, FAMILY_COLORS[f], "GS32", height))

    family_out_offsets = {f: family_pos[f][0] for f in families}
    outcome_offsets = {o: outcome_pos[o][0] for o in outcomes}
    for f in families:
        for o in outcomes:
            v = family_outcome.get((f, o), 0)
            if not v:
                continue
            h = v * scale
            y0a = family_out_offsets[f]
            y0b = y0a + h
            y1a = outcome_offsets[o]
            y1b = y1a + h
            family_out_offsets[f] = y0b
            outcome_offsets[o] = y1b
            parts.append(pdf_ribbon(x_family + node_w, y0a, y0b, x_outcome, y1a, y1b, OUTCOME_COLORS[o], "GS38", height))

    def add_nodes(labels: list[str], positions: dict[str, tuple[float, float]], totals: dict[str, int], x: float, colors: dict[str, str] | None = None) -> None:
        for label in labels:
            y0, y1 = positions[label]
            label_nudge = 6 if label == "Tooling / audit" else 0
            color = colors.get(label, BRANCH_COLOR) if colors else BRANCH_COLOR
            parts.append(pdf_rounded_rect(x, y0, node_w, y1 - y0, 4, color, height))
            parts.append(pdf_text(label, x + node_w + 10, (y0 + y1) / 2 - 2 + label_nudge, 10.5, INK_COLOR, height))
            parts.append(pdf_text(f"({totals[label]})", x + node_w + 10, (y0 + y1) / 2 + 5 + label_nudge, 8.8, MUTED_COLOR, height))

    branch_colors = {b: BRANCH_COLOR for b in branches}
    branch_label_x = x_branch - 23
    for b in branches:
        y0, y1 = branch_pos[b]
        parts.append(pdf_rounded_rect(x_branch, y0, node_w, y1 - y0, 4, branch_colors[b], height))
        parts.append(pdf_text(b, branch_label_x, (y0 + y1) / 2 - 2, 10.5, INK_COLOR, height, anchor="middle"))
        parts.append(pdf_text(f"({branch_totals[b]})", branch_label_x, (y0 + y1) / 2 + 5, 8.8, MUTED_COLOR, height, anchor="middle"))

    add_nodes(families, family_pos, family_totals, x_family, FAMILY_COLORS)
    add_nodes(outcomes, outcome_pos, outcome_totals, x_outcome, OUTCOME_COLORS)

    write_pdf(OUT_DIR / f"willow_simplified_sankey{OUTPUT_SUFFIX}.pdf", "\n".join(parts), width, height)


def write_summary(prs: list[PullRequest]) -> None:
    branch_family, family_outcome, state_counts = flow_counts(prs)
    family_counts = Counter(pr.family for pr in prs)
    lines = [
        "# Willow hosted-W&B compact Sankey data",
        "",
        f"Repository: `{REPO}`",
        "",
        f"Branches: {', '.join(f'`{b}`' for b in BRANCHES)}",
        "",
        f"Total PRs: {len(prs)}",
        "",
        "## Outcomes",
        "",
    ]
    for state, count in state_counts.most_common():
        lines.append(f"- {state}: {count}")
    lines.extend(["", "## Hypothesis Families", ""])
    for family, count in family_counts.most_common():
        merged = family_outcome.get((family, "Merged"), 0)
        closed = family_outcome.get((family, "Closed"), 0)
        open_ = family_outcome.get((family, "Open at cutoff"), 0)
        lines.append(f"- {family}: {count} PRs ({merged} merged, {closed} closed, {open_} open)")
    lines.extend(["", "## Branch Counts", ""])
    for b in BRANCHES:
        short = branch_label(b)
        lines.append(f"- {b}: {sum(v for (bb, _), v in branch_family.items() if bb == short)} PRs")
    lines.append("")
    (OUT_DIR / "willow_sankey_summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = collect_prs()
    write_csv(prs)
    draw_svg(prs)
    draw_pdf(prs)
    write_summary(prs)
    print(f"Wrote {OUT_DIR / f'willow_simplified_sankey{OUTPUT_SUFFIX}.svg'}")
    print(f"Wrote {OUT_DIR / f'willow_simplified_sankey{OUTPUT_SUFFIX}.pdf'}")
    print(f"Wrote {OUT_DIR / 'willow_prs_classified.csv'}")
    print(f"Wrote {OUT_DIR / 'willow_sankey_summary.md'}")


if __name__ == "__main__":
    main()
