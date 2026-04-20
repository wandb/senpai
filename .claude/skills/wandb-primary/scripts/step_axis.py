# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""X-axis step-metric detection for W&B runs.

Different stacks log different step keys (`_step`, `global_step`,
`trainer/global_step`, `epoch`, `train/step`, ...). Overlaying runs on the
wrong x-axis silently corrupts a verdict, so confirm with the user before
plotting. These helpers feed that confirmation step.
"""

from __future__ import annotations

from typing import Any

from wandb_helpers import fast_scan_history

# Common step-key names in order of likelihood.
KNOWN_STEP_KEYS: tuple[str, ...] = (
    "_step",
    "global_step",
    "trainer/global_step",
    "train/step",
    "step",
    "epoch",
    "train/epoch",
    "iteration",
    "iter",
)


def list_candidate_step_keys(run: Any, sample_rows: int = 50) -> list[str]:
    """Return plausible step-axis keys from the first `sample_rows` of history.

    Keeps known step-key names (in KNOWN_STEP_KEYS order) plus any other
    numeric monotonically-non-decreasing columns (alphabetical).
    """
    rows = []
    for i, row in enumerate(fast_scan_history(run)):
        rows.append(row)
        if i + 1 >= sample_rows:
            break
    if not rows:
        return []

    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())

    # Known keys that actually appear.
    known_hits = [k for k in KNOWN_STEP_KEYS if k in all_keys]

    # Monotonic-numeric columns not already matched.
    monotonic_extras: list[str] = []
    for key in sorted(all_keys):
        if key in known_hits:
            continue
        if key.startswith("_") and key != "_step":
            continue
        values = [r.get(key) for r in rows if r.get(key) is not None]
        if len(values) < 5:
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        if all(values[i + 1] >= values[i] for i in range(len(values) - 1)):
            monotonic_extras.append(key)

    return known_hits + monotonic_extras


def guess_step_key_from_workspace(entity: str, project: str) -> str | None:
    """Return the x-axis used by the project's W&B workspace line panels, or None.

    Reflects what the human actually looks at in the UI. Returns None only if
    `wandb_workspaces` isn't installed or the project has no line-plot panel.
    """
    try:
        import wandb_workspaces.workspaces as ws  # type: ignore
    except ImportError:
        return None

    workspaces = ws.Workspace.list(entity=entity, project=project)
    for wsp_ref in workspaces:
        wsp = wsp_ref.load() if hasattr(wsp_ref, "load") else wsp_ref
        for section in getattr(wsp, "sections", None) or []:
            for panel in getattr(section, "panels", None) or []:
                x = getattr(panel, "x", None)
                if isinstance(x, str) and x:
                    return x
    return None


def format_step_candidates(
    candidates: list[str],
    workspace_guess: str | None,
) -> list[tuple[str, str]]:
    """Format candidates as (label, description) pairs for AskUserQuestion.

    The workspace guess (if any) is placed first and labeled "(Recommended)".
    """
    seen: set[str] = set()
    ordered: list[str] = []
    if workspace_guess:
        ordered.append(workspace_guess)
        seen.add(workspace_guess)
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    pairs: list[tuple[str, str]] = []
    for key in ordered:
        if key == workspace_guess:
            label = f"{key} (Recommended)"
            desc = "Matches the x-axis used by this project's W&B workspace panels."
        elif key in KNOWN_STEP_KEYS:
            label = key
            desc = f"Standard step-key `{key}` logged in this run's history."
        else:
            label = key
            desc = f"Custom monotonic column `{key}` found in this run's history."
        pairs.append((label, desc))
    return pairs
