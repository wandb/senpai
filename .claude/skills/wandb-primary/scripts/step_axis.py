# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

"""X-axis step-metric detection for W&B runs.

Training curves are meaningless without knowing which column is the x-axis.
Different training stacks log different step keys — `_step` (wandb default),
`global_step` (HF Trainer), `trainer/global_step`, `epoch`, or custom names
like `train/step`. The skill MUST confirm this with the user before any
curve-shape or plotting work. These helpers feed that confirmation step.

Usage:
    from step_axis import (
        list_candidate_step_keys,
        guess_step_key_from_workspace,
        format_step_candidates,
    )

    candidates = list_candidate_step_keys(run)
    guess = guess_step_key_from_workspace(entity, project)
    # Then show `format_step_candidates(candidates, guess)` to the user via
    # AskUserQuestion and wait for confirmation.
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
    """Return plausible step-axis keys logged in this run's history.

    Scans the first `sample_rows` rows of history (no keys filter) to learn
    which columns the run actually logs, then keeps only those that match a
    known step-key name OR appear numeric and monotonically non-decreasing
    across the sample.

    Ordering: known-name matches first (in `KNOWN_STEP_KEYS` order), then
    any other monotonic-numeric candidates, sorted alphabetically.

    Args:
        run: A W&B Run object from api.run().
        sample_rows: How many rows to inspect (default 50).

    Returns:
        List of candidate column names. Never raises; returns [] if the
        history is empty or unreadable.
    """
    try:
        rows = []
        for i, row in enumerate(fast_scan_history(run)):
            rows.append(row)
            if i + 1 >= sample_rows:
                break
    except Exception:
        return []
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
    """Peek at the project's most recent workspace and return the x-axis
    its first line-plot uses. This reflects what the human actually looks
    at in the W&B UI.

    Degrades silently: returns None if `wandb_workspaces` isn't installed,
    if no workspace or line plot exists, or if anything else goes wrong.
    Never raises.

    Args:
        entity: W&B entity name.
        project: W&B project name.

    Returns:
        The x-axis metric name, or None.
    """
    try:
        import wandb_workspaces.workspaces as ws  # type: ignore
    except Exception:
        return None

    try:
        workspaces = ws.Workspace.list(entity=entity, project=project)
    except Exception:
        return None
    if not workspaces:
        return None

    # Most recent first; the list's order depends on the SDK version, so try
    # all of them until one yields a usable line plot.
    for wsp_ref in workspaces:
        try:
            wsp = wsp_ref.load() if hasattr(wsp_ref, "load") else wsp_ref
        except Exception:
            continue
        sections = getattr(wsp, "sections", None) or []
        for section in sections:
            panels = getattr(section, "panels", None) or []
            for panel in panels:
                x = getattr(panel, "x", None)
                if isinstance(x, str) and x:
                    return x
    return None


def format_step_candidates(
    candidates: list[str],
    workspace_guess: str | None,
) -> list[tuple[str, str]]:
    """Format candidates as (label, description) pairs ready to hand to
    AskUserQuestion.

    The workspace guess (if any) is placed first and labeled "(Recommended)".
    Duplicate entries are dropped.

    Args:
        candidates: Output of `list_candidate_step_keys`.
        workspace_guess: Output of `guess_step_key_from_workspace`.

    Returns:
        List of (label, description) tuples. Empty if no candidates at all.
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
    for i, key in enumerate(ordered):
        is_ws_guess = key == workspace_guess
        if is_ws_guess:
            label = f"{key} (Recommended)"
            desc = "Matches the x-axis used by this project's W&B workspace panels."
        elif key in KNOWN_STEP_KEYS:
            desc = f"Standard step-key `{key}` logged in this run's history."
            label = key
        else:
            desc = f"Custom monotonic column `{key}` found in this run's history."
            label = key
        pairs.append((label, desc))
    return pairs
