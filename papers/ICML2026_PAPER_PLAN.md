# SENPAI — ICML 2026 AI-for-Science Workshop Paper Plan

## Context

The ICML 2026 "AI for Science — AI Scientists: Tools, Co-authors, or Founders?" workshop accepts 4–8 page Original Research submissions. **The abstract has already been submitted and is fixed** (`papers/ABSTRACT_2026-02-22.md`, reproduced verbatim below); this plan designs a **5-page-main-text systems paper** whose body substantiates every claim in that submitted abstract. The body positions SENPAI as an autonomous ML-research harness whose distinguishing contribution is PR+W&B as first-class state (vs. scratchpad/in-context state in Sakana, Agent Laboratory, AgentRxiv). A 48–72h unattended sprint runs concurrently to land the abstract's at-risk claims (≥72h unsupervised with "minor course corrections", ≥8,000 runs, DrivAerML "preliminary runs nearing LES references").

**Constraints confirmed with user:**
- **Submitted abstract is immutable** — use verbatim, do not rewrite or soften. Body must land its claims.
- Systems paper framing (harness is THE contribution)
- 5 pages main text (leaning short for clarity)
- Failure-mode analysis as a ~0.4 page subsection inside Discussion
- Push the 48–72h sprint to substantiate claims — if results fall short, the response is run more experiments / frame honestly within the rhetorical room the abstract already provides (words like "preliminary", "nearing", "minor course corrections"), not change the abstract

### The submitted abstract (verbatim — do not edit)

> SENPAI (Self-Experimentation for Physical AI) is an autonomous research harness that has executed 8000+ training runs on ML surrogates for computational fluid dynamics (CFD) simulations. Training CFD surrogates encompasses standard ML training considerations - architecture, optimizer, schedule, and loss design - but also physics-aware considerations such as boundary conditions, symmetries, and rollout stability. The harness orchestration is deliberately thin: an Advisor agent proposes hypotheses as GitHub pull requests; Student agents check out each PR, execute training based on the hypothesis, and write results back to the PR. Every hypothesis is grounded by a literature-search sub-agent callable by the advisor or student. A key contribution is the observability-first memory and orchestration that goes beyond scratchpad files and in-context memory; SENPAI also grounds state in the artifacts ML researchers already use - PRs, git history, and an experiment logger (Weights & Biases) - producing an experiment ledger queryable by agents, researchers, and the harness itself. This design enables the system to learn from past experiments, reflect on its own trajectories, and keeps humans meaningfully in the research loop - properties that are difficult to achieve simultaneously in prior agentic research systems. We run SENPAI on a Transolver model jointly across three aerodynamic benchmarks: TandemFoilSet, AirfRANS, and DrivAerML. On the AirfRANS benchmark the final recipe outperforms the reported Transolver baseline on surface-MSE; on TandemFoilSet dataset it is competitive with the normalized full-field MSE reported in the TandemFoilSet paper benchmark.; on DrivAerML, preliminary runs show surface pressure relative-L2 nearing reported Large Eddy Simulation (LES) references for the Transolver model. Taken together, our results demonstrate that AI scientists function best as near-autonomous co-workers that can run up to 72 hours unsupervised, with only minor course corrections required after these durations. We release the harness, the PR-indexed experiment ledger, and a failure-mode analysis.

### Claim-to-evidence map (the contract the body must deliver)

| Abstract claim | Required evidence in the paper |
| --- | --- |
| "8000+ training runs" | Final ledger snapshot at submission time showing ≥8000 runs. Appendix: PR-indexed ledger. |
| "observability-first memory … grounds state in … PRs, git history, and … Weights & Biases" | §2.2 PR+W&B state section; Figure 1 system diagram; Figure 2 real PR+W&B artifact. |
| "literature-search sub-agent callable by the advisor or student" | §2.4 with appendix referring to `.claude/agents/researcher-agent.md`. |
| "AirfRANS … outperforms the reported Transolver baseline on surface-MSE" | §3.2 number with W&B run ID + comparison to published Transolver surface-MSE. |
| "TandemFoilSet … competitive with the normalized full-field MSE reported in the TandemFoilSet paper benchmark" | §3.3 citation-clean test-row on the paper-calibration contract (`tandemfoil_paper/` field_mse) — **provenance cleanup is a hard sprint requirement**. |
| "DrivAerML … preliminary runs show surface pressure relative-L2 nearing reported LES references" | §3.4 uses the abstract's "preliminary"/"nearing" latitude: report val improvement (currently ~4.0%) as the "preliminary nearing" evidence alongside best-checkpoint test; disclose test gap honestly without softening the abstract language. |
| "up to 72 hours unsupervised, with only minor course corrections" | §2.5 human-in-the-loop paragraph + appendix sprint log documenting ≥72h of wall-clock with ≤2 logged `urgent` GitHub issues as the "minor course corrections". |
| "We release the harness, the PR-indexed experiment ledger, and a failure-mode analysis" | Anonymized GitHub release at camera-ready time; ledger CSV in appendix; §4.2 failure-mode subsection + appendix taxonomy table. |

---

## Paper scaffold (5.0 pages main)

Page budgets are approximate; they sum to 5.0. References and appendix unlimited.

### Title & Abstract (not counted)
- Working title: **"SENPAI: An Autonomous Multi-Agent Research Harness with Pull-Request-Grounded State"** (title editable; abstract is not).
- **Abstract: use the submitted text verbatim** (reproduced in Context above). Do not rewrite. The body's job is to substantiate every one of its claims using the claim-to-evidence map above.

### §1. Introduction — 0.75 page
- Open motivation-first: CFD-surrogate research programmes have a large repetitive experiment surface (normalization, loss shaping, architecture ablations) and their state already lives in PRs + W&B.
- Gap sentence: prior autonomous-ML systems (Sakana AI Scientist v1/v2, Agent Laboratory, AgentRxiv) operate on in-context or scratchpad state; none use version-controlled PRs + an experiment tracker as **first-class** state.
- One-paragraph pipeline overview (Advisor triages → drafts PR → Student implements → trains → posts results → Advisor merges).
- **Contributions — exactly 3 bullets**:
  1. An Advisor+Student harness that grounds state in GitHub PRs and W&B runs, producing a queryable ledger without SENPAI-specific tooling.
  2. A 6-week deployment across three public CFD benchmarks (AirfRANS, DrivAerML) and an internal benchmark (TandemFoilSet), with autonomous hyperparameter and recipe search.
  3. A failure taxonomy from 241 agent sessions with phase-level evolution data — actionable for future agentic-research builders.
- **Tone rules**: never "fully autonomous"; use "autonomous under human PR review". Say what is autonomous (proposes, implements, evaluates) and what isn't (problem definition, dataset, merge ratification).

### §2. System Design — 1.5 pages
- **§2.1 Roles (0.25p)**: Advisor (CPU, persistent loop, triage+hypothesis) and N Students (GPU, short-context, stateless per session). Point at `system_instructions/CLAUDE-ADVISOR.md` and `CLAUDE-STUDENT.md` in appendix.
- **§2.2 PR + W&B as state (0.4p)** — load-bearing. The git history IS the research log; W&B runs ARE the experiment database; human reviewers and agents read the same artifacts. Cite AgentRxiv as a weaker related move. Verbatim claim: "we use GitHub pull requests as the primary state store."
- **§2.3 The loop and seven skills (0.35p)**: `survey-prs`, `assign-experiment`, `merge-winner`, `poll-for-work`, `submit-experiment-results`, `check-human-issues`, `senpai-gh`, `list-experiments`. Skills architecture (PR #1965) replaced a monolithic prompt — reference it as a turning point in the operational history.
- **§2.4 Literature-search subagent (0.15p)**: invoked by both roles at decision points (Opus, effort:high). Sources: arXiv, Semantic Scholar (citation-graph traversal), alphaXiv, GitHub, Kaggle. Defense against within-code-neighborhood search.
- **§2.5 Human-in-the-loop boundary (0.2p)** — dedicated paragraph ("what the human does"). Three channels: PR review, `check-human-issues` GitHub-issue channel, `launch.py` fleet control. Enumerate exactly what the human did during the reported sprint (T=0 infra merge, up to two logged GitHub issues, pod-restart watchdog). This paragraph is load-bearing for reviewer trust.
- **§2.6 Session mortality and best-checkpointing (0.15p)**: training outlives sessions; without best-checkpoint saving (PR #3029), final-epoch metrics underreport true minima. Acknowledge as a named limitation and the reason for the T=0 sprint merge.
- **Figure 1 (mandatory)**: system diagram — cluster, Advisor pod, N Student pods, GitHub PR store (state arrow), W&B run store (metrics arrow), human intervention channel. Must make the "state in PR+W&B" claim legible at a glance.

### §3. Experiments & Results — 1.5 pages
- **§3.1 Setup (0.2p)**: three benchmarks, single Transolver (ICML 2024) backbone across all. No manual hyperparameter tuning by authors — every hyperparameter is an agent decision. Benchmark splits: AirfRANS `full` task, DrivAerML public 400/34/50 split, TandemFoilSet kagent-v2 parity split + paper-calibration split. Always report the four surface-pressure metrics together; prefer p_tan over p_in if only one is shown.
- **§3.2 AirfRANS (0.3p)**: Surf MSE best = 0.00300 (run `3e0ce368`) vs. published Transolver surface reference ~0.0043 — this is the abstract's "outperforms the reported Transolver baseline on surface-MSE" claim, cleanly supported. State Vol MSE = 0.00764 vs. SpiderSolver 0.0017 honestly — the abstract only makes the surface claim, so the volume gap is disclosed without contradicting it.
- **§3.3 TandemFoilSet (0.3p)**: two rows — (a) internal parity split: test MAE 33.88 → 24.58 (≈27% autonomous improvement); (b) **paper-calibration split** (`tandemfoil_paper/` field_mse): citation-clean test-row with W&B run ID, pre-sprint sprint deliverable. This row is the abstract's "competitive with the normalized full-field MSE reported in the TandemFoilSet paper benchmark" claim — it must exist, be clean, and name its run.
- **§3.4 DrivAerML (0.3p)**: the abstract says "preliminary runs show surface pressure relative-L2 nearing reported LES references" — use that latitude honestly. Lead with the **best-checkpoint val** improvement (currently ~4.0%, targeting ≤3.99% in sprint) vs. the 3.71–3.82% reference band — this is the "nearing" evidence. Disclose test (post-#3029) in the same paragraph without framing as a contradiction: "preliminary" implies val-stage and acknowledges test is not yet closed. Every number names a W&B run ID + the 400/34/50 split contract.
- **§3.5 Cross-benchmark transfer + ERA effects (0.2p)**: the Advisor assigns cross-dataset experiments (e.g., PR #3021–#3041 wave) because state is a shared ledger. Two normalization-regime transitions (ERA A: PR #1935 Surface Refinement; ERA B: PR #2054 Asinh) produced discontinuous metric drops correctly propagated as new baselines.
- **§3.6 Harness-level metrics (0.2p)** — Table 1: ≥8000 W&B runs total at submission (substantiates abstract); PRs opened / merged / aborted; median wall-clock per merged PR; approximate $/PR; concurrent agents; unattended-operation windows (including the ≥72h documented sprint). Compare summary stats to Agent Laboratory/Sakana where available. If $/PR is too uncertain to cite cleanly, use a range.
- **Figure 2 (real artifact, mandatory)**: side-by-side — left shows an anonymized PR conversation (body hypothesis + student result comment + advisor merge); right shows the corresponding W&B run panel. Caption names the exact PR# and W&B run ID. This defuses the "vapourware" accusation and demonstrates the "experiment ledger queryable by agents, researchers, and the harness itself" claim.
- **Figure 3**: learning-curve / leaderboard plot — AirfRANS surface-MSE by PR-number or wall-clock, with ERA boundaries annotated and published reference lines. Alternative: a 3-panel bar chart across the three benchmarks with reference lines.

### §4. Discussion — 0.8 page
- **§4.1 What worked (0.2p)**: PR+W&B state survives pod restarts and session deaths losslessly; skills architecture made the Advisor loop modular; the ledger IS a scientific artifact (releasable).
- **§4.2 Failure-mode subsection (0.4p)** — three named categories with phase-level data from `analysis/AGENT_MISTAKES_REPORT.md` (241 sessions, P4/P5/P6):
  1. **Tool misuse (dominant, not fixed by prompting)**: 3.52 → 4.68 → 4.58 misuses/session across phases; a base-model property, motivating idempotent skill design.
  2. **Session mortality**: training outlives sessions; `poll-for-work` mitigated delegation loss; PR #3029 addresses the checkpointing root cause.
  3. **State/accounting drift**: the canonical state source must be the PR ledger itself, not derived summary documents, which silently drift.
  Additional named modes (brief mention): PR-switching after auto-compact; advisor spawning W&B verification subagents without awaiting results; premature PR closure.
- **§4.3 Limitations (0.2p)**: single problem family; reward-hacking risk on surrogate metrics (mitigated by four-metric reporting, not eliminated); compute/API cost; drafted with agent assistance; results not independently replicated.

### §5. Conclusion — 0.45 page
2–3 sentences restating the core claim (PR+W&B grounding as the architectural move that carries the paper), the empirical summary (AirfRANS surface clear win; TandemFoil +27%; DrivAerML gap closing but not closed), and the honest lesson (design for tool misuse and session mortality, not against them). Point at the open-source release and the ledger artifact. No new claims.

### References + Appendix (unlimited)
- Full `CLAUDE-ADVISOR.md`, `CLAUDE-STUDENT.md`, `researcher-agent.md` prompt files.
- PR-indexed ledger CSV (PR#, title, status, W&B run IDs, key metric delta) — the artifact `analysis/appendix_experiment_ledger.md` converted to a compact table.
- Per-benchmark result table with run-ID provenance and split-contract file references.
- Failure-mode full taxonomy table (counts per phase, per category).
- Two redacted real PR transcripts (one success, one failure).
- System YAML (k8s manifests).

---

## Writing order (cold-start sequence)

Write sections **against real numbers first**, framing last. This prevents the "optimistic intro walked back by results" failure.

1. §3 Experiments (commits to exact claims and gaps with provenance — maps directly to the abstract's three benchmark claims)
2. §4 Discussion (interprets while §3 is fresh)
3. §2 System Design (with hindsight on which design choices mattered)
4. §1 Introduction (contributions bullets anchored to the abstract's contributions)
5. Figures & tables (Fig 1 + Fig 2 + Fig 3 + Table 1) — draft captions alongside each section; finalize art after §3 numbers are frozen
6. Appendix assembly + References
7. Anonymization pass (last, on a dedicated `icml2026-anonymous` branch)

The abstract is **not** in the writing order — it is fixed.

---

## 48–72h substantiation sprint

**T=0 is the moment the clock starts (human-picked, log timestamp).**

### Human-intervention policy (defensible rule)
During unattended periods the human may:
- Read-only monitor W&B, GitHub, pod logs.
- Restart a crashed/zero-output pod (log: timestamp + pod name). Watchdog, not steering.
- Post at most **two** GitHub issues (T=0 and T+48h) with explicit steering — both disclosed in the paper as "operator intervention log". Bounded oversight is an honest claim; zero-touch is not.

The human may NOT: change training configs, merge research PRs, or post additional steering issues. The T=0 merge of infrastructure PR #3029 is pre-sprint, stated as "infrastructure landed at T=0 of the reported window".

### Day-by-day
- **T=0 → T+2h (pre-flight, human-allowed)**: Merge PR #3029 (best-checkpoint saving); read its diff, fix any test failure, merge. Post T=0 `urgent` issue directing the Advisor to (a) close stale old-wave PRs, (b) concentrate spare GPU on DrivAerML objective/sampling/stability lanes, (c) ensure best-checkpoint test numbers are reported. Snapshot `fetch_experiments.py` as `ledger_T0.json`. Start the clock.
- **T+2h → T+24h (unattended)**: harness runs. Advisor acts on T=0 issue, Students report best-checkpoint numbers for the first time.
- **T+24h → T+26h (read-only check-in)**: inspect DrivAerML best-checkpoint progress; verify stale PRs closed; at most 3 pod restarts if any pod has zero W&B points for >4h.
- **T+26h → T+48h (unattended)**: the scientific question resolves — does DrivAerML test follow val from ~4.0% downward with best-checkpoint saving?
- **T+48h → T+54h (human-allowed)**: snapshot ledger, check best-checkpoint DrivAerML; post second (final) `urgent` issue if needed (e.g., "confirm this test number; run two confirmation repeats"). Decide whether to extend to T+72h.
- **T+54h → T+72h (unattended)**: run-count accumulation + confirmation repeats.
- **T+72h → T+78h (artifact capture)**: final ledger snapshot (`ledger_T72.json`); W&B panel screenshots per benchmark; `git log --oneline origin/radford --since=<T0>` export (proves autonomous merges during sprint); GitHub issue timestamp log.
- **T+78h → submission**: paper drafting in the order above; anonymization; PDF compile.

### Contingency — what to do if sprint underdelivers (abstract does not change)

The abstract is locked. The contingencies are about what the BODY of the paper says inside the rhetorical space the abstract already provides, and about doubling-down on the sprint if needed. Never soften abstract language in the body.

- **If run count < 8,000 at T+72h**: extend the sprint and freeze the submission-ledger snapshot only once the count crosses 8,000. The abstract says "8000+" — the appendix table must prove it. If the cluster cannot deliver in the remaining window, diagnose the root cause (stalled pods, context loss) and fix it; do not cite a smaller number.
- **If DrivAerML best-checkpoint test stays above 5% but val reaches ≤4.0%**: §3.4 body leads with val as the "preliminary … nearing" evidence (which the abstract already flags as "preliminary"), reports test in the same paragraph for honesty, and attributes the val→test gap to checkpoint-policy history prior to PR #3029. This stays inside the abstract's promise because "preliminary" and "nearing" are the exact hedges present.
- **If DrivAerML val does not improve at all by T+48h**: post the second `urgent` issue instructing the Advisor to close lanes that are not moving val and concentrate GPUs on objective/sampling changes. Extend clock to T+72h+. This is substantiation work, not abstract rework.
- **If TandemFoil paper-calibration row is not citation-clean**: this is the single hardest requirement. If the provenance cannot be cleaned within the sprint, the human (not the Advisor) re-runs the winning config on the paper-calibration split with best-checkpoint enabled, captures the run ID, and records the result. This is allowed because the abstract's claim must stand.
- **If the 72h clock is interrupted**: restart the clock. The abstract says "up to 72 hours unsupervised" — at least one unattended window of that duration must appear in the appendix sprint log.

### Risk register (top 6, each with pre-committed mitigation)
1. **PR #3029 not mergeable at T=0** → human fixes the failing test in the T=0 window (infrastructure fix, not research); do not start the clock until it is merged.
2. **DrivAerML val does not close to ≤4.0%** → post second `urgent` issue redirecting GPUs to objective/sampling/stability lanes, extend clock.
3. **Run count below 8,000 at T+72h** → extend until 8,000+ is true; do not submit the ledger snapshot before that.
4. **Auto-compact / context loss re-opens stale PRs** → T=0 issue explicitly lists stale PRs to close; re-anchors the Advisor on next cycle.
5. **AirfRANS vol MSE stays at 0.00764** (expected) → abstract only makes the surface claim; volume gap stated honestly in §3.2 without contradicting the abstract.
6. **TandemFoil paper-calibration run not citation-clean** → human re-runs winning config on that split with best-checkpoint enabled; captures the clean row. This is the one sprint deliverable with no acceptable fallback.

---

## Anonymization checklist (mandatory — double-blind)

Work on branch `icml2026-anonymous` cut from `radford`. Verify every item; the submission fails if any identifier leaks.

1. `grep -rl "CoreWeave, Inc." .` → replace with `Anonymous Authors` (SPDX headers).
2. `grep -rn "wandb/senpai\|github.com/wandb/senpai" .` → redact in paper and committed configs to `<anonymous>/senpai`.
3. Container image `ghcr.io/wandb/senpai:latest` → `<anonymous-registry>/senpai:latest` in `k8s/*.yaml`, `launch.py`, `senpai.yaml`, `README.md`.
4. W&B project `senpai-v1` in paper figures/captions → `<anonymous-project>`.
5. W&B entity / org any committed example → `<anonymous-entity>`.
6. Author email `mmcguire@coreweave.com` → anonymous placeholder in paper metadata & acknowledgments.
7. Pod names (`frieren`, `kakashi`, etc.) stay in code (non-identifying) but referred to as "worker agents" in paper prose.
8. W&B run IDs in paper are fine (`nrn0q3ct`, `qx7z7if3`, etc.) — not identifying once project/entity are anonymized.
9. PR numbers fine — not identifying once the repo URL is anonymized.
10. CoreWeave-specific K8s API `agentic.coreweave.com/v1alpha1` → redact/genericize in figures; keep YAML in supplementary with a note.

Verification: `grep -r "CoreWeave\|mmcguire\|coreweave\.com\|wandb/senpai\|ghcr\.io/wandb" .` must return zero matches inside the submission artifact.

---

## Ten things not to write (pitfalls specific to this paper)

1. "Fully autonomous" — use "autonomous under human PR review" or specify the autonomous steps.
2. Claiming AirfRANS is a full benchmark win — surface is a win; volume is an open gap. State both.
3. Describing DrivAerML as "converging" or "nearly solved" — state the exact gap.
4. Comparing metrics across normalization eras (A/B) without labeling.
5. Citing a val number as if it were test (DrivAerML val 4.0% ≠ test 6.24%; AirfRANS val ≠ test).
6. Omitting PR #3029 as a stated limitation.
7. Writing the failure-modes subsection as if problems were solved — tool-misuse rate did NOT decrease across phases; say so.
8. Presenting SENPAI as a general-purpose scientist — it is a specialist harness for CFD surrogate improvement.
9. More than three datasets in main text unless each has citation-clean provenance.
10. Sacrificing the PR-as-state explanation (§2.2) to save space — every other cut is preferable. This is the paper's contribution.

Also avoid LLM-voice prose: "leveraging", "in the realm of", "paradigm shift", "unprecedented", "it is worth noting that".

---

## Critical files to read/modify

**Read for content**:
- `papers/ABSTRACT_2026-02-22.md` — current abstract (rewrite to matched claims)
- `system_instructions/CLAUDE-ADVISOR.md`, `CLAUDE-STUDENT.md` — role prompts (appendix)
- `.claude/agents/researcher-agent.md` — literature-search subagent (§2.4 + appendix)
- `analysis/AGENT_MISTAKES_REPORT.md` — the 241-session taxonomy (§4.2)
- `analysis/OUTER_LOOP_REPORT.md` — phase timeline, PR #1965 skills pivot
- `analysis/appendix_experiment_ledger.md` — 2,395 PRs / 7,223 runs ledger (Table 1 + appendix)
- `analysis/STATUS_2026-04-22-1745_radford_benchmark_contract_status_refresh.md` — live benchmark numbers
- `analysis/STATUS_2026-04-22-0923_radford_live_status_after_cross_dataset_wave.md` — per-benchmark W&B IDs
- `analysis/AIRFRANS_BENCHMARK.md`, `DRIVAERML_BENCHMARK.md`, `TANDEMFOILSET_BENCHMARK.md` — split contracts & reference numbers
- `system_instructions/radford_icml2026_airfrans_tandemfoil.md` — sprint operational contract
- `BASELINE.md` on `radford` — canonical current-best (access via `git show radford:BASELINE.md`)

**Modify**:
- Create `papers/icml2026/main.tex` (or chosen LaTeX skeleton) — new.
- Create `papers/icml2026/figures/` — system diagram (Fig 1), real-artifact composite (Fig 2), leaderboard plot (Fig 3).
- Cut the `icml2026-anonymous` branch at the end for submission.

**Reuse (do not re-author)**:
- `fetch_experiments.py` — use as-is for T=0 and T+72h ledger snapshots.
- `tools/hivemind_senpai/` — for session-log exports if needed.
- Existing per-benchmark BENCHMARK.md files — canonical contract descriptions; paraphrase into §3.1 setup.

---

## Verification (pre-submission)

End-to-end checks to run before OpenReview upload:

1. **Abstract unchanged**: diff `papers/ABSTRACT_2026-02-22.md` against the abstract in the submitted PDF — must be byte-identical.
2. **Claim-to-evidence matrix filled**: each row in the Context's claim-to-evidence table has a concrete §/figure/appendix-row pointer and, where numerical, a W&B run ID + split contract.
3. **Numbers pass**: search LaTeX for every numerical claim; verify each against `BASELINE.md` on `radford`, the T+72h ledger snapshot, and the per-benchmark STATUS memo.
4. **Metric-labeling pass**: every occurrence of "val" vs "test" is correctly labeled; DrivAerML §3.4 labels the "nearing" evidence as preliminary-val and discloses test in the same paragraph.
5. **Anonymization pass**: `grep -r "CoreWeave\|mmcguire\|coreweave\.com\|wandb/senpai\|ghcr\.io/wandb" papers/icml2026/ k8s/ target/icml2026/` returns zero matches in the submission bundle.
6. **Figure readability**: Fig 1 alone communicates the PR+W&B state contribution; Fig 2 shows real artifacts with PR# + run ID in caption.
7. **Related-work coverage**: Sakana v1/v2, Agent Laboratory, AgentRxiv, Zochi/Tempest, Kosmos (concurrent), ChemCrow, Coscientist, FunSearch, ScienceAgentBench, MLE-Bench, ML4CFD, AirfRANS, DrivAerML all cited.
8. **Human-in-the-loop paragraph** is present in §2.5 and enumerates the sprint's actual operator touches (≤2 `urgent` issues + pod-restart watchdog) — backs the abstract's "minor course corrections".
9. **Page count**: 5 pages main text ± 0.1 page; references and appendix unlimited.
10. **Sprint evidence appendix**: T=0 and T+72h `fetch_experiments.py` snapshots, W&B panel screenshots, `git log` export covering the unattended window, GitHub issue timestamp log. ≥72h wall-clock of unattended operation is provable from these artifacts.
