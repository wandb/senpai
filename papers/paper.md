SENPAI: Self-ExperimentatioN for Physical AI - an observability-based research harness


SENPAI (Self-Experimentation for Physical AI) is an observability-first research harness in which multi-agent state is grounded in pull requests and structured experiment logs rather than agent memory and scratchpads. Training CFD surrogates encompasses standard ML training considerations - architecture, optimizer, schedule, and loss design - but also physics-aware considerations such as boundary conditions, symmetries, and rollout stability. The harness orchestration is deliberately thin: an Advisor agent proposes hypotheses as GitHub pull requests; Student agents check out each PR, execute training based on the hypothesis, and write results back to the PR. Every hypothesis is grounded by a literature-search sub-agent callable by the advisor or student. A key contribution is the observability-first memory and orchestration that goes beyond scratchpad files and in-context memory; SENPAI also grounds state in the artifacts ML researchers already use - PRs, git history, and an experiment logger (Weights & Biases) - producing an experiment ledger queryable by agents, researchers, and the harness itself. This design enables the system to learn from past experiments, reflect on its own trajectories, and keeps humans meaningfully in the research loop - properties that are difficult to achieve simultaneously in prior agentic research systems. We run SENPAI on a Transolver model jointly across three aerodynamic benchmarks: TandemFoilSet, AirfRANS, and DrivAerML. On the AirfRANS benchmark the final recipe outperforms the reported Transolver baseline on surface-MSE; on TandemFoilSet dataset it is competitive with the normalized full-field MSE reported in the TandemFoilSet paper benchmark.; on DrivAerML, preliminary runs show surface pressure relative-L2 nearing reported Large Eddy Simulation (LES) references for the Transolver model. Taken together, our results demonstrate that AI scientists function best as near-autonomous co-workers that can run up to 72 hours unsupervised, with only minor course corrections required after these durations. We release the harness, the PR-indexed experiment ledger, and a failure-mode analysis. 


—--------------------

1. Introduction

CFD surrogates promise a practical acceleration layer for engineering design: inexpensive learned models can screen early-stage geometries as design candidates, explore diverse operating regimes, and focus high-fidelity CFD and physical testing for later decisions. The technology has advanced quickly: modern surrogates now handle larger meshes, irregular geometries, tighter accuracy requirements, and smaller training sets than were feasible only a few years ago [Zhou, Hang, et al. "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries." arXiv preprint arXiv:2602.04940 (2026).]. In principle, this makes a hybrid workflow attractive, whereby ML surrogates complement established simulation pipelines.

In practice, adoption remains difficult for many small/mid-size engineering organizations. Training a useful CFD surrogate is not a routine application of standard ML: it requires choices about architecture, optimization, normalization, boundary conditions, symmetries, rollout stability, and benchmark-specific evaluation [Wang, Haixin, et al. "Recent advances on machine learning for computational fluid dynamics: A survey." arXiv preprint arXiv:2408.12171 (2024)]. The best models, and usage thereof, are typically produced by teams that combine ML expertise with domain knowledge, for example aerodynamics or fluid mechanics [Hodges, Justin. Approaching machine learning problems in computational fluid dynamics and computer aided engineering applications: A Monograph for Beginners. Justin Hodges, 2024.
]. Large enterprises can build such teams internally, but most engineering groups are already constrained by safety-critical design processes, shortening development cycles, and expanding technical requirements. SENPAI targets this gap: it provides an autonomous research harness that can explore, document, and refine problem-specific surrogate recipes while keeping human engineers in the review loop.

Recent autonomous-research systems show that LLM agents can search literature, propose hypotheses, edit code, run experiments, and produce research reports: The AI Scientist generates ideas, implements experiments, writes papers, and runs automated review [Lu et al., 2024; SakanaAI/AI-Scientist], AI Scientist-v2 extends this with agentic tree search and workshop-level paper generation [Yamada et al., 2025; SakanaAI/AI-Scientist-v2], Agent Laboratory structures literature review, experimentation, and report writing around a human-provided idea [Schmidgall et al., 2025; AgentLaboratory], and Jr. AI Scientist iteratively improves a baseline paper using modern coding agents [Miyai et al., 2026]. Other systems emphasize collaboration or external tooling: ResearchAgent generates and revises research ideas over scientific-literature graphs [Baek et al., 2025], AgentRxiv lets agent laboratories upload and retrieve generated reports from a shared preprint server [Schmidgall & Moor, 2025], ChemCrow and Coscientist connect LLM planners to chemistry tools, web/documentation search, code execution, and automated or robotic laboratory workflows [Bran et al., 2024; Boiko et al., 2023], and FunSearch couples an LLM generator to a verifiable evaluator and a scored program database [Romera-Paredes et al., 2024]. These systems establish the value of tool use, execution, and persistent artifacts in autonomous research, but their state is typically organized as agent-owned files, generated papers, report stores, search populations, or conversation/workflow traces rather than as the ordinary audit trail of a multi-experiment ML research programme.

The closest recent systems to SENPAI also recognize that long-horizon agents need to store state outside the model context, but they store that state in different substrates: Kosmos maintains an internal structured memory of literature-search and data-analysis summaries to choose later tasks, and cites final report claims with papers or generated Jupyter notebooks [Mitchener et al., 2025]; AiScientist uses a permission-scoped File-as-Bus workspace so agents re-ground on file-based analyses, plans, code, logs, and experimental evidence [Chen et al., 2026]; and grounded autonomous research externalizes state as rerunnable simulation artifacts and comparison results within a single-paper reproduction loop [Huang, 2026]. SENPAI takes a complementary systems position: the authoritative state is the substrate human ML teams already inspect, namely pull requests for code review and discussion [GitHub Docs, 2026], git history for ordered code provenance [Chacon & Straub, 2014], and an experiment logger (W&B) for runs for metrics, hyperparameters, system metrics, and checkpoints [Biewald, 2020; Weights & Biases, 2026]. This makes each hypothesis, code diff, training result, review decision, failed run, and baseline update both machine-queryable and human-reviewable through standard tools, rather than hidden inside agent memory or an agent-specific workspace.

SENPAI is a deliberately thin research loop: a new research program git branch is first created, then a lightweight advisor agent reads the current experiment state from git (GitHub) and the experiment logger (Weights & Biases), proposes a literature-grounded hypothesis as a pull request, and assigns it to one of N GPU-backed student agents, with the students name as a label. Each student polls for its labeled PR, checks out the branch, modifies the training loop for the requested hypothesis, runs the experiment, and writes metrics, commands used, and analysis back into the PR as a comment. The student marks the PR as ready for review after which the advisor either merges the result, requests a revision, or closes the line of inquiry. Both roles can call a shared literature-search sub-agent to ground hypotheses and implementation choices, while humans stay in the loop through monitoring past and current experiments via GitHub PRs. If the advisor needs steering, a GitHub issue is opened and the advisor is tagged with a label. Scratchpad files are utilised as rough noisy and research logs and a brief, high-level research state file is maintained by the advisor. Ground-truth experiment state is externalized in PRs, git history, and W&B runs. This queryable ground-truth state has been found to make the system more robust to drift while updating large local scratchpad files, agent context compaction and file-loss due to restarts, and leaves behind a durable experiment ledger as a by-product.

Over the course of the research programme, SENPAI created 2,700+ PRs and recorded 10,700+ W&B runs while deploying at peak 59 distinct Student agents. 

Metric
Value
Total agent-generated PRs
2,700+
PRs with at least one completed run
2,300+
W&B runs recorded
10,700+
Peak concurrent Student agents deployed
59
Peak concurrent training runs
347
Merged run-bearing PR latency (median / p75 / p90)
48 min / 3.4 h / 6.6 h
Median Student input tokens per PR
1,421,948
Median Student output tokens per PR
6,012
Median Advisor input tokens per PR
187,453
Median Advisor output tokens per PR
1,621




Contributions 
[Observability Harness contribution]
SENPAI contributes a new observability-first harness for semi-autonomous research. By grounding state in structured, queryable units of data, GitHub PRs and commits and W&B runs, we produce and experiment ledger that is both researcher and agent-friendly. This structured logging of results enables the research harness to reproducibly query for past experiments, check the status of current experiments and identify idle resources. It also enables researchers to view the current experiments in flight as well as inspect successful and unsuccessful experiments. The centralisation of these results also enables researchers to do on-the-fly analysis of the current research progress which can then inform if the harness needs additional steering or guidance.

[PAI contribution]
SENPAI autonomously improves multiple CFD surrogate benchmarks starting from a basic Transolver model: Under the directly comparable aggregate AirfRANS full-regime Surface MSE protocol, SENPAI generated a solution with the lowest surface error we found (7.77e-4) however its volume MSE remains higher than the strongest reported volume results, at 3.33e-3 versus 1.1e-3 for LRSA (Yang et al., 2026) and 1.7e-3 for SpiderSolver (Qi et al., 2025). On DrivAerML, surface-pressure relative-L2 reaches 4.50%, improving on the reported Transolver DrivAerML baseline (4.81%) while still trailing Transolver++ (4.12%), AB-UPT (3.82%), and Transolver-3 (3.71%); On TandemFoilSet , SENPAI reaches normalized full-field test MSE of 1.78e-3, below the TandemFoilSet paper’s best Experiment-4 MSEs (0.10–0.36) across the Cruise Random and Race Car tasks (Lim et al., 2026), and on TandemFoilSet-Balanced (McGuire and Capelle, 2026), SENPAI reaches an average test surface-pressure MAE of 22.87 across the 4 sub-metics.


Failure-mode taxonomy 

Finally, SENPAI contributes an empirical failure-mode analysis of long-running agentic ML research systems. Because every agent trajectory, tool call, PR transition, W&B run, monitor event, and context compaction was logged, we can audit the system as an experimental object rather than describe failures anecdotally. We audit a 24-hour fleet trace containing 53,022 real Claude requests and 5.24B tokens across one Advisor and 57 token-emitting Student agents. The dominant failure case was monitor-driven context bloat: 28,247 model-facing monitor events from Claude Code’s Monitor tool generated 28,246 direct model responses and consumed 3.25B cache-inclusive tokens as brief training-log updates were passed to the full, long-lived agent context. Tool-use errors accounted for 892 of 25,365 tool uses (3.5%), and Student agents spent 3.8% of their usage records checking whether a PR had been assigned to them or whether a human had opened an issue requiring attention.


Routine coordination checks were a small part of the total workload: Student agents spent only 3.8% of their usage records checking whether a PR had been assigned to them or whether a human had opened an issue requiring attention.


These measurements inform the design lessons of this semi-autonomous system - ML agents need bounded monitoring, durable external state, and executable handoff protocols more than additional prompt prose. Tool errors were usually interface-contract failures - stale GitHub commands, incorrect paths, blocked wait patterns, brittle log probes, missing branch preconditions — rather than failures of scientific reasoning. Training crashes were a separate category again: often useful experimental evidence, provided the result was captured. SENPAI’s observability-first ledger made these distinctions measurable and actionable, leading to concrete harness changes including GitHub helpers, bounded monitoring design, and mandatory best-checkpoint capture. We therefore release not only the harness and experiment ledger, but also a failure-mode taxonomy for future autonomous-research systems.


2. Methodology: System Design
State lives in pull requests and W&B runs, not in agent memory or scratchpad files. Every other design choice follows.

Figure 1 — SENPAI deployment architecture. A thin harness deploys one Advisor and N Student pods; both roles can call a shared `researcher-agent`. All programme state lives outside the cluster in the experiment ledger (GitHub PRs, git history, W&B runs), through which humans steer the system.

SENPAI organises semi-autonomous CFD-surrogate research around artifacts human ML teams already produce — pull requests (PR), Git commits, and experiment-tracker runs — and two agent roles that read and write them (Figure 1). An Advisor runs a research subagent, proposes hypotheses by opening draft PRs; one of N Student agents claims each PR, implements the hypothesis, runs training, and writes results back as PR comments linked to W&B runs. A hypothesis's full trajectory — intent, metrics, review — is therefore a standard GitHub PR log rather than an agent-internal trace. 

The rest of this section describes the ledger (§2.1), the harness loop and its observability daemons (§2.2), and the human-in-the-loop channels (§2.3). Role prompts, the literature sub-agent design, the skill catalogue, and the Advisor's hypothesis-selection and merge rules are deferred to Appendix A.
2.1 SENPAI’s research loop

Each experiment follows six steps:

0. Research. The Advisor decides whether or not to run the research sub-agent to design the next batch of experiments(s)
1. Assignment. The Advisor drafts a PR containing a hypothesis, the exact training code guidelines to run against the current baseline, and the metrics to improve. A PR label names the target Student; 
2. Implementation and training. The Student polls for PRs assigned to it, checks out the PR branch once it discovers one, implements the changes and launches training experiments, with logging performed by W&B.
3. Reporting On experiment completion the Student posts PR comment with a summary of the experiment results and changes the PR status from wip to review in order to request a review from the Advisor.
4. Review The Advisor polls for PRs that are ready for review and reviews the work and results against the proposed hypothesis and validation metric (§2.3). From there it decides whether to merge the change, comment and ask for follow up work, or close the PR.
5. Baseline advance If a PR is merged then the Advisor also updates the current baseline metric on the Advisor branch, against which every subsequent PR is compared.

2.2 The Experiment Ledger - Pull requests and Experiment Logger 

Every experiment is a pull request and a set of W&B logs. The PR body holds the hypothesis, the experiment instructions, and the baseline metrics to improve upon; the comments thread carries the Student's results and any Advisor review notes. The Experiment logger (W&B) carries the quantitative side: each run documents the training config and results metric. The two tools together form an observable experiment ledger - researcher-inspectable through the GitHub and W&B UIs, and machine-queryable through the CLI. When researchers want to investigate an individual experiment they can open the GitHub PR, read the hypothesis, setup and results and open the associated W&B training run to view the configs and metrics used. The ease of research observability using tools researchers use daily lowers the barriers for researchers to understand what is going on in the system - a key feature for autoresearch systems that can produce hundreds or thousands of experiments during a research program. 

By using an experiment logger with agent observability tooling and mature CLIs, researches’ agents can also query for SENPAI agent trajectories as well as experiment results and status’ and quickly generate aggregate reports for researchers, again providing researchers with more visibility and understanding into what these autoresearch systems are doing. 

Finally, this structured, accessible logging also means that harness issues or failures such as pod restarts, context compactions, and fresh-container boots are less harmful for agent state and a research program can quickly recover even if agent context and local experiment scratchpad files are lost.

In contrast to AgentRxiv [Schmidgall et al., 2025], which shares papers between agents, SENPAI shares experiments within an existing code-review substrate.

In addition to the core experiment ledger, the Advisor also keeps a small set of focus-aid markdown files for research-state summary, baseline metric tracking, experiments scratchpad, dataset notes. These are not authoritative state: an earlier iteration of the harness treated them as durable, which produced silent drift when summaries disagreed with the ledger (§4.3). An anonymised end-to-end PR transcript is reproduced in Appendix E.

2.3 SENPAI’s harness loop

SENPAI deploys as a small set of Kubernetes workloads: one Advisor CPU pod, N Student GPU pods, and a shared persistent data volume for datasets and checkpoints. The harness image is stable across problems; a task repository is swapped in per programme (bring-your-own-repo), so every agent commit, branch, and PR lives in a self-contained target repository that a researcher can inspect independent of the harness. 

The outer harness loop is a shell wrapper, not an LLM. The Advisor entrypoint wraps Claude Code in a `while true` iteration, in the spirit of Huntley's "Ralph loop" pattern. We use programmatic triage to determine if Students are idle as well as new PRs, Issues and comments in order to remove the cost of using an LLM to do polling. We note elsewhere in this paper that despite this we still encountered high token usage via monitoring that we have subsequently addressed. The Claude Code agent is now encouraged to exit and the outer programmatic loop is used to identify updates that require the Claude Code session to be continued with the information from the update.

```
# Advisor outer loop 
# entrypoint_advisor.sh
iteration, last_check = 0, None
while True:
    iteration += 1
    review_ready   = list_review_ready_prs(branch, since=last_check)
    human_issues   = list_new_issues(branch, since=last_check)
    idle_students  = list_idle_students(student_names, branch)

    # Out-of-LLM triage: skip the agent entirely when nothing is actionable
    if iteration > 1 and not (review_ready or human_issues or idle_students):
        sleep(tick); continue

    triage_message = render_triage_message(review_ready, human_issues, idle_students)
    if iteration == 1:
        exit_code = run_agent(role_prompt + triage_message)              # fresh session
    else:
        exit_code = run_agent(heartbeat + triage_message, resume=True)   # resume prior session

    # watermark-incremental polling: advance only on success, so failures retry
    if exit_code == 0:
        last_check = max_updated_at(review_ready, human_issues)
    sleep(tick)
```

2.4 Human-in-the-loop

Researchers can steer and interact with SENPAI through GitHub. The Advisor and Students poll for GitHub Issues regularly - this is the primary mechanism to ask questions of the Advisor or Students or provide it a new research direction or idea to pursue. This is a deliberately low-bandwidth communication mechanism as SENPAI is designed to be semi-autonomous with only infrequent guidance or interruption from researchers. Researchers can also comment on individual experiment PRs in order to guide how an individual experiment is running, although this was a much less frequent interaction mechanism. During multi-day sessions, human actions were limited to i) research direction guidance corrections approximately once per day ii) as well as pod restarts to release stuck Claude Code instances; no training file or config edits were performed by researchers while the system was running (Appendix F).



3. Experiments and Results
3.1 Experiment setup

Claude Code (v2.1.85 to v2.1.117) was run in headless mode as the agent harness for both the Advisor and Students. The Claude Opus 4.6 LLM drove early experimentation whilst Claude Opus 4.7 drove the later experiments. Advisor and Student instructions used can be found in Appendix A. Student pods were scheduled on NVIDIA RTX 6000 Blackwell nodes (96 GB VRAM per GPU, 8 GPUs per Student node); we tested up to N = 59 concurrent Students, with peak concurrent training runs of 347. The harness experiments and iteration was run for 1 month and in that time 29 billion tokens were processed by Claude Code.

3.2 Data and Benchmarks

To demonstrate SENPAI’s CFD performance across multiple benchmarks we train and evaluate on three CFD surrogate datasets spanning 2D tandem-airfoil flow, 2D airfoil RANS, and 3D automotive aerodynamics: TandemFoilSet (Lim et al., 2026), AirfRANS (Bonnet et al., 2022), and DrivAerML (Ashton et al., 2024). 

TandemFoilSet: we use the paper’s original Experiment 4 partition as one of two TandemFoilSet datasets and measure denormalized surface pressure MAE. We generate a second TandemFoilSet dataset split, named TandemFoilSet-Balanced (CITE https://github.com/morganmcg1/tandemfoil2), with 1,499 training cases and four balanced test splits: single-foil in-distribution, two unseen-front-camber geometry holdouts (race-car and cruise), and a Reynolds-number holdout - an average of these 4 splits is taken as the primary test metric.

AirfRANS: we use the official full task with the last 10% of the official training list held out for validation, giving a 720/80/200 train/val/test split while keeping the official test set unchanged. We test on normalized targets on the official test split. 

DrivAerML: we use the public surface split on the available processed cases (400/34/50 train/val/test) and measure surface-pressure relative-L2 in percent, computed per case on unnormalized predictions and targets and then averaged over test cases.


3.2.1 AirfRANS: surface-MSE below published Transolver
< TO BE UPDATED>

The official AirfRANS `full` task benchmark is defined on the pair (surface-MSE, volume-MSE) evaluated on train-statistic-normalised targets [Bonnet et al. 2022]. Our best configuration achieves surface-MSE = 0.003 (W&B run `3e0ce368`, PR #2824), an improvement over the strongest published surface reference (SpiderSolver, 0.0043). Volume-MSE on the same run is 0.00764, ~4.5× SpiderSolver's 0.0017. We therefore report a surface result competitive with published references and do not claim a full-benchmark win; closing the volume gap is open work.

Across three independent seeds (PR #2831), surface-MSE distributes as 0.00333 / 0.00668 / 0.00857 (mean 0.0062) and volume-MSE as 0.00886 / 0.00901 / 0.01709 (mean 0.0117). Best-seed surface beats SpiderSolver; mean-seed does not. Seed-level volume remains above all published references.

Table 2 — AirfRANS full-task reference comparison (train-stat normalised, per-case mean, official test split; lower is better).

Method
Surf MSE
Vol MSE
Transolver [Wu et al. 2024]
0.0142
0.0037
GeoANF [Li et al. 2024]
0.0089
0.0062
SpiderSolver [NeurIPS 2025]
0.0043
0.0017
SENPAI, best run `3e0ce368`
0.003
0.00764
SENPAI, 3-seed mean
0.0062
0.0117


A later-cited Transolver pair (0.0080 / 0.0025) appears in follow-on work but is not the original published row; we cite the original following standard practice [details in Appendix B].

Winning recipe: 2-layer / 256-width / 4-head Transolver, AdamW (lr=7e-4), cosine schedule (T_max=5), grad-clip 1.0, weight decay 1e-2, EMA off, Fourier positional features. The Advisor arrived at the 2-layer depth autonomously through 4L/256d → 3L/256d (−46.6% surface-MSE) → 2L/256d (−16.4%); wider 2-layer variants (384d, 512d) diverged early.

3.2.2 TandemFoilSet: two benchmark contracts, reported separately
< TO BE UPDATED>

< RESULTS TABLE>

TandemFoilSet is evaluated in this paper under two distinct contracts — the paper-faithful Experiment-4 contract (normalised full-field MSE) and the packaged parity contract on the public `kagent` split family (denormalised surface-pressure MAE). They are not numerically interchangeable and we report them separately. Mixing the two under a single TandemFoilSet heading without relabelling the contracts would be scientifically misleading.

3.2.2a TandemFoilSet paper contract: preliminary

Following the TandemFoilSet paper's Experiment 4 [TandemFoilSet 2024], we evaluate normalised full-field MSE (`field_mse`) over the six task splits (cruise_random_uniform, cruise_random_{aoa, re, stagger, gap}_extrap, racecar_uniform). The harness-released contract lives in `target/icml2026/tandemfoil_paper/` with split-faithful train/test manifests; full provenance in Appendix B.

At the time of writing this lane remains immature. The best observed number is `test_primary/field_mse ≈ 0.151` on `cruise_random_uniform` on a non-converged development run, which sits between the paper-reported baseline (1.79±1.38) and best (0.10±0.13) on that task. We scope this benchmark as preliminary work and do not claim a competitive paper-contract result.

3.2.2b TandemFoilSet parity contract (kagent split): harness-driven improvement

The packaged parity target (`target/icml2026/tandemfoil/`) uses the public `kagent` v2 split — four balanced val/test tracks (single-in-dist, geom-camber-rc, geom-camber-cruise, re-rand) — and reports denormalised pressure-channel surface MAE (`surface_pressure_mae`, aggregated globally over valid surface nodes). This is an internal SENPAI benchmark, not a literature comparator; we report it to quantify harness-driven improvement over SENPAI's own prior best.

Test surface-pressure MAE improved from 33.88 (run `v6amjkh7`, PR #2810) to 24.58 (run `nrn0q3ct`, branch `robin/ema-warmup-tandem-0.999`) — a 27% reduction — entirely through PR-mediated Advisor–Student iteration, with zero manual intervention between the two numbers. The pathway is visible in the PR ledger. Early PRs stepped the learning rate down from 3e-4 to 1.25e-4 and added gradient clipping at 1.0; a subsequent EMA-warmup recipe family then crossed the 25.0 MAE threshold. Figure 2 overlays these milestones on the W&B loss curve.

3.2.3 DrivAerML: preliminary surface-pressure transfer result
< TO BE UPDATED>

DrivAerML is treated as a transfer benchmark in this paper: we apply the shared Transolver stack used for AirfRANS and TandemFoilSet, evaluate on the public 400-train / 34-val / 50-test split, and report mean per-case surface-pressure relative-L2 on unnormalised predictions (`surface_rel_l2_pct`). There is no single canonical public scalar in the DrivAerML literature, so we follow the AB-UPT and Transolver-3 reporting convention, which has converged on this metric, split family, and aggregation rule. We do not report a volume or multi-field comparison and scope this benchmark as surface-pressure transfer only, not a full multi-field result.

Our best-checkpoint numbers on the public split are val = 4.62% and test = 6.24%. The strongest directly-comparable published references are Transolver-3 (3.71%) and AB-UPT (3.82%). Our best-val result sits in the secondary-provenance band (Transolver at 4.81%, Transolver++ at 4.12%, both recovered from the AB-UPT / Transolver-3 comparison tables); our best-test is above that band. We therefore report DrivAerML as preliminary transfer work and do not claim a competitive benchmark result.

Table 3 — DrivAerML public-split surface-pressure relative-L2 (%). Mean per case on unnormalised predictions; lower is better.

Method
p_s rel-L2 (%)
Transolver-3 [2026]
3.71
AB-UPT [2025]
3.82
Transolver++ (secondary provenance)
4.12
Transolver (secondary provenance)
4.81
SENPAI, best val
4.62
SENPAI, best test
6.24



Figure x — real SENPAI artifact (deferred to Appendix E)

PR conversation (top): anonymised PR #X — Advisor hypothesis body, Student results comment with best-checkpoint metrics, Advisor merge decision
W&B panel (bottom): run `[run-id]` — surface-MSE curve annotated with PR-ledger events
- Substantiates abstract's "queryable ledger" claim

---

4. Discussion

4.1 What the PR+W&B grounding bought us

Lossless session boundaries — crashed pods, context-limit compactions, preempted Students all resumed by re-reading PR state on the next cycle
Human-interruptible autonomy — reviewers steer the Advisor by commenting on PRs, not by editing prompts or restarting services
Queryable retrospection — `list-experiments` answered cross-benchmark questions (e.g. "has any prior run combined EMA warmup with grad-clip on AirfRANS?") directly against the ledger, enabling the §3.5 transfer

4.1 Observable State as a Control Surface
The PR and W&B ledger made SENPAI useful as a research assistant rather than just an autonomous executor. By placing the core experiment ledger in systems already used by ML teams, the harness kept researchers close to the work without requiring them to supervise every experiment. Agents could run large numbers of experiments independently while researchers could still see, question, and redirect the broader research program via the tools they use daily.

This externalized state also reduced the fragility of this long-running system. When agents compacted, restarted, or lost local scratchpad context, the next cycle could reconstruct the experiment from the PR, git history, and W&B runs. 

Finally, the ledger turned a collection of individual experiments into a queryable research programme. Both researchers and agents could ask what had been tried, which failures were meaningful, which recipes transferred, and where progress had stalled. In this sense, SENPAI’s core contribution is not replacing the researcher, but increasing the bandwidth at which researchers can understand, guide, and accelerate autonomous experimentation.

4.2 kagent: parallel autonomous agents under a lighter harness

To contrast SENPAI's Advisor-mediated loop, we ran two cohorts under `kagent` — a lighter-weight harness that drops the Advisor/Student split and pull-request review logic in favour of a flat peer-competitive cohort and a scoring-driven leaderboard. Each kagent agent runs a *read-leaderboard → hypothesise → edit `train.py` → train → score → commit* loop with its own `EXPERIMENT_JOURNAL.md` as durable memory; agents see each other only through the shared leaderboard branch and their public commits — no pull-request review, no merges, no Advisor. We report these cohorts as comparators, not as benchmark claims. Every iteration the agents start by consulting the public leaderboard and are prompted to be competitive: 

> Your objective is to top the leaderboard. If you are stuck with a low scoring solution, don't be afraid to try radical changes, marginal improvements on your low scoring solution are not going to cut it!


**TandemFoilSet parity head-to-head (Appendix G.1).** An eight-agent Opus 4.7 cohort ran for 12 h (2026-04-23 → 2026-04-24) on the same dataset, split family, and primary metric as the SENPAI result in §3.2.2b. Head-to-head on denormalised pressure-channel surface MAE averaged over the four val/test tracks:

| Harness | Agents × GPUs | Scored runs | Best avg-surf-p MAE | Wall clock |
|---|---|---|---|---|
| SENPAI (PR-mediated) | 7 × 8 | 544 | **40.9** | 12h |
| kagent (flat cohort)  | 8 × 1 | 223 | **34.41** | 12 h |

SENPAI lands ~20% lower in absolute MAE; kagent reaches 3.5× over the Transolver starter in 12 wall-clock hours, with the decisive full-mesh-warm-start recipe rediscovered by three of the top four agents within a ~3 h window once one agent's leaderboard commits exposed it. The qualitative trade — review gating and indexed-into-the-ledger retrospection vs. raw peer-diffusion speed — is the subject of §4.3 and Appendix G.1. Leaderboard at [`apr23-leaderboard@6cfac55`](https://github.com/tcapelle/kagent/blob/6cfac55a619f3230a8549df3faabd058e2b402b6/leaderboard.md); per-agent journals are SHA-pinned in Appendix G.1.

**GRaM ICLR 2026 open-competition run (Appendix G.2).** As a second demonstration we pointed sixteen Opus 4.6 agents at an active external competition — GRaM ICLR 2026, 3D velocity-field prediction on F1 front-wing meshes (~100k points, five-step horizon) — for ~30 h on 2026-03-29/30. The winner scored 0.8526 mean L2 against a ~1.75 reference-MLP baseline; the strongest kagent checkpoint was packaged and submitted to the upstream competition as [PR #4](https://github.com/gram-competition/iclr-2026/pull/4) — a cold-start kagent-to-external-submission pipeline executed end-to-end in under two days.

4.3 Failure modes: where long-running research agents break

The recurring problems SENPAI encountered were harness engineering monitoring long-running jobs, recovering across compaction or restart, enforcing brittle tool interfaces, and keeping PR and W&B state consistent - not failures of experimental reasoning.

The dominant cost was monitor-driven context bloat. Students monitored for progress, errors, checkpoint detection, and completion. However, persistent `tail -f | grep` monitor event over training logs caused small log events to resume full long-lived agent sessions. In a 24-hour fleet trace, 615 Student monitor setups produced 28,247 model-facing monitor events and 28,246 direct responses, consuming 3.25B cache-inclusive tokens. The median response reloaded roughly 110k cached tokens while adding only 347 uncached input tokens. This is an architectural failure mode: training logs should be reduced by low-context processes that escalate only decision-relevant events such as new-best checkpoints, milestones, errors, completion, or timeout.

Tool-interface errors were smaller but systematic: 892 of 25,365 tool results, or 3.5%. The main causes were failed shell commands, GitHub scope errors, blocked wait patterns, wrong paths, failed pushes, oversized reads, and edit guards. A broader scan found 2,156 failure-like tool results, or 8.5%, but many were training outcomes such as NaNs, OOMs, tracebacks, killed jobs, or divergent metrics. This distinction matters: failed runs can be scientific evidence if they leave metrics and logs; failed control actions require narrower, idempotent tool contracts.

Compaction introduced another boundary risk. The trace contained 240 automatic compactions, with main Student contexts reduced from roughly 196k tokens to 3k-token continuation summaries. These summaries were sufficient for resumption, but unsafe as ground truth. Decisions had to be revalidated against PR labels, PR comments, W&B configuration, W&B history, and live run state.

The ledger mitigated these failures. Among 39 Student sessions with final-like monitor signals, 38 had explicit PR-comment result evidence and 33 had ready or status-review handoff evidence. Thus, even when sessions compacted, restarted, or delegated work, experiments often survived as inspectable PR/W&B artifacts.

The lesson is that long-running research agents need observable infrastructure more than additional prompt prose. Monitor bloat requires bounded observation; tool errors require executable helpers; training failures require mandatory result and checkpoint capture; state drift requires a single authoritative ledger, with summaries treated as caches. SENPAI did not eliminate failure, but it made failure measurable, attributable, and repairable.




4.4 Limitations

Students select from a fixed training entry point with flag-gated features; autonomous code generation for new architectures / loss functions is out of scope
"Near-autonomous" phrasing is literal: every result required an initial problem spec, a dataset, and downstream human PR review
Reward hacking on surrogate metrics is mitigated (not eliminated) by full-metric reporting
Compute + API cost were substantial; released ledger makes future systems' efficiency auditable
Results not independently replicated
Paper drafted with SENPAI's assistance; all prose reviewed and edited by authors

- Students operate on a shared `core/` library (architectures, datasets, features, optimizer, metric contracts) plus a per-target `train.py` entry point. Students can in principle modify any of these and frequently do (feature additions, loss reformulations, transform hooks); what remains out of scope is *de novo* architecture generation — invention of new model classes from a blank file — and cross-cutting refactors that would conflict with other live PRs.
- *Near-autonomous* is a literal description: every result required an initial problem specification, a dataset, and downstream human PR review. Our unattended windows measure *continuity between touchpoints*, not unsupervised research.
- Reward hacking on surrogate metrics is mitigated but not eliminated by full-metric reporting (surface + volume + primary alias, per-split breakdowns) and by the ledger's retroactive query capability; reviewers should audit the ledger rather than trust the summary scalars.
- Compute and API costs were substantial; the released ledger makes future systems' per-merged-PR efficiency auditable. See §3.3 for aggregate numbers.
- Results are not independently replicated at the time of writing.
- This paper was drafted with SENPAI-style AI assistance; all prose was reviewed and edited by the human authors in accordance with ICML 2026 LLM-use guidelines.

5. Conclusion

Core claim
Grounding multi-agent research state in with an obvervability focus via PRs and an experiment tracker (tools researchers already use) is sufficient to run a 3 week, ≥8,000-run semi-autonomous programme across three CFD-surrogate benchmarks, with researchers intervening through ordinary PR review + a bounded issue channel

Hard-won design lessons:
Tool misuse did not decrease under prompt hardening
Session mortality closed only by infrastructure fix (PR #3029 best-checkpoint saving)
Derived summary state drifted silently from the authoritative ledger
⇒ agentic research systems should be designed for these failure modes, not against them

Release commitment: 
harness, PR-indexed experiment ledger, failure-mode analysis all released as open source on <URL>

---

References

Updated paragraph sentence with `ml-intern` included:

```text
Long-running ML-engineering agents have also begun to appear outside the paper-generation setting: Hugging Face's `ml-intern` is an open-source agent that researches papers, writes ML code, launches training through the Hugging Face ecosystem, and maintains session history with auto-compaction and optional session upload [Hugging Face, 2026]. Like AiScientist's File-as-Bus and AI Scientist-v2's tree/journal artifacts, this is durable agent state, but it is not a PR-indexed experiment ledger coupled to an experiment tracker; reviewability depends on inspecting session traces or generated files rather than following the same pull-request, git-history, and W&B substrate used by human ML teams.
```

BibTeX-style raw citations:

```bibtex
@misc{lu2024aiscientist,
  title        = {The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author       = {Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  year         = {2024},
  eprint       = {2408.06292},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2408.06292}
}

@software{sakana2024aiscientist,
  title        = {The AI Scientist},
  author       = {{Sakana AI}},
  year         = {2024},
  url          = {https://github.com/SakanaAI/AI-Scientist}
}

@misc{yamada2025aiscientistv2,
  title        = {The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author       = {Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  year         = {2025},
  eprint       = {2504.08066},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2504.08066}
}

@software{sakana2025aiscientistv2,
  title        = {The AI Scientist-v2},
  author       = {{Sakana AI}},
  year         = {2025},
  url          = {https://github.com/SakanaAI/AI-Scientist-v2}
}

@misc{schmidgall2025agentlaboratory,
  title        = {Agent Laboratory: Using LLM Agents as Research Assistants},
  author       = {Schmidgall, Samuel and Su, Yusheng and Wang, Ze and Sun, Ximeng and Wu, Jialian and Yu, Xiaodong and Liu, Jiang and Moor, Michael and Liu, Zicheng and Barsoum, Emad},
  year         = {2025},
  eprint       = {2501.04227},
  archivePrefix= {arXiv},
  primaryClass = {cs.HC},
  url          = {https://arxiv.org/abs/2501.04227}
}

@software{schmidgall2025agentlaboratorycode,
  title        = {Agent Laboratory},
  author       = {Schmidgall, Samuel},
  year         = {2025},
  url          = {https://github.com/SamuelSchmidgall/AgentLaboratory}
}

@misc{baek2024researchagent,
  title        = {ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models},
  author       = {Baek, Jinheon and Jauhar, Sujay Kumar and Cucerzan, Silviu and Hwang, Sung Ju},
  year         = {2024},
  eprint       = {2404.07738},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2404.07738}
}

@misc{schmidgall2025agentrxiv,
  title        = {AgentRxiv: Towards Collaborative Autonomous Research},
  author       = {Schmidgall, Samuel and Moor, Michael},
  year         = {2025},
  eprint       = {2503.18102},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2503.18102}
}

@article{bran2024chemcrow,
  title        = {Augmenting large language models with chemistry tools},
  author       = {Bran, Andres M. and Cox, Sam and Schilter, Oliver and Baldassari, Carlo and White, Andrew D. and Schwaller, Philippe},
  journal      = {Nature Machine Intelligence},
  volume       = {6},
  pages        = {525--535},
  year         = {2024},
  doi          = {10.1038/s42256-024-00832-8},
  url          = {https://www.nature.com/articles/s42256-024-00832-8}
}

@article{boiko2023coscientist,
  title        = {Autonomous chemical research with large language models},
  author       = {Boiko, Daniil A. and MacKnight, Robert and Kline, Ben and Gomes, Gabe},
  journal      = {Nature},
  volume       = {624},
  pages        = {570--578},
  year         = {2023},
  doi          = {10.1038/s41586-023-06792-0},
  url          = {https://www.nature.com/articles/s41586-023-06792-0}
}

@article{romeraparedes2024funsearch,
  title        = {Mathematical discoveries from program search with large language models},
  author       = {Romera-Paredes, Bernardino and Barekatain, Mohammadamin and Novikov, Alexander and Balog, Matej and Kumar, M. Pawan and Dupont, Emilien and Ruiz, Francisco J. R. and Ellenberg, Jordan S. and Wang, Pengming and Fawzi, Omar and Kohli, Pushmeet and Fawzi, Alhussein},
  journal      = {Nature},
  volume       = {625},
  pages        = {468--475},
  year         = {2024},
  doi          = {10.1038/s41586-023-06924-6},
  url          = {https://www.nature.com/articles/s41586-023-06924-6}
}

@misc{mitchener2025kosmos,
  title        = {Kosmos: An AI Scientist for Autonomous Discovery},
  author       = {Mitchener, Ludovico and Yiu, Angela and Chang, Benjamin and others},
  year         = {2025},
  eprint       = {2511.02824},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2511.02824}
}

@misc{miyai2026jraiscientist,
  title        = {Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper},
  author       = {Miyai, Atsuyuki and Toyooka, Mashiro and Otonari, Takashi and Zhao, Zaiying and Aizawa, Kiyoharu},
  year         = {2026},
  eprint       = {2511.04583},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2511.04583}
}

@misc{chen2026aiscientist,
  title        = {Toward Autonomous Long-Horizon Engineering for ML Research},
  author       = {Chen, Guoxin and Chen, Jie and Chen, Lei and Zhao, Jiale and Meng, Fanzhe and Zhao, Wayne Xin and Song, Ruihua and Chen, Cheng and Wen, Ji-Rong and Jia, Kai},
  year         = {2026},
  eprint       = {2604.13018},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2604.13018}
}

@misc{huang2026groundedautonomousresearch,
  title        = {Towards grounded autonomous research: an end-to-end LLM mini research loop on published computational physics},
  author       = {Huang, Haonan},
  year         = {2026},
  eprint       = {2604.12198},
  archivePrefix= {arXiv},
  primaryClass = {physics.comp-ph},
  url          = {https://arxiv.org/abs/2604.12198}
}

@software{huggingface2026mlintern,
  title        = {ml-intern: an open-source ML engineer that reads papers, trains models, and ships ML models},
  author       = {{Hugging Face}},
  year         = {2026},
  url          = {https://github.com/huggingface/ml-intern},
  note         = {GitHub repository}
}

@misc{githubdocs_pullrequests,
  title        = {Reviewing proposed changes in a pull request},
  author       = {{GitHub Docs}},
  year         = {2026},
  url          = {https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/reviewing-proposed-changes-in-a-pull-request}
}

@misc{gitbook_log,
  title        = {Git Basics: Viewing the Commit History},
  author       = {Chacon, Scott and Straub, Ben},
  booktitle    = {Pro Git},
  year         = {2014},
  url          = {https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History}
}

@misc{wandbdocs_experiments,
  title        = {Experiments overview},
  author       = {{Weights \& Biases}},
  year         = {2026},
  url          = {https://docs.wandb.ai/models/track}
}

@misc{githubdocs_pullrequests,
  title        = {About pull requests},
  author       = {{GitHub Docs}},
  year         = {2026},
  url          = {https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests},
  note         = {Accessed 2026-04-24}
}

@misc{githubdocs_repositories,
  title        = {About repositories},
  author       = {{GitHub Docs}},
  year         = {2026},
  url          = {https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories},
  note         = {Accessed 2026-04-24}
}

@book{chacon2014progit,
  title        = {Pro Git},
  author       = {Chacon, Scott and Straub, Ben},
  edition      = {2},
  publisher    = {Apress},
  year         = {2014},
  url          = {https://git-scm.com/book/en/v2}
}

@misc{biewald2020wandb,
  title        = {Experiment Tracking with Weights and Biases},
  author       = {Biewald, Lukas},
  year         = {2020},
  note         = {Software available from wandb.com},
  url          = {https://www.wandb.com/}
}

@software{wandb2026software,
  title        = {Weights \& Biases},
  author       = {{Weights \& Biases}},
  year         = {2026},
  url          = {https://github.com/wandb/wandb},
  note         = {Python SDK and platform documentation repository; accessed 2026-04-24}
}

@misc{wandbdocs_experiments,
  title        = {Experiments overview},
  author       = {{Weights \& Biases}},
  year         = {2026},
  url          = {https://docs.wandb.ai/models/track},
  note         = {Accessed 2026-04-24}
}

@article{yang2026simple,
  title   = {Simple yet Effective: Low-Rank Spatial Attention for Neural Operators},
  author  = {Yang, Zherui and Xin, Haiyang and Du, Tao and Liu, Ligang},
  journal = {arXiv preprint arXiv:2604.03582},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.03582},
  url     = {https://arxiv.org/abs/2604.03582}
}

@inproceedings{qi2025spidersolver,
  title     = {SpiderSolver: A Geometry-Aware Transformer for Solving {PDE}s on Complex Geometries},
  author    = {Qi, Kai and Wang, Fan and Dong, Zhewen and Sun, Jian},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=hWtvsL51hO}
}

@inproceedings{lim2026tandemfoilset,
  title        = {{TandemFoilSet}: Datasets for Flow Field Prediction of Tandem-Airfoil Through the Reuse of Single Airfoils},
  author       = {Lim, Wei Xian and Loh, Sher En Jessica and Li, Zenong and Oo, Thant Zin and Chan, Wai Lee and Kong, Adams Wai-Kin},
  booktitle    = {The Fourteenth International Conference on Learning Representations},
  year         = {2026},
  url          = {https://openreview.net/forum?id=4Z0P4Nbosn}
}

@misc{mcguire2026tandemfoilsetbalanced,
  title        = {{TandemFoilSet-Balanced}: Balanced Split Design and CFD Surrogate Benchmark Package},
  author       = {McGuire, Morgan and Capelle, Thomas},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/morganmcg1/TandemFoilSet-Balanced},
  note         = {Accessed 2026-04-24}
}



```
Appendix (unlimited)


A. Role prompts, sub-agent, and skill catalogue.**

*A.1 Role prompts.* Advisor, Student, and `researcher-agent` prompt files — verbatim at repo release and pinned to the versions in effect at T=0 of the sprint. Includes the Advisor's plateau-escalation protocol (hyperparameter → architecture → loss reformulation → data representation after five non-improving experiments) and the Student's result-comment schema.

*A.2 Skill catalogue.* The seven composable skills and the shared `senpai-gh` library referenced in §2.3. Each entry lists the caller, inputs, effect on the PR/W&B ledger, and label transitions.

- `survey-prs` — Advisor-side. Reads `$ADVISOR_BRANCH` and `$STUDENT_NAMES` from the pod environment and calls the `senpai-gh` queries `list_all_prs`, `list_ready_for_review_prs`, and `list_idle_students`. Returns a compact markdown summary categorising open PRs as review-ready (`status:review`), WIP (`status:wip` + `student:<name>`), or draft/stalled, followed by a list of idle Students. Read-only.

- `poll-for-work` — Student-side. Takes `<student-name>`. Calls `student_poll_for_work`, which returns PRs carrying both `student:<name>` and `status:wip`. If a PR is returned, the skill additionally reads PR comments so the caller can distinguish a fresh assignment from a revision request. Returns `WORK_AVAILABLE`, `WORK_AVAILABLE (REVISION)`, or `NO_WORK`. Read-only.

- `check-human-issues` — both roles. Takes `<name> <ADVISOR|STUDENT>`. Returns the deduplicated union of issues tagged `human` + `<name>` and `human` + `team`. For each, the skill checks whether the caller's most recent comment is newer than the latest human comment; if not, it posts a reply prefixed `ADVISOR:` or `STUDENT <name>:`. Never closes issues — that is reserved for the human researcher team.

- `list-experiments` — invoked primarily by the `researcher-agent` sub-agent (§2.4) during hypothesis generation. Runs a paginated GraphQL query against the repo for all PRs (OPEN, CLOSED, MERGED) based against the Advisor branch and writes three files under `experiment_log/`: (i) merged winners in chronological order with hypothesis and results, (ii) a compact results table with `val/loss` extracted by regex from PR bodies and comments, (iii) full PR bodies for experiments that ran. Read-only on GitHub; writes only to local disk.

- `assign-experiment` — Advisor-side. Takes `<student-name> <hypothesis-slug> <problem-dir>`. Pulls the Advisor branch, creates branch `<student>/<slug>`, pushes it, and opens a draft PR with labels `status:wip`, `student:<name>`, and the Advisor-branch tag. The PR body — authored by the Advisor — contains the hypothesis, full experiment instructions (typically a hyperparameter diff against `BASELINE.md`), and the current baseline metrics.

- `submit-experiment-results` — Student-side. Takes `<pr-number> <problem-dir>`. Stages `<problem-dir>/train.py` and any other modified files (e.g. `pyproject.toml`), commits, pushes the branch, marks the PR ready for review, and swaps `status:wip` → `status:review`. Assumes the caller has already posted a results comment on the PR with metrics, W&B run ID, analysis, and suggested follow-ups.

- `merge-winner` — Advisor-side. Takes `<pr-number> <problem-dir>`. Squash-merges the PR (`gh pr merge --squash`), pulls the updated Advisor branch, appends the new best metrics to `BASELINE.md` as a dated entry, and commits and pushes. If the squash fails due to conflicts, the skill instead calls `send_pr_back_to_student_with_comment` with a rebase request and stops without updating the baseline.

- `senpai-gh` — shared bash library, sourced by both entrypoint scripts and by every skill above. Provides label-swap, send-back, close-with-comment, ready-for-review, and read-only query primitives. `swap_gh_pr_label` is implemented as two REST calls (DELETE old, POST new) rather than `gh pr edit --remove-label --add-label`, which silently strips all other labels on the PR.

- Plateau Protocol (inside ADVISOR.md prompt)

“When you observe 5 or more consecutive experiments with no improvement, **escalate — do not stop**:

1. **Change strategy tier.** If you have been tuning hyperparameters, move to architecture changes. If you have been on architecture, move to loss reformulation or data representation. Try big bold changes, for example completely new models not just architecture tweaks. Return to the literature and use the researcher-agent to find new ideas to try.
2. **Revisit first principles.** What does the model fundamentally struggle with? Read the worst predictions. What pattern do failed experiments share? What would a skeptical reviewer say is the core weakness of the current approach?
3. **Think bigger.** What techniques in fluid dynamics, numerical simulation, mathematics, physics, computer science, machine learning or optimization have not been tried?
4. **Try bold ideas.** A plateau is permission to take bigger swings. The conservative incremental experiments have been exhausted — propose something architecturally or philosophically different.

**A plateau is never a completion signal. It is a map telling you where not to look, which makes it an asset.**

Use the researcher-agent to explore new ideas and research directions and other sub-agents to do reviews of large amounts of data such as W&B logs, PR logs or many code diffs.”



B. Full benchmark result tables
AirfRANS frontier (surface + volume MSE, W&B run IDs)
TandemFoilSet parity progression by PR number
TandemFoilSet paper-calibration row (sprint-populated)
DrivAerML val/test trajectory (sprint-populated)
C. Failure-mode full taxonomy — subcategories beneath each of the three main categories

D. Experiment ledger sample — CSV schema: PR#, title, status, Student, W&B run IDs, primary metric name/value, baseline delta, era tag, merge-decision timestamp

E. Real SENPAI artifact (Figure 2) — PR conversation + W&B panel, redacted for double-blind review

F. Sprint evidence log — four auditable artifacts:
`fetch_experiments.py` snapshots at T=0 and T+72h (run-count delta)
Per-benchmark W&B run panels from the window
`git log --oneline` on advisor branch restricted to window
Log of 2 urgent-labelled GitHub issues (T=0, T+48h) with timestamps and Advisor actions

G. kagent comparator case studies (§4.2).

*G.1 TandemFoilSet parity apr23 completed run.* An eight-agent kagent cohort on the TandemFoilSet parity benchmark of §3.2.2b — same dataset, same public `kagent` v2 split family, same primary metric (denormalised pressure-channel surface MAE averaged over the four val/test tracks `single_in_dist`, `geom_camber_rc`, `geom_camber_cruise`, `re_rand`). Each agent received a single GPU pod, a 30-minute-per-training-run budget, and the Transolver starter (`n_hidden=128, n_layers=5`, ~120 MAE). The cohort ran from 2026-04-23 15:53 UTC to 2026-04-24 04:00 UTC (~12 h) without human intervention before the scheduled stop. Total activity: **330 agent commits**, **223 scored leaderboard updates** from the organiser (~one scored submission per 2.4 min), one launcher invocation, one scheduled kill, zero manual edits on any agent branch.

Agents and organiser communicate through three channels only: a shared volume for data, predictions, and logs; the git remote for code and the leaderboard; and a W&B project for training telemetry. There is no network path between agents, and each agent sees only its own competition-facing working directory — the organiser area holding ground truth and scoring code is invisible from inside an agent pod. Scoring is the only privileged operation in the system.

At the scheduled stop, seven of eight agents finished below 70 avg-surf-p MAE and four below 50, versus the Transolver starter's ~120; the winning agent's 6-way weighted ensemble scored **34.41**, a 3.5× improvement over the starter in 13 wall-clock hours. SENPAI's multi-week PR-mediated anchor on the same benchmark is 24.58 (§3.2.2b) — ~40% lower in absolute terms but obtained over roughly 40× more wall clock with code review and merge gating. 

https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil
https://github.com/tcapelle/kagent

**Final leaderboard** (lower is better; all splits in MAE):

| Rank | Agent    | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand | Commits |
|-----:|----------|-----------:|---------------:|--------:|------------:|--------:|--------:|
|    1 | frieren  |  **34.41** |          38.73 |   47.92 |       18.21 |   32.78 |      98 |
|    2 | fern     |      42.62 |          46.80 |   55.53 |       26.65 |   41.50 |      47 |
|    3 | edward   |      44.00 |          48.00 |   59.19 |       26.44 |   42.35 |      28 |
|    4 | askeladd |      48.89 |          52.09 |   62.46 |       32.02 |   49.01 |      33 |
|    5 | alphonse |      56.21 |          57.41 |   70.36 |       40.80 |   56.29 |      41 |
|    6 | thorfinn |      62.57 |          45.81 |   77.35 |       40.44 |   86.70 |      20 |
|    7 | tanjiro  |      68.25 |          55.52 |   84.19 |       45.00 |   88.27 |      24 |
|    8 | nezuko   |      85.84 |          96.56 |   96.43 |       63.43 |   86.94 |      39 |

https://github.com/tcapelle/kagent/blob/apr23-leaderboard/leaderboard.md


Seven of eight agents finish below 70; four below 50; the winner at 34.41 is a 3.5× improvement over the starter. Kagent Frieren is best on every test split, and the gap to fern widened in the final hour rather than narrowing. For calibration, SENPAI's multi-week PR-mediated target reached 24.58 MAE (§3.2.2b).

Under kagent, the decisive recipe — full-mesh training with `batch_size=2` warm-started from a subsample-trained checkpoint — was independently rediscovered by three of the top four agents within a ~3 h window once one agent's visible leaderboard commits exposed it; idea diffusion is faster than under SENPAI's PR-review loop. Under SENPAI, the same class of unlock would be routed through an Advisor merge — slower per step, but indexed into the ledger and queryable across future benchmarks. The winning solution's experiment journal: [`frieren/EXPERIMENT_JOURNAL.md`](https://github.com/tcapelle/kagent/blob/cdfa5d78e04aa2b6977e3d5ab8a8f4d1c0dfcc91/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md).


**Four phases of the run.** (i) *Quick baselines (first ~90 min):* all eight agents land a Transolver submission at 88–135 MAE; the cohort independently converges on bf16 autocast + volume-point subsampling as the dominant throughput lever, unlocking 3–4× more epochs inside the 30 min cap. (ii) *Recipe convergence (~16:30–18:00 UTC):* agents read each other's leaderboard commits and converge on a near-identical config (`n_hidden=192, n_layers=6, slice_num=64, mlp_ratio=4, lr≈5-7e-4, warmup+cosine-by-steps, surf_weight≈20, surf_p_weight≈2`); nezuko, tanjiro and edward explicitly reference thorfinn's config. (iii) *Warm-start fine-tune chains (~18:00–23:00 UTC):* the top three agents chain multiple 30-minute runs, each warm-starting from the previous best at roughly half the previous peak LR. Askeladd's six-link chain takes #1 at 56.07; frieren replies and overtakes at 55.32. The leaders compress to <1 MAE apart and run out of low-LR headroom. (iv) *Full-mesh unlock (~23:00 UTC–04:00 UTC):* three agents independently discover that the subsampling relied on for speed was itself the ceiling, and step-change 15–25% in a single run.

**The defining finding — the subsampling trap, rediscovered three ways.** The slice-weight tensors inside Transolver's `PhysicsAttention` are computed across the full node population; training on 40K subsampled points per sample while evaluating on 240K silently caps `re_rand` generalisation. Three of the top four agents rediscovered this independently, with three different chains of reasoning:

- *Frieren* (iter93, inference from askeladd's W&B config): *"bs=2 + no-subsample + warm-start = BREAKTHROUGH. Subsampling was the root cause of my re_rand weakness — dropping 60% of volume nodes left the model unable to learn Re-dependent field structure. With no subsampling the model sees the full 240K-node grid. Askeladd's edge was entirely this config difference."* [[journal L39–44]](https://github.com/tcapelle/kagent/blob/cdfa5d78e04aa2b6977e3d5ab8a8f4d1c0dfcc91/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L39-L44) Scored 35.27 (was 52); `re_rand` dropped 73 → 33.43. Sixty further commits polished this into the 6-way ensemble at 34.41.
- *Alphonse* (v11, direct hypothesis about slice-weight distribution shift): *"The PhysicsAttention slice weights are computed on a different density of nodes between train and val … the hypothesis that subsampling 'just acts like a regulariser' was wrong in this domain; it creates a real train/eval distribution gap for attention models that pool over the node set."* [[journal L72–76]](https://github.com/tcapelle/kagent/blob/d145d312660459324e386042eaa69a14f3a48d48/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L72-L76) 22% single-run gain; `val_re_rand` 2.70 → 1.25.
- *Edward* (v13, inferred from askeladd's W&B runs): *"Sub40k was great for pre-training, but fine-tuning at full mesh preserves fine surface detail that's essential for pressure MAE … 9 epochs at full mesh beats 35 at sub40k in this regime."* [[journal L32]](https://github.com/tcapelle/kagent/blob/a76f87a17574d9490d868edad359460f22a1465c/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L32) 15.7% drop in one run, lifting edward from rank 7 to rank 3.

The three agents who did not find the unlock stalled. *Thorfinn* documented the plateau precisely (*"clean logarithmic decay; we hit the architecture's capacity floor"* [[journal L37]](https://github.com/tcapelle/kagent/blob/aa76905ad4467a95bcd914b588ef077802ce6c88/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L37)) without testing whether the floor was the subsampling, and fell from rank 3 to rank 6. *Tanjiro* never tried full-mesh. *Nezuko* went in the opposite direction, tuning `surf_weight` down from 10 → 1.5; her own earlier-formulated rule — *"compute-per-epoch is the binding constraint: any change that slows each batch needs a matching epochs adjustment and usually nets negative"* [[journal L151]](https://github.com/tcapelle/kagent/blob/4e69f0371b01da656cd1dc89a3295cc0be50e3f9/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L151) — kept her away from a change (full mesh) that violated it.

**Two robust secondary findings.**

1. *Same-lineage ensembles regress.* Every agent that ensembled checkpoints from a single warm-start chain reported regression, and several explicitly formulated the rule — *"members have to be individually close in quality, or strongly decorrelated"* (alphonse [[journal L110]](https://github.com/tcapelle/kagent/blob/d145d312660459324e386042eaa69a14f3a48d48/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L110)); *"v1 is a strictly-worse ancestor of v2"* (askeladd [[journal L125]](https://github.com/tcapelle/kagent/blob/054c788eb583dd052b87ca3a4dabcbcf75efa672/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L125)); *"only ensemble strong models"* (frieren [[journal L99]](https://github.com/tcapelle/kagent/blob/cdfa5d78e04aa2b6977e3d5ab8a8f4d1c0dfcc91/tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md?plain=1#L99)). Frieren eventually made ensembling pay, but only after introducing architectural diversity (`slice=64` vs `slice=128` chains).
2. *Documentation-vs-data no-slip red herring.* The starter `README.md` frames no-slip (`velocity = 0` on airfoil surface) as a hard physical constraint to exploit. Three independent agents (*askeladd*, *tanjiro*, *thorfinn*) inspected the data in the first hour, found that `is_surface=True` covers inlet/outlet/walls with non-zero velocity, and disabled the constraint before it caused harm. This is the inverse of a failure mode: enough harness latitude for the agents to sanity-check a wrong prior.

**Scorer-race incident.** `predict.py` writes the four `test_*.pt` files sequentially (300–500 MB each); the organiser's 60 s scorer stamps any commit directory with a missing file as `"incomplete"` in `scores.json`, and the "already scored" check then treats `"incomplete"` as terminal even once all files are present. Every agent accumulated at least one `incomplete` entry; two agents lost roughly one hour of board time. At least one agent diagnosed the bug in its journal, correctly chose not to chase it, and submitted a fresh commit hash instead. We revisit this in §4.3 as a sibling of SENPAI's checkpoint-contract failure — in both cases the agent's correctness depends on an infrastructure contract the agent cannot inspect, and the fix is at the infrastructure layer (re-score any `incomplete` entries whose files are now all present).

**References.** Per-agent branches: `github.com/tcapelle/kagent/tree/apr23/kaggler/<name>`; leaderboard: `github.com/tcapelle/kagent/tree/apr23-leaderboard`; W&B project: `wandb.ai/wandb-applied-ai-team/kagent-tandemfoil`. Per-agent journals totalling ~1,100 lines are retained on those branches at `tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md`; every numeric claim in this appendix is traceable to the commit-pinned permalinks inline above.



G.2 GRaM ICLR 2026 open-competition run.* We pointed sixteen Claude Opus 4.6 agents at the [GRaM ICLR 2026 competition](https://github.com/gram-competition/iclr-2026) — predicting the 3D velocity field around Formula-1 front-wing geometries for five future timesteps given five past, on ~100k-point meshes, scored by mean L2 velocity error — on `gram-mar29/kaggler/<name>` branches from 2026-03-29 to 2026-03-30 (~30 h wall clock, single-GPU pod each, 30-minute-per-training-run budget, starting from the reference MLP baseline shipped with the competition). The [final leaderboard](https://github.com/tcapelle/kagent/blob/20a631f83958a912e8ba1dcc0d7f372a99c2d3f4/leaderboard.md) shows a ~2× spread across the cohort: winner *violet* at 0.8526 L2, *gilbert* at 1.0230, [*thorfinn*](https://github.com/tcapelle/kagent/commit/e087be89ed601deda6770d25aefde9597409062b) at 1.0742 (`lr=1e-3 + dropout=0.02 + aggressive val skipping`); [*frieren*](https://github.com/tcapelle/kagent/commit/731a5c0aafbb5635330c2e81a7cb17c82c368312) landed mid-pack at 1.1535 (rank 5) after twelve iterations through loss-function variants (L1+L2 → MSE+L2 → pure L2-norm); *alphonse* finished last at 1.6226. The reachable scored commits span MLP-with-regularisation, [residual-from-mean-velocity](https://github.com/tcapelle/kagent/commit/9833cd308a) (*emma* iter35), [subsample-and-tune](https://github.com/tcapelle/kagent/commit/ac1af4d388) (*haku*), and a [GNN run](https://github.com/tcapelle/kagent/commit/89f15ea165) (*nezuko* v12d). The strongest kagent checkpoint was packaged and [submitted to the upstream GRaM competition as a live PR](https://github.com/gram-competition/iclr-2026/pull/4) — end-to-end, cold-start-to-external-submission in under two days. This run predates SENPAI's experiment-journal convention, so there is no per-agent self-commentary to quote; numeric claims here resolve to the leaderboard commit and the per-agent scored SHAs cited inline.

