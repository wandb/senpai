# Runnable Autoresearch Workshop

This folder is the source package for the SENPAI autoresearch workshop. It contains:

- a Slidev deck
- live-service Python notebook-style lessons
- a credential setup script
- facilitator resources and offline fixtures
- review tooling for PDF export quality checks

Core thesis:

> Autoresearch is a distributed ML control system, not a long prompt.

The curriculum walks from a single LLM call to tool calls, single-agent loops, multi-agent routing, W&B/Weave memory, Kubernetes dry-runs, and physical-AI claim review.

## Who This Is For

| Audience | What they should get |
| --- | --- |
| Senior/staff ML engineers | A concrete architecture for letting agents run ML experiments without losing evidence or control. |
| W&B users | A model for joining training runs, configs, histories, and Weave traces into an autoresearch ledger. |
| CoreWeave platform users | A view of GPU pods, PVC-backed data, Kubernetes dry-runs, and resource policy as part of the research system. |
| DevRel and solution architects | A teachable storyline with slides, notebooks, and fallback resources. |
| Research leads | A framework for deciding which autonomous results are defensible, proxy-only, or misleading. |

Assumed background: Python, GitHub PRs, basic W&B usage, and ML training concepts such as validation/test splits, checkpoints, and metrics.

## Folder Map

| Path | Purpose |
| --- | --- |
| `slides.md` | Slidev deck for the 3-hour workshop. |
| `package.json` / `package-lock.json` | Local Slidev toolchain and pinned npm dependency graph. |
| `setup_credentials.py` | One-command credential setup and live validation. |
| `.env.example` | Credential template. |
| `common/` | Shared helpers for config, LLM calls, GitHub, W&B, and teaching protocols. |
| `notebooks/` | Numbered `.py` notebook-style lessons with `# %%` cells. |
| `artifacts/` | Generated outputs from notebook runs. |
| `resources/` | Facilitator guides, optional paper labs, offline artifact pack, and checklist. |
| `scripts/` | Slide export/review and cleanup tooling. |

Generated paths such as `node_modules/`, `review/`, `dist/`, PDFs, PNGs, and notebook artifacts are ignored by `workshop/.gitignore`.

## Quick Start For Learners

From the repo root:

```bash
uv run python workshop/setup_credentials.py
uv run python workshop/notebooks/00_environment_check.py
```

Then run the notebooks in order:

```bash
uv run python workshop/notebooks/01_llm_calls_to_hypotheses.py
uv run python workshop/notebooks/02_tool_calls_and_contracts.py
uv run python workshop/notebooks/03_single_agent_student_loop.py
uv run python workshop/notebooks/04_multiagent_advisor_student_flow.py
uv run python workshop/notebooks/05_wandb_and_weave_as_memory.py
uv run python workshop/notebooks/06_k8s_autoresearch_dry_run.py
uv run python workshop/notebooks/07_physical_ai_claim_review.py
uv run python workshop/notebooks/08_full_autoresearch_trace.py
```

To check credentials without prompting:

```bash
uv run python workshop/setup_credentials.py --check-only
```

## Run The Slidev Deck

Install Node dependencies from `workshop/`:

```bash
npm --prefix workshop install
```

Start the deck:

```bash
npm --prefix workshop run dev
```

Build or export manually:

```bash
npm --prefix workshop run build
npm --prefix workshop run export
```

Export and review the deck with screenshots, console capture, and a contact sheet:

```bash
npm --prefix workshop run review:slides
```

This writes:

- `workshop/autoresearch-workshop.pdf`
- `workshop/review/autoresearch-workshop/review-report.md`
- `workshop/review/autoresearch-workshop/contact-sheet.html`
- one screenshot per slide under `workshop/review/autoresearch-workshop/slides/`

The deck uses standard Slidev Markdown in `workshop/slides.md`: frontmatter, slide separators, Mermaid diagrams, code highlighting, and presenter notes in HTML comments.

Clean generated Slidev outputs:

```bash
npm --prefix workshop run clean
```

## Required Credentials

Copy `workshop/.env.example` to `workshop/.env` or run `setup_credentials.py`.

Required:

- `ANTHROPIC_API_KEY`
- `GITHUB_TOKEN`
- `WANDB_API_KEY`
- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `TARGET_REPO_URL`
- `TARGET_REPO_BRANCH`
- `ADVISOR_BRANCH`

Optional:

- `EXA_API_KEY`
- `ANTHROPIC_MODEL`

`workshop/.env` is ignored and written with `0600` permissions by the setup script.

## Safety Defaults

- Live GitHub and W&B reads are enabled when credentials are configured.
- GitHub mutations are disabled unless a notebook explicitly sets a visible mutation flag.
- No notebook pushes commits, creates PRs, or changes labels by default.
- Kubernetes examples render dry-run manifests only.
- Generated notebook outputs go under `workshop/artifacts/`.

## Notebook Curriculum

| Notebook | Teaches |
| --- | --- |
| `00_environment_check.py` | Validate Anthropic, GitHub, and W&B access. |
| `01_llm_calls_to_hypotheses.py` | Turn LLM ideas into falsifiable research assignments. |
| `02_tool_calls_and_contracts.py` | Build a toy tool registry and teach workflow invariants. |
| `03_single_agent_student_loop.py` | Model a bounded student-agent result handoff. |
| `04_multiagent_advisor_student_flow.py` | Route PRs through advisor/student labels. |
| `05_wandb_and_weave_as_memory.py` | Treat W&B and Weave as research memory. |
| `06_k8s_autoresearch_dry_run.py` | Inspect SENPAI launch resources without applying them. |
| `07_physical_ai_claim_review.py` | Classify benchmark claims as defensible or misleading. |
| `08_full_autoresearch_trace.py` | Run the full educational trace end to end. |

## Facilitator Resources

Use these when teaching:

- `resources/instructor-guide.md` - timing, scripts, transitions, misconceptions, and audience-specific guidance.
- `resources/architecture-checklist.md` - take-home review checklist.
- `resources/legacy-labs.md` - optional paper/group labs from the earlier workshop design.
- `resources/optional-live-demo.md` - fallback demo paths and dry-run delivery options.
- `resources/artifact-pack/` - offline PR/W&B/Weave/dry-run fixtures.

## Source vs Generated Files

Keep these in version control:

- `README.md`
- `.env.example`
- `.gitignore`
- `package.json`
- `package-lock.json`
- `slides.md`
- `setup_credentials.py`
- `common/`
- `notebooks/`
- `resources/`
- `scripts/`
- `artifacts/README.md`

Do not commit:

- `workshop/.env`
- `workshop/node_modules/`
- `workshop/artifacts/*` except `artifacts/README.md`
- `workshop/review/`
- `workshop/dist/`
- exported PDFs, PNGs, or PPTX files

## Delivery Paths

W&B-heavy session:

- Emphasize notebooks `05` and `08`.
- Show how W&B metrics and Weave traces complement each other.
- Use `resources/artifact-pack/wandb-run-summary.json` and `weave-trace-excerpt.json`.

CoreWeave-heavy session:

- Emphasize notebooks `06` and `08`.
- Walk through GPU requests, PVC mounts, secrets, tags, and teardown.
- Use `resources/optional-live-demo.md`.

Research-lead session:

- Emphasize notebooks `01`, `07`, and `08`.
- Focus on metric contracts, validation vs test, and claim discipline.
- Use `resources/architecture-checklist.md`.

DevRel session:

- Start from `slides.md`.
- Use notebook outputs only where they clarify the flow.
- Keep `resources/artifact-pack/` open as a no-network fallback.

## Validation

Before teaching, run:

```bash
python3 -c 'import compileall, sys; ok=compileall.compile_dir("workshop", quiet=1); sys.exit(0 if ok else 1)'
python3 workshop/setup_credentials.py --help
python3 workshop/setup_credentials.py --check-only
python3 -m json.tool workshop/package.json >/dev/null
npm --prefix workshop run review:slides
```

Only run the live notebooks after `--check-only` succeeds.

If you only want a source-tree cleanup after export/review:

```bash
npm --prefix workshop run clean
```
