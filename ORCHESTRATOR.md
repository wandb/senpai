# Orchestrator Context — Living Document

Last updated: 2026-03-25

## Repo Architecture

### What this is
**senpai** — An autonomous ML research loop for training CFD surrogate neural networks (Transolver) on the TandemFoilSet dataset. Claude Code agents (advisor + students) coordinate through GitHub PRs on a Kubernetes cluster.

### Core files (modify boundaries)

| File | Role | Modifiable? |
|---|---|---|
| `senpai.yaml` | Problem selector (points to active problem directory) | Config |
| `cfd_tandemfoil/train.py` | Training script + Transolver model (inline). 645 lines. | YES — students modify this |
| `cfd_tandemfoil/data/prepare.py` | Original dataset loading, `pad_collate`, `FullFieldDataset` (18-dim x) | READ-ONLY |
| `cfd_tandemfoil/data/prepare_multi.py` | Extended preprocessing (24-dim x, dual-foil features), `MultiFieldDataset`, `load_data()` | READ-ONLY |
| `cfd_tandemfoil/data/utils.py` | Visualization (`visualize()`, `plot_samples()`) | READ-ONLY |
| `cfd_tandemfoil/data/split.py` | One-time split manifest generator | READ-ONLY |
| `cfd_tandemfoil/data/split_manifest.json` | Committed train/val indices | READ-ONLY |
| `cfd_tandemfoil/data/split_stats.json` | Committed normalization stats | READ-ONLY |
| `cfd_tandemfoil/program.md` | Research context, metrics, constraints | Reference |
| `CLAUDE.md` | Gets overwritten at pod launch with role-specific instructions | Templated |

### Model architecture (in train.py)
- **Transolver**: Physics-aware attention over irregular meshes
- Key classes: `MLP`, `Physics_Attention_Irregular_Mesh`, `TransolverBlock`, `Transolver`
- Default config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Input: 24-dim x (pos, saf, dsdf, is_surface, log_Re, AoA0, NACA0, AoA1, NACA1, gap, stagger)
- Output: 3 channels (Ux, Uy, p)
- Loss: MSE volume + `surf_weight` * MSE surface (default surf_weight=10)

### Training config (in train.py)
- `Config` dataclass parsed by `simple_parsing`
- Defaults: lr=5e-4, weight_decay=1e-4, batch_size=4, surf_weight=10
- Env-controlled limits: `SENPAI_TIMEOUT_MINUTES` (30), `SENPAI_MAX_EPOCHS` (50)
- 4 validation splits: `val_in_dist`, `val_tandem_transfer`, `val_ood_cond`, `val_ood_re`
- Key metric: **Surface MAE pressure** (`mae_surf_p`) — lower is better
- `WeightedRandomSampler` for balanced domain sampling across racecar_single, racecar_tandem, cruise

### Data pipeline
- Raw data: pickled PyG samples at `/mnt/new-pvc/datasets/tandemfoil/`
- `prepare_multi.py` is the active data loader (24-dim x, includes foil-2 features)
- `prepare.py` is the original (18-dim x), still imported for `load_pickle`, `parse_naca`, `pad_collate`
- `pad_collate` handles variable-length mesh samples → padded batches with masks

### Coordination system (advisor/student)
- **Advisor** (CPU pod): Creates hypothesis PRs, reviews results, merges winners, generates new hypotheses
- **Students** (8x GPU pods): Poll for assigned PRs, implement hypotheses, run training, report results
- Coordination via GitHub labels: `<advisor-branch>`, `student:<name>`, `status:wip`, `status:review`
- PR lifecycle: Draft → WIP → Review → Merge/Close/RequestChanges
- Advisor branch (default: `noam`) — all PRs target this, not main
- 24 named students available (anime characters: frieren, fern, tanjiro, etc.)

### K8s infrastructure
- `k8s/launch.py` — Deploys advisor + student pods via ConfigMaps + Deployments
- `k8s/advisor-deployment.yaml` — CPU pod (4-8 cores, 8-16GB)
- `k8s/student-deployment.yaml` — GPU pod (120 cores, 960GB RAM, 8x GPU with 96GB VRAM each)
- `k8s/entrypoint-advisor.sh` — Clones repo, reads `senpai.yaml` for problem dir, installs deps, runs Claude Code in loop (5min interval)
- `k8s/entrypoint-student.sh` — Same but for students (5s restart interval)
- `k8s/install-weave-cc-plugin.sh` — Runtime Weave plugin registration
- Secrets: `senpai-secrets` (wandb-api-key, anthropic-api-key, github-token, exa-api-key)
- PVC: `new-pvc` mounted at `/mnt/new-pvc` for dataset

### Docker
- `Dockerfile` — Based on CoreWeave ML containers (CUDA 13.2, PyTorch 2.10)
- Bakes in: Node.js 22, kubectl, uv, Claude Code, gh, weave-claude-plugin, yq
- Build trigger: push to main/docker branches when Dockerfile changes, or manual dispatch
- Image: `ghcr.io/wandb/senpai:latest`

### Claude Code setup
- `.claude/settings.json` — High effort, always thinking, agent teams enabled
- `.claude/agents/researcher-agent.md` — Deep literature research agent (uses Exa, arxiv, Semantic Scholar, AlphaXiv)
- `.claude/skills/wandb-primary/` — W&B + Weave query skill
- `.claude/skills/list-experiments/` — Fetches all experiment PRs, generates 3 report files
- `.mcp.json` — Exa MCP server for web search

### Tests
- `tests/test_docker_image.py` — E2E: deploys test pod on k8s, verifies tools installed, weave plugin ready, Claude creates Weave trace
- `tests/test-pod.yaml` — Pod template for e2e test

### CI/CD
- `.github/workflows/build.yaml` — Docker image build+push on Dockerfile changes
- `.github/workflows/cla.yaml` — CLA signature check

## Conventions

1. **Python 3.12+** — modern typing (`list`, `dict`, `a | b`, `a | None`), no `from __future__`
2. **simple_parsing** for CLI args — `sp.parse(Config)`, underscore flag names
3. **uv** as package manager
4. **python-dotenv** for env vars
5. **rich.Console** for script output
6. **wandb** for experiment tracking
7. **SPDX headers** on all files (Apache-2.0, CoreWeave)
8. **No mocking** in tests — full e2e with live API calls
9. **Fail-fast** — no defensive try/except patterns
10. **One hypothesis per PR** — clean attribution

## Known fragilities

1. **CLAUDE.md gets overwritten** at pod launch — entrypoints copy role-specific instructions over it. Branch checkouts also clobber it, so entrypoints restore it each iteration.
2. **prepare.py SURFACE_IDS=(5,6) misses boundary ID 7** (foil 2 surface in tandem data). Fixed in prepare_multi.py but prepare.py is read-only.
3. **pad_collate re-exported** from prepare.py via prepare_multi.py — both files define the same collation, but train.py imports from prepare_multi.
4. **Data paths hardcoded** to `/mnt/new-pvc/datasets/tandemfoil/` in prepare.py. Only works on k8s with the PVC mounted.
5. **Docker build frees disk space** aggressively in CI to fit the large base image.
6. **Weave inactivity timeout patched** in Dockerfile with sed — fragile if plugin updates.
7. **Label swapping** must use GitHub API (not `gh pr edit --remove-label --add-label`) to avoid stripping other labels.
8. **`claude -c`** (continue) falls back to fresh `-p` if continuation fails — entrypoints handle this but context may be lost.

## Current state

- **Branch**: `refactor-problem` (moving problem-specific files into `cfd_tandemfoil/` directory)
- **Recent commits**: Refactor to multi-problem layout — problem files under `cfd_tandemfoil/`, `senpai.yaml` selector
- **Advisor branch**: `noam` (many experiment branches: `exp-noam/*`)

## Decisions log

(Will be updated as orchestrator thread progresses)
