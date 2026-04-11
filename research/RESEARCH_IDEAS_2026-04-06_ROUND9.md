# SENPAI Research Ideas — 2026-04-06 (Round 9, Human Team Suggestions — Issue #1926)

Generated from human researcher team suggestions in GitHub Issue #1926. Each idea has been researched against the original papers and cross-referenced against the full PR history to avoid duplicates.

**Current baseline (PR #2104, +aft_foil_srf, 8-seed mean, seeds 42-49):**
p_in=13.19 ± 0.33, p_oodc=7.92 ± 0.17, p_tan=28.50–30.05, p_re=6.45 ± 0.07

**Flags enabled in baseline:** `--residual_prediction --surface_refine --aft_foil_srf`

**Definitively closed dead ends (DO NOT retry):**
- XSA (PR #2007): +8.2% val/loss, +24.4% p_in — catastrophic failure
- Muon/Gram-NS (PR #2006): 30–70% regression across all metrics
- SOAP/CauchyAdamW/Schedule-Free AdamW (PR #2010): 2–6% worse than Lion across all seeds

**Not applicable to CFD mesh surrogate:**
- MSA (Memory Sparse Attention, arXiv 2603.23516): designed for 100M-token LLM context — not relevant to fixed-size mesh graphs
- HyperP/ArchScale (microsoft/ArchScale): T^{-0.32} LR + MuonH for matrices — validated only for 100B–600B token LLM training; no mesh-surrogate mechanism

---

## Ranked Hypotheses

### 1. noble-branches — NOBLE Nonlinear Low-Rank Branches in TransolverBlock MLPs

**Priority: HIGH. ~20 LoC. Expected -2 to -5% p_tan. MEDIUM risk.**

**What it is:** NOBLE (arXiv 2603.06492, "Nonlinear Low-Rank Branches") adds a residual branch `σ(x W_down) W_up` alongside each existing linear layer in transformer MLPs, where `W_down` projects to rank r (e.g., r=16) and `W_up` projects back. The activation `σ` is CosNet: `cos(ω·x + φ)` with learnable frequency `ω` and phase `φ` per channel. This makes each MLP layer a sum of a linear map and a nonlinear low-rank correction — the correction branch is initialized to near-zero so training starts from the pretrained/baseline linear behavior.

**Why it helps p_tan:** The tandem pressure distribution has sharp nonlinear dependencies on wake interaction geometry that a purely linear projection cannot represent efficiently. The NOBLE branch provides each TransolverBlock's MLP with a dedicated nonlinear capacity — a thin but expressive correction — without disrupting the well-conditioned linear path. CosNet's periodic activation is physically motivated: pressure fluctuations in the inter-foil channel are quasi-periodic functions of gap/stagger. Paper reports 1.47× step speedup and 32% fewer steps to convergence at equivalent parameter count, which at fixed epoch budget translates to better final accuracy.

**Key paper:** "NOBLE: Nonlinear Low-Rank Branches for Transformers" (arXiv 2603.06492, 2026). Reports +4% parameter overhead, 1.47× step speedup, validated on language and vision tasks.

**Implementation — exactly where to add code:**

Add to `Transolver` class in `train.py`, modifying the `TransolverBlock`'s two MLP `nn.Linear` layers. The `TransolverBlock` uses `nn.Sequential` MLPs inside `Physics_Attention_Irregular_Mesh`. The cleanest implementation is a drop-in `NOBLELinear` wrapper:

```python
class NOBLELinear(nn.Module):
    """Linear layer + low-rank nonlinear branch: y = xW + σ(xW_down)W_up."""
    def __init__(self, in_f, out_f, rank=16, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.down = nn.Linear(in_f, rank, bias=False)
        self.up = nn.Linear(rank, out_f, bias=False)
        self.omega = nn.Parameter(torch.ones(rank))   # CosNet freq
        self.phi = nn.Parameter(torch.zeros(rank))     # CosNet phase
        # Zero-init the correction branch so training starts from baseline
        nn.init.zeros_(self.up.weight)
    def forward(self, x):
        h = self.down(x)
        h = torch.cos(self.omega * h + self.phi)       # CosNet activation
        return self.linear(x) + self.up(h)
```

In `Physics_Attention_Irregular_Mesh.__init__` and the two `nn.Linear` calls in `ffn` (the feed-forward network after attention), replace `nn.Linear(in_dim, hidden_dim)` and `nn.Linear(hidden_dim, in_dim)` with `NOBLELinear(in_dim, hidden_dim, rank=cfg.noble_rank)` and `NOBLELinear(hidden_dim, in_dim, rank=cfg.noble_rank)` respectively.

Add config flag:
```python
noble_branches: bool = False   # Enable NOBLE nonlinear low-rank branches in MLP layers
noble_rank: int = 16           # Rank of the low-rank correction branch
```

Then wrap the `NOBLELinear` usage with `if cfg.noble_branches`.

**Caution:** CosNet's `omega` parameter can diverge if LR is too high. Set a lower LR for `omega` and `phi` (0.1× base LR, same group as attention params). Also: the `up.weight` zero-init means gradients through the correction branch are zero at step 0 — this is correct, the `down` and CosNet parameters will receive gradients through `up` only after `up.weight` escapes zero, which happens naturally within a few hundred steps due to gradient noise.

**Suggested experiment:**
```
--noble_branches --noble_rank 16
```
Run 2 seeds. If p_tan improves ≥1%, run 8 seeds. Also sweep `--noble_rank 8` vs `--noble_rank 32` in a follow-up.

**Confidence:** Moderate. NOBLE is validated on language/vision but not CFD mesh surrogates. The CosNet periodic activation is well-motivated for pressure fields. Zero-init residual ensures no regression risk at initialization. Main uncertainty: whether the nonlinear correction captures something new vs. just adding parameters.

---

### 2. geotransolver-gale — Geometry-Aware Context Cross-Attention (GALE)

**Priority: HIGH. ~50 LoC. Expected -3 to -8% p_tan. MEDIUM-HIGH risk.**

**What it is:** GeoTransolver (arXiv 2512.20399) introduces GALE (Geometry-Aware Latent Encoding) — a shared geometry context computed once per mesh, then cross-attended to at each Transolver block. The geometry context is a small set of geometric summary tokens (k=32 or 64) derived from multi-scale ball queries on mesh node positions and boundary features. Each TransolverBlock gets a cross-attention sublayer: `x_out = x + CrossAttn(x, geometry_ctx)`, where `geometry_ctx` is the same for all blocks in one forward pass. This is architecturally motivated: the current Transolver processes each slice independently without an explicit global geometry context — GALE provides this missing context with minimal overhead.

**Why it helps p_tan:** Tandem pressure predictions fail because the aft-foil sits in a complex wake that depends on the full upstream geometry. Each Transolver block currently has no mechanism to reference a global geometry summary — it only sees local slice statistics. GALE provides a fixed, geometry-conditioned context that each block can query. This is directly analogous to how human engineers reason: they first build a mental model of the full geometry, then reason about local flow at each point. The paper demonstrates GeoTransolver outperforming vanilla Transolver on DrivAerML and ShiftML automotive aerodynamics benchmarks — a directly comparable setting to TandemFoilSet.

**Key paper:** "GeoTransolver: Geometric-Aware Neural Operator for Complex Geometry Flow Simulation" (arXiv 2512.20399, 2025). Beats Transolver on DrivAerML by a meaningful margin. Code available in NVIDIA PhysicsNeMo: `nvidia/physicsnemo`.

**Implementation sketch:**

Add a `GeometryContextEncoder` module that takes mesh node positions and boundary features (x[:,:,0:10] raw) and produces a geometry context of shape `[B, n_ctx, n_hidden]`:

```python
class GeometryContextEncoder(nn.Module):
    """
    Build geometry context tokens from mesh positions and boundary features.
    Uses a simple mean-pool over K ball-query neighborhoods at n_ctx anchor points.
    Anchor points = farthest point sampling of the mesh (FPS), or fixed quantiles of xy.
    """
    def __init__(self, in_dim, n_hidden, n_ctx=32):
        super().__init__()
        self.n_ctx = n_ctx
        self.proj = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
        )

    def forward(self, raw_geo_feats):  # [B, N, in_dim]
        # Simple global average pool per-batch + learnable context queries
        # (full GALE uses ball queries; this minimal version uses global pool)
        ctx = raw_geo_feats.mean(dim=1, keepdim=True).expand(-1, self.n_ctx, -1)  # [B, n_ctx, in_dim]
        return self.proj(ctx)  # [B, n_ctx, n_hidden]
```

In each `TransolverBlock.forward()`, after the slice-attention sublayer and before the FFN, add:
```python
# GALE cross-attention: x queries geometry context
if self.use_gale:
    x = x + self.gale_cross_attn(x, geometry_ctx, geometry_ctx)
```

Add `gale_cross_attn = nn.MultiheadAttention(n_hidden, n_head, batch_first=True)` to `TransolverBlock.__init__`.

Add config flags:
```python
gale_context: bool = False   # Enable geometry context cross-attention per block
gale_n_ctx: int = 32         # Number of geometry context tokens
```

**Implementation caution:** The geometry context must be computed from pre-standardization features (raw xyz + boundary info) to preserve geometric meaning. Pass the context through `Transolver.forward()` to each block as an additional argument. Start with the simplest version (global mean pool as context anchor) before attempting full FPS + ball queries.

**Suggested experiment:**
```
--gale_context --gale_n_ctx 32
```
Run 2 seeds. If promising, run 8 seeds and try `--gale_n_ctx 64`.

**Confidence:** Moderate. Strong architectural motivation and domain-relevant prior validation (automotive CFD). The implementation risk is medium — cross-attention changes the information flow significantly and the FPS/ball-query version adds complexity. Recommend starting with the simplified global-pool context (avoids FPS implementation), then upgrading to multi-scale if the simplified version shows signal.

---

### 3. piratenets-adaptive-residuals — PirateNets Adaptive Residual Connections

**Priority: MEDIUM-HIGH. ~25 LoC. Expected -1 to -4% p_tan. LOW-MEDIUM risk.**

**What it is:** PirateNets (arXiv 2402.00326, 2024) introduces adaptive residual connections for physics-informed neural networks. Instead of fixed skip connections `h_{l+1} = F(h_l) + h_l`, each layer learns a scalar gate `α_l ∈ [0,1]` that interpolates between the identity and the transformation: `h_{l+1} = α_l · F(h_l) + (1 - α_l) · h_l`. The gates are initialized at `α_l = 0` (pure identity), so early training uses shallow-network dynamics (fast convergence, stable gradients), then progressively deepens as `α_l` learns to open. This avoids the "spectral bias" trap in deep PINNs where all layers try to learn features simultaneously.

**Why it helps here:** The current Transolver uses standard residual connections with fixed `skip_gate` (initialized to negative bias for approximate identity). PirateNets' adaptive gating extends this with a per-layer learnable interpolation that starts purely as identity. For our 2-layer Transolver, this provides each block an explicit "depth switch" — early in training both blocks act shallow (stable), then progressively specialize. The paper shows significant improvements for multi-physics PDE settings, which parallels our tandem pressure problem (multiple interacting flow regimes).

**Key paper:** "PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks" (arXiv 2402.00326, 2024). Shows consistent improvements on Navier-Stokes, Maxwell, and multi-scale PDE benchmarks.

**Prior attempt:** PR #2004 attempted PirateNets but **crashed due to implementation bugs, not concept failure**. This is a fresh re-implementation from scratch with the correct gating mechanism.

**Implementation — where to add code:**

In `TransolverBlock.__init__`, add:
```python
if cfg.pirate_residuals:
    self.pirate_alpha = nn.Parameter(torch.zeros(1))  # init α=0 → pure identity
```

In `TransolverBlock.forward()`, replace the current skip_gate pattern:
```python
# Current (approximate identity via biased sigmoid):
# out = skip_gate(out) * out + ...
# Replace with PirateNet adaptive residual:
if cfg.pirate_residuals:
    alpha = torch.sigmoid(self.pirate_alpha)  # ∈ (0, 1)
    out = alpha * attn_out + (1 - alpha) * x_in
else:
    # existing skip_gate logic
```

Apply the same pattern to the FFN sublayer within each block.

Add config flag:
```python
pirate_residuals: bool = False  # Enable PirateNets adaptive residual gating
```

**Key constraint:** The `pirate_alpha` parameters should use 10× higher LR than the main params (they need to escape 0 quickly enough to be useful by epoch 140 when EMA begins). Add them to the attention param group.

**Suggested experiment:**
```
--pirate_residuals
```
Run 2 seeds. Do NOT repeat PR #2004 — this is a clean re-implementation, not a debug of the crashed code.

**Confidence:** Moderate. Well-supported for PDE settings. The prior crash was a code bug, not a concept failure. The connection to our setting is direct: Transolver already uses residuals; making them adaptive is a natural extension.

---

### 4. moon-optimizer — Moon Optimizer (Muon Variant with Corrected Update Rule)

**Priority: MEDIUM. ~15 LoC. Expected -1 to -3% p_in, -1 to -2% p_tan. LOW risk.**

**What it is:** Moon is a Muon variant that modifies a single argument in the Nesterov update step. Where Muon applies orthogonalization (Gram-NS) to the gradient before the momentum update, Moon applies orthogonalization after the Nesterov step — a subtle but meaningful change that avoids gradient interference between momentum direction and the orthogonalization transform. Moon was proposed as a direct fix for a known instability in Muon when momentum accumulates orthogonalized directions.

**Why it helps:** Muon/Gram-NS failed catastrophically (PR #2006: 30–70% regression). However, Moon is **architecturally distinct from Muon** — it changes the update order in a way that the Muon paper itself identified as a potential improvement. The original Muon failure was traced to gradient instability in the attention weight matrices (Wqkv, slice_weight). Moon's corrected update order may resolve this. At worst it replicates Muon's failure, but the failure mode is different and the LR sensitivity may be lower.

**Implementation:**

```python
# In train.py, optimizer setup section (lines ~1431-1464):
# Replace Lion with Moon for matrix parameters, keep Lion for vector/bias params

# Install: pip install muon  (Moon is available as muon.Moon)
from muon import Moon

# Separate matrix params (weight matrices) from vector params (biases, norms, embeddings)
matrix_params = [p for n, p in model.named_parameters()
                 if p.ndim >= 2 and p.requires_grad and 'bias' not in n]
vector_params = [p for n, p in model.named_parameters()
                 if not (p.ndim >= 2) or 'bias' in n and p.requires_grad]

moon_opt = Moon(matrix_params, lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay,
                momentum=0.95)
lion_opt = Lion(vector_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=(0.9, 0.99))
# Use both optimizers with matching schedulers
```

Add config flag:
```python
moon_optimizer: bool = False  # Use Moon for matrix params, Lion for vector params
```

**Caution:** Moon is distinct from Muon in update order only — if Moon is not available in the current environment (`pip show muon`), it can be implemented manually (~10 lines). The LR split (Moon at 0.5× base, Lion at 1.0× base) mirrors the existing attn/non-attn split and should be a safe starting point.

**Suggested experiment:**
```
--moon_optimizer
```
Run 2 seeds. Compare directly against Lion-only baseline. If stable (no NaN, no >5% regression), run 4 more seeds.

**Confidence:** Low-moderate. Muon failed badly; Moon's single-argument change may or may not resolve the instability. However, the risk is bounded: if it diverges, stop early. The theoretical motivation is sound — update order in orthogonal gradient methods matters significantly for convergence stability.

---

### 5. mhc-residuals — Manifold-Constrained Hyper-Connections (mHC)

**Priority: MEDIUM. ~35 LoC. Expected -1 to -3% p_tan. MEDIUM risk.**

**What it is:** Manifold-Constrained Hyper-Connections (mHC, arXiv 2512.24880, 2024) replace standard residual connections with a learned per-layer combination matrix `A_l ∈ R^{(l+1)×(l+1)}` that routes information from ALL previous layer outputs, not just the immediately preceding one. Unlike vanilla hyper-connections (which add a full dense matrix over all layers and scale poorly), mHC constrains `A_l` to lie on a Stiefel manifold (orthogonal subspace), which prevents redundant routing and maintains gradient norm stability. This enables each block to learn which prior representations are most useful for its computation.

**Why it helps p_tan:** The current Transolver uses a simple 2-layer stack with `skip_gate` connections. The aft-foil pressure task likely requires information from both the "global geometry understanding" (block 1) and "local flow correction" (block 2) simultaneously — a pattern that flat sequential residuals cannot represent efficiently. mHC's manifold-constrained combination allows block 2 to explicitly learn a mixture of block 1's output and the raw input, potentially enabling a more effective fore-aft decomposition.

**Key paper:** "Manifold-Constrained Hyper-Connections" (arXiv 2512.24880, 2024). Reports "tangible performance improvements" on language tasks. CFD application is novel.

**Implementation sketch:**

For a 2-layer Transolver, mHC introduces a 2×2 combination matrix per block:

```python
class mHCResidual(nn.Module):
    """Manifold-constrained hyper-connection for 2-layer Transolver."""
    def __init__(self, n_hidden):
        super().__init__()
        # Learnable combination weights, initialized to identity (standard residual)
        self.A = nn.Parameter(torch.eye(2))  # 2×2 for 2-layer stack
        self.norm = nn.LayerNorm(n_hidden)

    def forward(self, x_prev, x_curr):
        # x_prev: [B, N, n_hidden] — output of block l-1 (or raw input for block 1)
        # x_curr: [B, N, n_hidden] — output of block l's transformation
        # Project A onto Stiefel manifold via QR decomposition (cheap for 2×2)
        Q, _ = torch.linalg.qr(self.A)
        a_prev, a_curr = Q[0, 0], Q[0, 1]
        return self.norm(a_prev * x_prev + a_curr * x_curr)
```

In `Transolver.forward()`, track `h_prev` (input to block 1) and pass to block 2's mHC layer. This requires a small change to how `fx` is passed between blocks.

Add config flag:
```python
mhc_residuals: bool = False  # Enable manifold-constrained hyper-connections
```

**Caution:** The QR decomposition at every forward pass adds overhead (negligible for 2×2 matrices). The Stiefel constraint means `a_prev^2 + a_curr^2 = 1` — this is correct orthogonal normalization. For a 2-layer model, this reduces to a rotation in the (x_prev, x_curr) plane, which is interpretable. The main risk is that with only 2 layers, the information routing benefit of mHC is smaller than in deeper networks where the full prior layer history is more diverse.

**Suggested experiment:**
```
--mhc_residuals
```
Run 2 seeds. The implementation is non-trivial — verify forward pass shapes before full run with a short smoke test (1 epoch).

**Confidence:** Low-moderate. Strong theoretical motivation but validated only on language tasks. The 2-layer Transolver limits the benefit compared to deeper architectures where mHC shines. If the Transolver is ever scaled to n_layers=4+, mHC would become higher priority.

---

## Summary Table

| Rank | Slug | Code size | Risk | Expected p_tan gain | Priority |
|------|------|-----------|------|--------------------:|----------|
| 1 | noble-branches | ~20 LoC | MEDIUM | -2 to -5% | HIGH |
| 2 | geotransolver-gale | ~50 LoC | MEDIUM-HIGH | -3 to -8% | HIGH |
| 3 | piratenets-adaptive-residuals | ~25 LoC | LOW-MEDIUM | -1 to -4% | MEDIUM-HIGH |
| 4 | moon-optimizer | ~15 LoC | LOW | -1 to -3% | MEDIUM |
| 5 | mhc-residuals | ~35 LoC | MEDIUM | -1 to -3% | MEDIUM |

**Excluded from list (not applicable or definitively failed):**
- XSA: closed dead end (PR #2007, +24.4% p_in)
- Muon/Gram-NS: closed dead end (PR #2006, 30–70% regression)
- MSA: designed for LLM 100M-token context, no mechanism relevant to mesh surrogate
- HyperP/ArchScale: LLM-specific optimizer scaling, not applicable

**Recommended first assignments:**
1. `noble-branches` — highest expected impact per line of code, zero-init residual ensures no regression risk at initialization, clear mechanism (periodic activation for pressure fields)
2. `geotransolver-gale` — most architecturally motivated for CFD geometry (the paper tests on automotive CFD which is directly comparable), starts as a global-pool simplification to reduce implementation risk
3. `piratenets-adaptive-residuals` — prior crash (PR #2004) was code bugs, not concept failure; clean re-implementation is a different experiment
