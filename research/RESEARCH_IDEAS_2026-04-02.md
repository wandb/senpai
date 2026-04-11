# Phase 6 Research Ideas — 2026-04-02

## Researcher-Agent Analysis (Ranked by Confidence)

Based on deep literature review and analysis of 1,615+ prior experiments.

### Rank 1 — NOBLE: Nonlinear Low-Rank Branches → **ASSIGNED to alphonse (#2011)**
- Adds parallel nonlinear branch: `output = xW + b + CosNet(xW_down) * W_up`
- CosNet uses cosine activations (bounded, captures periodic patterns)
- 21-32% fewer steps to baseline in LLM evals
- Apply to attention projection layers. Flag: `--noble --noble_rank 32`
- **Risk:** Medium (10-15% step overhead). **Confidence:** Medium-high.

### Rank 2 — HeavyBall Optimizers (LaProp+MARS, SOAP) → **ASSIGNED to nezuko (#2010)**
- LaProp+MARS+caution: numerically stable Adam with variance reduction, zero overhead
- SOAP: Shampoo-based full-matrix preconditioning, more aggressive
- Do NOT re-run plain Muon (that's PR #2006)
- **Risk:** Low (LaProp) to Medium (SOAP). **Confidence:** Medium.

### Rank 3 — PirateNets Adaptive Residual Gate → **ASSIGNED to frieren (#2008)**
- RWF: W = diag(exp(s)) * V (fixed random V, learned scale s)
- Also adaptive residual gate: `x_{l+1} = sigmoid(alpha) * F(x) + (1-sigmoid(alpha)) * x`
- Apply to surface refine MLP and TransolverBlock MLPs
- **Risk:** Low. **Confidence:** Low-medium (PINN-to-surrogate transfer uncertain).

### Rank 4 — mHC: Manifold-Constrained Hyper-Connections → **DEFER**
- Expands residual stream to n×C dims with Birkhoff-polytope mixing
- Too complex for 3-layer model; DomainLayerNorm already handles domain specifics
- **Risk:** High (significant refactor). Only if NOBLE/HeavyBall plateau.

### Rank 5 — HyperP: Hypersphere Optimization → **WAIT FOR PR #2006**
- Weight matrices on fixed-norm Frobenius sphere + MuonH optimizer
- Only relevant as follow-up if Muon shows benefits
- **Risk:** Medium (overlaps PR #2006). **Confidence:** Low.

### Rank 6 — Geosolver / GeoTransolver → **DO NOT ASSIGN**
- **Already tried in PR #1989 and failed decisively** (+9.8% worse, gate values near zero, model rejected context)
- 24-dim input features already encode geometry (dsdf, saf, curvature, coordinates)
- Originally assigned to alphonse (#2009), closed and replaced with NOBLE.

### Rank 7 — MSA: Memory Sparse Attention → **DO NOT ASSIGN**
- Wrong problem class. Solves cross-session memory recall for LLMs.
- Our slice attention already handles node compression (25k → 96 slices).
- PR #1877 and #1878 explored similar ideas and found slice mechanism superior.

## Key Insights from Analysis
1. **Throughput is king**: any technique adding >15% step time costs epochs, typically hurts final metrics
2. **24-dim input already encodes geometry**: dsdf, saf, curvature, coordinates — adding more geometry context doesn't help
3. **3-layer model limits some techniques**: mHC, deep residual approaches designed for 12+ layers won't help
4. **Cosine activations match physics**: periodic activation functions capture pressure distributions naturally
