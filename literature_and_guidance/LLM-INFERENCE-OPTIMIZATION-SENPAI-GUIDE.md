<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Large Language Model Inference Optimization Guide

This guide distills the inference-optimization lessons from the Senpai Fast
Gemma Challenge autoresearch run. The source material is the full GitHub pull
request corpus for `morganmcg1/gemma-challenge-senpai`: 795 pull requests,
numbered from [#1](https://github.com/morganmcg1/gemma-challenge-senpai/pull/1) through
[#832](https://github.com/morganmcg1/gemma-challenge-senpai/pull/832), plus 2,643 issue and pull request comments. Most pull requests were closed
or merged between June 13 and June 20, 2026, and 754 pull requests contained a
structured `SENPAI-RESULT` marker.

The original project optimized one Gemma model for one benchmark and one
hardware target. The lessons below are written to transfer to a different large
language model, with its own architecture, serving stack, quality contract, and
deployment hardware.

The guide also incorporates the speculative decoding lessons from Modal's
[Speculation Is All You Need](https://modal.com/blog/spec-is-all-u-need), a
June 19, 2026 research post that argues for treating speculation as a central
inference optimization and as part of a continuous data, evaluation, and
distillation flywheel.

## Terms Used In This Guide

- Large language model: a transformer-based language model served for text
  generation.
- Tokens per second: generated output tokens per second. This was the main speed
  score in the challenge.
- Perplexity: the main likelihood-based quality gate. Lower is better, and in
  this challenge the submission had to stay below a fixed cap.
- Greedy-token identity: the served system must emit the same tokens as plain
  greedy autoregressive decoding for the same checkpoint.
- int4, int8, bf16, and fp32: four-bit integer, eight-bit integer, bfloat16, and
  float32 numeric formats. Lower precision can reduce memory traffic, but it can
  also change model quality or token choices.
- Output head: the final matrix multiplication that turns hidden states into
  vocabulary scores.
- Key-value cache: the cached attention keys and values reused during decoding.
- Drafter: a smaller or auxiliary model path that proposes future tokens.
- Verifier: the main model path that checks proposed tokens.
- Speculative decoding: a decoding strategy where a drafter proposes multiple
  tokens and the verifier accepts or rejects them.
- Acceptance length: how many proposed tokens are accepted per verifier pass.
  It is the main numerator for speculative decoding speedup.
- Draft length: how many tokens the drafter proposes before the verifier runs.
  The best value is usually measured, not assumed.
- Block drafter: a drafter that proposes a block of future tokens in one pass
  rather than paying a separate autoregressive drafter pass for each token.
- SplitK or fixed-split reduction: a kernel strategy that splits a matrix
  multiplication over the reduction dimension. It can improve speed, but it can
  also change numeric reduction order.
- CUDA graph capture: recording a repeated GPU workload so it can run with less
  launch overhead.
- EAGLE-style drafting: a trained speculative-drafting family that predicts
  future-token candidates from intermediate model features.

## Executive Summary

Large language model optimization is not one trick. It is a loop:

1. Define the exact contract that makes an optimization valid.
2. Build a local harness that reproduces the measured path.
3. Decompose decode time into named costs.
4. Test one optimization idea at a time.
5. Reject ideas quickly when the quality, determinism, or speed accounting fails.
6. Compose surviving levers only after each lever has an honest measurement.

In the Fast Gemma 4 case study, the strongest observed speed gains came from
reducing bytes moved per token, especially weight bytes in the decode path, and
from amortizing verifier work through speculative decoding. The strongest
research gains came from evaluation work: same-path perplexity, greedy-token
identity checks, downstream quality panels, private-set drift estimates, launch
gates, and paired speed measurements.

Speculative decoding deserves special treatment. Kernel work and runtime tuning
often produce small percentage gains once the obvious bottlenecks are addressed.
A high-quality speculator can produce factor-level gains because it changes the
shape of decode from one accepted token per target pass to several accepted
tokens per target pass. The price is that the whole system must measure
acceptance, verifier width, drafter cost, and correctness together.

The most important practical lesson is that speed and quality are not separate
work streams. Every optimization changes the measurement problem. If the quality
metric is too narrow, the search will find changes that look fast but are not
valid. If the speed metric is noisy or measured on the wrong path, the search
will chase phantom gains.

## Fast Gemma 4 Case Study: What It Showed

The Fast Gemma 4 run is useful as an imperfect case study, not as a template for
the optimal way to run autoresearch. It produced strong technical evidence, but
it also spent too much time on tiny projected gains, rare edge cases, and
over-analysis of eventualities that should have been killed earlier. The best
lesson is not "copy this exact path." The lesson is to keep the useful parts:
measurement discipline, explicit validity gates, structured negative results,
and break-even math, while avoiding the infinite loop of low-value analysis.

The run moved through these example phases:

- Baseline reproduction and harness work: [#1](https://github.com/morganmcg1/gemma-challenge-senpai/pull/1), [#2](https://github.com/morganmcg1/gemma-challenge-senpai/pull/2), [#8](https://github.com/morganmcg1/gemma-challenge-senpai/pull/8), [#21](https://github.com/morganmcg1/gemma-challenge-senpai/pull/21).
- Weight-byte reduction: [#3](https://github.com/morganmcg1/gemma-challenge-senpai/pull/3) moved from about 44 output tokens per second to
  95.46 by using int4 quantization-aware weights. [#4](https://github.com/morganmcg1/gemma-challenge-senpai/pull/4) moved to 126.38 by using
  coarser quantization groups and an int4 output head.
- Speculative decoding and verifier work: [#18](https://github.com/morganmcg1/gemma-challenge-senpai/pull/18), [#19](https://github.com/morganmcg1/gemma-challenge-senpai/pull/19), [#24](https://github.com/morganmcg1/gemma-challenge-senpai/pull/24), [#30](https://github.com/morganmcg1/gemma-challenge-senpai/pull/30), [#43](https://github.com/morganmcg1/gemma-challenge-senpai/pull/43), [#52](https://github.com/morganmcg1/gemma-challenge-senpai/pull/52), and
  many later pull requests measured accepted-token economics, verifier cost,
  and exactness failures.
- Tree, EAGLE, and drafter research: [#71](https://github.com/morganmcg1/gemma-challenge-senpai/pull/71) through roughly [#350](https://github.com/morganmcg1/gemma-challenge-senpai/pull/350) explored whether
  larger speculative structures could clear the next frontier. Much of the work
  became break-even analysis, correctness gating, and build-risk accounting.
- Strict validity and quality accounting: [#343](https://github.com/morganmcg1/gemma-challenge-senpai/pull/343) onward repeatedly showed that
  speed ideas must pass the real contract, not only a proxy. Pull requests such
  as [#456](https://github.com/morganmcg1/gemma-challenge-senpai/pull/456), [#561](https://github.com/morganmcg1/gemma-challenge-senpai/pull/561), [#583](https://github.com/morganmcg1/gemma-challenge-senpai/pull/583), [#587](https://github.com/morganmcg1/gemma-challenge-senpai/pull/587), and [#594](https://github.com/morganmcg1/gemma-challenge-senpai/pull/594) turned ambiguous speed ideas into closed
  decisions.
- Fire-candidate packaging and preflight work: [#499](https://github.com/morganmcg1/gemma-challenge-senpai/pull/499), [#519](https://github.com/morganmcg1/gemma-challenge-senpai/pull/519), [#770](https://github.com/morganmcg1/gemma-challenge-senpai/pull/770), [#801](https://github.com/morganmcg1/gemma-challenge-senpai/pull/801), [#815](https://github.com/morganmcg1/gemma-challenge-senpai/pull/815),
  [#819](https://github.com/morganmcg1/gemma-challenge-senpai/pull/819), and [#825](https://github.com/morganmcg1/gemma-challenge-senpai/pull/825) show that final speed is won or lost in reproducibility,
  launch gating, runner parity, and benchmark-faithful measurement.

The experiment also showed that negative results are first-class research
assets. A killed idea that explains the blocker is useful because it narrows the
search space for the next agent.

## Core Principles

### Optimize The Scored Path

Measure the path that will actually be scored. In this challenge, local Amazon
A10G runs were useful for profiling and smoke tests, but official numbers came
from Hugging Face Jobs on `a10g-small`. The team repeatedly separated local
exploratory numbers from official leaderboard numbers.

Same-path evaluation mattered. Pull request [#21](https://github.com/morganmcg1/gemma-challenge-senpai/pull/21) showed that the perplexity
measurement used for validation had to match the timed model path rather than a
convenient alternate path. Pull request [#52](https://github.com/morganmcg1/gemma-challenge-senpai/pull/52) produced a valid official run at
481.53 output tokens per second with perplexity 2.377, but later work showed
that a fast public score still needed private-drift and quality checks.

For a new model, establish this before optimizing:

- the exact serving entrypoint,
- the exact model artifact and tokenizer,
- the exact request shapes,
- the hardware and driver stack,
- the official speed metric,
- the official quality metric,
- any deterministic or token-identity requirements.

If the local harness cannot reproduce the baseline, do not trust local
optimization deltas.

### Prove Hardware And Kernel Reachability

Record the exact base commit, hardware, and kernel variant for every result. A
measurement supports only the path that ran; a local fallback does not validate
a target-only kernel.

Before accepting a result, prove that:

- the changed control reaches the timed path,
- the intended kernel dispatched,
- the required workload was preserved,
- the compiler and runtime used the intended implementation.

### Treat Quality As A System, Not One Number

Perplexity is necessary when the contract requires it, but it is not sufficient.
The run found several cases where perplexity passed while generated tokens,
answer quality, or private stability were still suspect.

The quality system eventually included:

- perplexity on the official token set,
- greedy-token identity for served decoding,
- completion count and output length checks,
- downstream tasks such as MMLU-Pro, GPQA-Diamond, AIME, and GSM8K,
- private-set drift estimates,
- run-to-run determinism checks,
- confidence intervals and paired comparisons for small margins.

Pull request [#6](https://github.com/morganmcg1/gemma-challenge-senpai/pull/6) is the clean example of why one metric is not enough. A
greedy-safe vocabulary prune preserved perplexity but slowed the system down
because the fallback guard fired too often. Pull request [#605](https://github.com/morganmcg1/gemma-challenge-senpai/pull/605) showed that a
speculative decoding path still needed downstream quality evidence. Pull
requests [#622](https://github.com/morganmcg1/gemma-challenge-senpai/pull/622) and [#636](https://github.com/morganmcg1/gemma-challenge-senpai/pull/636) showed that rescuing greedy identity can be possible, but
the rescue cost can erase the speed gain.

For a new model, build the quality gate before aggressive optimization. Use the
cheapest gate for fast iteration, but keep a stronger gate for any candidate
that might be launched or shipped.

### Decompose Before You Tune

Optimization becomes tractable when each millisecond has an owner. The run used
decode-step profiles, roofline checks, hardware-bandwidth estimates, and
component-specific experiments to avoid arguing from intuition.

Examples:

- [#30](https://github.com/morganmcg1/gemma-challenge-senpai/pull/30) found that verifier body matrix multiplication was about 53 percent of
  the frontier decode step.
- [#68](https://github.com/morganmcg1/gemma-challenge-senpai/pull/68) measured how close the verifier matrix multiplication was to hardware
  memory bandwidth limits.
- [#75](https://github.com/morganmcg1/gemma-challenge-senpai/pull/75) and [#77](https://github.com/morganmcg1/gemma-challenge-senpai/pull/77) showed that the drafter was not an unlimited source of recoverable
  speed.
- [#806](https://github.com/morganmcg1/gemma-challenge-senpai/pull/806) split the non-matrix-multiplication residual into attention and other
  terms after several large byte levers had already landed.

This pattern transfers well. Before proposing a kernel, quantization scheme, or
new drafter, ask which budget line it attacks:

- model weight reads,
- output head reads,
- key-value cache reads,
- attention computation,
- matrix multiplication scheduling,
- dequantization,
- sampling and token selection,
- host-side orchestration,
- launch overhead,
- prefill or warmup,
- network or benchmark overhead.

If the idea does not map to a measured cost, it is probably premature.

### Treat Speculation As A Data Flywheel

Speculative decoding is not just an engine feature. It is a machine learning
problem attached to the inference system. Modal's
[Speculation Is All You Need](https://modal.com/blog/spec-is-all-u-need)
emphasizes that this is why speculation can keep improving with more data and
compute: the target model can generate labels, the speed metric is direct, and
acceptance length is easy to measure.

This matters for autoresearch. A custom speculator can be trained from the same
application traces used to build evaluations. The loop is:

1. Serve the target model with a generic or built-in speculator.
2. Collect representative prompts, outputs, acceptance traces, and quality
   evaluations.
3. Train or fine-tune a speculator on that application distribution.
4. Re-measure acceptance length, end-to-end speed, and quality on the same
   evaluation ladder.
5. Use the improved latency or cost profile to collect more data.
6. Distill the target or specialize the speculator again when the traffic mix
   changes.

The data loop changes the optimization strategy. Speculator training is often
more tractable than ordinary model training because the teacher is the deployed
model, labels are cheap to generate, and the target metric is acceptance by the
verifier rather than a vague proxy. It still needs production discipline:
traffic can drift, user populations can split into distinct regimes, and a
single global speculator may be worse than a small set of domain or region
specific speculators.

### Model Speculation Before Spending Training Compute

The first-order intuition is simple: if one verifier pass accepts four tokens,
the idealized speedup is close to four. That rule is useful for intuition but
too optimistic for planning. Real systems pay extra target cost for wider
verification, extra drafter cost, and extra host or kernel overhead.

A useful speculative-decoding model includes:

- acceptance length,
- draft length,
- verifier latency as a function of verification width,
- drafter latency per token or per block,
- memory bandwidth and compute roofline limits,
- context length and batch or concurrency,
- communication overhead if the drafter and verifier are disaggregated.

The Modal post highlights two practical modeling moves that match the Senpai
run. First, simulate or mock acceptance length before training a new speculator
so the team can estimate the possible speedup. Second, use a roofline model
rather than only a toy acceptance-length model, because target latency and
drafter latency decide where the optimum draft length sits.

For an autoresearch system, every speculative decoding proposal should answer:
what acceptance length is needed, how much drafter latency is affordable, what
verification width costs, and whether the best draft length is stable across the
actual workload.

### Separate Speed Levers From Validity Levers

Some changes make the model faster by changing what is computed. Other changes
make the same computation cheaper. The distinction is crucial.

By-construction safe levers are preferred:

- fewer bytes for weights when dequantized values are unchanged or quality is
  proven,
- deterministic reductions that preserve the selected token,
- kernel schedule changes with exact output checks,
- serving configuration changes with identical request behavior.

Riskier levers need stronger gates:

- speculative decoding,
- relaxed acceptance,
- vocabulary pruning,
- output head pruning,
- layer dropping,
- sub-int4 or codebook quantization,
- new drafter training,
- tree or multi-candidate verification.

The run repeatedly found that a speed lever can turn into a validity lever. For
example, batch-invariant verification could reduce token divergence, but [#122](https://github.com/morganmcg1/gemma-challenge-senpai/pull/122)
measured a very large speed cost. A change that restores correctness after
another optimization is part of the optimization cost, not a separate free fix.

### Use Break-Even Math Before Building

The best pull requests often answered one priced question: how much does this
idea need to improve before it matters?

Examples:

- [#100](https://github.com/morganmcg1/gemma-challenge-senpai/pull/100) composed the known levers into an official tokens-per-second landscape.
- [#102](https://github.com/morganmcg1/gemma-challenge-senpai/pull/102) computed the tree accept-length needed to clear 500.
- [#105](https://github.com/morganmcg1/gemma-challenge-senpai/pull/105) priced a path to 500 without the tree and solved for the needed SplitK
  contribution.
- [#106](https://github.com/morganmcg1/gemma-challenge-senpai/pull/106) compared the tree path against the tree-free path and gave the build team
  a milestone ladder.
- [#119](https://github.com/morganmcg1/gemma-challenge-senpai/pull/119) closed a drafter accept-length ceiling instead of continuing to chase an
  unpriced hope.

A good break-even pull request includes:

- the baseline cost,
- the maximum plausible gain,
- the minimum gain needed to matter,
- the quality cost,
- the implementation cost,
- the decision rule for green, amber, or red.

This kind of analysis prevents the search from spending days on a lever that
can only move the final metric by half a percent.

### Prefer Representative Metrics Over Convenient Metrics

The experiment repeatedly caught metrics that were easy to collect but not
representative:

- public prompts did not always predict private prompts,
- local tokens per second did not always transfer to official tokens per
  second,
- greedy self-checks could compare against the wrong reference,
- partial-prompt measurements could miss long-context behavior,
- small downstream evaluation sets could create misleading pass or fail calls,
- warm benchmark paths could hide cold-start or runner-environment failures.

Representative metrics do not need to be expensive on every iteration. A good
system uses a ladder:

1. Static checks: manifest, dependency, model-load, tokenizer, request shape.
2. Microbenchmarks: isolated kernel or component timing.
3. Local correctness: perplexity, completion count, greedy identity.
4. Local speed: paired wall-clock comparisons with a known noise floor.
5. Expanded quality: downstream task panels and confidence intervals.
6. Private or shifted distribution probes.
7. Official benchmark launch.

The fast loop should catch obvious failures. The slow loop should prevent
expensive false positives.

### Make Negative Results Easy To Reuse

The corpus contains many dead ends that are useful precisely because they were
made explicit:

- static vocabulary pruning with a full-vocabulary fallback was slower ([#6](https://github.com/morganmcg1/gemma-challenge-senpai/pull/6)),
- double-quantized scales did not round-trip exactly enough to be worthwhile
  ([#104](https://github.com/morganmcg1/gemma-challenge-senpai/pull/104)),
- persistent-kernel idle recovery was small ([#97](https://github.com/morganmcg1/gemma-challenge-senpai/pull/97)),
- some speculative decoding fixes recovered correctness only by paying too much
  speed ([#122](https://github.com/morganmcg1/gemma-challenge-senpai/pull/122), [#636](https://github.com/morganmcg1/gemma-challenge-senpai/pull/636)),
- several quality-safe full-head paths could not beat the faster int4 baseline
  ([#544](https://github.com/morganmcg1/gemma-challenge-senpai/pull/544), [#561](https://github.com/morganmcg1/gemma-challenge-senpai/pull/561), [#594](https://github.com/morganmcg1/gemma-challenge-senpai/pull/594)),
- some draft-head pruning ideas did not reduce the active token work ([#821](https://github.com/morganmcg1/gemma-challenge-senpai/pull/821)).

A negative result should state whether the idea failed because of quality,
speed, implementation feasibility, hardware transfer, or measurement mismatch.
Those are different failures with different follow-up actions.

## A Warm-Start Playbook For A New Model

### Phase 1: Reproduce The Baseline

Start by reproducing the unoptimized model on the exact serving stack.

Deliverables:

- one command to serve the model,
- one command to run the local validation harness,
- a baseline speed number,
- a baseline perplexity or task-quality number,
- a completion-count check,
- a small profile of decode time,
- a written list of what differs between local and official scoring.

Do not begin speculative decoding or custom kernels until this exists.

### Phase 2: Build The Evaluation Ladder

Create a staged gate that gets more expensive only when a candidate survives.

Minimum gates:

- same-path perplexity or the task-equivalent metric,
- token identity or the model-specific validity condition,
- downstream task checks if the benchmark quality metric is narrow,
- private or shifted distribution checks when public prompts are known,
- paired speed tests with a measured noise floor,
- final launch preflight.

The gate should fail closed. Missing artifacts, partial completion, or a
different request path should block promotion.

### Phase 3: Make A Decode Budget

Profile one token at a time. For transformer inference, split the budget into:

- embedding and input preparation,
- attention,
- feed-forward layers,
- output head,
- key-value cache reads and writes,
- dequantization,
- sampling,
- host scheduling,
- kernel launch overhead,
- logging and benchmark overhead.

Then assign each proposed optimization to a budget line. This keeps the research
portfolio balanced and avoids over-investing in an already-small component.

### Phase 4: Establish The Weight-Byte Floor

For memory-bandwidth-bound decoding, weight movement is usually the first major
target.

Try:

- quantization-aware checkpoints,
- int4 or int8 weight-only quantization,
- group-size sweeps,
- output-head quantization,
- module sensitivity maps,
- mixed precision for sensitive modules,
- scale-storage compression,
- activation quantization when the serving stack supports it,
- structured sparsity if kernels exist.

Measure:

- perplexity,
- downstream quality,
- greedy-token stability,
- real tokens per second,
- bytes read per token,
- whether the kernel is actually using the intended quantized path.

The Gemma run found a strong early path: bf16 at about 44 output tokens per
second, int4 quantization-aware weights at 95.46, then coarser groups plus an
int4 output head at 126.38. That pattern is not guaranteed for another model,
but it is a strong first search direction.

### Phase 5: Add Speculative Decoding With Honest Economics

Speculative decoding can be a major lever when accepted tokens amortize an
expensive verifier. It can also be a mirage when the drafter is costly, the
acceptance rate is low, or correctness repair erases the gain.

Treat acceptance length as the main numerator and end-to-end step time as the
main denominator. The idealized speedup follows acceptance length, but the
realized speedup is reduced by drafter work, wider target verification, sampling
logic, host overhead, and any correctness repair needed to preserve the quality
contract.

Measure:

- accepted tokens per verifier step,
- acceptance by position,
- acceptance by prompt family,
- acceptance under long contexts and agentic workloads,
- drafter cost,
- drafter cost per proposed token versus per proposed block,
- verifier cost at each verification width,
- token identity or quality drift,
- the best draft length rather than the largest draft length,
- whether acceptance mocking or a roofline model predicts the measured speed.

Try:

- trained drafters,
- multi-token prediction heads,
- block drafters that produce several candidate tokens per pass,
- n-gram or prompt-lookup drafters,
- dynamic draft length,
- margin-gated acceptance,
- early exit,
- multi-candidate or tree verification,
- distillation objectives that optimize acceptance rather than likelihood.

Stop when the break-even math says the drafter cannot repay its cost. The
experiment repeatedly showed that accepted-token count alone is not a sufficient
metric.

### Phase 6: Train And Adapt The Speculator

Once a generic speculator is working, use application data to specialize it.
Modal argues that custom speculators can move acceptance length materially with
data volumes that real applications can produce. The important operational move
is to reuse the same traces for both evaluation and training, while keeping
public, private, and shifted-distribution checks separate.

Deliverables:

- a trace schema for prompts, outputs, accepted tokens, rejected tokens, latency,
  and quality labels,
- a training set that excludes evaluation leakage,
- a held-out workload that represents the application rather than only the
  public benchmark,
- acceptance-length and speed measurements before and after training,
- a drift policy for deciding when to retrain, fork, or retire a speculator.

Prefer adaptive speculators when the workload has persistent domains. Prefer
multiple stable speculators when the traffic mix alternates between distinct
regions, languages, tools, or product modes. Retraining every time the average
acceptance length shifts can be a mistake when the real issue is a mixture of
workloads.

### Phase 7: Work The Kernel And Runtime Stack

After the byte floor and speculative decoding are in place, the remaining gains
often come from kernel and runtime details.

Try:

- attention backend selection,
- sliding-window attention kernels,
- SplitK or fixed-split reductions,
- CUDA graph capture,
- fused dequantization,
- fused drafter operations such as key-value injection or output projection,
- fused sampling or token-selection epilogues,
- key-value cache precision or prefetching,
- output-head read reductions,
- launch-overhead reduction,
- host-loop removal.

Cost a kernel change before building it:

- bytes moved,
- dispatches removed and whether they serialized or overlapped,
- work replicated per threadgroup,
- barriers or synchronization added,
- register pressure and occupancy,
- materialization or graph construction removed.

Every kernel optimization needs a numerical-equivalence gate. If the gate is
"same selected token" rather than bit-exact equality, state that explicitly and
test the margin where token choices can flip.

Drafters need their own runtime scrutiny. They are smaller than the target, so
they may fail to saturate the accelerator and may pay proportionally more host
overhead. Moving the drafter to separate hardware can look appealing, but
communication latency can erase the gain unless the workload and deployment
topology are unusually favorable.

### Phase 8: Compose Conservatively

Do not add measured gains linearly without checking interactions. Quantization
can change drafter acceptance. A faster verifier can change the best speculative
draft length. A serving flag can change deterministic behavior. A quality
recovery patch can erase the speed gain it was meant to unlock.

Use a composition table with:

- each lever's measured gain,
- each lever's confidence interval,
- the budget line it attacks,
- expected interactions,
- the validity gates it depends on,
- the smallest useful official benchmark to confirm transfer.

## Transformer-Specific Research Categories

### Weight And Activation Precision

Promising experiments:

- weight-only int4 quantization with a quantization-aware checkpoint,
- group-size sweeps such as group size 32 versus 128,
- output-head quantization,
- module-level precision sensitivity maps,
- mixed precision for sensitive layers,
- activation quantization,
- key-value cache quantization for long-context workloads,
- codebook or sub-int4 quantization only when serving kernels are real.

Common failure modes:

- quality loss concentrated in reasoning-heavy tasks,
- fake-quant results that do not survive the serving kernel,
- quantized output heads that reduce acceptance or answer quality,
- savings too small to matter after dequantization overhead,
- "lossless" scale compression that is not actually bit-exact.

### Speculative Decoding And Drafters

Promising experiments:

- multi-token prediction drafters,
- DFlash-style or other block drafters that avoid paying a full drafter pass per
  proposed token,
- acceptance-optimized drafter training,
- distillation on representative prompts and application traces,
- prompt-lookup or n-gram drafters as cheap baselines,
- dynamic draft length,
- margin-aware acceptance,
- tree or multi-candidate verification when verifier width is cheap,
- acceptance mocking to estimate speedup before training,
- roofline modeling to choose draft length and detect when the drafter is too
  expensive.

Common failure modes:

- high public acceptance that does not transfer to private prompts,
- drafter cost larger than the verifier work it saves,
- drafter kernels that are too small to use the accelerator efficiently,
- treating a traffic-mix shift as a retraining problem when it really calls for
  multiple domain-specific speculators,
- numerical differences between batch widths that break token identity,
- acceptance gains that degrade downstream task quality,
- complex tree builds whose realized accepted-token count is far below the
  analytical model.

When the application owns the model behavior contract, lossy speculation becomes
a possible research lane. It should not be mixed into a lossless API-serving
contract by accident. If lossy acceptance is allowed, define the behavior budget
explicitly with downstream evaluations, canary traffic, and rollback criteria.

### Output Head And Vocabulary Work

Promising experiments:

- output-head quantization,
- faster output-head matrix multiplication,
- candidate-limited verification when the validity contract allows it,
- exact or guarded pruning with measured fallback rates.

Common failure modes:

- static keep sets that miss rare but important tokens,
- fallback rates high enough to erase the savings,
- downstream reasoning collapse from head pruning or baked artifacts,
- full-vocabulary materialization hidden in the serving stack.

### Attention And Key-Value Cache

Promising experiments:

- choosing the right attention backend for local and global attention layers,
- sliding-window attention kernels,
- key-value cache layout and precision,
- attention path graph capture,
- fixed-order reductions when deterministic output matters,
- long-context crossover measurements.

Common failure modes:

- one backend forcing the whole model onto a slower path,
- attention wins that disappear at the scored context length,
- deterministic settings that restore correctness but impose a large speed tax,
- private prompts with longer context changing the bottleneck.

### Decode Loop, Scheduling, And Host Overhead

Promising experiments:

- full decode-step graph capture,
- persistent kernels only after proving there is meaningful idle time,
- fused dequantization and sampling,
- host-loop removal,
- benchmark warmup and cold-start separation,
- paired comparison runners to measure small deltas.

Common failure modes:

- optimizing a component smaller than the measurement noise,
- mistaking benchmark warmup for real throughput,
- breaking multimodal load paths or prefill while optimizing decode,
- changing logging or runner behavior instead of model speed,
- disaggregating drafter and verifier work before proving that communication
  latency is small enough.

## Experiment Template For An Autoresearch System

Each experiment should be small enough to answer one question.

Use this schema:

```text
Hypothesis:
  The specific cost or quality issue this experiment addresses.

Base and hardware:
  Exact commit, architecture, and kernel family exercised.

Target cost:
  The measured budget line, in milliseconds, bytes, or percentage of decode time.

Expected gain:
  The break-even gain and the optimistic gain.

Validity gate:
  The exact quality, identity, completion, and private-stability checks.

Implementation scope:
  Files, model artifacts, kernels, flags, or training jobs allowed to change.

Measurement plan:
  Static checks, microbenchmarks, local validation, paired speed test, and
  official launch conditions.

Speculation plan:
  Acceptance length target, draft length sweep, drafter cost, verifier-width
  cost, workload drift plan, and whether the lane is lossless or lossy.

Stop rule:
  Green, amber, and red criteria, including when to kill the idea.

Result:
  Structured metrics, artifact paths, and a plain-language conclusion.
```

The system should keep a ledger of rejected ideas. The ledger should include the
reason an idea failed, not only the result number.

## High-Value Defaults For The Next Model

Start with these unless the new model gives a clear reason not to:

- Build the validation harness first.
- Reproduce the unoptimized baseline locally and officially.
- Profile decode time before proposing optimizations.
- Try quantization-aware int4 or int8 weight paths early.
- Quantize or optimize the output head if it is a large fraction of decode time.
- Measure group-size and module-sensitivity tradeoffs.
- Use speculative decoding only after measuring drafter acceptance and verifier
  width cost.
- For interactive serving, treat speculation as a primary optimization lane, not
  a secondary polish step after kernel work.
- Model acceptance length and roofline costs before training a new speculator.
- Reuse application traces for evaluation, speculator training, and later
  distillation, while keeping held-out quality gates clean.
- Prefer prompt-invariant levers for private stability.
- Use downstream quality gates when perplexity is a weak proxy.
- Use paired speed measurements for small deltas.
- Require launch preflight before spending scarce official benchmark runs.

Avoid these habits:

- optimizing against a different request path,
- treating local exploratory speed as official speed,
- trusting one public prompt set as the whole distribution,
- accepting a perplexity pass as proof of generation quality,
- treating acceptance length from one prompt set as universal,
- stacking projected gains before measuring interactions,
- retraining a speculator for transient traffic shifts without checking whether
  the workload is really a mixture of stable domains,
- leaving dead ends undocumented.

## Evidence Appendix

Selected pull request evidence:

| Pull request | Lesson |
| --- | --- |
| [#2](https://github.com/morganmcg1/gemma-challenge-senpai/pull/2) | Local baseline validation produced perplexity 2.3012 and established the need for reusable validation. |
| [#3](https://github.com/morganmcg1/gemma-challenge-senpai/pull/3) | Int4 quantization-aware weights produced 95.46 official output tokens per second with perplexity 2.0057. |
| [#4](https://github.com/morganmcg1/gemma-challenge-senpai/pull/4) | Coarser groups plus an int4 output head produced 126.38 official output tokens per second with perplexity 2.019. |
| [#6](https://github.com/morganmcg1/gemma-challenge-senpai/pull/6) | Greedy-safe vocabulary pruning was slower because the safety fallback erased the theoretical gain. |
| [#21](https://github.com/morganmcg1/gemma-challenge-senpai/pull/21) | Same-path perplexity prevented validation from drifting away from the timed serving path. |
| [#24](https://github.com/morganmcg1/gemma-challenge-senpai/pull/24) | Verify rollback removed token flips but introduced an explicit rollback-rate cost. |
| [#30](https://github.com/morganmcg1/gemma-challenge-senpai/pull/30) | Decode profiling found verifier body matrix multiplication at roughly 53 percent of the frontier step. |
| [#52](https://github.com/morganmcg1/gemma-challenge-senpai/pull/52) | The FlashAttention sliding-window and split key-value cache stack reached 481.53 official output tokens per second with perplexity 2.377. |
| [#82](https://github.com/morganmcg1/gemma-challenge-senpai/pull/82) | Paired wall-clock testing established a low-noise way to compare local speed deltas. |
| [#90](https://github.com/morganmcg1/gemma-challenge-senpai/pull/90) | A speculative decoding draft-length sweep found the best local wall speed at draft length 7. |
| [#96](https://github.com/morganmcg1/gemma-challenge-senpai/pull/96) | Small per-layer numerical perturbations could compound into many token flips. |
| [#100](https://github.com/morganmcg1/gemma-challenge-senpai/pull/100) | Lever composition turned isolated measurements into a decision surface. |
| [#104](https://github.com/morganmcg1/gemma-challenge-senpai/pull/104) | Double-quantized scales were killed because only 13.09 percent round-tripped bit-exactly. |
| [#105](https://github.com/morganmcg1/gemma-challenge-senpai/pull/105) | A tree-free path to 500 was priced before continuing risky tree work. |
| [#122](https://github.com/morganmcg1/gemma-challenge-senpai/pull/122) | Batch-invariant verification had a very large speed cost, showing correctness repair must be priced. |
| [#456](https://github.com/morganmcg1/gemma-challenge-senpai/pull/456) | A closed-lever annex recorded twenty strict speed levers and their closure status. |
| [#499](https://github.com/morganmcg1/gemma-challenge-senpai/pull/499) | A human-approved strict draw reached 375.86 tokens per second, showing the cost of stricter validity. |
| [#519](https://github.com/morganmcg1/gemma-challenge-senpai/pull/519) | Full served recertification measured 442.35 warm median tokens per second and perplexity 2.376981. |
| [#561](https://github.com/morganmcg1/gemma-challenge-senpai/pull/561) | A quality-safe full-head path was capped around 311.25 tokens per second and did not beat faster candidates. |
| [#583](https://github.com/morganmcg1/gemma-challenge-senpai/pull/583) | A speculative decoding closure reconciled the speed and identity gates. |
| [#770](https://github.com/morganmcg1/gemma-challenge-senpai/pull/770) | A guarded fire candidate reached 218.02 tokens per second with perplexity 2.0058. |
| [#805](https://github.com/morganmcg1/gemma-challenge-senpai/pull/805) | Input-gate dequantization produced a measured local kernel lever around 265.61 decode tokens per second. |
| [#825](https://github.com/morganmcg1/gemma-challenge-senpai/pull/825) | Benchmark-faithful 128-prompt testing measured a fire candidate at 268.26 tokens per second with a 260.7 lower bootstrap bound. |

External research evidence:

| Source | Lesson |
| --- | --- |
| [Modal, "Speculation Is All You Need"](https://modal.com/blog/spec-is-all-u-need), June 19, 2026 | Speculative decoding should be modeled and trained as a first-class inference optimization. Acceptance length drives the numerator, roofline and drafter cost determine the realized speedup, custom speculators can be trained from application data, and adaptive speculators should follow real workload structure rather than noisy short-term drift. |

Corpus statistics:

- Pull requests reviewed: 795.
- Pull requests with comments: 783.
- Pull requests with a `SENPAI-RESULT` marker: 754.
- Open pull requests at review time: [#824](https://github.com/morganmcg1/gemma-challenge-senpai/pull/824), [#826](https://github.com/morganmcg1/gemma-challenge-senpai/pull/826), [#829](https://github.com/morganmcg1/gemma-challenge-senpai/pull/829), [#830](https://github.com/morganmcg1/gemma-challenge-senpai/pull/830), [#831](https://github.com/morganmcg1/gemma-challenge-senpai/pull/831), [#832](https://github.com/morganmcg1/gemma-challenge-senpai/pull/832).
- Main recurring themes: quantization, speculative decoding, kernel and runtime
  work, quality gates, launch gates, private stability, and composition models.
