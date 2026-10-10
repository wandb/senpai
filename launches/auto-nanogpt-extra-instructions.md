# Auto-nanoGPT Launch Extra Instructions

This target is not a physical AI, CFD, or surrogate-modelling task. When the
generic Senpai system prompt mentions physical metrics, CFD, datasets, or
architecture search for physical modelling, reinterpret that guidance as:
optimize the fixed `modded-nanogpt` track 3 optimizer benchmark described in
`$PROBLEM_DIR/program.md`.

Keep the task close to the public `modded-nanogpt` track 3 benchmark:

- The objective is lower optimizer steps to FineWeb validation loss below 3.28.
- Keep data, batch size, model architecture, and one forward-backward pass per
  optimizer step fixed.
- Focus on optimizer algorithms, optimizer hyperparameters, schedules, and
  initialization.
- Do not invent extra data configs, data files, or benchmark rules beyond the
  target repo's `program.md` and the official track 3 README.

Keep the research portfolio balanced between exploitation and exploration.
Retuning LR/WD/cooldown is important when giving a new method a fair shot, but
do not let the run become mostly scalar hyperparameter search. Assign fresh
optimizer mechanisms, preconditioners, schedule ideas, initialization ideas,
and pruning ablations of complex stacks.

Be deliberate about step budgets. The launch timeout and max-epoch values are
hard ceilings, not instructions to run forever. Use tiny runs for smoke tests,
shorter step budgets for uncertain optimizer screening, and longer predeclared
seed batches for serious confirmation. Think clearly before each PR about
whether the run is exploration, tuning, pruning, or confirmation.

Early kill gates are encouraged for obvious crashes, non-finite loss, exploding
gradients, or hopeless screening runs, but do not use per-run validation loss to
cherry-pick final steps or seeds. Final claims must report all non-cherry-picked
runs at a predeclared step count and evaluate the benchmark statistical rule.

Use W&B aggressively. Preserve and extend the starter script's telemetry for
losses, validation loss, steps-to-target, learning rates, weight decay, gradient
norms and distributions, parameter norms and distributions, and any
optimizer-specific diagnostics that help explain why an idea worked or failed.
The starter script also logs trailing-window train-loss slopes every 10% of the
run and at the final step; use these slopes to reason about curve shape without
overreacting to individual noisy steps.

Focus only on your assigned advisor branch, research tag, student list, PR
stream, and W&B runs. Ignore work that is not labeled with your branch/tag. Use
the benchmark snapshot checked into this target repo; do not refresh, browse,
fetch, or mine new upstream PRs, branches, records, issues, or post-launch
updates during this run.

Prime Intellect's autonomous-run materials are explicitly banned sources for
agents during this launch. Do not open, fetch, browse, search within, clone,
cite, summarize, or use:

- `https://www.primeintellect.ai/auto-nanogpt`
- `https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning`
- any raw GitHub URLs, files, branches, issues, pull requests, or archives under
  that repository

Those links are named only so you know what not to read. They are comparison
artifacts for humans after the run, not part of the active experimental
context.
