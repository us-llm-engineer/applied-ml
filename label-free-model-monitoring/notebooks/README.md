# Notebooks

| Notebook | Shows | Runtime |
|---|---|---|
| `01_research_foundations.ipynb` | Each source paper's estimand, identifiability boundary, and guarantee, derived and checked on seeded synthetic data; the conditional retrain-count ceiling and its assumption check; literature trace citations for every source used. | CPU, executed |
| `02_baseline_deployment_study.ipynb` | A synthetic deployed classifier with known latent outcomes, a naive monitor's failure modes, the delayed-audit queue, the finite-capacity training/deployment queue, and the immediate-vs-delayed retraining-decision comparison. | CPU, executed |
| `03_monitoring_and_retraining_policies.ipynb` | The retraining controller's full lifecycle, paired policy comparisons, cost and hindsight-regret, uncertainty/practical-significance checks, and a reuse of NB2's delayed-decision evidence to price the cost of detection failure correctly. | CPU, executed |

Each claim-bearing section cites its source, notes what is `derived here`
versus paper-stated, recomputes an assertion from an artifact on disk, and ends
with a "how to read this chart" note and a rendered figure. Figure links in this folder are
relative: `../label_free_monitoring/figures/fig-aN.png`.

## 01 — Research foundations

- **Three estimands, one deployment setting.** SEES's identifiable performance
  gap under Sparse Joint Shift, a sequential detector's time-uniform false-alarm
  control, and D3M's disagreement-based deterioration test are derived and run
  on the same seeded synthetic streams, each against the observable-vs-latent
  boundary it actually respects.
- **The retraining cost objective.** A finite-horizon retrain-schedule
  optimizer matches an exhaustive oracle exactly, including its tie-breaking.
- **The conditional count bound.** A Regol-style ceiling on the number of
  optimal retrains is checked only where its uniform adjacent-model-gap
  assumption holds; the one setting that violates it is drawn with no ceiling.

## 02 — Baseline deployment study

- **Data and preprocessing.** Six shift regimes share one source sample and
  one classifier; a naive feature-drift monitor's failure modes are shown
  with honest intervals before any label-free alarm is introduced.
- **Observable vs. latent outcomes.** A counterfactual label-replay check
  proves every alarm decision uses only observable signals, never the latent
  truth available solely because this is a seeded study.
- **Delayed audit labels.** A label submitted for audit stays hidden until its
  release boundary and is released exactly once; duplicate submissions and
  over-budget audits reject without changing state.
- **Finite queue capacity.** A retraining request holds the serving model in
  place until training and deployment both finish; a request arriving while
  one is in flight is rejected without spending budget or moving the model.
- **Immediate vs. delayed decisions.** The delayed-audit queue is connected to
  the actual retraining decision for the first time: on a ten-seed paired
  grid, delaying the label by 2–4 steps measurably shifts detection timing
  and turns some "detected, later" outcomes into "never detected."

## 03 — Monitoring and retraining policies: controller, policy comparison, and cost

- **Controller lifecycle.** The legal path stable → audit → candidate →
  promote/rollback → cooldown is exercised end to end; every illegal
  transition is rejected with the full state left untouched.
- **Paired policy comparison.** Five policies are compared on the same
  paired-seed streams, keeping false retraining, missed deterioration,
  detection delay, recovery, and cost reported separately rather than folded
  into one score.
- **Cost, hindsight regret, and uncertainty.** A replication-spread-vs-policy
  effect-size check is run against the literature's own practical-significance
  bar (Cohen's |d| > 0.5); this notebook's own `no_action` vs. `threshold`
  separation clears it by a wide margin (d ≈ 2.80).
- **Reusing NB2's delayed-decision evidence.** NB2's raw mean cost under label
  delay is non-monotonic because a missed detection stops charging later
  intervention costs. Reusing NB2's artifact verbatim and pricing cost per
  successfully detected seed instead resolves the puzzle: every delayed arm
  costs more than the immediate one.

## Limitations

- All data is synthetic and CPU-only; nothing here is evidence about a real
  deployed model, a real label-arrival process, or real retraining costs.
- The Regol-style count bound is a decision-theoretic comparator conditional
  on a checked assumption, not a label-free detection guarantee; it is never
  invoked where the assumption fails.
- The immediate-vs-delayed comparison covers one policy (`threshold`), ten
  paired seeds, and three delay values — a measured trade-off on this grid,
  not a general claim about label delay.
- Dasari's empirical retraining-policy findings (queueing, incremental vs.
  static learning) are kept as external evidence with their own assumptions
  (immediate labels, a fixed-cost linear learner); this project's synthetic
  sweep is not presented as a reproduction of them.
