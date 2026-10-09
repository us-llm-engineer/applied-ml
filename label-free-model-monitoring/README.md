# Label-free post-deployment monitoring and retraining

This directory contains the estimators, the retraining lifecycle controller,
the delayed-audit queue, the finite-capacity retraining/deployment queue, and
the figure evidence used by the three executed notebooks. The study is
synthetic and CPU-only: it derives three papers' label-free monitoring
guarantees, then measures where a `derived here` retraining controller built
on top of them actually helps, costs, or fails.

## The five sources

| arXiv ID | Paper | Role in the notebooks |
| --- | --- | --- |
| 2209.08436 | Chen, Zaharia & Zou, *Estimating and Explaining Model Performance When Both Covariates and Labels Shift* (NeurIPS 2022) | SEES: identifiable performance-gap estimation under Sparse Joint Shift |
| 2412.12910 | Amoukou et al., *Sequential Harmful Shift Detection Without Labels* (NeurIPS 2024) | time-uniform sequential false-alarm control (Theorem 4.2) |
| 2506.05047 | Nguyen et al., *Reliably Detecting Model Failures in Deployment Without Labels* (NeurIPS 2025) | D3M: disagreement-based deterioration detection and its named failure regime |
| 2505.14903 | Regol, Schwinn, Sprague, Coates & Markovich, *When to retrain a machine learning model* (ICML 2025 / PMLR 267) | the conditional retrain-count ceiling (Proposition 3.1) and the UPF decision form |
| 2608.19488 | Dasari, *When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints* (2026 preprint) | empirical evidence on queueing, latency, and incremental- vs. static-learner retraining policies |

None of the five defines a delayed-label, finite-budget, deployment-latency
retraining controller end to end; the controller, the delayed-audit queue,
and every cost model in this directory are `derived here`.

## Contents

| Path | What it holds |
| --- | --- |
| `synthetic.py` | seeded shift scenarios (covariate/label/concept/dense-joint/D3M-regime-2) and deployment streams with a latent, unlabeled truth |
| `estimators.py` | the SEES gap estimator, the sequential tail monitor, and the D3M disagreement monitor |
| `controller.py` | the retraining lifecycle state machine — stable → audit → candidate → promote/rollback → cooldown |
| `audit.py` | the delayed-label queue: an audited label stays hidden until its release boundary, releases exactly once |
| `policy.py` | five monitoring policies, the audit/promotion engine (`run_stream`), and the `label_delay` extension connecting delayed evidence to real retraining decisions |
| `retraining_economics.py` | the exact finite-horizon retrain-schedule optimizer, the conditional count bound, and the finite-capacity training/deployment queue |
| `figures.py` | renders every figure below from real run data, never from a mock |
| `run_study.py` | regenerates the full figure set (`ensure_figures()` is idempotent) |
| `figures/` | each rendered PNG with its source CSV |

## Headline claims and the runs behind them

| Claim | Evidence |
| --- | --- |
| A label-free alarm can be time-uniform | the sequential monitor's empirical ever-alarm rate is 0 at alpha 0.10 across 20 no-harm streams, with detection after a genuinely harmful shift |
| The Regol-style retrain-count ceiling holds only when its uniform adjacent-model-gap assumption is checked | 4 of 5 measured settings satisfy the checked bound (optimized counts 0–6 against ceilings 3.9–10.9); the assumption-violating setting is drawn with no ceiling, not a false one |
| A finite-capacity retraining queue can strand budget three different ways | budget is now split into completed, still-in-flight-at-the-horizon, and never-requested capacity, instead of one conflated "stranded" number |
| Delaying an audited label changes both when and whether deterioration is caught | `threshold` policy, 10 paired seeds, delay ∈ {0, 2, 4}: median detection delay 12 → 14 → 16 steps, miss rate 0% → 10% → 10% |
| Raw average cost under delay is not the right number to read | mean cost is non-monotonic (13.46 → 14.01 → 13.07); priced per successfully detected seed instead, every delayed arm costs strictly more (13.46 → 15.56 → 14.52) |

## Notebook relationship

The publication layer is in `../notebooks/`. Notebook 01 derives each paper's
estimator/guarantee and checks it on synthetic data. Notebook 02 builds a
synthetic deployed classifier and shows a naive monitor's failure modes.
Notebook 03 reuses notebook 02's artifacts verbatim and applies the retraining
controller, the finite-capacity queue, and the immediate-vs-delayed decision
comparison. The section-by-section guide is in `../notebooks/README.md`.

## What this is not

No result here is a production monitoring or retraining certificate. The
delay-vs-miss-rate trade-off is measured only for one policy, ten seeds, and
three delay values on a synthetic grid — it is not a general law about label
delay. The Regol-style count bound is never claimed where its adjacent-model
assumption is unchecked, and Dasari's own empirical result (queueing can more
than halve effective retraining budget) is kept as external evidence, not
reproduced on real data. All data is synthetic and CPU-only.
