# Execution evidence — training-dynamics observability

This directory contains the analytical objects, seeded experiment pipeline, machine-readable results, and figure evidence used by the three executed notebooks. The study is synthetic, CPU-only, and deliberately scope-limited: it demonstrates observability methods rather than claiming production-model stability.

## The three sources

| arXiv ID | Paper | Role in the notebooks |
| --- | --- | --- |
| 2406.08307 | Banerjee et al., *Measuring training variability from stochastic optimization using robust nonparametric testing* | bounded-null variability diagnostics and alpha trimming |
| 2412.20553 | Andreyev & Beneventano, *Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD* | batch sharpness, the one-sided catapult condition, and interpretation of sharpness traces |
| 2609.02728 | Wang et al., *Momentum in large-batch training* | order-level critical-batch and learning-rate transitions for SGD, Polyak, and Nesterov momentum |

## Contents

| Path | What it holds |
| --- | --- |
| `foundations/` | Rayleigh-quotient checks, EoSS simulations, momentum boundaries, variability diagnostics, and source-default audits |
| `walkthrough/` | train-only preprocessing, the seeded teacher–student task, incident logic, artifact reuse, and paired-seed comparisons |
| `foundations_results.json` | machine-readable outputs behind notebook 01 |
| `walkthrough_results.json` | data, seed, incident, and reuse evidence behind notebooks 02–03 |
| `run_ledger.json` | run IDs, configurations, hashes, and result hashes |
| `figures/` | each rendered PNG with its source CSV and, for most figures, a caption JSON |

## Headline results

| Claim | Evidence |
| --- | --- |
| The analytic and computed batch Rayleigh quotient agree | 1.2013422819 in both paths |
| The sufficient EoSS condition is non-vacuous in the controlled quadratic | 14/14 above-threshold seeded trials catapult |
| Momentum changes the order-level batch-scaling transitions | the illustrated reference gives $b_1=0.40$, $b_2=0.58$, $b_3=0.67$ |
| The real walkthrough arms are not forced into an instability story | learning rates 0.04 and 0.02 remain below estimated boundaries 1.77 and 1.81 |
| A single seed does not establish a winner | across eight paired seeds, four favor each arm; B − A is −0.0030 with 95% interval [−0.0252, +0.0192] |
| Notebook 03 audits the unchanged notebook-02 run | the notebook hash, data hash, config hash, and shared run IDs are recorded under `reuse` |

## Notebook relationship

The notebooks are in `../notebooks/`. Notebook 01 builds the analytical vocabulary; notebook 02 creates the seeded task and telemetry; notebook 03 reuses notebook 02's artifacts and applies the sharpness, decision-surface, and paired-run audits. The detailed section-by-section guide is in `../notebooks/README.md`.

## What this is not

The deterministic boundary is order-level, a below-threshold sharpness trace is inconclusive because the cited condition is one-sided, and eight paired seeds do not provide paper-scale power. No result here is a production safety certificate or evidence about a real model or dataset.
