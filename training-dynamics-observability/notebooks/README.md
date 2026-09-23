# Notebooks

| Notebook | Shows | Runtime |
|---|---|---|
| `01_research_foundations.ipynb` | Analytic Rayleigh checks, momentum recurrences and their order-level scaling, bounded-null variability diagnostics, and trace-labelled cross-paper context. | CPU, executed |
| `02_project_walkthrough_part1.ipynb` | A seeded teacher-student task with train-only preprocessing, an experiment ledger, and incident telemetry. | CPU, executed |
| `03_project_walkthrough_part2.ipynb` | Sharpness, batch/LR/momentum decision surfaces, and paired-run comparisons applied to the unchanged NB2 artifacts. | CPU, executed |

Each figure is preceded by the claim it checks, a printed self-check tied to a saved CSV artifact, and a “How to read this chart” note. Figure links in this folder are relative: `../exec/figures/fig-rXX-*.png`.

## 01 — Research foundations

- **Analytic objects.** Validates a batch Rayleigh quotient against a known quadratic, makes the three momentum recurrences explicit, and contrasts a classical KS result with alpha trimming under a bounded null.
- **Order-level boundaries.** Treats batch-scaling transitions `b1`, `b2`, and `b3` as a qualitative/order-level map, not a universal exact constant.
- **Trace context.** Separates loss, curvature, and noise; compares reported EoSS paper settings with the toy; and records specified versus unspecified reproducibility defaults.
- **Figures.** Includes the formal setup, paper-versus-toy comparison, momentum transitions, window/transient sensitivity, and bounded-null coverage diagnostics.

## 02 — Project walkthrough, part 1: seeded task and telemetry

- **Data and preprocessing.** Generates a synthetic teacher-student problem with a known latent structure and applies preprocessing on training data only.
- **Experiment discipline.** Records seeds, configurations, hashes, and run outcomes in an auditable ledger rather than conflating reruns with new evidence.
- **Incident telemetry.** Creates an incident bank whose online diagnosis is restricted to the telemetry available at that time.
- **Figures.** Publishes momentum-grid, transient, trim-sensitivity, and detector-replay artifacts from live computation.

## 03 — Project walkthrough, part 2: observability on the unchanged run

- **Reuse proof.** Loads NB2’s published artifacts and verifies its pinned notebook hash; it does not regenerate data or quietly select a friendlier sample.
- **Sharpness and catapult evidence.** Computes minibatch Hessian/Rayleigh evidence on the real seed/arm trace; below-threshold results remain inconclusive, not safe.
- **Decision and comparison views.** Applies the deterministic batch/LR/momentum boundary to the real arms and compares the real paired seed metrics through classical KS and alpha trimming.
- **Figures.** Includes the sharpness trace, decision surface, and honest paired-run comparison.

## Limitations

- All data are synthetic and CPU-only; no result establishes production stability, false-alarm control, or real-model performance.
- The momentum statements are order-level and scope-limited; adaptive optimizers and non-quadratic losses remain open.
- Eight paired seeds support a transparent portfolio diagnostic, not the power of a paper-scale ensemble.
