# Claims payment-integrity labels

This is a synthetic-data research study — not production claims-review
code, and not evidence about real payment-integrity performance. Everything runs offline on CPU against a
deterministic generator: no real claims data, no PHI, no network access. It focuses on noisy reviewer labels,
selective review under a finite capacity, and a joint finite-sample certificate over risk, acceptance and
dollar-weighted utility, with every claim kept separate from what the synthetic study derives on its own.

## The three papers

| arXiv ID | Paper | Role in the notebooks |
| --- | --- | --- |
| 2306.03288 | Ibrahim, Nguyen & Fu, *Deep Learning from Crowdsourced Labels: Coupled Cross-Entropy Minimization, Identifiability, and Regularization* | CCEM noisy-reviewer model, Theorem 1 permutation identifiability, the specialist/weak-anchor stress arm (NB1 §B, NB2 Stage 3) |
| 2512.12844 | Xu, Guo & Wei, *Selective Conformal Risk Control* | SCRC's selective coverage / conditional selected risk, the exchangeable-calibration guarantee (NB1 §C, NB3 Stage 3a) |
| 2606.08517 | Yu & Liu, *A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control* | SCoRC's Algorithm 1 finite-grid certificate (Clopper-Pearson + empirical-Bernstein, `INFEASIBLE` branch) (NB1 §D, NB3 Stage 3b) |

## Notebooks

| Notebook | What it does |
| --- | --- |
| `01_research_foundations.ipynb` | derives each paper's model/procedure/theorem and checks it against a hand-computed case or a toy simulation, with paper-reported numbers kept in their own labelled blocks, never mixed into the synthetic numbers |
| `02_claim_stream_and_scorers.ipynb` | builds the synthetic claim stream with known latent truth, fits a naive and a noise-aware (CCEM) scorer, and evaluates ranking quality (ROC, precision@K, a reliability diagram) against a fresh unseen stream |
| `03_selective_review_and_certificates.ipynb` | reuses notebook 02's stream and scorers unchanged, then runs the selective-capacity, joint-certificate and dollar-utility studies, closing with a calibration-sampling falsifier and a cost accounting section |

## Headline results

| Result | Numbers | Where |
| --- | --- | --- |
| CCEM identifiability, control vs. coverage-break stress | aligned error 0.020-0.027 (anchored) vs. 0.156-0.186 (specialist/weak-anchor), 4 seeds x 2 regimes, n=2000; every seed/metric effect size excludes 0 | `01_identifiability` figure; NB2 forest plot |
| Reviewer confusion-matrix recovery, aligned | mean \|true - recovered\| = 0.082-0.086 (anchored) vs. 0.207-0.211 (stress) | NB1 confusion-heatmap figure |
| Noise-aware vs. naive scorer, fresh stream | AUROC naive 0.46 (at/below chance) vs. noise-aware 0.59; precision@K noise-aware beats naive and the 0.063 base rate at every K in {10,30,60,100} | NB2 ROC + precision@K figures |
| Selective capacity study | noise-aware accepted risk 0.058-0.069 < naive 0.064-0.070 at every capacity/seed; SCoRC certificate at alpha=0.05 is `INFEASIBLE` at every capacity, shown with `feasible=false`, never hidden | `02_capacity_risk` figure |
| Dollar-weighted utility (synthetic proxy) | seed 11: noise-aware $412/$1,363/$2,383 recovered vs. naive $0/$0/$5 at 30/90/180 reviewer minutes; both catch a large claim by seed 23 | `03_dollars_recovered` figure |
| Calibration-sampling falsifier | randomized-audit control exceedance 0.63 [0.31, 0.86] vs. selected-high-risk 1.00, selected-low-risk 0.88 (8 seeds, alpha=0.05); calibration-risk differences +0.048 [0.039, 0.057] and -0.026 [-0.040, -0.012] | NB3 falsifier figure + per-seed spread chart |

## Contents

- `claims_integrity/` is the package: CCEM (`ccem.py`), the SCRC/SCoRC selective machinery (`selective.py`),
  the synthetic claim-stream generator (`synthetic.py`), pre-processing/scoring (`scoring.py`), the four studies
  behind the figures and notebook numbers (`experiments.py`), the two observability monitors (`monitors.py`),
  the run-ledger helpers (`ledger.py`: seed, config hash, capacity, wall-clock runtime), and the figure builders/renderers (`figures.py`).
- `figures/` pairs every rendered PNG with its underlying CSV.
- `make_figures.py` is the CLI entry point that rebuilds everything in `figures/`.

The executed notebooks are in `notebooks/`.

## Reproduce

Runs on CPython 3.12 with the packages in `requirements.txt` — no GPU, no network.

```bash
cd claims-payment-integrity-labels
pip install -r requirements.txt
python3 make_figures.py                     # rebuilds figures/*.csv and *.png
python3 -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_research_foundations.ipynb \
    notebooks/02_claim_stream_and_scorers.ipynb \
    notebooks/03_selective_review_and_certificates.ipynb
```

Notebook 03 regenerates notebook 02's canonical stream (seed 11) from the same config and asserts the
digest matches before reusing it, rather than reading a saved file — so it should run after notebook 02, but it
does not depend on notebook 02's process state.

## What this is not

Every default the three papers do not settle — the synthetic claim stream, the specialist/weak-anchor stress
queue, the dollar-recovery proxy, the one-stage SCRC routing collapse, and the two observability monitors — is
labelled **derived here** inside the notebooks, distinct from every **source-backed** quote.

## Limitations

The strict `alpha=0.05` certificate is `INFEASIBLE` at every studied capacity — the first certifying target in
the probed bracket is close to `alpha≈0.11`, so the SCoRC branch does not certify anything deployable at the
paper's stricter target. Provider false-positive burden and rejected-case safety are not instrumented. The SCoRC
paper's own benchmark numbers are not reproduced or quoted.
