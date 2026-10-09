# Leakage-proof time-series validation — code and artifacts

This repository is a synthetic-data research study — not production trading
code, and not evidence of live performance. Everything runs offline on CPU against a deterministic generator: no
market data, no network access, no external spend.
It focuses on leakage-proof validation on time-ordered data, regime-aware testing, baselines, versioned
data, seeds and ablations.

## The three papers

| arXiv ID | Paper | Role in the notebooks |
| --- | --- | --- |
| 2408.04607 | Atanasov, Zavatone-Veth & Pehlevan, *Risk and cross validation in ridge regression with correlated samples* | ordinary GCV vs the CorrGCV estimator for correlated samples (NB1 §2, NB3 §2) |
| 2606.01650 | Pav, *Post-Selection Estimation of Sharpe Ratios* | winner-selection bias and shrinkage of the selected Sharpe (NB1 §3, NB3 §3) |
| 2309.09111 | Shekhar & Ramdas, *Reducing sequential change detection to sequential estimation* | repeated-FCS detector, ARL control and detection delay (NB1 §4, NB3 §4) |

## Notebooks

| Notebook | What it does |
| --- | --- |
| `01_research_foundations.ipynb` | derives each paper's estimator/test and verifies its guarantee by simulation, with the paper numbers kept separate from the toy numbers |
| `02_naive_pipeline_leakage_audit.ipynb` | generates the synthetic market with known latent truth and runs one pipeline under paired clean/leaky protocols |
| `03_three_audits_on_the_pipeline.ipynb` | reuses notebook 02's artifacts unchanged and applies the three audits: correlation-aware ridge risk, selection-corrected Sharpe, single-change monitoring — then a cost docket |

## Headline results

The measured results and their figures are summarised in the project README and recorded in `artifacts/` and `figures/`.

## Contents

- `*.py` contains the estimators, detectors, synthetic generator, audit logic, and shared notebook utilities.
- `artifacts/` contains the versioned datasets, splits, predictions and metrics consumed by the walkthroughs, plus the run ledger.
- `figures/` pairs every rendered PNG with its underlying CSV and provenance metadata.

The executed notebooks sit one level above this directory.

## Reproduce

Run from the repository root; regenerated artifacts remain inside this collection:

```
uv sync --project leakage-proof-timeseries/leakage_study
uv run --project leakage-proof-timeseries/leakage_study jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=480 leakage-proof-timeseries/01_research_foundations.ipynb
uv run --project leakage-proof-timeseries/leakage_study jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=480 leakage-proof-timeseries/02_naive_pipeline_leakage_audit.ipynb
uv run --project leakage-proof-timeseries/leakage_study jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=480 leakage-proof-timeseries/03_three_audits_on_the_pipeline.ipynb
```

Notebook 03 reads `nb2_dataset.csv`, `nb2_profile.json`, `nb2_splits.json`, `nb2_runs.json`,
`nb2_predictions.csv` and `nb1_fcs.json`, so it must run after notebooks 01 and 02; it writes no dataset of its
own. Every run appends a row to `leakage-proof-timeseries/leakage_study/artifacts/run_ledger.jsonl` with the dataset hash, protocol, fold
boundaries, seed, metric and wall time.

## What this is not

Every operational default the three papers do not settle (purge and embargo widths, fold-local normalisation,
sequence design, detector tolerances, transaction costs) is labelled **derived here** inside the notebooks. The
walkthroughs close with "Limitations — what the three papers do not settle", listing the gaps that remain: production purge/embargo widths,
dependence-valid monitoring for heavy-tailed live metrics, sequence architecture and tuning budget beyond the CPU
demo, transaction costs and turnover, the generality of the leak-null results, and CorrGCV beyond the matched
linear-ridge setting.
