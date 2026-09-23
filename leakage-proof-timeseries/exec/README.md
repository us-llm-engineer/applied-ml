# Execution evidence — leakage-proof time-series validation

This repository is a synthetic-data research study and a starting point for further work — not production trading
code, and not evidence of live performance. Everything runs offline on CPU against a deterministic generator: no
market data, no network access, no external spend (the ledger records `external_spend_usd = 0.0` on every row).
It focuses on leakage-proof validation on time-ordered data, regime-aware testing, honest baselines, versioned
data, seeds, ablations, and every claim backed by the run that produced it.

## The three sources

| arXiv ID | Paper | Role in the notebooks |
| --- | --- | --- |
| 2408.04607 | Atanasov, Zavatone-Veth & Pehlevan, *Risk and cross validation in ridge regression with correlated samples* | ordinary GCV vs the CorrGCV estimator for correlated samples (NB1 §2, NB3 §2) |
| 2606.01650 | Pav, *Post-Selection Estimation of Sharpe Ratios* | winner-selection bias and shrinkage of the selected Sharpe (NB1 §3, NB3 §3) |
| 2309.09111 | Shekhar & Ramdas, *Reducing sequential change detection to sequential estimation* | repeated-FCS detector, ARL control and detection delay (NB1 §4, NB3 §4) |

## Notebooks

| Notebook | What it does |
| --- | --- |
| `01_research_foundations.ipynb` | derives each paper's estimator/test and verifies its guarantee by simulation, with the paper numbers kept separate from the toy numbers |
| `02_project_walkthrough_part1.ipynb` | generates the synthetic market with known latent truth and runs one pipeline under paired clean/leaky protocols |
| `03_project_walkthrough_part2.ipynb` | reuses part-1's artifacts unchanged and applies the three audits: correlation-aware ridge risk, selection-corrected Sharpe, single-change monitoring — then a cost docket |

## Headline claims and the runs behind them

| Claim | What it says | Run ID |
| --- | --- | --- |
| C6 / C14 | CorrGCV is attached to the linear ridge only; on an empirical Gram the part-2 comparison is a derived-here diagnostic with a mismatch warning | nb3-corrgcv, nb1-corrgcv |
| C10 | The synthetic dataset carries the planted ingredients: AR(1) edge, regimes, a zero-edge segment and a look-ahead feature | nb2-generator |
| C12 | Ridge and a small CPU GRU trained over 3 seeds x 6 protocols, with per-fold and per-seed results stored | nb2-clean-ridge-s1, nb2-clean-seqnn-s1 |
| C13 | On this generator only the label-touching leaks inflate mean IC (look-ahead features +0.51, overlapping labels +0.11); a random split and global normalisation do not measurably move it, and a leaky run still outscores every clean run | nb2-future-ridge-s1, nb2-all-ridge-s1 |
| C15 | The candidate set is declared (hash and sequence tick) before any score is inspected; shrinkage carries an applicability caveat | nb3-selection |
| C16 | One bounded monitored stream, one alarm at series t = 946 (delay 126), no restart; the theorem ARL lower bound 1/alpha = 20 is reported separately from the fully censored restricted mean (not an ARL estimate), and the stream's dependence validity is not established | nb3-monitor |
| C17 | Every docket statement is backed by a recorded run, a raw trace, a derived-here default, or left unsettled | nb3-docket |
| C20 / C21 | The renormalised CorrGCV solves kappa and kappa_tilde jointly so the subordination identity kappa*kappa_tilde/lambda = 1/df1_tilde and the duality q*df1 = df1_tilde hold to stated residuals on every configuration; with K = I it reduces to ordinary GCV within solver tolerance | nb1-identities, nb1-corrgcv |
| C22 / C24 | All four of q3's estimators (GCV1, GCV2 = S^2 R_in, Carmack GCCV, CorrGCV) are reported against latent risk in one figure; in the xi = 1e2 strong-correlation regime CorrGCV tracks R_out while GCV1/GCV2 fail | nb1-corrgcv |
| C27 / C28 | The James-Stein shrinkage uses q5's (k - 2)/n constant, and four of Pav's family (naive, expected-max, James-Stein, SURE) are compared against the latent selected Sharpe over >= 500 seeded repetitions per regime | nb1-selection |
| C32 / C34 | Theorem 2.5's bound 3/(1-alpha)*u is computed from the notebook's own width envelope and reported against the measured mean delay with its slack; the null-run ARL check is informative (a non-degenerate run-length distribution) | nb1-fcs |

## Contents

- `*.py` contains the estimators, detectors, synthetic generator, audit logic, and shared notebook utilities.
- `artifacts/` contains the versioned datasets, splits, predictions, metrics, and run ledger consumed by the walkthroughs.
- `figures/` pairs every rendered PNG with its underlying CSV and provenance metadata.

The executed notebooks themselves sit one level above this directory so the entire study remains under `leakage-proof-timeseries/`.

## Reproduce

Run from the repository root; regenerated artifacts remain inside this collection:

```
uv sync --project leakage-proof-timeseries/exec
uv run --project leakage-proof-timeseries/exec jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=480 leakage-proof-timeseries/01_research_foundations.ipynb
uv run --project leakage-proof-timeseries/exec jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=480 leakage-proof-timeseries/02_project_walkthrough_part1.ipynb
uv run --project leakage-proof-timeseries/exec jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=480 leakage-proof-timeseries/03_project_walkthrough_part2.ipynb
```

Notebook 03 reads `nb2_dataset.csv`, `nb2_profile.json`, `nb2_splits.json`, `nb2_runs.json`,
`nb2_predictions.csv` and `nb1_fcs.json`, so it must run after notebooks 01 and 02; it writes no dataset of its
own. Every run appends a row to `leakage-proof-timeseries/exec/artifacts/run_ledger.jsonl` with the dataset hash, protocol, fold
boundaries, seed, metric, wall time, claim ids and zero external spend.

## What this is not

Every operational default the three sources do not settle (purge and embargo widths, fold-local normalisation,
sequence design, detector tolerances, transaction costs) is labelled **derived here** inside the notebooks. The
walkthroughs close with "Limitations — what the three sources do not settle", listing only the gaps that survived
an explicit attempt, each with its raw trace and a `tried:` record: production purge/embargo widths,
dependence-valid monitoring for heavy-tailed live metrics, sequence architecture and tuning budget beyond the CPU
demo, transaction costs and turnover, the generality of the leak-null results, and CorrGCV beyond the matched
linear-ridge setting.
