# Notebooks

| Notebook | Shows | Runtime |
|---|---|---|
| `01_research_foundations.ipynb` | Derivation and reproduction of the three papers' core identities: CorrGCV, post-selection Sharpe estimators, and the repeated-FCS change detector, each checked against its source's reported numbers. | ≈190 s |
| `02_project_walkthrough_part1.ipynb` | A synthetic regime-switching dataset, a naive validation pipeline built on it, and a paired-switch audit of six leakage protocols against that pipeline. | ≈220 s |
| `03_project_walkthrough_part2.ipynb` | The three audits from notebook 01 (correlated-sample ridge risk, selection-corrected Sharpe, sequential change monitoring) applied to the unchanged pipeline from notebook 02, including the honest negative on non-stationary CorrGCV. | ≈25 s |

Each figure is preceded by the claim it is checking, a self-check computation against the source paper's identity or reported number, and a "How to read this chart" note.

Figure links in this file are relative: `exec/figures/fig-XX.png`.

## 01 — Research foundations

- **§1 Source identities and the renormalised-ridge identities.** States CorrGCV's Eq. 30 and solves the coupled renormalisation — duality $q\,df_1=\tilde{df}_1$ and subordination $\kappa\tilde\kappa/\lambda=1/\tilde{df}_1$ — jointly rather than substituting one into the other; checks the identity residual ($4.3\times10^{-16}$).
- **§2 The four ridge-risk estimators under a power-law spectrum.** Ordinary GCV, Altman GCV, CorrGCV, and the latent (oracle) risk, defined on a common power-law spectral model ($\alpha=1.8$) used throughout the reproduction.
- **§3 Post-selection family.** The six Sharpe-ratio estimators from Pav (naive, expected-max, James–Stein, SURE, GMLEB, polyhedral) and the rank-agreement metrics (Kendall $\tau$, Spearman $\rho$) used to compare them.
- **§4 Repeated-FCS algorithm and the null-run ARL check.** The repeated forward-confidence-sequence detector construction and a null-hypothesis run confirming its average run length behaves as the theory predicts before any change is introduced.
  - **§4b Theorem 2.5 bound and Proposition 2.7 horizon.** The detection-delay upper bound and the horizon condition it depends on, stated ahead of the sweep in §7.
- **§5 Reproduction (Figs A1–A3, D2).** Reproduces the paper's Fig. 1 (risk vs. $T$ across three correlation families and two spectral regimes; `fig-a1.png`), the CorrGCV-advantage phase diagram over $(\xi,\lambda)$ (`fig-a2.png`), a third supporting sweep (`fig-a3.png`), and a deterministic-equivalent-vs-simulation check (`fig-d2.png`).
- **§6 Post-selection rank metrics (Figs B1–B3).** Rank agreement vs. backtest length with the paper's $n=1008$ point overlaid (`fig-b1.png`), a second rank-metric view (`fig-b2.png` — Dolan–Moré performance profiles), and the polyhedral estimator's breakdown as the top-two gap shrinks (`fig-b3.png`).
- **§7 e-detector, ARL across α, PFA detector (Figs C1–C3, D1).** Detection delay vs. change size against Theorem 2.5's bound (`fig-c3.png`, with two supporting views in `fig-c1.png`/`fig-c2.png`), the average-run-length check across significance levels $\alpha$, the Remark 2.3 PFA-controlled detector's null false-alarm rate (Wilson upper bound 0.0019 at $\alpha=0.2$ over 2000 runs), and the Eq. (5) e-detector inclusion check across the full run pool (`fig-d1.png`: 120 runs total, 60 alarmed, and on all 60 alarmed runs the e-detector crossed its threshold no later than the alarm, strictly earlier on 28).
- **Source gaps.** Records where a source paper leaves a parameter, bound, or regime unspecified (e.g. purge/embargo widths, non-stationary CorrGCV) rather than filling the gap silently.

## 02 — Project walkthrough, part 1: naive pipeline and leakage audit

- **§1 Data and planted regimes.** Generates the synthetic regime-switching series used by both walkthrough notebooks, with the planted regime boundaries recorded for later comparison.
- **§2 Pre-processing and paired leakage switches.** Builds paired variants of the pre-processing and splitting logic that differ only in one leakage-relevant switch at a time, covering the six protocols audited in §5, including look-ahead features, overlapping labels, random split, and global normalisation.
- **§3 Ridge and small GRU training.** Trains the two model families used throughout the walkthrough on each of the six pipeline variants.
- **§4 Seed/split distributions.** Characterises how much of the measured effect is seed noise versus a genuine effect of the leakage switch, across repeated seeds and splits.
- **§5 Metric observability and the experiment ledger (Figs E3, E2).** Reports the leakage-protocol audit as a forest plot of IC inflation relative to the leak-free baseline (`fig-e3.png`: +0.51 for look-ahead features, +0.11 for overlapping labels, no measurable inflation for random split or global normalisation) and evaluates Theorem VI.1 on a near-horizon test fold, finding a risk ratio of 0.80 against an independent fold (about 20% optimistic), inside the theorem's predicted band (`fig-e2.png`). All runs are recorded in an experiment ledger with seed, config, and run id for provenance.
- **Limitations.** The six protocols cover common leakage failure modes on this generator, not an exhaustive taxonomy; a null result for a given protocol is evidence about this generator, not a general guarantee that protocol is leakage-proof.

## 03 — Project walkthrough, part 2: three audits on the part-1 pipeline

- **§1 Reuse part-1 unchanged (provenance).** Loads the exact dataset and pipeline artifacts produced by notebook 02, verified against the run ledger, so the three audits below are evaluated on identical data rather than a re-generated copy.
- **§2 Correlation-aware ridge risk.** Applies CorrGCV, ordinary GCV, and Altman GCV to the walkthrough's ridge model.
- **§3 Selection-corrected Sharpe.** Applies the six post-selection Sharpe estimators from notebook 01 to the walkthrough's model-selection step.
- **§4 Monitoring a bounded metric across a regime change (Fig E1 + embedded E2/E3).** Runs the repeated-FCS detector on a bounded performance metric spanning the planted regime change; reports the honest negative that estimating a stationary Toeplitz $\hat K$ reduces CorrGCV's mean relative error 1.56× (85.9 → 54.9) on this non-stationary data but leaves it far worse than ordinary GCV (0.32) (`fig-e1.png`), alongside the embargo-optimism and leakage-audit figures from notebook 02 (`fig-e2.png`, `fig-e3.png`) for side-by-side reading.
- **§5 Costs, contradictions, and open decisions.** Discusses trade-offs and open modelling decisions across the three audits, including embargo width and $\hat K$ estimation under non-stationarity, left unresolved by the source papers.
- **Limitations.** CorrGCV's failure under non-stationarity is a property of the estimator as specified in its source paper, not a bug in this implementation; the monitoring audit covers one change point with no restart policy.
