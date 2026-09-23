# Execution evidence — fraud/identity-risk model training under delayed, censored and selective labels

This directory contains the estimators, monitors, seeded synthetic-stream generator, machine-readable results, and figure evidence used by the three executed notebooks. The study is synthetic, CPU-only, and deliberately scope-limited: it demonstrates recover/evaluate/monitor methodology for fraud-label delay and selection rather than claiming production-model performance.

## The three sources

| arXiv ID | Paper | Role in the notebooks |
| --- | --- | --- |
| 2201.06277 | Coudray, Keribin, Massart, Pamphile, *Risk bounds for PU learning under Selected At Random assumption* | the propensity-explicit unbiased SAR risk (Eq. 12–13) and its excess-risk bound (Eq. 15–16); the Cannings-condition mechanism distinguishing naive from aware training |
| 2407.13009 | Kozodoi, Lessmann, Alamgir, Moreira-Matias, Papakonstantinou, *Fighting Sampling Bias* | Algorithm 1 reject inference with a non-oracle, accepts-trained prior, and the BASL self-learning stop rule |
| 2505.04608 | Prinster, Han, Liu, Saria, *WATCH: Adaptive Monitoring for AI Deployments* (corrected version, as ingested) | the weighted-conformal test martingale and its one-sided anytime false-alarm guarantee `P(ever alarm) <= 1/c` |

## Contents

| Path | Employer-facing evidence |
| --- | --- |
| `config.py` | canonical and notebook-demo configurations, seeds, and run-identity hashing |
| `data.py` | the seeded synthetic fraud stream: delayed/selective labels, typologies, an adversarial drift, a trained model score |
| `recovery.py` | the SAR risk estimator, its independent-truth unbiasedness check, sabotage variants, and the Eq. 15 excess-risk study |
| `evaluation.py` | Kozodoi's reject-inference Algorithm 1 with a non-oracle prior, BASL, and the naive-vs-aware training mechanism restricted to where Coudray's theory predicts it matters |
| `monitoring.py` | the WCTM built from Prinster's Eq. 6/9 betting martingale, the one-sided alarm-rate criterion, and the equal-footing comparison against label-free monitors |
| `policies.py` | equal-cost label-budget policies (random, risk-first, uncertainty-first, diverse-typology) |
| `ledger.py` | a per-run cost/wall-clock ledger with cache-aware cost accounting |
| `figures/` | the twelve frozen figures (three per round, one shared triptych set from round 1) with their sidecar CSV data and provenance (seed, config hash, run id) |
| `results/run_manifest.json` | the canonical run's identity and the manifest of every figure it produced |
| `results/notebook_ledger.jsonl`, `results/r2_monitoring_ledger.jsonl` | the run-cost ledger evidence behind the notebooks' observability sections |

## What this is not

Not production-model experience: all data is synthetic, all figures are seeded and reproducible, and every claim not directly stated by one of the three papers is labelled "derived here" in the notebooks. This is a starting point for the reader to extend into their own portfolio project.
