# Notebooks

| Notebook | Shows | Runtime |
|---|---|---|
| `NB1_mathematical_foundations.ipynb` | Derivation and simulation-based verification of all three papers' core objects: the SAR risk identity and its excess-risk bound (Coudray), Algorithm 1 reject inference and the BASL stop rule pursued across seven conditions (Kozodoi), and the weighted-conformal test martingale's one-sided alarm guarantee (Prinster). | ~90 s |
| `NB2_baseline_walkthrough.ipynb` | A synthetic fraud stream with known latent truth (0.4% prevalence, four typologies, two deliberately ambiguous pairs, an adversarial drift), the naive delayed-label pipeline built on it, and the mechanism-restricted comparison showing where naive training genuinely under-calls fraud. | ~90 s |
| `NB3_corrected_walkthrough.ipynb` | Re-executes NB2 from disk (not copied), then applies the corrected SAR/Bayesian/WCTM estimators to the same stream, an equal-footing comparison of three monitors on one score definition, and a reconciliation check against the canonical figures. | ~7 min |

Each figure is preceded by the claim it checks, a printed self-check computed against a canonical sidecar CSV or an independent ground truth, and a "How to read this chart" note. Anything not stated by one of the three papers is labelled "derived here". Figure links in this folder's exec companion are relative: `../exec/figures/*.png`.

## NB1 — Mathematical foundations

- **SAR risk and its excess-risk bound.** Rebuilds Eq. 12/13 term by term against a hand example, checks the Monte-Carlo oracle gap against an independent large-sample truth (with a sabotage variant that must fail the same check), and measures Eq. 15's excess-risk ratio against a free-constant reference across a training-size/propensity-floor grid.
- **The Cannings condition.** Derives exactly where naive training on the observed label differs from the population-optimal classifier — `e(x) < 1/(2*eta(x))` where `eta(x) >= 1/2` — on an `(eta, e)` grid.
- **Reject inference under a non-oracle prior.** Algorithm 1 with an accepts-trained, never-oracle prior, judged against an unselected holdout; a prior-corruption sweep.
- **BASL, pursued not assumed.** The stop rule's first draw is checked against seven conditions (three `j_max` values, two `gamma` values, a wider sampling rate, a 5.5x larger applicant pool) — all non-improving, which is the paper's own criterion working correctly, not a bug.
- **WCTM validity.** The betting function and weighted-conformal p-value are checked for null uniformity, and the one-sided alarm-rate bound is checked against a peeking-monitor sabotage that must exceed it.

## NB2 — Baseline walkthrough

- **The stream.** A synthetic fraud stream with known latent truth: label delay by typology, a maturation curve, an approval rule that never reads the latent outcome, and two genuinely ambiguous typology pairs so a perfect classifier is not achievable by construction.
- **The naive pipeline.** Treats every unresolved label as a confirmed negative and evaluates itself only on labels it happens to have — precision/recall gaps against the population truth, a selection-damage and a delay-damage view.
- **The Cannings-violating subset.** Builds the exact subset Coudray's mechanism predicts naive training under-calls, and shows the gap concentrates there (with the whole-population contrast shown for scale, not required to be significant).
- **BASL across seeds.** The same never-regresses claim as NB1, checked across six seeds instead of six hyperparameter conditions.

## NB3 — Corrected walkthrough

- **Reuses NB2 by execution**, not by copying its cells, so both notebooks report from the identical cached run.
- **Corrected estimators on the shared stream.** SAR with a known one-sided propensity, the excess-risk bound, Algorithm 1's non-oracle prior, and WCTM on matured labels — ready, selective, and lagged label lanes reported separately, with no validity claim outside the paper's own ready-label reference.
- **Equal-footing monitor comparison.** Three monitors (a label-free alert rate, a label-free drift statistic, and WCTM) on the same classification score, same stream, same three metrics (false-alarm incidence, detection delay, miss rate) in one composed figure — deliberately run at the canonical stream scale, because the delay/miss ranking needs the drift's signal-to-noise ratio, which does not converge from repetition count alone at a reduced demo stream size.
- **Label-budget policies and a run ledger.** Four equal-cost policies compared, and cost/wall-clock evidence with cached-rerun cost accounting.
- **Limitations.** Closes with what the three sources do not settle: unknown-propensity guarantees, prior dependence, delayed/selective monitoring validity, and the absence of any label-focusing guarantee — each with its trace and the pursuit attempted before it was called a limit.

## Executed evidence and reproduction

The committed notebooks retain their printed checks, tables, and embedded figures: NB1 has 76 cells (34 with outputs), NB2 has 80 (37 with outputs), and NB3 has 85 (39 with outputs). There are no saved error outputs or failing self-check lines in the published runs. Their canonical PNG/CSV figure pairs and provenance live in `../exec/figures/` and `../exec/results/run_manifest.json`.

From the collection root, rerun them on CPU with the already-installed scientific Python stack:

```bash
python3 -m exec.render_all_figures
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/NB1_mathematical_foundations.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/NB2_baseline_walkthrough.ipynb
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/NB3_corrected_walkthrough.ipynb
```

The generator, labels, and outcomes are synthetic. These notebooks demonstrate a reproducible analytical workflow, not live fraud-model performance or a deployed monitoring guarantee.
