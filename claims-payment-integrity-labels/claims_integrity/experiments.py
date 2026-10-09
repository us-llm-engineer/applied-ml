"""Studies: capacity comparison, certificate control, calibration falsifier.

All studies are *derived here* adaptations over synthetic data with known latent
truth.  They are designed to exercise the source-backed machinery (CCEM Eq. 6;
SCRC Eq. 11/15; SCoRC Algorithm 1) and to surface assumption failures rather
than hide them.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .ccem import fit_ccem, vote_posterior
from .config import CONFIG
from .scoring import (
    build_features,
    ccem_posterior,
    fit_naive_scorer,
    identify_clean_class,
    overpayment_score,
    three_way_split,
)
from .selective import (
    acceptance_rate,
    compare_calibration_sampling,
    scorec_certificate,
    scrc_inductive_policy,
)
from .synthetic import make_claim_stream

#: Data-independent finite candidate grids on the derived scores: accept when
#: the model-implied overpayment probability is at most the msp threshold, or
#: when the clean-vs-next margin is at least the margin threshold.
MSP_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
MARGIN_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
CERT_GRID = tuple(("msp", t) for t in MSP_GRID) + tuple(("margin", t) for t in MARGIN_GRID)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (deterministic, no resampling)."""

    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def fit_scorers(stream: dict[str, Any], seed: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit the naive (single-verdict) and noise-aware (CCEM) scorers on one stream."""

    cfg = dict(CONFIG if config is None else config)
    features, names = build_features(stream)
    split = three_way_split(features.shape[0], seed)
    train = split["train"]
    k = int(cfg["k_latent"])
    naive = fit_naive_scorer(features[train], stream["observed_overpayment"][train], stream["reviewer_id"][train] >= 0, seed)
    ccem = fit_ccem(
        features[train],
        stream["reviewer_label_matrix"][train],
        k=k,
        n_reviewers=int(cfg["n_reviewers"]),
        seed=seed,
        n_restarts=2,
        posterior_init=vote_posterior(stream["reviewer_label_matrix"][train], k, stream["class_prior"]),
    )
    tune = split["tune"]
    clean_class, class_mapping = identify_clean_class(ccem, features[tune], stream["reviewer_labels"][tune])
    return {
        "features": features,
        "names": names,
        "split": split,
        "naive": naive,
        "ccem": ccem,
        "clean_class": clean_class,
        "class_mapping": class_mapping,
    }


def _policy_row(
    score: np.ndarray,
    y_overpaid: np.ndarray,
    capacity: int,
    seed: int,
    policy: str,
    *,
    feasible: bool = True,
    accept: np.ndarray | None = None,
    row_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = score.size
    if accept is None:
        order = np.argsort(-score, kind="stable")
        reviewed = np.zeros(n, dtype=bool)
        reviewed[order[:capacity]] = True
        accepted = ~reviewed
    else:
        accepted = np.asarray(accept, dtype=bool)
        reviewed = ~accepted
    accepted_count = int(accepted.sum())
    errors = int(np.sum(y_overpaid[accepted])) if accepted_count else 0
    risk = errors / accepted_count if accepted_count else float("nan")
    low, high = wilson_interval(errors, accepted_count)
    row = {
        "capacity_claims": int(capacity),
        "policy": policy,
        "seed": int(seed),
        "risk": float(risk),
        "risk_low": low,
        "risk_high": high,
        "acceptance": float(accepted_count / n),
        "acceptance_low": float(accepted_count / n),
        "acceptance_high": float(accepted_count / n),
        "feasible": bool(feasible),
        "accepted_count": accepted_count,
        "errors": errors,
        "reviewed_count": int(reviewed.sum()),
    }
    if row_extra:
        row.update(row_extra)
    return row


def certificate_accept(
    score_name: str,
    threshold: float,
    posterior: np.ndarray,
    clean_class: int | None,
) -> np.ndarray:
    """Auto-pay rule for one certificate candidate.

    ``msp`` accepts when ``P(clean) >= 1 - threshold`` (i.e. the model-implied
    overpayment probability is at most ``threshold``); ``margin`` accepts when
    the clean class leads the best other class by at least ``threshold``.
    """

    p = np.asarray(posterior, dtype=float)
    index = 0 if clean_class is None else int(clean_class)
    clean = p[:, index]
    if score_name == "msp":
        return clean >= 1.0 - float(threshold)
    others = np.delete(p, index, axis=1)
    return clean - np.max(others, axis=1) >= float(threshold)


def _scorec_candidates(
    fit: dict[str, Any],
    features: np.ndarray,
    cert_idx: np.ndarray,
    y_overpaid: np.ndarray,
    config: dict[str, Any],
    review_budget: int | None = None,
    clean_class: int | None = None,
) -> list[dict[str, Any]]:
    """Finite (threshold, score-function) grid evaluated on the certification split.

    ``review_budget`` (claims the capacity allows the certificate to route to
    review on the certification split) filters candidates deterministically;
    candidates that would exceed the budget are not offered to Algorithm 1, so
    an empty grid returns the documented INFEASIBLE branch.
    """

    posterior = ccem_posterior(fit, features)
    value = float(config["value_per_autopay"])
    cost = float(config["cost_per_review"])
    loss = y_overpaid.astype(float)
    candidates = []
    for score_name, threshold in CERT_GRID:
        accept = certificate_accept(score_name, threshold, posterior, clean_class)
        routed = int((~accept[cert_idx]).sum())
        if review_budget is not None and routed > review_budget:
            continue
        candidates.append(
            {
                "name": f"{score_name}@{threshold:.3f}",
                "accept": accept[cert_idx],
                "loss": loss[cert_idx],
                "value": np.full(cert_idx.size, value),
                "cost": np.full(cert_idx.size, cost),
                "routed_on_cert": routed,
            }
        )
    return candidates


def capacity_study(
    seeds: Sequence[int],
    n: int | None = None,
    capacities: Sequence[int] | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Naive / noise-aware / certificate-aware policies at equal review capacity.

    Policies are fit on stream A (reviewer-selected world) and evaluated on a
    fresh stream B with latent truth.  The certificate uses only the disjoint
    certification split of stream A (SCoRC Assumption 6); the randomized-audit
    SCRC arm is kept for the falsifier study.
    """

    cfg = dict(CONFIG if config is None else config)
    n = int(cfg["n_claims"] if n is None else n)
    caps = tuple(cfg["capacity_claims"] if capacities is None else capacities)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        stream_a = make_claim_stream(seed=seed, n=n, regime="anchored")
        stream_b = make_claim_stream(seed=seed + 5000, n=n, regime="anchored")
        fitted = fit_scorers(stream_a, seed=seed, config=cfg)
        x_b = build_features(stream_b)[0]
        naive_score = fitted["naive"].predict_proba(x_b)[:, 1]
        aware_score = overpayment_score(fitted["ccem"], x_b)
        y_b = stream_b["latent_overpayment"].astype(bool)
        cert_idx = fitted["split"]["cert"]
        y_a = stream_a["latent_overpayment"].astype(bool)
        for capacity in caps:
            rows.append(_policy_row(naive_score, y_b, capacity, seed, "naive"))
            rows.append(_policy_row(aware_score, y_b, capacity, seed, "noise-aware"))
            budget = int(round(capacity * cert_idx.size / n))
            cert = scorec_certificate(
                _scorec_candidates(
                    fitted["ccem"], fitted["features"], cert_idx, y_a, cfg,
                    review_budget=budget, clean_class=fitted["clean_class"],
                ),
                alpha=float(cfg["alpha"]),
                pi_min=float(cfg["pi_min"]),
                delta=float(cfg["delta"]),
                loss_bound=1.0,
                value_bound=float(cfg["value_per_autopay"]),
                cost_bound=float(cfg["cost_per_review"]),
            )
            if cert["feasible"]:
                chosen = cert["chosen"]
                tau_name, threshold = chosen["name"].split("@")
                accept = certificate_accept(
                    tau_name, float(threshold), ccem_posterior(fitted["ccem"], x_b), fitted["clean_class"]
                )
                usable = int(accept.sum())
                if usable == 0:
                    accept = np.ones_like(accept, dtype=bool)
                    usable = accept.size
                rows.append(
                    _policy_row(
                        aware_score,
                        y_b,
                        capacity,
                        seed,
                        "selective",
                        accept=accept,
                        row_extra={
                            "acceptance_low": float(chosen["acceptance_lcb"]),
                            "certified_threshold": float(threshold),
                            "certified_score": tau_name,
                            "risk_ucb": float(chosen["risk_ucb"]),
                            "utility_lcb": float(chosen["utility_lcb"]),
                            "certified_acceptance": float(chosen["acceptance_lcb"]),
                        },
                    )
                )
            else:
                rows.append(
                    _policy_row(
                        aware_score,
                        y_b,
                        capacity,
                        seed,
                        "selective",
                        feasible=False,
                        accept=np.ones_like(y_b, dtype=bool),
                        row_extra={
                            "acceptance_low": float(cert["ledger"][0]["acceptance_lcb"]),
                            "certified_threshold": float("nan"),
                            "certified_score": "infeasible",
                            "risk_ucb": float("nan"),
                            "utility_lcb": float("nan"),
                            "certified_acceptance": float(cert["ledger"][0]["acceptance_lcb"]),
                        },
                    )
                )
    return rows


def dollars_study(
    seeds: Sequence[int],
    minutes_grid: Sequence[int] = (30, 90, 180),
    n: int | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Dollars recovered and net utility for naive vs noise-aware ranking."""

    cfg = dict(CONFIG if config is None else config)
    n = int(cfg["n_claims"] if n is None else n)
    minutes_per_review = float(cfg["minutes_per_review"])
    cost_per_minute = float(cfg["cost_per_minute"])
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        stream_a = make_claim_stream(seed=seed, n=n, regime="anchored")
        stream_b = make_claim_stream(seed=seed + 5000, n=n, regime="anchored")
        fitted = fit_scorers(stream_a, seed=seed, config=cfg)
        x_b = build_features(stream_b)[0]
        scores = {
            "naive": fitted["naive"].predict_proba(x_b)[:, 1],
            "noise-aware": overpayment_score(fitted["ccem"], x_b, clean_class=fitted["clean_class"]),
        }
        dollars = stream_b["recoverable_dollars"]
        y_b = stream_b["latent_overpayment"].astype(bool)
        rng = np.random.default_rng(seed + 3)
        for minutes in minutes_grid:
            capacity = max(1, int(round(minutes / minutes_per_review)))
            for policy, score in scores.items():
                order = np.argsort(-score, kind="stable")
                found = order[:capacity][y_b[order[:capacity]]]
                recovered = float(dollars[found].sum())
                draws = np.empty(2000)
                for i in range(2000):
                    sample = rng.choice(order[:capacity], size=capacity, replace=True)
                    hits = sample[y_b[sample]]
                    draws[i] = dollars[hits].sum()
                low, high = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
                review_cost = minutes * cost_per_minute
                rows.append(
                    {
                        "reviewed_minutes": int(minutes),
                        "policy": policy,
                        "seed": int(seed),
                        "dollars_recovered": recovered,
                        "recovered_low": low,
                        "recovered_high": high,
                        "net_utility": recovered - review_cost,
                        "utility_low": low - review_cost,
                        "utility_high": high - review_cost,
                        "summary": (
                            f"mean recovered ${recovered:,.0f}; median recovered "
                            f"${float(np.median(dollars[found])) if found.size else 0.0:,.0f}; "
                            "heavy-tailed summary"
                        ),
                        "review_cost": review_cost,
                        "claims_reviewed": capacity,
                    }
                )
    return rows


def calibration_falsifier_study(
    seeds: Sequence[int],
    n: int | None = None,
    config: dict[str, Any] | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Randomized-audit vs scorer-selected calibration, evaluated on fresh truth.

    All arms run the *identical* SCRC-I calibration code and are evaluated on a
    fresh stream with latent truth.  Only the sampling of the calibration set
    changes, so the comparison isolates the selection mechanism:

    * ``randomized_audit`` -- independent audit sample, the exchangeable control;
    * ``selected_high_risk`` -- the scorer routes its riskiest claims to review,
      so calibration over-states the base risk (conservative direction);
    * ``selected_low_risk`` -- the scorer routes its most-confident-clean claims,
      so calibration under-states the base risk (permissive direction).

    Calibration labels are the audited (confirmed) outcomes, i.e. the latent
    truth of the synthetic stream; reviewer-opinion noise is studied separately
    in the CCEM identifiability stress, so this experiment attributes the change
    to the selection mechanism alone.
    """

    cfg = dict(CONFIG if config is None else config)
    n = int(cfg["n_claims"] if n is None else n)
    alpha = float(cfg["alpha"] if alpha is None else alpha)
    xi = 0.5
    grid = tuple(np.round(np.linspace(0.0, 1.0, 101), 4))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        stream_a = make_claim_stream(seed=seed, n=n, regime="anchored")
        stream_b = make_claim_stream(seed=seed + 5000, n=n, regime="anchored")
        fitted = fit_scorers(stream_a, seed=seed, config=cfg)
        features_a = fitted["features"]
        features_b = build_features(stream_b)[0]
        score_a = overpayment_score(fitted["ccem"], features_a, clean_class=fitted["clean_class"])
        score_b = overpayment_score(fitted["ccem"], features_b, clean_class=fitted["clean_class"])
        clean_a = 1.0 - score_a
        clean_b = 1.0 - score_b
        y_a = stream_a["latent_overpayment"].astype(float)
        y_b = stream_b["latent_overpayment"].astype(bool)
        audit_idx = np.flatnonzero(stream_a["audit_mask"])
        size = audit_idx.size
        order_high = np.argsort(-score_a, kind="stable")[:size]
        order_low = np.argsort(score_a, kind="stable")[:size]
        arms = {}
        losses = {}
        for arm_name, idx in (
            ("randomized_audit", audit_idx),
            ("selected_high_risk", order_high),
            ("selected_low_risk", order_low),
        ):
            loss = y_a[idx]
            losses[arm_name] = loss
            policy = scrc_inductive_policy(clean_a[idx], loss, grid=grid, alpha=alpha, xi=xi)
            if policy["feasible"]:
                accept_b = clean_b >= policy["threshold"]
                accepted = int(accept_b.sum())
                errors = int(y_b[accept_b].sum())
                risk = errors / accepted if accepted else float("nan")
                low, high = wilson_interval(errors, accepted)
                coverage = accepted / n
            else:
                risk = float(y_b.mean())
                low, high = wilson_interval(int(y_b.sum()), n)
                coverage = 1.0
            exceeds = bool(np.isfinite(risk) and risk > alpha)
            class_share = np.bincount(stream_a["y_latent"][idx], minlength=int(cfg["k_latent"])) / idx.size
            arms[arm_name] = {
                "arm": arm_name,
                "seed": int(seed),
                "calibration_size": int(idx.size),
                "calibration_risk_observed": float(loss.mean()),
                "mean_calibration_score": float(score_a[idx].mean()),
                "median_calibration_score": float(np.median(score_a[idx])),
                "selected_share_by_class": class_share.tolist(),
                "population_share_by_class": (np.bincount(stream_a["y_latent"], minlength=int(cfg["k_latent"])) / n).tolist(),
                "ambiguous_share": float(stream_a["ambiguous"][idx].mean()),
                "population_ambiguous_share": float(stream_a["ambiguous"].mean()),
                "feasible": bool(policy["feasible"]),
                "threshold": float(policy["threshold"]) if policy["feasible"] else float("nan"),
                "test_risk": float(risk),
                "test_risk_low": low,
                "test_risk_high": high,
                "test_acceptance": float(coverage),
                "risk_target": alpha,
                "exceeds_target": exceeds,
                "validity_status": (
                    "assumption falsifier: scorer-selected calibration is out of SCRC/SCoRC scope"
                    if arm_name != "randomized_audit"
                    else "control: randomized audit calibration is the exchangeable arm"
                ),
            }
        comparisons = {
            arm: compare_calibration_sampling(losses[arm], losses["randomized_audit"], seed=seed)
            for arm in ("selected_high_risk", "selected_low_risk")
        }
        rows.append({"seed": int(seed), "comparison": comparisons, "arms": arms})

    summary: dict[str, Any] = {"rows": rows, "alpha": alpha, "xi": xi, "n_seeds": len(rows)}
    for arm in ("randomized_audit", "selected_high_risk", "selected_low_risk"):
        risks = np.array([row["arms"][arm]["test_risk"] for row in rows])
        exceed = float(np.mean([row["arms"][arm]["exceeds_target"] for row in rows]))
        summary[f"{arm}_risk_mean"] = float(risks.mean())
        summary[f"{arm}_exceedance_rate"] = exceed
        summary[f"{arm}_exceedance_interval"] = wilson_interval(int(round(exceed * len(rows))), len(rows))
    summary["diff_high_vs_audit"] = {
        "mean": float(np.mean([row["comparison"]["selected_high_risk"]["risk_difference"] for row in rows])),
        "interval": _mean_interval(
            np.array([row["comparison"]["selected_high_risk"]["risk_difference"] for row in rows])
        ),
    }
    summary["diff_low_vs_audit"] = {
        "mean": float(np.mean([row["comparison"]["selected_low_risk"]["risk_difference"] for row in rows])),
        "interval": _mean_interval(
            np.array([row["comparison"]["selected_low_risk"]["risk_difference"] for row in rows])
        ),
    }
    return summary


def _mean_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return float(values.mean()) if values.size else float("nan"), float("nan")
    half = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    return float(values.mean() - half), float(values.mean() + half)


def scrc_exchangeable_control(
    n_seeds: int = 200,
    n_cal: int = 1000,
    n_test: int = 4000,
    alpha: float = 0.10,
    xi: float = 0.5,
    seed0: int = 0,
) -> dict[str, Any]:
    """Exchangeable control for the SCRC-I routing procedure (derived simulation).

    Scores and losses are i.i.d. by construction: ``g ~ U(0,1)`` and
    ``loss ~ Bernoulli(0.16 - 0.12 g)``.  The paper's control is a property of
    the procedure under exchangeability; these numbers are our simulation and
    the mean realized test risk is compared against the target ``alpha``
    (single-run exceedance is dominated by test-sample noise at this size and
    is reported only as a diagnostic).
    """

    risks, exceed, coverages, infeasible = [], [], [], 0
    for step in range(n_seeds):
        rng = np.random.default_rng(seed0 + step)
        g = rng.random(n_cal + n_test)
        loss = (rng.random(n_cal + n_test) < (0.16 - 0.12 * g)).astype(float)
        policy = scrc_inductive_policy(
            g[:n_cal], loss[:n_cal], grid=tuple(np.round(np.linspace(0.0, 1.0, 101), 4)), alpha=alpha, xi=xi
        )
        if policy["feasible"]:
            accept = g[n_cal:] >= policy["threshold"]
            risk = float(loss[n_cal:][accept].mean()) if accept.any() else float("nan")
            coverages.append(float(accept.mean()))
        else:
            infeasible += 1
            risk = float("nan")
        risks.append(risk)
        exceed.append(1.0 if (np.isfinite(risk) and risk > alpha) else 0.0)
    finite = np.array([r for r in risks if np.isfinite(r)])
    return {
        "n_seeds": n_seeds,
        "alpha": alpha,
        "xi": xi,
        "n_cal": n_cal,
        "n_test": n_test,
        "infeasible_runs": infeasible,
        "mean_test_risk": float(finite.mean()) if finite.size else float("nan"),
        "risk_interval": _mean_interval(finite) if finite.size > 1 else (float("nan"), float("nan")),
        "mean_test_coverage": float(np.mean(coverages)) if coverages else float("nan"),
        "exceedance_rate": float(np.mean(exceed)),
        "exceedance_interval": wilson_interval(int(round(np.mean(exceed) * n_seeds)), n_seeds),
        "note": (
            "single-run test-sample noise dominates the raw exceedance rate; the reported control "
            "is the mean realized selected risk against the target alpha"
        ),
    }


def scorec_certificate_control(
    n_seeds: int = 200,
    n_cert: int = 1500,
    n_test: int = 1500,
    alpha: float = 0.05,
    pi_min: float = 0.50,
    delta: float = 0.10,
    loss_prob: float = 0.01,
    seed0: int = 0,
) -> dict[str, Any]:
    """Joint certificate coverage under i.i.d. certification data (SCoRC Theorem 1).

    Each replication draws an independent certification split and test split
    from the same exchangeable distribution, runs Algorithm 1 on the
    certification split, and evaluates the returned policy on the untouched test
    split.  The reported joint coverage is the fraction of replications where
    the certificate's three claims held simultaneously (risk <= alpha,
    acceptance >= pi_min, utility >= its LCB); Theorem 1 promises at least
    ``1 - delta``.  All quantities are derived from this synthetic construction.
    """

    thresholds = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    m = len(thresholds)
    infeasible = 0
    certified = 0
    joint_ok = 0
    risk_bad = 0
    acc_bad = 0
    util_bad = 0
    slacks_eb, slacks_hoeffding = [], []
    for step in range(n_seeds):
        rng = np.random.default_rng(seed0 + step)
        cert_scores = rng.random(n_cert)
        cert_loss = (rng.random(n_cert) < loss_prob).astype(float)
        test_scores = rng.random(n_test)
        test_loss = (rng.random(n_test) < loss_prob).astype(float)
        value = np.ones(n_cert)
        cost = np.full(n_cert, 0.5)
        candidates = [
            {
                "name": f"thr{threshold:.2f}",
                "accept": cert_scores >= threshold,
                "loss": cert_loss,
                "value": value,
                "cost": cost,
            }
            for threshold in thresholds
        ]
        cert = scorec_certificate(
            candidates, alpha=alpha, pi_min=pi_min, delta=delta, loss_bound=1.0, value_bound=1.0, cost_bound=0.5
        )
        if not cert["feasible"]:
            infeasible += 1
            continue
        certified += 1
        chosen = cert["chosen"]
        threshold = float(chosen["name"].removeprefix("thr"))
        accept_test = test_scores >= threshold
        risk_test = float(test_loss[accept_test].mean()) if accept_test.any() else float("nan")
        acc_test = float(accept_test.mean())
        utility_test = float(np.mean(np.where(accept_test, 1.0, -0.5)))
        ok = True
        if not (np.isfinite(risk_test) and risk_test <= alpha):
            risk_bad += 1
            ok = False
        if acc_test < pi_min:
            acc_bad += 1
            ok = False
        if utility_test < chosen["utility_lcb"]:
            util_bad += 1
            ok = False
        joint_ok += int(ok)
        slacks_eb.append(chosen["slack_risk"])
        slacks_hoeffding.append(np.sqrt(np.log(3 * m / delta) / (2 * n_cert)))
    return {
        "n_seeds": n_seeds,
        "n_cert": n_cert,
        "n_test": n_test,
        "grid_size": m,
        "alpha": alpha,
        "pi_min": pi_min,
        "delta": delta,
        "loss_prob": loss_prob,
        "infeasible_runs": infeasible,
        "certified_runs": certified,
        "risk_violation_rate": float(risk_bad / certified) if certified else float("nan"),
        "acceptance_violation_rate": float(acc_bad / certified) if certified else float("nan"),
        "utility_violation_rate": float(util_bad / certified) if certified else float("nan"),
        "joint_coverage": float(joint_ok / certified) if certified else float("nan"),
        "joint_coverage_interval": wilson_interval(joint_ok, certified) if certified else (float("nan"), float("nan")),
        "guarantee_target": 1.0 - delta,
        "mean_eb_slack": float(np.mean(slacks_eb)) if slacks_eb else float("nan"),
        "mean_hoeffding_slack": float(np.mean(slacks_hoeffding)) if slacks_hoeffding else float("nan"),
        "slack_ratio": float(np.mean(slacks_eb) / np.mean(slacks_hoeffding)) if slacks_eb else float("nan"),
        "note": "independent certification/test splits",
    }
