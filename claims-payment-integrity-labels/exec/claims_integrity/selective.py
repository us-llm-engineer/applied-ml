"""Selective-risk and certificate machinery.

Source-backed pieces (raw traces in ``nlm/responses/``):

* SCRC (Xu, Guo, Wei, 2512.12844), q5/q7: bounded monotone loss, conditional
  selected risk ``E[l(C, Y) | g(X) >= 1 - lambda_1]`` (Eq. 3/6), coverage
  (Eq. 2/7), first-stage selection threshold (Eq. 11), second-stage conformal
  counting quantile (Eq. 15).  Theorems 2-3 control risk/coverage under
  exchangeable calibration.
* SCoRC (Yu, Liu, 2606.08517), q8/q9: Algorithm 1 -- failure budget
  ``delta/3m`` per grid point, exact Clopper-Pearson acceptance LCB, empirical-
  Bernstein risk UCB and utility LCB, deterministic feasibility filter,
  INFEASIBLE branch, and the sample-size precondition
  ``n_cert >= 32 log(32m/delta)/pi_min``.

Adaptations are labelled *derived here*: the synthetic binary action space has
no prediction sets, so SCRC is collapsed onto a single routing threshold, and
the dollar quantities are bounded proxies.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import beta

_EPS = 1e-12


def selected_risk(losses: np.ndarray, selected: np.ndarray) -> float:
    """Mean loss among selected items; empty selection is rejected, not zero."""

    loss = np.asarray(losses, dtype=float)
    mask = np.asarray(selected, dtype=bool)
    if loss.shape != mask.shape:
        raise ValueError("losses and selected must have the same shape")
    if not mask.any():
        raise ValueError("selected risk is undefined when no items are selected")
    return float(loss[mask].mean())


def acceptance_rate(selected: np.ndarray) -> float:
    """Fraction of items selected (coverage)."""

    return float(np.asarray(selected, dtype=bool).mean())


def bounded_loss(y_true: np.ndarray, decision: np.ndarray, bound: float) -> float:
    """Mean per-item loss ``min(|y_true - decision|, bound)``."""

    y = np.asarray(y_true, dtype=float)
    d = np.asarray(decision, dtype=float)
    return float(np.clip(np.abs(y - d), 0.0, float(bound)).mean())


def utility(recovered_dollars, reviewed, review_cost: float):
    """Recovered dollars minus review cost times reviews performed."""

    value = np.asarray(recovered_dollars, dtype=float) - float(review_cost) * np.asarray(reviewed, dtype=float)
    return float(value) if np.ndim(value) == 0 else value


def select_scorc_policy(
    grid: Sequence[float],
    risk: Sequence[float],
    acceptance: Sequence[float],
    utility: Sequence[float],
    risk_level: float,
    min_acceptance: float,
) -> dict[str, Any]:
    """Deterministic finite-grid selection with an explicit INFEASIBLE branch."""

    grid_arr = np.asarray(grid, dtype=float)
    risk_arr = np.asarray(risk, dtype=float)
    acc_arr = np.asarray(acceptance, dtype=float)
    util_arr = np.asarray(utility, dtype=float)
    if not (grid_arr.shape == risk_arr.shape == acc_arr.shape == util_arr.shape):
        raise ValueError("grid, risk, acceptance and utility must share one shape")
    feasible = (risk_arr <= float(risk_level)) & (acc_arr >= float(min_acceptance))
    if not feasible.any():
        return {
            "feasible": False,
            "threshold": None,
            "risk": None,
            "acceptance": None,
            "utility": None,
            "reason": "INFEASIBLE: no grid point satisfies risk <= risk_level and acceptance >= min_acceptance",
        }
    idx = int(np.argmax(np.where(feasible, util_arr, -np.inf)))
    return {
        "feasible": True,
        "threshold": float(grid_arr[idx]),
        "risk": float(risk_arr[idx]),
        "acceptance": float(acc_arr[idx]),
        "utility": float(util_arr[idx]),
        "reason": "utility-maximising certified grid point",
    }


def clopper_pearson_lcb(k: int, n: int, quantile: float) -> float:
    """Exact Clopper-Pearson lower confidence bound (SCoRC Algorithm 1, step 2)."""

    if n <= 0:
        return 0.0
    k = int(np.clip(k, 0, n))
    if k == 0:
        return 0.0
    q = float(np.clip(quantile, 1e-12, 1.0 - 1e-12))
    if k == n:
        return float(q ** (1.0 / n))
    return float(beta.ppf(q, k, n - k + 1))


def empirical_bernstein_slack(sigma2: float, bound: float, n: int, log_term: float) -> float:
    """Variance-adaptive empirical-Bernstein slack (SCoRC Theorem 1, Eq. 3)."""

    if n < 2:
        return float("inf")
    return float(np.sqrt(2.0 * max(sigma2, 0.0) * log_term / n) + 7.0 * bound * log_term / (3.0 * (n - 1)))


def scorec_certificate(
    candidates: Sequence[dict[str, Any]],
    alpha: float,
    pi_min: float,
    delta: float,
    loss_bound: float,
    value_bound: float,
    cost_bound: float,
    n_cert: int | None = None,
) -> dict[str, Any]:
    """Run SCoRC Algorithm 1 over a finite candidate grid.

    Each candidate must provide ``name``, ``accept`` (bool array), ``loss``
    (bounded, in ``[0, loss_bound]``), ``value`` (in ``[0, value_bound]``) and
    ``cost`` (in ``[0, cost_bound]``) evaluated on the certification split.
    Returns the full per-candidate ledger plus the chosen policy or the
    deterministic INFEASIBLE result.
    """

    m = len(candidates)
    if m == 0:
        raise ValueError("scorec_certificate needs at least one candidate")
    n = int(len(candidates[0]["accept"]) if n_cert is None else n_cert)
    log_term = float(np.log(3.0 * m / delta))
    n_star = 32.0 * np.log(32.0 * m / delta) / float(pi_min)
    ledger = []
    for cand in candidates:
        accept = np.asarray(cand["accept"], dtype=bool)
        loss = np.asarray(cand["loss"], dtype=float)
        value = np.asarray(cand["value"], dtype=float)
        cost = np.asarray(cand["cost"], dtype=float)
        count = int(accept.sum())
        p_lcb = clopper_pearson_lcb(count, n, delta / (3.0 * m))
        shifted = accept * (loss - alpha)
        z_bar = float(shifted.mean())
        sigma_z = float(shifted.var(ddof=1)) if n > 1 else 0.0
        eps_z = empirical_bernstein_slack(sigma_z, loss_bound, n, log_term)
        risk_ucb = float(alpha + (z_bar + eps_z) / max(p_lcb, _EPS))
        per_sample = accept * value - (~accept) * cost
        v_bar = float(per_sample.mean())
        sigma_v = float(per_sample.var(ddof=1)) if n > 1 else 0.0
        eps_v = empirical_bernstein_slack(sigma_v, value_bound + cost_bound, n, log_term)
        utility_lcb = float(v_bar - eps_v)
        feasible = bool(risk_ucb <= alpha and p_lcb >= pi_min and n >= n_star)
        ledger.append(
            {
                "name": cand["name"],
                "accept_count": count,
                "routed_on_cert": int(n - count),
                "acceptance_lcb": p_lcb,
                "risk_ucb": risk_ucb,
                "utility_lcb": utility_lcb,
                "empirical_utility": v_bar,
                "empirical_risk": float(loss[accept].mean()) if count else float("nan"),
                "feasible": feasible,
                "delta_per_point": delta / (3.0 * m),
                "slack_risk": eps_z,
                "slack_utility": eps_v,
                "sample_size_ok": bool(n >= n_star),
            }
        )
    feasible_rows = [row for row in ledger if row["feasible"]]
    base = {
        "ledger": ledger,
        "n_cert": n,
        "n_star": float(n_star),
        "log_term": log_term,
        "delta_per_point": delta / (3.0 * m),
        "alpha": float(alpha),
        "pi_min": float(pi_min),
    }
    if not feasible_rows:
        return {**base, "feasible": False, "chosen": None, "reason": "INFEASIBLE: no certified grid point"}
    best = max(feasible_rows, key=lambda row: row["utility_lcb"])
    return {**base, "feasible": True, "chosen": best, "reason": "utility-maximising certified grid point"}


def scrc_inductive_policy(
    scores: np.ndarray,
    losses: np.ndarray,
    grid: Sequence[float],
    alpha: float,
    xi: float,
) -> dict[str, Any]:
    """SCRC-I-style calibration-only routing (derived one-stage collapse).

    Accept claim ``i`` when ``scores[i] >= t``.  Choose the smallest grid
    threshold that keeps the coverage floor ``xi`` and satisfies the conformal
    counting quantile of Eq. (15) on the accepted subset; return INFEASIBLE
    when no threshold does.  Under exchangeable calibration this controls the
    selected risk at ``alpha`` up to the usual finite-sample slack; under
    reviewer-selected calibration it does not (see the calibration falsifier).
    """

    s = np.asarray(scores, dtype=float)
    l = np.asarray(losses, dtype=float)
    n = s.size
    for threshold in sorted(float(t) for t in grid):
        accept = s >= threshold
        count = int(accept.sum())
        if count == 0:
            continue
        if count / n < xi:
            continue
        if count < int(np.ceil(1.0 / alpha)) - 1:
            continue
        budget = int(np.ceil((count + 1) * alpha)) - 1
        if budget <= 0:
            continue
        if l[accept].sum() <= budget:
            return {
                "feasible": True,
                "threshold": threshold,
                "accept": accept,
                "risk": selected_risk(l, accept),
                "coverage": acceptance_rate(accept),
                "accepted": count,
                "budget": budget,
                "reason": "first grid threshold meeting coverage floor and Eq. (15) counting quantile",
            }
    return {
        "feasible": False,
        "threshold": None,
        "accept": np.zeros_like(s, dtype=bool),
        "risk": float("nan"),
        "coverage": 0.0,
        "accepted": 0,
        "budget": 0,
        "reason": "INFEASIBLE: no grid threshold meets the coverage floor and Eq. (15)",
    }


def compare_calibration_sampling(
    selected_losses: np.ndarray,
    audit_losses: np.ndarray,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Compare reviewer-selected vs randomized-audit calibration losses.

    The reviewer-selected arm is an assumption falsifier: it is out of the
    exchangeability conditions required by SCRC/SCoRC, so a met target there is
    not a restored guarantee and a missed target is not a theorem failure.
    """

    sel = np.asarray(selected_losses, dtype=float)
    aud = np.asarray(audit_losses, dtype=float)
    rng = np.random.default_rng(seed)
    difference = float(sel.mean() - aud.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(sel, size=sel.size, replace=True).mean() - rng.choice(aud, size=aud.size, replace=True).mean()
    low, high = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return {
        "risk_difference": difference,
        "interval": (low, high),
        "ci": (low, high),
        "selected_mean": float(sel.mean()),
        "audit_mean": float(aud.mean()),
        "n_selected": int(sel.size),
        "n_audit": int(aud.size),
        "validity_status": "assumption falsifier: reviewer-selected calibration is out of SCRC/SCoRC scope",
        "certificate_valid": False,
        "note": (
            "Reviewer-selected labels violate the exchangeability / selector-independence "
            "conditions (SCRC q5-q7; SCoRC Assumption 6-7, q9); the randomized audit is the "
            "control arm and no guarantee is inherited in either direction."
        ),
    }
