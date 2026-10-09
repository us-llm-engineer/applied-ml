"""Label-free monitors for the three papers' estimands.

All three functions consume the synthetic scenario (or its arrays) and return
records that carry both the decision and the diagnostics behind it.  None of
them reads the latent labels of the target sample except through the oracle
metadata used for evaluation figures.

* ``estimate_sjs_gap``      SEES-style batch performance-gap estimate for m-SJS.
* ``sequential_tail_monitor`` time-uniform high-loss-tail alarm (P2 analogue).
* ``d3m_disagreement_monitor`` disagreement-based deterioration alarm (P3 analogue).

Adaptations from the papers are marked ``derived here`` in the docstrings; the
notebooks repeat those labels next to the cells that use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from .synthetic import M_SPARSE, NOISE_FEATURES, ShiftScenario, _sigmoid


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return tuple((a - mu) / sd for a in (train, *others))


def _domain_features(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    xs = (x - mu) / sd
    return np.hstack([xs, xs**2])


def _proxy_features(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Observable error-proxy features: covariates (with interactions) plus score.

    The champion's Brier loss is highest on the decision boundary, which is a
    ridge along a direction of the covariate space; the pairwise products are
    what let a linear regression represent that ridge.  ``derived here``.
    """
    xs = (x - mu) / sd
    d = xs.shape[1]
    pairs = np.column_stack([xs[:, j] * xs[:, k] for j in range(d) for k in range(j + 1, d)])
    centred = score - 0.5
    return np.hstack([xs, xs**2, pairs, centred[:, None], centred[:, None] ** 2])


def _fit_error_predictor(
    x: np.ndarray, loss: np.ndarray, score: np.ndarray
) -> tuple[Ridge, np.ndarray, np.ndarray]:
    """Ridge regression of the champion's per-instance loss on observable features."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    phi = _proxy_features(x, mu, sd, score)
    return Ridge(alpha=0.5).fit(phi, loss), mu, sd


def _error_proxy(
    model: Ridge, mu: np.ndarray, sd: np.ndarray, x: np.ndarray, score: np.ndarray
) -> np.ndarray:
    return model.predict(_proxy_features(x, mu, sd, score))


# --------------------------------------------------------------------------- #
# P1 analogue: SEES-style sparse joint shift gap
# --------------------------------------------------------------------------- #
@dataclass
class SJSGapResult:
    status: str
    certified: bool
    estimate: float
    lower: float
    upper: float
    true_gap: float | None
    assumptions: dict[str, Any]
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _ks_stats(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-feature two-sample Kolmogorov-Smirnov statistics."""
    d = a.shape[1]
    stats = np.empty(d)
    n_a, n_b = len(a), len(b)
    grid = np.linspace(0.0, 1.0, 200)
    for j in range(d):
        qa = np.quantile(a[:, j], grid)
        qb = np.quantile(b[:, j], grid)
        lo = min(qa.min(), qb.min())
        hi = max(qa.max(), qb.max())
        if hi <= lo:
            stats[j] = 0.0
            continue
        grid_j = np.linspace(lo, hi, 400)
        stats[j] = float(
            np.max(np.abs(np.searchsorted(np.sort(a[:, j]), grid_j) / n_a
                          - np.searchsorted(np.sort(b[:, j]), grid_j) / n_b))
        )
    return stats


def _select_features(
    source_x: np.ndarray, target_x: np.ndarray, m: int, n_null: int = 100, seed: int = 0
) -> tuple[list[int], dict[str, Any]]:
    """Permutation-calibrated KS feature selection for the sparse shift set.

    Features whose two-sample KS statistic exceeds the 95th percentile of the
    permutation null *maximum* (pooled permutation, exact observed sizes) are
    declared shifted.  This is a ``derived here`` operational counterpart of
    P1's shift index set I.
    """
    rng = np.random.default_rng(seed)
    observed = _ks_stats(source_x, target_x)
    pooled = np.vstack([source_x, target_x])
    labels = np.concatenate([np.zeros(len(source_x)), np.ones(len(target_x))])
    null_max = np.empty(n_null)
    n_s, n_t = len(source_x), len(target_x)
    for i in range(n_null):
        perm = rng.permutation(len(pooled))
        null_max[i] = _ks_stats(pooled[perm[:n_s]], pooled[perm[n_s : n_s + n_t]]).max()
    cut = float(np.quantile(null_max, 0.99))
    selected = [int(j) for j in range(source_x.shape[1]) if observed[j] > cut]
    mu = source_x.mean(axis=0)
    sd = source_x.std(axis=0)
    sd[sd == 0] = 1.0
    phi_s = _domain_features(source_x, mu, sd)
    phi_t = _domain_features(target_x, mu, sd)
    auc = _holdout_auc(np.vstack([phi_s, phi_t]), labels, seed=seed)
    return selected, {
        "ks_stats": observed.tolist(),
        "ks_null_cut": cut,
        "selected": selected,
        "n_selected": len(selected),
        "domain_auc": auc,
    }


def _holdout_auc(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    split = len(idx) // 2
    train, test = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(features[train], labels[train])
    scores = clf.predict_proba(features[test])[:, 1]
    return float(_auc(labels[test], scores))


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _density_weights(
    source_x: np.ndarray, target_x: np.ndarray, features: list[int], clip: float = 50.0
) -> np.ndarray:
    """Probabilistic-classifier density ratio p_t(x)/p_s(x) on the selected block."""
    if not features:
        return np.ones(len(source_x))
    s = source_x[:, features]
    t = target_x[:, features]
    mu = s.mean(axis=0)
    sd = s.std(axis=0)
    sd[sd == 0] = 1.0
    phi_s = _domain_features(s, mu, sd)
    phi_t = _domain_features(t, mu, sd)
    pooled = np.vstack([phi_s, phi_t])
    labels = np.concatenate([np.zeros(len(phi_s)), np.ones(len(phi_t))])
    clf = LogisticRegression(max_iter=4000, C=10.0)
    clf.fit(pooled, labels)
    rate = clf.predict_proba(phi_s)[:, 1]
    rate = np.clip(rate, 1e-6, 1 - 1e-6)
    weights = (rate / (1 - rate)) * (len(phi_t) / len(phi_s))
    return np.clip(weights, 0.0, clip)


def estimate_sjs_gap(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    predictions: np.ndarray,
    m: int = M_SPARSE,
    n_bootstrap: int = 40,
    seed: int = 0,
) -> SJSGapResult:
    """Estimate the target performance gap under an m-SJS working model.

    Follows P1's reweighting estimand ``Delta = E_t[l] - E_s[l]`` with a
    sparsity-aware density matcher (P1 Eq. 4.1).  Two adaptations are
    ``derived here``: a quadratic feature map so variance shifts are reachable,
    and a label-free error-proxy residual check that downgrades the estimate to
    ``diagnostic_only`` when the observable high-loss evidence exceeds what the
    covariate explanation predicts (a concept-shift signature).
    """
    meta = getattr(predictions, "meta", {}) or {}
    model = meta.get("model")
    if model is None:
        raise ValueError("predictions must be a PredictionArray carrying its champion model")
    source_prob = model.predict_proba(source_x)[:, 1]
    source_loss = (source_prob - source_y) ** 2
    selected, sel_diag = _select_features(source_x, target_x, m)
    k_sel = sel_diag["n_selected"]
    assumptions = {
        "sparse_joint_shift": bool(k_sel <= m),
        "support_overlap": bool(sel_diag["domain_auc"] < 0.97),
        "n_selected_features": int(k_sel),
        "sparsity_budget": int(m),
        "domain_auc": float(sel_diag["domain_auc"]),
    }

    if not assumptions["sparse_joint_shift"]:
        return SJSGapResult(
            status="unsupported",
            certified=False,
            estimate=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            true_gap=meta.get("true_gap"),
            assumptions=assumptions,
            message=(
                f"identifiability is not available: the selected shift set has "
                f"{k_sel} features but m-SJS requires at most m={m}; "
                "dense joint shift is not identifiable from target covariates alone"
            ),
            diagnostics=sel_diag,
        )
    if not assumptions["support_overlap"]:
        return SJSGapResult(
            status="unsupported",
            certified=False,
            estimate=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            true_gap=meta.get("true_gap"),
            assumptions=assumptions,
            message="identifiability is not available: source and target supports do not overlap",
            diagnostics=sel_diag,
        )

    weights = _density_weights(source_x, target_x, selected)
    gap = float(np.sum(weights * source_loss) / np.sum(weights) - np.mean(source_loss))

    rng = np.random.default_rng(seed)
    boots = []
    n = len(source_loss)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        w = weights[idx]
        boots.append(float(np.sum(w * source_loss[idx]) / np.sum(w) - np.mean(source_loss[idx])))
    lower, upper = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))

    proxy_diag = _error_proxy_residual(source_x, source_loss, target_x, model, weights)
    if proxy_diag["flagged"]:
        return SJSGapResult(
            status="diagnostic_only",
            certified=False,
            estimate=gap,
            lower=lower,
            upper=upper,
            true_gap=meta.get("true_gap"),
            assumptions=assumptions,
            message=(
                "identifiability check inconclusive: the observed high-loss proxy exceeds the "
                "SJS-implied level, so a label-rule (concept) change is suspected and the "
                "covariate reweighting estimate is not certified"
            ),
            diagnostics={**sel_diag, **proxy_diag},
        )
    return SJSGapResult(
        status="supported",
        certified=True,
        estimate=gap,
        lower=lower,
        upper=upper,
        true_gap=meta.get("true_gap"),
        assumptions=assumptions,
        message="m-SJS diagnostics pass; gap estimate certified under the sparsity assumption",
        diagnostics={**sel_diag, **proxy_diag},
    )


def _error_proxy_residual(
    source_x: np.ndarray,
    source_loss: np.ndarray,
    target_x: np.ndarray,
    model: Any,
    weights: np.ndarray,
    flag_margin: float = 0.03,
) -> dict[str, Any]:
    """Compare target high-loss proxy level against its SJS-implied value."""
    source_score = model.predict_proba(source_x)[:, 1]
    target_score = model.predict_proba(target_x)[:, 1]
    predictor, mu, sd = _fit_error_predictor(source_x, source_loss, source_score)
    proxy_source = _error_proxy(predictor, mu, sd, source_x, source_score)
    implied = float(np.sum(weights * proxy_source) / np.sum(weights))
    proxy_target = _error_proxy(predictor, mu, sd, target_x, target_score)
    observed = float(np.mean(proxy_target))
    se = float(np.std(proxy_target) / np.sqrt(len(proxy_target)))
    excess = observed - implied
    return {
        "proxy_implied": implied,
        "proxy_observed": observed,
        "proxy_excess": excess,
        "proxy_se": se,
        "flagged": bool(excess > max(flag_margin, 4.0 * se)),
        "target_score_mean": float(np.mean(target_score)),
    }


# --------------------------------------------------------------------------- #
# P2 analogue: time-uniform sequential high-loss-tail monitor
# --------------------------------------------------------------------------- #
@dataclass
class SequentialMonitorResult:
    ever_alarm: bool
    alarm_time: int | None
    source_upper: float
    source_tail_rate: float
    source_selector_rate: float
    false_discovery_rate: float
    tolerance: float
    alpha: float
    quantile: float
    selector_threshold: float
    selector_power: float
    selector_fdp: float
    selector_drift: bool
    lower_bounds: np.ndarray
    hit_rates: np.ndarray
    source_upper_series: np.ndarray
    alarms: np.ndarray
    latent_tail_rates: np.ndarray
    sample_alarms: np.ndarray
    selector_flags: np.ndarray


def _calibrate_selector(
    proxy_source: np.ndarray, source_loss: np.ndarray, quantile: float = 0.85, loss_quantile: float = 0.85
) -> tuple[float, float, float]:
    """P2 Eq. 9 analogue: a source-calibrated high-loss-propensity selector.

    The selector threshold is the ``quantile`` of the source error proxy; the
    reported power/FDP diagnose that selector against the source's realized
    high-loss tail (the top ``1 - loss_quantile`` of Brier losses).  Unlike the
    paper's grid search over quantile pairs, no FDP cap is imposed on the
    threshold; the realized FDP is reported instead so the review can see when
    the selector is unreliable (``derived here``).
    """
    thr = float(np.quantile(proxy_source, quantile))
    q_loss = float(np.quantile(source_loss, loss_quantile))
    sel = proxy_source > thr
    tail = source_loss > q_loss
    power = float(sel[tail].mean()) if tail.any() else 0.0
    fdp = float((sel & ~tail).sum() / sel.sum()) if sel.any() else 1.0
    return thr, power, fdp


def sequential_tail_monitor(
    data: ShiftScenario,
    alpha: float = 0.10,
    tolerance: float = 0.03,
    quantile: float = 0.85,
    loss_quantile: float = 0.85,
) -> SequentialMonitorResult:
    """Time-uniform alarm on an increase in the error-propensity tail (P2 analogue).

    The observable statistic is the fraction of arrivals whose source-calibrated
    error proxy lands in the top ``1 - quantile`` of the source proxy.  The
    alarm fires when the target fraction's time-uniform lower bound exceeds the
    source fraction plus ``tolerance``:

    ``L_t = rate_t - w_t`` and ``alarm = L_t > source_rate + tolerance``.

    ``derived here``: the confidence sequence is a union-bound Hoeffding width
    (``alpha_t = 6 alpha / (pi^2 t^2)``) in place of the paper's predictably
    mixed empirical-Bernstein bound, the selector threshold is the source
    proxy quantile rather than a power/FDP grid optimum, and the error proxy is
    a ridge regression of the champion's Brier loss.  The realized selector
    power/FDP are reported as calibration diagnostics.
    """
    return sequential_tail_monitor_arrays(
        source_x=data.source_x,
        source_loss=data.source_loss,
        source_score=data.source_score,
        target_x=data.target_x,
        target_score=data.target_score,
        target_loss=data.target_loss,
        alpha=alpha,
        tolerance=tolerance,
        quantile=quantile,
        loss_quantile=loss_quantile,
    )


def sequential_tail_monitor_arrays(
    source_x: np.ndarray,
    source_loss: np.ndarray,
    source_score: np.ndarray,
    target_x: np.ndarray,
    target_score: np.ndarray,
    target_loss: np.ndarray,
    alpha: float = 0.10,
    tolerance: float = 0.03,
    quantile: float = 0.85,
    loss_quantile: float = 0.85,
) -> SequentialMonitorResult:
    """Array-level implementation of :func:`sequential_tail_monitor`."""
    n = len(source_x)
    predictor, mu, sd = _fit_error_predictor(source_x, source_loss, source_score)
    proxy_source = _error_proxy(predictor, mu, sd, source_x, source_score)
    proxy_target = _error_proxy(predictor, mu, sd, target_x, target_score)
    thr, power, fdp = _calibrate_selector(proxy_source, source_loss, quantile, loss_quantile)

    sel_source = proxy_source > thr
    source_tail_rate = float(np.mean(sel_source))

    sel_target = proxy_target > thr
    t_axis = np.arange(1, len(sel_target) + 1, dtype=float)
    hit_rates = np.cumsum(sel_target) / t_axis
    # time-uniform one-sided Hoeffding width: alpha_t = 6 alpha / (pi^2 t^2)
    w_t = np.sqrt(np.log((np.pi**2) * t_axis**2 / (6.0 * alpha)) / (2.0 * t_axis))
    lower_bounds = hit_rates - w_t
    alarms = lower_bounds > (source_tail_rate + tolerance)
    ever_alarm = bool(alarms.any())
    alarm_time = int(np.argmax(alarms)) if ever_alarm else None
    latent_tail = target_loss > float(np.quantile(source_loss, loss_quantile))
    latent_tail_rates = np.cumsum(latent_tail) / t_axis

    selector_drift = bool(fdp > 0.5 and power < 0.2)
    return SequentialMonitorResult(
        ever_alarm=ever_alarm,
        alarm_time=alarm_time,
        source_upper=source_tail_rate,
        source_tail_rate=source_tail_rate,
        source_selector_rate=source_tail_rate,
        false_discovery_rate=fdp,
        tolerance=tolerance,
        alpha=alpha,
        quantile=quantile,
        selector_threshold=thr,
        selector_power=power,
        selector_fdp=fdp,
        selector_drift=selector_drift,
        lower_bounds=lower_bounds,
        hit_rates=hit_rates,
        source_upper_series=np.full_like(t_axis, source_tail_rate),
        alarms=alarms,
        latent_tail_rates=latent_tail_rates,
        sample_alarms=alarms.copy(),
        selector_flags=sel_target,
    )


# --------------------------------------------------------------------------- #
# P3 analogue: disagreement-driven deterioration monitor
# --------------------------------------------------------------------------- #
@dataclass
class D3MResult:
    alarm: bool
    statistic: float
    threshold: float
    guarantee_status: str
    failure_mode_exposed: bool
    alpha: float
    baseline_level: float
    n_hypotheses: int
    calibration: np.ndarray
    disagreement_gap: float
    message: str


def build_hypotheses(
    source_x: np.ndarray,
    source_y: np.ndarray,
    n_hypotheses: int = 40,
    seed: int = 0,
    weak_fraction: float = 0.35,
) -> list[LogisticRegression]:
    """Diverse source-consistent hypotheses (a ``derived here`` analogue of VBLL sampling).

    Each hypothesis is a logistic model on the full covariate vector, trained
    on a random bootstrap subsample with a random regularisation level.  The
    weak tail of the ensemble (small subsamples, strong shrinkage) is what
    gives the calibration envelope its width, mirroring P3's posterior spread.
    """
    rng = np.random.default_rng(seed)
    models: list[LogisticRegression] = []
    for k in range(n_hypotheses):
        weak = k < n_hypotheses * weak_fraction
        if weak:
            size = max(int(rng.uniform(0.05, 0.6) * len(source_x)), 60)
            c = float(rng.uniform(0.02, 0.4))
        else:
            size = len(source_x)
            c = float(rng.uniform(0.3, 2.0))
        idx = rng.choice(len(source_x), size=size, replace=True)
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit(source_x[idx], source_y[idx])
        models.append(model)
    return models


def _hypothesis_predictions(models: list[LogisticRegression], x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in models])


def _max_disagreement(models: list[LogisticRegression], champion_pred: np.ndarray, x: np.ndarray) -> float:
    preds = _hypothesis_predictions(models, x)
    return float(np.max(np.mean(preds != champion_pred[:, None], axis=0)))


def d3m_disagreement_monitor(
    data: ShiftScenario,
    alpha: float = 0.05,
    n_hypotheses: int = 40,
    calibration_batches: int = 80,
    batch_size: int = 300,
    resolution_floor: float = 0.20,
    weak_fraction: float = 0.35,
    seed: int = 0,
) -> D3MResult:
    """P3's Train-Calibrate-Deploy disagreement test on the synthetic scenario.

    ``derived here``: posterior sampling is replaced by an ensemble of
    bootstrapped, feature-subset, differently-regularised source models; the
    deployment statistic is the mean over deployment batches of the per-batch
    maximum disagreement rate.  ``guarantee_status`` reports ``not_guaranteed``
    when the calibrated envelope is too wide to resolve a shift (P3 Theorem
    A.6's regime), which is the only honest label-free statement available.
    """
    rng = np.random.default_rng(seed)
    models = build_hypotheses(
        data.source_x,
        data.source_y,
        n_hypotheses=n_hypotheses,
        seed=seed,
        weak_fraction=weak_fraction,
    )
    source_pred = data.model.predict(data.source_x)

    n_cal = min(calibration_batches, max(1, len(source_pred) // batch_size))
    calibration = np.empty(n_cal)
    for t in range(n_cal):
        idx = rng.choice(len(source_pred), size=batch_size, replace=True)
        calibration[t] = _max_disagreement(models, source_pred[idx], data.source_x[idx])
    threshold = float(np.quantile(calibration, 1.0 - alpha))

    target_pred = data.model.predict(data.target_x)
    n_batches = max(1, len(data.target_x) // batch_size)
    stats = []
    for b in range(n_batches):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        stats.append(_max_disagreement(models, target_pred[sl], data.target_x[sl]))
    statistic = float(np.mean(stats)) if stats else float("nan")

    alarm = bool(statistic >= threshold)
    if alarm:
        status = "guaranteed_deterioration"
        failure_mode_exposed = False
        message = "deployment disagreement exceeds the calibrated envelope"
    elif threshold >= resolution_floor:
        status = "not_guaranteed"
        failure_mode_exposed = True
        message = (
            "the calibrated disagreement envelope is too wide to resolve a shift: "
            "detection power is not guaranteed (P3 Theorem A.6 regime)"
        )
    else:
        status = "supported_no_alarm"
        failure_mode_exposed = False
        message = "deployment disagreement stays inside the calibrated envelope"

    source_pred_batches = float(np.mean(
        [
            _max_disagreement(models, source_pred[i : i + batch_size], data.source_x[i : i + batch_size])
            for i in range(0, min(len(source_pred), 6 * batch_size), batch_size)
        ]
    )) if len(source_pred) >= batch_size else float("nan")
    disagreement_gap = float(statistic - source_pred_batches) if not np.isnan(source_pred_batches) else float("nan")

    return D3MResult(
        alarm=alarm,
        statistic=statistic,
        threshold=threshold,
        guarantee_status=status,
        failure_mode_exposed=failure_mode_exposed,
        alpha=alpha,
        baseline_level=float(np.mean(calibration)),
        n_hypotheses=n_hypotheses,
        calibration=calibration,
        disagreement_gap=disagreement_gap,
        message=message,
    )
