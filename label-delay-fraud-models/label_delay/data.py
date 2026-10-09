"""Synthetic delayed-and-selective fraud-label stream for the NB2 audit.

The stream is deliberately *selective* and *delayed*: a transaction's label
arrival time depends on its latent outcome and on the approval decision taken
from its risk score, so the order in which labels arrive differs from the order
in which transactions arrive. ``Y`` and the true delay are audit-only ground
truth and are never handed to the model-facing preprocessing step.

Timeline model: transactions arrive uniformly across an ``audit_horizon_days``
window (one arrival slot per transaction), so a transaction at arrival index
``i`` is ``audit_horizon_days * (n - i) / n`` days old at the audit time. A
transaction's label arrives ``delay_days`` after arrival through one of two
paths: a fast investigation queue whose propensity rises steeply with the risk
decile, or a slow dispute cycle. Low-decile (mostly non-fraud) labels therefore
stay unresolved at the audit horizon while high-decile labels arrive quickly,
which is the selection/delay structure the NB2 audit must expose.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from label_delay.config import config_hash, config_key, seed_of

DEFAULT_HORIZON_DAYS = 30.0
DEFAULT_APPROVAL_BOUNDARY = 0.70

#: Investigation-queue propensity grows as (risk_decile / 10) ** URGENCY_POWER,
#: so the queue concentrates on the top deciles; everyone else waits on the slow
#: dispute cycle, whose mean differs by latent outcome.
URGENCY_POWER = 3.0
INVESTIGATION_DELAY_MEAN_DAYS = 3.0
SLOW_DELAY_MEAN_DAYS = {"fraud": 45.0, "non_fraud": 90.0}

#: Model-facing feature columns produced by ``preprocess_for_model``.
FEATURE_NAMES = (
    "amount",
    "velocity",
    "account_age_days",
    "is_new_account",
    "risk_score",
    "log_amount",
    "amount_per_velocity",
    "velocity_sqrt",
    "risk_amount_interaction",
    "account_age_log",
)


def generate_synthetic_data(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically generate a delayed/selective fraud transaction stream."""
    seed = seed_of(config)
    n_transactions = int(config["n_transactions"])
    if n_transactions <= 0:
        raise ValueError("config['n_transactions'] must be a positive integer")
    approval_boundary = float(config.get("approval_boundary", DEFAULT_APPROVAL_BOUNDARY))
    horizon_days = float(config.get("audit_horizon_days", DEFAULT_HORIZON_DAYS))

    rng = np.random.default_rng(seed)

    amount = np.round(np.exp(rng.normal(4.1, 1.05, n_transactions)), 2)
    velocity = (rng.poisson(2.5, n_transactions) + 1).astype(np.int64)
    account_age_days = rng.integers(1, 1460, n_transactions).astype(np.int64)
    is_new_account = (account_age_days < 30).astype(np.int64)

    log_amount = np.log1p(amount)
    z_amount = (log_amount - log_amount.mean()) / (log_amount.std() + 1e-12)
    z_velocity = (velocity - velocity.mean()) / (velocity.std() + 1e-12)

    # risk_score is a model-facing score: it is a noisy function of the visible
    # features only, never of the latent outcome.
    latent_logit = (
        -0.70
        + 0.85 * z_amount
        + 0.70 * z_velocity
        + 0.95 * is_new_account
        + rng.normal(0.0, 0.7, n_transactions)
    )
    risk_score = 1.0 / (1.0 + np.exp(-latent_logit))

    fraud_prob = 1.0 / (1.0 + np.exp(-(-1.90 + 1.90 * latent_logit)))
    Y = (rng.random(n_transactions) < fraud_prob).astype(np.int64)

    days_per_transaction = horizon_days / n_transactions
    arrival_index = np.arange(n_transactions)
    arrival_day = arrival_index * days_per_transaction

    decile_order = np.argsort(risk_score, kind="stable")
    risk_decile = np.empty(n_transactions, dtype=np.int64)
    risk_decile[decile_order] = np.arange(n_transactions) * 10 // n_transactions + 1

    urgency = (risk_decile / 10.0) ** URGENCY_POWER
    investigated = rng.random(n_transactions) < urgency
    slow_mean = np.where(
        Y == 1,
        SLOW_DELAY_MEAN_DAYS["fraud"],
        SLOW_DELAY_MEAN_DAYS["non_fraud"],
    )
    delay_days = 1.0 + np.where(
        investigated,
        rng.exponential(INVESTIGATION_DELAY_MEAN_DAYS, n_transactions),
        rng.exponential(slow_mean),
    )
    label_day = arrival_day + delay_days
    label_observed = (label_day <= horizon_days).astype(np.int64)
    # Label arrival is expressed in the same slot scale as arrival_index so the
    # two orderings stay comparable through audit_synthetic_data.
    label_arrival_index = np.rint(label_day / days_per_transaction).astype(np.int64)

    transactions: list[dict[str, Any]] = []
    for i in range(n_transactions):
        transactions.append(
            {
                "transaction_id": int(i + 1),
                "arrival_index": int(arrival_index[i]),
                "amount": float(amount[i]),
                "velocity": int(velocity[i]),
                "account_age_days": int(account_age_days[i]),
                "is_new_account": int(is_new_account[i]),
                "risk_score": round(float(risk_score[i]), 6),
                "risk_decile": int(risk_decile[i]),
                "age_days": round(float(horizon_days - arrival_day[i]), 6),
                "Y": int(Y[i]),
                "label_delay_days": round(float(delay_days[i]), 6),
                "label_arrival_index": int(label_arrival_index[i]),
                "label_observed": int(label_observed[i]),
            }
        )

    return {
        "transactions": transactions,
        "seed": seed,
        "n_transactions": n_transactions,
        "approval_boundary": approval_boundary,
        "audit_horizon_days": horizon_days,
        "days_per_transaction": days_per_transaction,
        "label_delay_model": {
            "description": (
                "delay_days = 1 + Exponential(mean); mean is the investigation "
                "queue mean for transactions drawn into it with propensity "
                "(risk_decile / 10) ** power, else the outcome-specific dispute "
                "mean. Labels arriving after the audit horizon are generated but "
                "not observed."
            ),
            "urgency_power": URGENCY_POWER,
            "investigation_delay_mean_days": INVESTIGATION_DELAY_MEAN_DAYS,
            "slow_delay_mean_days": dict(SLOW_DELAY_MEAN_DAYS),
            "approval_boundary": approval_boundary,
            "horizon_days": horizon_days,
            "arrival_model": "uniform arrival slots over the audit horizon",
        },
        "config": dict(config),
        "config_hash": config_hash(config),
    }


def audit_synthetic_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Audit provenance, ordering, and observed-label accounting of a stream."""
    transactions = data["transactions"]
    seed = int(data["seed"])
    config = data.get("config")
    c_hash = data.get("config_hash")
    if not c_hash and config is not None:
        c_hash = config_hash(config)
    if not c_hash:
        c_hash = ""

    payload = json.dumps(transactions, sort_keys=True, separators=(",", ":"), default=str)
    data_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    transaction_order = [
        int(t["transaction_id"])
        for t in sorted(
            transactions,
            key=lambda t: (int(t["arrival_index"]), int(t["transaction_id"])),
        )
    ]
    label_arrival_order = [
        int(t["transaction_id"])
        for t in sorted(
            transactions,
            key=lambda t: (int(t["label_arrival_index"]), int(t["transaction_id"])),
        )
    ]

    return {
        "seed": seed,
        "config_hash": str(c_hash),
        "data_hash": data_hash,
        "transaction_order": transaction_order,
        "label_arrival_order": label_arrival_order,
        "n_transactions": int(len(transactions)),
        "n_observed_labels": int(sum(int(t["label_observed"]) for t in transactions)),
    }


def preprocess_for_model(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return model-facing features only; latent truth and label fields never leak."""
    records: list[dict[str, Any]] = []
    for t in data["transactions"]:
        amount = float(t["amount"])
        velocity = int(t["velocity"])
        account_age_days = int(t["account_age_days"])
        risk_score = float(t["risk_score"])
        records.append(
            {
                "transaction_id": int(t["transaction_id"]),
                "amount": amount,
                "velocity": velocity,
                "account_age_days": account_age_days,
                "is_new_account": int(t["is_new_account"]),
                "risk_score": risk_score,
                "log_amount": round(math.log1p(amount), 6),
                "amount_per_velocity": round(amount / (1.0 + velocity), 6),
                "velocity_sqrt": round(math.sqrt(velocity), 6),
                "risk_amount_interaction": round(risk_score * math.log1p(amount), 6),
                "account_age_log": round(math.log1p(account_age_days), 6),
            }
        )
    return {"feature_names": list(FEATURE_NAMES), "records": records}


# ===========================================================================
# Shared realistic fraud stream, a trained score, and a deterministic audit.
#
# Story: transactions arrive in order 0..n-1. Each record carries a typology
# (latent to the model), a latent risk factor, model-facing features, a latent
# logit, a bank score produced by a *trained* sklearn model, an approval
# decision taken from that score only, and a label delay drawn from a
# typology-specific maturation curve. Drift is injected: exactly one
# fraud typology shifts its features at ``t_drift``.
# ===========================================================================

STUDY_FEATURE_NAMES: tuple[str, ...] = (
    "amount_log",
    "velocity_7d",
    "velocity_30d",
    "account_age_log",
    "device_risk",
    "ip_risk",
    "email_risk",
    "bin_risk",
    "chargeback_history",
    "night_share",
    "merchant_risk",
)

STUDY_FRAUD_TYPOLOGIES: tuple[str, ...] = (
    "first_party_fraud",
    "synthetic_identity",
    "account_takeover",
)
STUDY_LEGIT_TYPOLOGY = "legitimate"
STUDY_CONFUSER_TYPOLOGIES: tuple[str, ...] = ("plain_default", "thin_file")
STUDY_DRIFT_TYPOLOGY = "account_takeover"

STUDY_AMBIGUOUS_REGIMES: tuple[dict[str, str], ...] = (
    {
        "name": "first-party fraud vs plain default",
        "fraud_typology": "first_party_fraud",
        "confuser_typology": "plain_default",
    },
    {
        "name": "synthetic identity vs thin file",
        "fraud_typology": "synthetic_identity",
        "confuser_typology": "thin_file",
    },
)

#: Position of each typology on the shared latent risk axis. The two ambiguity
#: pairs sit close together (so a strong learner has a real Bayes-error floor),
#: while each fraud typology is clearly but not perfectly above the legit mass.
STUDY_TYPOLOGY_LEVEL: dict[str, float] = {
    "legitimate": 0.00,
    "plain_default": 0.72,
    "thin_file": 0.75,
    "first_party_fraud": 1.98,
    "synthetic_identity": 2.02,
    "account_takeover": 2.35,
}

#: Per-feature loading on the shared risk factor; the residual sigma is chosen
#: so each feature has unit within-typology variance.
STUDY_FEATURE_WEIGHTS: tuple[float, ...] = (
    0.80,
    0.55,
    0.62,
    0.28,
    0.82,
    0.68,
    0.40,
    0.66,
    0.45,
    0.20,
    0.58,
)

#: Adversary step for the drift typology at t_drift (per-feature standardised
#: mean shift; zero means the feature is untouched).
STUDY_DRIFT_SHIFT: dict[str, float] = {
    "amount_log": 0.0,
    "velocity_7d": 1.25,
    "velocity_30d": 0.0,
    "account_age_log": 0.0,
    "device_risk": 1.35,
    "ip_risk": 1.20,
    "email_risk": 0.0,
    "bin_risk": 0.90,
    "chargeback_history": 1.15,
    "night_share": 0.0,
    "merchant_risk": 0.0,
}

#: Typology-specific maturation curves: (age, fraction_confirmed) pairs, ages
#: strictly increasing, fractions in [0,1]. Delays are drawn from exactly these
#: curves (inverse CDF), so the empirical confirmation at each age matches.
STUDY_MATURATION_CURVES: dict[str, tuple[tuple[int, float], ...]] = {
    "first_party_fraud": ((1, 0.15), (4, 0.32), (10, 0.55), (21, 0.74), (45, 0.90), (90, 0.98)),
    "synthetic_identity": ((2, 0.05), (7, 0.16), (18, 0.32), (40, 0.52), (80, 0.76), (150, 0.92)),
    "account_takeover": ((0, 0.30), (2, 0.55), (5, 0.76), (10, 0.89), (20, 0.96), (40, 0.99)),
    "legitimate": ((0, 0.45), (1, 0.62), (3, 0.76), (7, 0.88), (14, 0.96), (30, 0.995)),
    "plain_default": ((1, 0.35), (3, 0.58), (8, 0.78), (18, 0.90), (40, 0.97), (90, 0.995)),
    "thin_file": ((2, 0.28), (5, 0.50), (12, 0.72), (28, 0.87), (60, 0.95), (120, 0.99)),
}

STUDY_APPROVAL_ALPHA = 1.70
STUDY_APPROVAL_BETA = 1.20
STUDY_INCUMBENT_WEIGHT_NOISE = 2.50
STUDY_INCUMBENT_APPROVAL = (2.50, 0.60)
STUDY_TRAIN_SHARE = 0.70
STUDY_SCORE_CLIP = (1e-6, 1.0 - 1e-6)

#: Small LRU cache: stream generation is the costly step and several
#: modules ask for the same config. Public callables deep-copy on return, so no
#: caller can mutate the cached object.
_STUDY_STREAM_CACHE: dict[str, dict[str, Any]] = {}
_STUDY_STREAM_CACHE_ORDER: list[str] = []
_STUDY_STREAM_CACHE_MAX = 4

#: Infrastructure keys are not part of the stream's data-producing config; the
#: ledger contract already requires run identity to ignore them, so the stream
#: hashes and memoises the same way.
_STUDY_INFRA_KEYS = ("ledger_path", "cache_dir")


def _study_prevalence(n: int) -> float:
    """Fraud prevalence: inside the fixed band at the smoke-test/canonical sizes.

    Small smoke-test streams are allowed a slightly higher prevalence (still
    under the 1% band edge) so the replicated studies keep enough positives to
    measure anything.
    """
    if n >= 60_000:
        return 0.004
    return min(0.009, max(0.006, 120.0 / n))


def _study_effective_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dict(config).items() if k not in _STUDY_INFRA_KEYS}


def _study_split_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    names = list(weights)
    raw = np.asarray([weights[k] for k in names], dtype=float)
    raw = raw / raw.sum()
    exact = raw * total
    counts = np.floor(exact).astype(np.int64)
    remainder = int(total - counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    for i in range(remainder):
        counts[int(order[i % len(names)])] += 1
    return {name: int(c) for name, c in zip(names, counts)}


def _study_curve_arrays(typology: str) -> tuple[np.ndarray, np.ndarray]:
    points = STUDY_MATURATION_CURVES[typology]
    ages = np.asarray([p[0] for p in points], dtype=float)
    fractions = np.asarray([p[1] for p in points], dtype=float)
    return ages, fractions


def _study_sample_delays(rng: np.random.Generator, typology: str, size: int) -> np.ndarray:
    """Inverse-CDF draw from the typology curve, with an exponential tail."""
    ages, fractions = _study_curve_arrays(typology)
    probs = np.diff(np.concatenate([[0.0], fractions]))
    probs = np.clip(probs, 0.0, None)
    cdf = np.cumsum(probs)
    cdf[-1] = fractions[-1]
    draws = rng.random(size)
    idx = np.clip(np.searchsorted(cdf, draws, side="left"), 0, len(ages) - 1)
    delays = ages[idx]
    tail = draws > fractions[-1]
    if tail.any():
        scale = max(1.0, float(ages[-1]) * 0.35)
        extra = 1.0 + rng.exponential(scale, int(tail.sum()))
        delays[tail] = ages[-1] + np.ceil(extra)
    return delays.astype(np.int64)


def _study_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _study_approval_probability(score: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Operational approval rule: reject on the standardised score log-odds.

    The deployed cut-off is calibrated on the score's own spread (standardised
    logit), so the model's absolute calibration cannot silently turn the rule
    into 'approve everything' or 'approve nothing'. Approval depends on the
    score only, never on the latent outcome.
    """
    log_odds = _study_logit(score)
    standardised = (log_odds - log_odds.mean()) / (log_odds.std() + 1e-12)
    return 1.0 / (1.0 + np.exp(-(alpha - beta * standardised)))


def _build_realistic_stream(config: Mapping[str, Any]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression  # local: keep import light

    seed = seed_of(config)
    n = int(config.get("stream_n", 0))
    if n <= 0:
        raise ValueError("config['stream_n'] must be a positive integer")
    t_drift = int(config.get("stream_t_drift", n // 2))
    if not 0 <= t_drift <= n:
        raise ValueError("config['stream_t_drift'] must lie inside [0, stream_n]")

    rng = np.random.default_rng(seed)
    m = len(STUDY_FEATURE_NAMES)

    # -- typologies and the latent outcome --------------------------------
    # Optional, default-unchanged prevalence override: mechanism study
    # needs >= 1000 confirmed frauds inside the Cannings-violating subset, and
    # passes ``fraud_prevalence`` to raise the internal fraud rate for that
    # study only. Without the key the ``_study_prevalence`` band is used.
    prevalence_override = config.get("fraud_prevalence")
    prevalence = (
        _study_prevalence(n)
        if prevalence_override is None
        else min(max(float(prevalence_override), 1e-9), 0.5)
    )
    n_fraud = int(round(prevalence * n))
    fraud_counts = _study_split_counts(
        n_fraud, {"first_party_fraud": 0.30, "synthetic_identity": 0.30, "account_takeover": 0.40}
    )
    legit_counts = _study_split_counts(
        n - n_fraud, {"legitimate": 0.88, "plain_default": 0.06, "thin_file": 0.06}
    )
    typology = np.asarray(
        [name for name, count in fraud_counts.items() for _ in range(count)]
        + [name for name, count in legit_counts.items() for _ in range(count)],
        dtype=object,
    )
    if len(typology) != n:
        raise RuntimeError("internal typology allocation error")
    typology = typology[rng.permutation(n)]
    Y = np.isin(typology, np.asarray(STUDY_FRAUD_TYPOLOGIES, dtype=object)).astype(np.int64)
    level = np.asarray([STUDY_TYPOLOGY_LEVEL[t] for t in typology], dtype=float)

    # -- latent-free features ---------------------------------------------
    weights = np.asarray(STUDY_FEATURE_WEIGHTS, dtype=float)
    sigmas = np.sqrt(np.maximum(1e-6, 1.0 - weights**2))
    risk_factor = rng.normal(size=n)
    features = np.outer(level + risk_factor, weights) + rng.normal(size=(n, m)) * sigmas

    arrival_index = np.arange(n, dtype=np.int64)
    drift_mask = (typology == STUDY_DRIFT_TYPOLOGY) & (arrival_index >= t_drift)
    drift_shift = np.asarray([STUDY_DRIFT_SHIFT[name] for name in STUDY_FEATURE_NAMES], dtype=float)
    if drift_shift.any():
        features[drift_mask] += drift_shift

    # -- latent logit (partly unobservable from the features) --------------
    latent_logit = 2.2 * level + 1.1 * risk_factor + 0.8 * rng.normal(size=n)

    # -- label delays from the typology curves ----------------------------
    label_delay = np.empty(n, dtype=np.int64)
    for typ in sorted(set(typology.tolist())):
        mask = typology == typ
        label_delay[mask] = _study_sample_delays(rng, typ, int(mask.sum()))
    label_arrival_index = arrival_index + label_delay

    # -- incumbent (pre-stream) score selects the historical label sample ---
    incumbent_weights = weights + rng.normal(0.0, STUDY_INCUMBENT_WEIGHT_NOISE, size=m)
    incumbent_score = 1.0 / (1.0 + np.exp(-(features @ incumbent_weights)))
    incumbent_alpha, incumbent_beta = STUDY_INCUMBENT_APPROVAL
    incumbent_approved = rng.random(n) < _study_approval_probability(
        incumbent_score, incumbent_alpha, incumbent_beta
    )
    observed_incumbent = incumbent_approved & (label_arrival_index < n)

    # -- train the deployed score on the approved, label-observed sample ---
    train_split = rng.random(n) < STUDY_TRAIN_SHARE
    subset = train_split & observed_incumbent
    mu, sd = features[subset].mean(axis=0), features[subset].std(axis=0) + 1e-12
    standardised = (features - mu) / sd
    score_model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    score_model.fit(standardised[subset], Y[subset])
    model_score = np.clip(score_model.predict_proba(standardised)[:, 1], *STUDY_SCORE_CLIP)

    # -- the deployed decision: approval from the trained score only ------
    approved = rng.random(n) < _study_approval_probability(model_score, STUDY_APPROVAL_ALPHA, STUDY_APPROVAL_BETA)
    label_observed = approved & (label_arrival_index < n)

    # -- plain-Python records ---------------------------------------------
    names = list(STUDY_FEATURE_NAMES)
    feature_rows = np.round(features, 8).tolist()
    score_values = np.round(model_score, 8).tolist()
    latent_values = np.round(latent_logit, 8).tolist()
    typology_values = typology.tolist()
    approved_values = approved.tolist()
    observed_values = label_observed.tolist()
    delay_values = label_delay.tolist()
    arrival_values = label_arrival_index.tolist()
    outcome_values = Y.tolist()

    transactions: list[dict[str, Any]] = []
    for i in range(n):
        transactions.append(
            {
                "transaction_id": i + 1,
                "arrival_index": i,
                "features": dict(zip(names, feature_rows[i])),
                "model_score": score_values[i],
                "approved": approved_values[i],
                "Y": outcome_values[i],
                "typology": typology_values[i],
                "latent_logit": latent_values[i],
                "label_delay": delay_values[i],
                "label_arrival_index": arrival_values[i],
                "label_observed": observed_values[i],
            }
        )

    score_model_meta = {
        "name": "sklearn.linear_model.LogisticRegression",
        "params": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "random_state": seed},
        "feature_names": list(STUDY_FEATURE_NAMES),
        "trained": True,
        "train_subset": "incumbent_approved_and_label_observed_train_split",
        "train_share_requested": STUDY_TRAIN_SHARE,
        "n_train": int(subset.sum()),
        "n_train_positives": int(Y[subset].sum()),
        "n_holdout": int((~train_split).sum()),
        "standardised": True,
    }

    return {
        "seed": seed,
        "n": n,
        "config_hash": config_hash(config),
        "config_name": str(config.get("config_name", "r2-stream")),
        "t_drift": t_drift,
        "feature_names": list(STUDY_FEATURE_NAMES),
        "fraud_typologies": list(STUDY_FRAUD_TYPOLOGIES),
        "legit_typology": STUDY_LEGIT_TYPOLOGY,
        "ambiguous_regimes": [dict(r) for r in STUDY_AMBIGUOUS_REGIMES],
        "drift_typology": STUDY_DRIFT_TYPOLOGY,
        "maturation_curves": {
            typ: [{"age": int(age), "fraction_confirmed": float(frac)} for age, frac in points]
            for typ, points in STUDY_MATURATION_CURVES.items()
        },
        "score_model": score_model_meta,
        "prevalence": float(Y.mean()),
        "transactions": transactions,
    }


def generate_realistic_stream(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically generate the shared fraud stream."""
    effective = _study_effective_config(config)
    key = config_key(effective)
    cached = _STUDY_STREAM_CACHE.get(key)
    if cached is None:
        cached = _build_realistic_stream(effective)
        if len(_STUDY_STREAM_CACHE_ORDER) >= _STUDY_STREAM_CACHE_MAX:
            oldest = _STUDY_STREAM_CACHE_ORDER.pop(0)
            _STUDY_STREAM_CACHE.pop(oldest, None)
        _STUDY_STREAM_CACHE[key] = cached
        _STUDY_STREAM_CACHE_ORDER.append(key)
    else:
        _STUDY_STREAM_CACHE_ORDER.remove(key)
        _STUDY_STREAM_CACHE_ORDER.append(key)
    return copy.deepcopy(cached)


def _study_as_stream(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a generated stream or the config that produces it."""
    if "transactions" in obj:
        return obj
    if "stream_n" in obj:
        return generate_realistic_stream(obj)
    raise ValueError("expected a realistic stream (or the config that generates one)")


def preprocess_stream(stream: Mapping[str, Any]) -> dict[str, Any]:
    """Model-facing view of the shared stream; audit-only fields never leak."""
    stream = _study_as_stream(stream)
    names = list(stream["feature_names"])
    records: list[dict[str, float]] = []
    for transaction in stream["transactions"]:
        features = transaction["features"]
        records.append({name: float(features[name]) for name in names})
    return {"feature_names": names, "records": records}


def audit_stream(stream: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance and observed-label accounting for the shared stream."""
    stream = _study_as_stream(stream)
    transactions = stream["transactions"]
    payload = json.dumps(transactions, sort_keys=True, separators=(",", ":"), default=str)
    data_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    typology_counts: dict[str, int] = {}
    for transaction in transactions:
        name = str(transaction["typology"])
        typology_counts[name] = typology_counts.get(name, 0) + 1
    return {
        "seed": int(stream["seed"]),
        "config_hash": str(stream["config_hash"]),
        "config_name": str(stream.get("config_name", "")),
        "data_hash": data_hash,
        "n": int(stream["n"]),
        "n_observed_labels": int(sum(bool(t["label_observed"]) for t in transactions)),
        "n_approved": int(sum(bool(t["approved"]) for t in transactions)),
        "n_fraud": int(sum(int(t["Y"]) for t in transactions)),
        "prevalence": float(sum(int(t["Y"]) for t in transactions) / len(transactions)),
        "t_drift": int(stream["t_drift"]),
        "drift_typology": str(stream["drift_typology"]),
        "feature_names": list(stream["feature_names"]),
        "typology_counts": typology_counts,
        "score_model_name": str(stream["score_model"]["name"]),
    }
