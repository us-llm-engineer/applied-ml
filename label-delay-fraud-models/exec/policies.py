"""C7 equal-cost label-budget policies: a comparative synthetic experiment.

Three policies spend exactly the same number of observed labels and are compared
on three synthetic outcomes. This is deliberately comparative only: no policy is
declared optimal and no focusing/optimality guarantee is claimed. The internal
simulator is self-contained (it never imports ``exec.data`` or ``exec.recovery``).
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from exec.config import config_hash, config_key, seed_of

POLICIES: tuple[str, ...] = ("random", "risk_first", "uncertainty_first")
LABEL_BUDGET_FRACTION = 0.20
POPULATION_FRAUD_BASE = 0.03
POPULATION_FRAUD_RISK_SLOPE = 0.15
PRE_CHANGE_FRAUD_RATE = 0.05
POST_CHANGE_FRAUD_RATE = 0.25
MONITORING_WINDOW = 20
DETECTION_THRESHOLD = (PRE_CHANGE_FRAUD_RATE + POST_CHANGE_FRAUD_RATE) / 2.0

OUTCOME_RULE = (
    "recovery_outcome=absolute error of an inverse-propensity fraud-risk "
    "estimate against the synthetic population rate; "
    "evaluation_outcome=absolute latent-fraud selection gap against the "
    "synthetic population rate; monitoring_outcome=change-detection delay "
    "(updates) of a rolling-mean detector fed through this policy's label delays"
)

SELECTION_RULE = {
    "random": "uniformly drawn transaction ids",
    "risk_first": "transactions with the largest synthetic risk score",
    "uncertainty_first": "transactions with the most uncertain risk score (4*p*(1-p))",
}

_CACHE: dict[str, dict[str, Any]] = {}


def run_equal_cost_policy_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic, comparative-only equal-cost policy experiment."""
    key = config_key(config)
    if key not in _CACHE:
        _CACHE[key] = _run(config)
    return copy.deepcopy(_CACHE[key])


def _simulate_population(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compact synthetic population: risk score, latent fraud, label propensity."""
    seed = seed_of(config)
    n_transactions = int(config["n_transactions"])
    rng = np.random.default_rng([seed, 101])
    risk = rng.random(n_transactions)
    fraud_probability = POPULATION_FRAUD_BASE + POPULATION_FRAUD_RISK_SLOPE * risk**2
    latent_fraud = (rng.random(n_transactions) < fraud_probability).astype(float)
    # Higher-risk transactions are harder to confirm, so the propensity to
    # observe a label falls with the risk score.
    propensity = np.clip(
        0.92 - 0.55 * risk + 0.06 * rng.standard_normal(n_transactions), 0.05, 0.95
    )
    label_delay = 1.0 + 4.0 * (1.0 - propensity)
    return {
        "risk": risk,
        "latent_fraud": latent_fraud,
        "propensity": propensity,
        "label_delay": label_delay,
    }


def _select(policy: str, population: Mapping[str, np.ndarray], budget: int, rng: np.random.Generator) -> np.ndarray:
    if policy == "random":
        return rng.choice(population["risk"].size, size=budget, replace=False)
    if policy == "risk_first":
        return np.argsort(-population["risk"], kind="stable")[:budget]
    uncertainty = 4.0 * population["risk"] * (1.0 - population["risk"])
    return np.argsort(-uncertainty, kind="stable")[:budget]


def _monitoring_outcome(
    fraud: np.ndarray,
    selection: np.ndarray,
    label_delay: np.ndarray,
) -> float:
    """Detection delay of a rolling-mean detector fed through the policy's delays."""
    m = int(selection.size)
    if m == 0:
        return 0.0
    change = max(1, m // 2)
    delays = np.maximum(np.round(label_delay[selection]).astype(int), 0)
    horizon = m + int(delays.max()) if delays.size else m
    seen = np.zeros(horizon, dtype=float)
    seen[np.arange(m) + delays] = fraud
    window = min(MONITORING_WINDOW, max(3, m // 5))
    alarm: int | None = None
    for update in range(change, horizon):
        low = max(0, update - window + 1)
        if float(seen[low : update + 1].mean()) >= DETECTION_THRESHOLD:
            alarm = update
            break
    return float(alarm - change) if alarm is not None else float(m - change)


def _run(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = seed_of(config)
    population = _simulate_population(config)
    n_transactions = int(config["n_transactions"])
    latent_fraud = population["latent_fraud"]
    propensity = population["propensity"]
    label_delay = population["label_delay"]
    truth = float(latent_fraud.mean())
    label_budget = max(2, int(round(LABEL_BUDGET_FRACTION * n_transactions)))

    # One shared change-point fraud stream across policies so the monitoring
    # comparison isolates each policy's label-delay profile, not the draw.
    monitoring_rng = np.random.default_rng([seed, 303])
    monitoring_change = max(1, label_budget // 2)
    monitoring_fraud = (
        monitoring_rng.random(label_budget)
        < np.where(
            np.arange(label_budget) >= monitoring_change,
            POST_CHANGE_FRAUD_RATE,
            PRE_CHANGE_FRAUD_RATE,
        )
    ).astype(float)

    rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(POLICIES):
        selection_rng = np.random.default_rng([seed, 202, policy_index])
        selection = _select(policy, population, label_budget, selection_rng)
        observed_fraud = latent_fraud[selection]
        observed_propensity = propensity[selection]
        # S_i = 1 only when the picked transaction is truly fraudulent and its
        # label is actually revealed, with P(S_i = 1) = propensity_i * Y_i.
        revealed_rng = np.random.default_rng([seed, 404, policy_index])
        revealed = observed_fraud * (
            revealed_rng.random(selection.size) < observed_propensity
        )
        recovered_rate = (
            float(revealed.sum() / observed_propensity.sum()) * (selection.size / n_transactions)
            if observed_propensity.sum() > 0
            else 0.0
        )
        recovery_outcome = abs(recovered_rate - truth)
        evaluation_outcome = abs(float(observed_fraud.mean()) - truth)
        monitoring_outcome = _monitoring_outcome(monitoring_fraud, selection, label_delay)
        rows.append(
            {
                "policy": policy,
                "observed_label_count": int(selection.size),
                "selection_rule": SELECTION_RULE[policy],
                "recovery_outcome": float(recovery_outcome),
                "evaluation_outcome": float(evaluation_outcome),
                "monitoring_outcome": float(monitoring_outcome),
                "outcome_rule": OUTCOME_RULE,
                "selected_latent_fraud_rate": float(observed_fraud.mean()),
                "population_latent_fraud_rate": truth,
                "mean_label_delay": float(label_delay[selection].mean()),
            }
        )

    return {
        "policies": rows,
        "comparative_only": True,
        "label_budget_fraction": LABEL_BUDGET_FRACTION,
        "n_transactions": int(n_transactions),
        "seed": int(seed),
        "config_hash": config_hash(config),
        "experiment_note": (
            "comparative synthetic selection experiment only; equal observed-label "
            "cost and no optimality or focusing guarantee"
        ),
    }


# ===========================================================================
# Round 2 (C19/C21) -- appended below the Round-1 C7 experiment, which is
# untouched. One realistic shared stream (exec.data), a trained aware trainer,
# >= 20 seeds and replicated policy x budget comparisons.
#
# Everything here is a comparative synthetic measurement with
# `claim_status: "hypothesis"`: no policy is declared optimal, no optimality
# and no label-focusing guarantee is claimed. All randomness is seeded from the
# config seed; both public callables memoise on `exec.config.config_key`.
# ===========================================================================

from scipy import stats as _stats  # noqa: E402  (appended section; Round-1 imports above)

from exec.data import generate_realistic_stream as _generate_realistic_stream  # noqa: E402
from exec.data import preprocess_stream as _preprocess_stream  # noqa: E402
from exec.data import R2_MATURATION_CURVES as _R2_MATURATION_CURVES  # noqa: E402
from exec.evaluation import precision_recall_at_fpr as _precision_recall_at_fpr  # noqa: E402

R2_POLICY_NAMES: tuple[str, ...] = (
    "random",
    "risk_first",
    "uncertainty_first",
    "diverse_typology",
)
R2_DEFAULT_FPR_TARGET = 0.01
R2_INTERVAL_LEVEL = 0.95
R2_PROPENSITY_CLIP: tuple[float, float] = (0.02, 1.0)
R2_RIDGE = 1e-3
R2_NEWTON_MAX_ITER = 40
R2_NEWTON_TOL = 1e-9
R2_SEED_STRIDE = 1_000_003

#: One frozen description per policy (the allocator, not an optimality claim).
R2_SELECTION_RULES: dict[str, str] = {
    "random": "a seeded uniform permutation of the matured label pool",
    "risk_first": "descending trained model score (highest risk first)",
    "uncertainty_first": (
        "scores closest to the deployed operating boundary first (|p - boundary| "
        "ascending, boundary = the 1 - fpr_target score quantile): the cases a "
        "deployed rare-event classifier is least sure about"
    ),
    "diverse_typology": (
        "round-robin over the stream's audit-only typology groups (each group's "
        "highest-risk member in turn) so every typology is covered from a budget "
        "of at least the typology count; an oracle-diversity reference used for "
        "coverage diagnostics, not a deployable allocator"
    ),
}

R2_NAIVE_RULE = (
    "logistic regression on the approved population with every unresolved label "
    "encoded as a negative (Y * label_observed) and no selection correction"
)
R2_AWARE_RULE = (
    "inverse-propensity-weighted logistic regression on the matured (approved "
    "and label-observed) rows; P(label_observed = 1 | x, score) is estimated by "
    "logistic regression on the full population and clipped to [0.02, 1], so the "
    "weights lie in [1, 50]"
)

_R2_CACHE: dict[str, dict[str, Any]] = {}


def run_naive_vs_aware_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """C19: naive vs aware training over >= 20 seeded streams of the shared generator.

    For every seed a stream is generated from a per-seed derivation of the base
    seed. Both models see only model-facing features and the stream's observed
    labels (never the latent ``Y``, the typology or the delay fields):

    * ``naive`` -- trains on the approved population with every unresolved label
      treated as a negative;
    * ``aware`` -- IPW logistic regression on the matured labels only.

    Both are scored on every transaction and evaluated against the stream's
    latent ``Y`` with :func:`exec.evaluation.precision_recall_at_fpr` at
    ``config["fpr_target"]``. The paired differences (aware minus naive) are
    reported with a paired t interval over the seeds. Comparative only: the
    direction is recorded, never required.
    """
    key = "naive_vs_aware::" + config_key(config)
    if key not in _R2_CACHE:
        _R2_CACHE[key] = _r2_naive_vs_aware(config)
    return copy.deepcopy(_R2_CACHE[key])


def run_replicated_policy_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """C21: four label-buying policies x every budget, paired over >= 20 seeds.

    Per seed, every policy buys exactly ``budget`` matured labels from the same
    stream, the SAME aware trainer (see :data:`R2_AWARE_RULE`) is fitted on the
    bought rows, and ``recall_at_fpr`` is computed against the stream's latent
    ``Y``. Equal cost per cell is the point of the comparison: no row ever
    reports an ``observed_label_count`` other than its budget. Comparative only
    (``comparative_only: True``): no optimality key or claim is emitted.
    """
    key = "replicated_policy::" + config_key(config)
    if key not in _R2_CACHE:
        _R2_CACHE[key] = _r2_replicated_policy(config)
    return copy.deepcopy(_R2_CACHE[key])


def _r2_stream_config(config: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Per-seed derivation of the shared stream: same size/drift, new seed."""
    return {
        "seed": int(config["seed"]) + R2_SEED_STRIDE * (index + 1),
        "stream_n": int(config["stream_n"]),
        "stream_t_drift": int(config["stream_t_drift"]),
    }


def _r2_stream_arrays(stream: Mapping[str, Any]) -> dict[str, Any]:
    """Model-facing feature matrix plus audit-only arrays for evaluation only."""
    names = list(stream["feature_names"])
    records = _preprocess_stream(stream)["records"]
    transactions = stream["transactions"]
    return {
        "X": np.asarray([[record[name] for name in names] for record in records], dtype=float),
        "y": np.asarray([int(t["Y"]) for t in transactions], dtype=float),
        "approved": np.asarray([bool(t["approved"]) for t in transactions], dtype=bool),
        "observed": np.asarray([bool(t["label_observed"]) for t in transactions], dtype=bool),
        "model_score": np.asarray([float(t["model_score"]) for t in transactions], dtype=float),
        "typology": np.asarray([str(t["typology"]) for t in transactions], dtype=object),
    }


def _r2_sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _r2_newton_fit(design: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Ridge-stabilised Newton (IRLS) logistic fit on a standardised design."""
    n_rows, n_columns = design.shape
    penalty = R2_RIDGE * np.eye(n_columns + 1)
    penalty[0, 0] = 1e-8
    full = np.column_stack([np.ones(n_rows), design])
    beta = np.zeros(n_columns + 1)
    for _ in range(R2_NEWTON_MAX_ITER):
        probabilities = _r2_sigmoid(full @ beta)
        variances = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        gradient = full.T @ (weights * (labels - probabilities)) - penalty @ beta
        hessian = (full * (weights * variances)[:, None]).T @ full + penalty
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < R2_NEWTON_TOL:
            break
    return beta


def _r2_logistic_scores(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    weights_fit: np.ndarray | None,
    X_score: np.ndarray,
) -> np.ndarray:
    """Fit on ``(X_fit, y_fit, weights)``, standardised on the fit rows, and score ``X_score``.

    A single-class fit carries no ranking information, so it returns the weighted
    observed positive rate as a constant score rather than a fitted direction.
    """
    labels = np.asarray(y_fit, dtype=float)
    weights = (
        np.ones(labels.size, dtype=float)
        if weights_fit is None
        else np.asarray(weights_fit, dtype=float)
    )
    if np.unique(labels).size < 2:
        constant = float(np.clip(np.average(labels, weights=weights), 0.0, 1.0))
        return np.full(X_score.shape[0], constant, dtype=float)
    centre = X_fit.mean(axis=0)
    spread = X_fit.std(axis=0) + 1e-12
    beta = _r2_newton_fit((X_fit - centre) / spread, labels, weights)
    return _r2_sigmoid(((X_score - centre) / spread) @ beta[1:] + beta[0])


def _r2_propensity_scores(X: np.ndarray, score: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """P(label_observed = 1 | features, score), the selection propensity."""
    augmented = np.column_stack([X, np.clip(score, 1e-6, 1.0 - 1e-6)])
    propensity = _r2_logistic_scores(augmented, observed.astype(float), None, augmented)
    return np.clip(propensity, *R2_PROPENSITY_CLIP)


def _r2_ipw_weights(propensity: np.ndarray, selection: np.ndarray) -> np.ndarray:
    return np.clip(1.0 / propensity[selection], 1.0, 1.0 / R2_PROPENSITY_CLIP[0])


def _r2_interval(values: Any) -> dict[str, float]:
    """Paired t interval (and mean) over per-seed values; degenerate-safe."""
    diffs = np.asarray(values, dtype=float).reshape(-1)
    n = int(diffs.size)
    mean = float(diffs.mean()) if n else 0.0
    if n < 2:
        return {"n": n, "mean": mean, "low": mean, "high": mean}
    half = (
        float(_stats.t.ppf(0.5 + R2_INTERVAL_LEVEL / 2.0, n - 1))
        * float(diffs.std(ddof=1))
        / float(np.sqrt(n))
    )
    return {"n": n, "mean": mean, "low": mean - half, "high": mean + half}


def _r2_naive_vs_aware(config: Mapping[str, Any]) -> dict[str, Any]:
    n_seeds = int(config["n_seeds"])
    fpr_target = float(config["fpr_target"])
    rows: list[dict[str, Any]] = []
    per_method: dict[str, dict[int, dict[str, Any]]] = {"naive": {}, "aware": {}}
    for index in range(n_seeds):
        stream_config = _r2_stream_config(config, index)
        stream = _generate_realistic_stream(stream_config)
        data = _r2_stream_arrays(stream)
        seed = int(stream_config["seed"])
        X, y, observed = data["X"], data["y"], data["observed"]
        approved = data["approved"]
        propensity = _r2_propensity_scores(X, data["model_score"], observed)

        naive_rows = approved
        naive_labels = y[naive_rows] * observed[naive_rows]
        naive_scores = _r2_logistic_scores(X[naive_rows], naive_labels, None, X)
        observed_rows = observed
        aware_scores = _r2_logistic_scores(
            X[observed_rows], y[observed_rows], _r2_ipw_weights(propensity, observed_rows), X
        )
        for method, scores, n_train, n_train_positives in (
            ("naive", naive_scores, int(naive_rows.sum()), int(naive_labels.sum())),
            ("aware", aware_scores, int(observed_rows.sum()), int(y[observed_rows].sum())),
        ):
            metrics = _precision_recall_at_fpr(y, scores, fpr_target)
            row = {
                "seed": seed,
                "method": method,
                "precision_at_fpr": float(metrics["precision"]),
                "recall_at_fpr": float(metrics["recall"]),
                "achieved_fpr": float(metrics["achieved_fpr"]),
                "n_train": n_train,
                "n_train_positives": int(n_train_positives),
            }
            rows.append(row)
            per_method[method][seed] = row

    seeds = sorted(per_method["naive"])
    paired: dict[str, dict[str, Any]] = {}
    for metric, field in (("precision", "precision_at_fpr"), ("recall", "recall_at_fpr")):
        interval = _r2_interval(
            [per_method["aware"][s][field] - per_method["naive"][s][field] for s in seeds]
        )
        paired[metric] = {
            "n_pairs": interval["n"],
            "mean": interval["mean"],
            "low": interval["low"],
            "high": interval["high"],
            "level": R2_INTERVAL_LEVEL,
            "method": "paired_t_interval_over_seeds",
        }

    return {
        "fpr_target": fpr_target,
        "n_seeds": n_seeds,
        "claim_status": "hypothesis",
        "rows": rows,
        "paired_differences": paired,
        "seeds": seeds,
        "naive_rule": R2_NAIVE_RULE,
        "aware_rule": R2_AWARE_RULE,
        "scoring_rule": (
            "both models score every transaction from model-facing features only; "
            "precision and recall are measured against the latent Y of the full "
            "stream at the configured fpr target"
        ),
    }


def _r2_diverse_order(pool: np.ndarray, score: np.ndarray, typology: np.ndarray) -> np.ndarray:
    """Round-robin over typology groups, each group ordered by descending score."""
    groups: dict[str, list[int]] = {}
    for index in pool.tolist():
        groups.setdefault(str(typology[index]), []).append(int(index))
    queues = [
        sorted(members, key=lambda i: (-float(score[i]), i))
        for _, members in sorted(groups.items())
    ]
    order: list[int] = []
    positions = [0] * len(queues)
    while sum(len(queue) - position for queue, position in zip(queues, positions)) > 0:
        for queue_index, queue in enumerate(queues):
            if positions[queue_index] < len(queue):
                order.append(queue[positions[queue_index]])
                positions[queue_index] += 1
    return np.asarray(order, dtype=int)


def _r2_policy_order(
    data: Mapping[str, Any], policy: str, seed: int, fpr_target: float
) -> np.ndarray:
    """The policy's full ordering of the matured label pool; budgets take prefixes."""
    pool = np.flatnonzero(data["observed"])
    score = data["model_score"][pool]
    if policy == "random":
        rng = np.random.default_rng([int(seed), 9001])
        return pool[rng.permutation(pool.size)]
    if policy == "risk_first":
        return pool[np.argsort(-score, kind="stable")]
    if policy == "uncertainty_first":
        boundary = float(np.quantile(data["model_score"], 1.0 - float(fpr_target)))
        return pool[np.argsort(np.abs(score - boundary), kind="stable")]
    if policy == "diverse_typology":
        return _r2_diverse_order(pool, data["model_score"], data["typology"])
    raise ValueError(f"unknown policy {policy!r}; expected one of {R2_POLICY_NAMES}")


def _r2_replicated_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    n_seeds = int(config["n_seeds"])
    budgets = [int(budget) for budget in config["label_budgets"]]
    fpr_target = float(config.get("fpr_target", R2_DEFAULT_FPR_TARGET))
    rows: list[dict[str, Any]] = []
    cell: dict[tuple[int, str, int], float] = {}
    for index in range(n_seeds):
        stream_config = _r2_stream_config(config, index)
        stream = _generate_realistic_stream(stream_config)
        data = _r2_stream_arrays(stream)
        seed = int(stream_config["seed"])
        X, y = data["X"], data["y"]
        pool_size = int(data["observed"].sum())
        if budgets and max(budgets) > pool_size:
            raise ValueError(
                f"budget {max(budgets)} exceeds the {pool_size} matured labels "
                "available in this stream; equal-cost buying is impossible"
            )
        propensity = _r2_propensity_scores(X, data["model_score"], data["observed"])
        for policy in R2_POLICY_NAMES:
            order = _r2_policy_order(data, policy, seed, fpr_target)
            for budget in budgets:
                selection = order[:budget]
                scores = _r2_logistic_scores(
                    X[selection], y[selection], _r2_ipw_weights(propensity, selection), X
                )
                metrics = _precision_recall_at_fpr(y, scores, fpr_target)
                rows.append(
                    {
                        "seed": seed,
                        "budget": int(budget),
                        "policy": policy,
                        "observed_label_count": int(selection.size),
                        "recall_at_fpr": float(metrics["recall"]),
                        "precision_at_fpr": float(metrics["precision"]),
                        "n_train_positives": int(y[selection].sum()),
                    }
                )
                cell[(seed, policy, budget)] = float(metrics["recall"])

    seeds = [int(_r2_stream_config(config, index)["seed"]) for index in range(n_seeds)]
    performance_vs_budget: list[dict[str, Any]] = []
    for policy in R2_POLICY_NAMES:
        for budget in budgets:
            interval = _r2_interval([cell[(seed, policy, budget)] for seed in seeds])
            performance_vs_budget.append(
                {
                    "policy": policy,
                    "budget": int(budget),
                    "mean": interval["mean"],
                    "low": interval["low"],
                    "high": interval["high"],
                    "n": interval["n"],
                    "level": R2_INTERVAL_LEVEL,
                    "interval_method": "t_interval_over_seeds",
                }
            )

    paired_differences: list[dict[str, Any]] = []
    for policy in R2_POLICY_NAMES:
        if policy == "random":
            continue
        for budget in budgets:
            interval = _r2_interval(
                [
                    cell[(seed, policy, budget)] - cell[(seed, "random", budget)]
                    for seed in seeds
                ]
            )
            paired_differences.append(
                {
                    "policy_a": policy,
                    "policy_b": "random",
                    "budget": int(budget),
                    "n_pairs": interval["n"],
                    "mean_difference": interval["mean"],
                    "low": interval["low"],
                    "high": interval["high"],
                    "level": R2_INTERVAL_LEVEL,
                    "method": "paired_t_interval_over_seeds",
                }
            )

    return {
        "policies": list(R2_POLICY_NAMES),
        "budgets": budgets,
        "n_seeds": n_seeds,
        "comparative_only": True,
        "claim_status": "hypothesis",
        "rows": rows,
        "paired_differences": paired_differences,
        "performance_vs_budget": performance_vs_budget,
        "selection_rules": dict(R2_SELECTION_RULES),
        "aware_trainer_rule": R2_AWARE_RULE,
        "fpr_target": fpr_target,
        "seed": int(config["seed"]),
        "experiment_note": (
            "comparative synthetic label-allocation experiment only; equal observed-"
            "label cost per cell, same aware trainer and seeds per cell, no optimality "
            "and no label-focusing guarantee"
        ),
    }


# ===========================================================================
# Round 3 (C24) -- naive vs aware recall at 1% FPR, restricted to the
# Cannings-violating subset, with both models trained at a label-maturation
# cutoff. Appended below the Round-1/Round-2 studies, which are untouched.
#
# Mechanism (audit-side; Coudray q11): with eta(x) = P(Y=1 | x) and
# e(x) = P(label observed at the training cutoff | Y=1, x), the naive
# observed-label classifier converges to 1{e(x) eta(x) >= 1/2}, so it agrees
# with 1{eta(x) >= 1/2} exactly where e(x) >= 1/(2 eta(x)); the
# Cannings-violating subset is {eta(x) >= 1/2 and e(x) < 1/(2 eta(x))}. The
# DIRECTION is grounded, the size is measured. Neither trained model reads an
# audit-only field directly: both are fitted only on the model-facing features
# plus the observed-label convention (Y x observed); the typology, the delay and
# the latent Y enter only through this study's audit-side subset definition (and
# the final latent-truth evaluation).
#
# Cutoff semantics: ``training_cutoff_frac`` is the fraction of the frozen
# maturation window (the slowest typology curve's final age, 150 slots) at
# which the model is trained. A label counts at training time only if its own
# delay is below that cutoff age, so slow-maturing typologies (synthetic
# identity, first-party fraud) are still partly immature -- the regime the plan
# asks for -- while the stream's selection (approval) mechanism also enters e.
# Population size: tests/r3-interface.md calls the config sizes advisory and
# asks for >= 1000 confirmed frauds inside the subset; the study therefore runs
# on an internal population of at least ``C24_MIN_STREAM_N`` rows at the
# ``C24_PREVALENCE`` fraud rate (the frozen stream default is untouched and the
# default ``generate_realistic_stream`` behaviour is unchanged).
# ===========================================================================

C24_PREVALENCE = 0.06
C24_TRAIN_SHARE = 0.40
C24_ETA_THRESHOLD = 0.5
C24_DEFAULT_CUTOFF_FRACTION = 0.35
C24_MIN_STREAM_N = 60_000
C24_SEED_STRIDE = R2_SEED_STRIDE
C24_MATURATION_WINDOW = float(
    max(point[0] for curve in _R2_MATURATION_CURVES.values() for point in curve)
)

C24_ETA_THRESHOLD_RULE = (
    "eta(x) = P(Y=1 | x) (audit-side ridge-logistic fit on the model-facing "
    "features); e(x) = P(label observed at the training cutoff | Y=1, x) "
    "(audit-side ridge-logistic fit among the Y=1 rows); the Cannings-violating "
    "subset is {eta(x) >= 1/2 and e(x) < 1/(2 eta(x))} (Cannings et al. "
    "condition, Eq. 8, Coudray q11)"
)
C24_CUTOFF_RULE = (
    "trains at training_cutoff_frac x the 150-slot maturation window: a label is "
    "available only if its own delay is below that cutoff age, so later-maturing "
    "typologies are still partly unresolved at training time; the naive model "
    "encodes every unresolved approved row as a negative"
)
C24_TRAINING_DRAW_RULE = (
    "one audit population is generated once; each seed redraws a random "
    "C24_TRAIN_SHARE of it as the training pool (same draw for both models), so "
    "the paired per-seed differences come from training draws on a fixed "
    "evaluation population rather than from regenerating the stream"
)


def run_naive_vs_aware_mechanism_study(config: Mapping[str, Any]) -> dict[str, Any]:
    """C24: naive vs aware recall at ``fpr_target`` on the Cannings-violating subset.

    One audit population is generated once from the config seed at the study's
    internal prevalence; the audit-side fits define

    * ``eta(x)`` -- a ridge-logistic probability of latent fraud on the
      model-facing features, and
    * ``e(x)`` -- a ridge-logistic probability of being labelled by the
      training cutoff among the ``Y=1`` rows,

    and the subset is ``{eta(x) >= 1/2 and e(x) < 1/(2 eta(x))}``. Both trained
    models re-use the Round-2 machinery: ``naive`` is the unresolved-as-negative
    logistic on the approved rows (``R2_NAIVE_RULE`` at the cutoff) and
    ``aware`` is the clipped-IPW logistic on the matured labels
    (``R2_AWARE_RULE``). For every seed each model is scored on the whole fixed
    population; recall is the share of the partition's latent frauds flagged at
    the partition's own ``fpr_target`` threshold (see ``_c24_recall_at_fpr``).

    Returns the interface shape of ``tests/r3-interface.md`` C24 plus the
    audit-side rules and the paired (naive - aware) summary. Comparative and
    directional only: the size of the gap is measured, never required to exceed
    a magnitude, and ``whole_population`` is reported for contrast.
    """
    key = "mechanism_study::" + config_key(config)
    if key not in _R2_CACHE:
        _R2_CACHE[key] = _c24_mechanism_study(config)
    return copy.deepcopy(_R2_CACHE[key])


def _c24_recall_at_fpr(labels: Any, scores: Any, fpr: float) -> float:
    """Recall branch of :func:`exec.evaluation.precision_recall_at_fpr`, O(n log n).

    The reference walks the distinct thresholds from the highest score downward,
    keeps the last one whose flagged-negative share is at most ``fpr`` and stops
    at the first one that breaks it. Ties are one block, so on the descending
    score order that is exactly the block before the first whose cumulative
    flagged-negative share exceeds ``fpr`` (or the last block if none does).
    Verified equal to the reference on random data; kept local because the
    reference's per-threshold scan is ``O(n * distinct scores)`` while this
    study evaluates a 60k-row population per seed.
    """
    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    n_neg = float(np.sum(y == 0.0))
    n_pos = float(np.sum(y == 1.0))
    if y.size == 0 or n_neg <= 0.0:
        raise ValueError("C24 recall needs at least one negative example")
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    block_end = np.empty(s_sorted.size, dtype=bool)
    block_end[-1] = True
    block_end[:-1] = s_sorted[1:] != s_sorted[:-1]
    flagged_negatives = np.cumsum(y_sorted == 0.0)[block_end] / n_neg
    flagged_positives = np.cumsum(y_sorted == 1.0)[block_end]
    breaking = np.flatnonzero(flagged_negatives > fpr + 1e-12)
    last_block = (int(breaking[0]) - 1) if breaking.size else int(flagged_negatives.size - 1)
    if last_block < 0:
        return 0.0
    return float(flagged_positives[last_block] / n_pos) if n_pos > 0.0 else 0.0


def _c24_paired_summary(differences: list[float]) -> dict[str, Any]:
    """One-sided 95% paired-t upper bound on the mean of ``differences``."""
    values = np.asarray(differences, dtype=float)
    n = int(values.size)
    mean = float(values.mean()) if n else 0.0
    if n < 2:
        return {"n_seeds": n, "mean": mean, "one_sided_upper_95": mean,
                "method": "paired_t_upper_bound"}
    half = float(_stats.t.ppf(0.95, n - 1)) * float(values.std(ddof=1)) / float(np.sqrt(n))
    return {"n_seeds": n, "mean": mean, "one_sided_upper_95": mean + half,
            "method": "paired_t_upper_bound"}


def _c24_mechanism_study(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    n_seeds = int(config["n_seeds"])
    fpr_target = float(config.get("fpr_target", R2_DEFAULT_FPR_TARGET))
    cutoff_fraction = float(config.get("training_cutoff_frac", C24_DEFAULT_CUTOFF_FRACTION))
    stream_n = max(int(config["stream_n"]), C24_MIN_STREAM_N)
    stream = _generate_realistic_stream(
        {
            "seed": seed,
            "stream_n": stream_n,
            "stream_t_drift": int(config.get("stream_t_drift", stream_n // 2)),
            "fraud_prevalence": C24_PREVALENCE,
        }
    )
    names = list(stream["feature_names"])
    transactions = stream["transactions"]
    X = np.asarray(
        [[float(t["features"][name]) for name in names] for t in transactions], dtype=float
    )
    y = np.asarray([int(t["Y"]) for t in transactions], dtype=float)
    approved = np.asarray([bool(t["approved"]) for t in transactions], dtype=bool)
    delay = np.asarray([float(t["label_delay"]) for t in transactions], dtype=float)
    model_score = np.asarray([float(t["model_score"]) for t in transactions], dtype=float)
    n = int(y.size)

    cutoff_days = cutoff_fraction * C24_MATURATION_WINDOW
    matured = approved & (delay <= cutoff_days)

    eta = _r2_logistic_scores(X, y, None, X)
    e = _r2_logistic_scores(X[y == 1.0], matured[y == 1.0], None, X)
    subset = (eta >= C24_ETA_THRESHOLD) & (e < 1.0 / (2.0 * eta))
    if not 0 < int(subset.sum()) < n:
        raise RuntimeError("C24: the Cannings-violating subset must be a nonempty proper part")
    propensity = _r2_propensity_scores(X, model_score, matured)

    subset_rows: list[dict[str, Any]] = []
    whole_rows: list[dict[str, Any]] = []
    for index in range(n_seeds):
        draw_seed = seed + C24_SEED_STRIDE * (index + 1)
        rng = np.random.default_rng([draw_seed, 41])
        train = rng.random(n) < C24_TRAIN_SHARE
        naive_rows = approved & train
        naive_scores = _r2_logistic_scores(X[naive_rows], (y * matured)[naive_rows], None, X)
        aware_rows = matured & train
        aware_scores = _r2_logistic_scores(
            X[aware_rows], y[aware_rows], _r2_ipw_weights(propensity, aware_rows), X
        )
        subset_rows.append(
            {
                "seed": int(draw_seed),
                "naive_recall_at_1pct_fpr": _c24_recall_at_fpr(
                    y[subset], naive_scores[subset], fpr_target
                ),
                "aware_recall_at_1pct_fpr": _c24_recall_at_fpr(
                    y[subset], aware_scores[subset], fpr_target
                ),
            }
        )
        whole_rows.append(
            {
                "seed": int(draw_seed),
                "naive_recall_at_1pct_fpr": _c24_recall_at_fpr(y, naive_scores, fpr_target),
                "aware_recall_at_1pct_fpr": _c24_recall_at_fpr(y, aware_scores, fpr_target),
            }
        )

    def partition(rows: list[dict[str, Any]], instances: int, frauds: int) -> dict[str, Any]:
        diffs = [
            float(row["naive_recall_at_1pct_fpr"]) - float(row["aware_recall_at_1pct_fpr"])
            for row in rows
        ]
        return {
            "n_instances": int(instances),
            "n_confirmed_frauds": int(frauds),
            "per_seed": rows,
            "naive_minus_aware": _c24_paired_summary(diffs),
        }

    return {
        "n_seeds": n_seeds,
        "fpr_target": fpr_target,
        "eta_threshold_rule": C24_ETA_THRESHOLD_RULE,
        "cannings_subset": partition(
            subset_rows, int(subset.sum()), int((subset & (y == 1.0)).sum())
        ),
        "whole_population": partition(whole_rows, n, int(y.sum())),
        "claim_status": "hypothesis",
        "cutoff_rule": C24_CUTOFF_RULE,
        "training_draw_rule": C24_TRAINING_DRAW_RULE,
        "eta_model": "ridge_logistic_on_model_facing_features",
        "e_model": "ridge_logistic_among_Y1_rows",
        "e_condition_binding": bool(
            int(((eta >= C24_ETA_THRESHOLD) & (e < 1.0 / (2.0 * eta))).sum())
            < int((eta >= C24_ETA_THRESHOLD).sum())
        ),
        "internal_population": {
            "n": n,
            "prevalence": C24_PREVALENCE,
            "t_drift": int(stream["t_drift"]),
            "seed": seed,
            "maturation_cutoff_slots": float(cutoff_days),
            "maturation_window_slots": float(C24_MATURATION_WINDOW),
            "train_share_per_seed": C24_TRAIN_SHARE,
        },
    }
