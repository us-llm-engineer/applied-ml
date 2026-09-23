"""Synthetic shift regimes with latent truth for label-free monitoring.

Every regime exposes the two objects the papers keep separate: the observable
unlabeled target stream ``x`` (plus the frozen champion's scores) and the latent
target risk that is never available to a monitor.  The latent fields are used
only for evaluation, never inside an estimator.

Regimes
-------
``none``              no shift control.
``sparse_joint``      m-SJS-legal shift confined to two informative features
                      (location and scale), noise features untouched.
``benign_covariate``  strong mean shift on three label-irrelevant features:
                      feature drift without performance drift.
``dense_joint``       all-feature mixing shift: no subset of size <= m can
                      satisfy the SJS invariance, so the gap is not identifiable.
``harmful_concept``   the target label boundary itself moves and the covariate
                      mass concentrates on the stale boundary region.
``d3m_regime_2``      an under-trained champion plus a real deterioration whose
                      disagreement signature stays inside the champion's own
                      inflated calibration envelope (P3 Regime 2 analogue).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

D = 8
M_SPARSE = 3
INFORMATIVE = (0, 1)
NOISE_FEATURES = (2, 3, 4, 5, 6, 7)
LABEL_SCALE = 0.35
REGIMES = (
    "none",
    "sparse_joint",
    "benign_covariate",
    "dense_joint",
    "harmful_concept",
    "d3m_regime_2",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class PredictionArray(np.ndarray):
    """Champion scores on the target batch, with provenance metadata.

    The array behaves exactly like ``np.ndarray``; ``meta`` carries the fitted
    champion (so an estimator can score the labeled source sample) and the
    scenario's latent truth, which only evaluation code may read.
    """

    def __new__(cls, values: np.ndarray, meta: dict[str, Any] | None = None) -> "PredictionArray":
        obj = np.asarray(values, dtype=float).view(cls)
        obj.meta = dict(meta or {})
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.meta = getattr(obj, "meta", {})


def _label_probability(x: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    """True conditional P(y = 1 | x) for the generative label rule."""
    score = x[:, 0] + x[:, 1] - intercept
    return _sigmoid(score / LABEL_SCALE)


def _draw_source(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, D))
    y = (rng.random(n) < _label_probability(x)).astype(int)
    return x, y


def _draw_target(rng: np.random.Generator, n: int, regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Draw a target batch and its latent labels under the regime's law."""
    if regime == "none":
        return _draw_source(rng, n)

    z = rng.normal(size=(n, D))
    intercept = 0.0
    x = z.copy()

    if regime == "sparse_joint":
        # SJS-legal: features 0,1 move jointly (location + scale); 2..7 unchanged.
        x[:, 0] = 0.30 + 0.55 * z[:, 0]
        x[:, 1] = -0.20 + 0.55 * z[:, 1]
    elif regime == "benign_covariate":
        # Label-irrelevant mean shift on noise features: drift without risk.
        x[:, 5] = z[:, 5] + 0.70
        x[:, 6] = z[:, 6] + 0.55
        x[:, 7] = z[:, 7] + 0.85
    elif regime == "dense_joint":
        # All features are mixed and translated: no sparse invariant subset.
        mixing = 0.72 * np.eye(D) + 0.30 * np.triu(np.ones((D, D)), k=1) / D
        x = z @ mixing.T + np.linspace(0.45, -0.45, D)
    elif regime == "harmful_concept":
        # The label boundary moves (p(y|x) changes) and covariate mass
        # concentrates onto the stale boundary region.
        x[:, 0] = 0.10 + 0.40 * z[:, 0]
        x[:, 1] = 0.10 + 0.40 * z[:, 1]
        intercept = 0.95
    elif regime == "d3m_regime_2":
        # A pure label-rule move (no covariate change), framed by an
        # under-trained champion whose own disagreement noise already fills the
        # calibration envelope.  Predictions are unchanged, so the disagreement
        # statistic cannot see the deterioration: the P3 Regime-2 analogue.
        intercept = 1.50
    else:
        raise ValueError(f"unknown regime {regime!r}")

    y = (rng.random(n) < _label_probability(x, intercept=intercept)).astype(int)
    return x, y


def _fit_champion(x: np.ndarray, y: np.ndarray, weak: bool = False) -> LogisticRegression:
    """Fit the frozen champion; ``weak`` is the under-trained P3 Regime-2 case."""
    if weak:
        subset = np.random.default_rng(0).choice(len(x), size=min(60, len(x)), replace=False)
        model = LogisticRegression(C=0.01, max_iter=2000)
        model.fit(x[subset], y[subset])
    else:
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(x, y)
    return model


def _brier(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (prob - y) ** 2


def _stable_seed(*parts: object) -> int:
    """Process-stable seed (Python's ``hash`` is salted per process)."""
    import zlib

    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) % (2**32)


@dataclass
class DeploymentStream:
    """Time-ordered unlabeled arrivals with latent labels for auditing only."""

    x: np.ndarray
    y: np.ndarray
    champion_score: np.ndarray
    loss: np.ndarray
    batch_size: int
    n_steps: int
    onset_step: int
    regime: str
    seed: int
    run_id: str

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo = step * self.batch_size
        hi = (step + 1) * self.batch_size
        return self.x[lo:hi], self.y[lo:hi], self.champion_score[lo:hi], self.loss[lo:hi]

    def recent(self, step: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Most recent ``size`` labeled arrivals (used by the audit budget)."""
        hi = (step + 1) * self.batch_size
        lo = max(0, hi - size)
        return self.x[lo:hi], self.y[lo:hi]

    def risk_series(self, batch_size: int | None = None) -> np.ndarray:
        b = batch_size or self.batch_size
        n = (self.n_steps * self.batch_size) // b
        return self.loss[: n * b].reshape(n, b).mean(axis=1)

    def window(self, step: int, window_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rolling unlabeled window ending at ``step`` (no labels exposed)."""
        hi = (step + 1) * self.batch_size
        lo = max(0, hi - window_steps * self.batch_size)
        return self.x[lo:hi], self.champion_score[lo:hi], self.loss[lo:hi]


@dataclass
class ShiftScenario:
    """Reference/source sample, a frozen champion and a target sample."""

    regime: str
    seed: int
    run_id: str
    n_source: int
    n_target: int
    source_x: np.ndarray
    source_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    model: LogisticRegression
    predictions: PredictionArray
    source_score: np.ndarray
    source_loss: np.ndarray
    target_score: np.ndarray
    target_loss: np.ndarray
    shift_features: tuple[int, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ truth
    @property
    def source_risk(self) -> float:
        return float(np.mean(self.source_loss))

    @property
    def target_risk(self) -> float:
        return float(np.mean(self.target_loss))

    @property
    def true_gap(self) -> float:
        return self.target_risk - self.source_risk

    def summary(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "seed": self.seed,
            "run_id": self.run_id,
            "source_risk": self.source_risk,
            "target_risk": self.target_risk,
            "true_gap": self.true_gap,
        }

    # -------------------------------------------------------------- streaming
    def sample_deployment(
        self,
        seed: int,
        n_steps: int = 120,
        batch_size: int = 64,
        onset_frac: float = 0.30,
        ramp_steps: int = 4,
    ) -> DeploymentStream:
        """Draw a deployment stream: source law before onset, target law after.

        The shift is phased in over ``ramp_steps`` so the stream is not a step
        function, while delay is still measured from the onset step.
        """
        rng = np.random.default_rng(_stable_seed(self.regime, self.seed, seed))
        onset = int(round(onset_frac * n_steps))
        xs, ys = [], []
        for t in range(n_steps):
            if t < onset:
                frac = 0.0
            else:
                frac = min(1.0, (t - onset + 1) / max(1, ramp_steps))
            n_shift = int(round(batch_size * frac))
            n_base = batch_size - n_shift
            parts_x, parts_y = [], []
            if n_base:
                bx, by = _draw_source(rng, n_base)
                parts_x.append(bx)
                parts_y.append(by)
            if n_shift:
                tx, ty = _draw_target(rng, n_shift, self.regime)
                parts_x.append(tx)
                parts_y.append(ty)
            x_t = np.vstack(parts_x)
            y_t = np.concatenate(parts_y)
            perm = rng.permutation(batch_size)
            xs.append(x_t[perm])
            ys.append(y_t[perm])
        x = np.vstack(xs)
        y = np.concatenate(ys)
        score = self.model.predict_proba(x)[:, 1]
        loss = _brier(score, y)
        return DeploymentStream(
            x=x,
            y=y,
            champion_score=score,
            loss=loss,
            batch_size=batch_size,
            n_steps=n_steps,
            onset_step=onset,
            regime=self.regime,
            seed=seed,
            run_id=f"{self.run_id}+stream{seed}",
        )


def make_shift_scenario(seed: int, regime: str, n_source: int, n_target: int) -> ShiftScenario:
    """Build one seeded shift scenario: source sample, champion, target sample."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    rng = np.random.default_rng(seed)
    source_x, source_y = _draw_source(rng, n_source)
    target_x, target_y = _draw_target(rng, n_target, regime)
    weak = regime == "d3m_regime_2"
    model = _fit_champion(source_x, source_y, weak=weak)
    source_score = model.predict_proba(source_x)[:, 1]
    target_score = model.predict_proba(target_x)[:, 1]
    source_loss = _brier(source_score, source_y)
    target_loss = _brier(target_score, target_y)
    run_id = f"t5-r1-{regime}-s{seed}"
    shift_features = {
        "none": (),
        "sparse_joint": INFORMATIVE,
        "benign_covariate": (5, 6, 7),
        "dense_joint": tuple(range(D)),
        "harmful_concept": INFORMATIVE,
        "d3m_regime_2": INFORMATIVE,
    }[regime]
    predictions = PredictionArray(
        target_score,
        meta={
            "model": model,
            "regime": regime,
            "seed": seed,
            "run_id": run_id,
            "true_gap": float(np.mean(target_loss) - np.mean(source_loss)),
            "target_y": target_y,
            "target_loss": target_loss,
            "source_loss": source_loss,
        },
    )
    return ShiftScenario(
        regime=regime,
        seed=seed,
        run_id=run_id,
        n_source=n_source,
        n_target=n_target,
        source_x=source_x,
        source_y=source_y,
        target_x=target_x,
        target_y=target_y,
        model=model,
        predictions=predictions,
        source_score=source_score,
        source_loss=source_loss,
        target_score=target_score,
        target_loss=target_loss,
        shift_features=tuple(shift_features),
        notes={
            "label_scale": LABEL_SCALE,
            "weak_champion": weak,
            "champion_source_risk": float(np.mean(source_loss)),
        },
    )


def feature_summary(scenario: ShiftScenario) -> list[dict[str, Any]]:
    """Per-feature reference/deployment statistics for Fig A1."""
    rows: list[dict[str, Any]] = []
    for j in range(D):
        for window, xs in (("reference", scenario.source_x), ("deployment", scenario.target_x)):
            rows.append(
                {
                    "regime": scenario.regime,
                    "feature": f"x{j}",
                    "window": window,
                    "feature_mean": float(np.mean(xs[:, j])),
                    "feature_std": float(np.std(xs[:, j])),
                    "score_mean": float(
                        np.mean(scenario.source_score if window == "reference" else scenario.target_score)
                    ),
                    "latent_risk": scenario.source_risk if window == "reference" else scenario.target_risk,
                    "seed": scenario.seed,
                    "run_id": scenario.run_id,
                }
            )
    return rows
