"""corrgcv_study -- CorrGCV risk-vs-T curves (C38-C40), the (xi, lambda)
advantage region (C41), and Theorem VI.1's near-horizon test-point optimism (C42).
Also includes fig-d2 (C53): the Carmack estimate against the paper's unsquared GCCV bracket.

Sources:
- ``leakage-proof-ts-q3.json`` -- Fig. 1's two regimes (left: xi=1e-2, lambda=1e-2, sigma_eps=0.05;
  right: xi=1e2, lambda=1e-4, sigma_eps=0.05), N=100, T swept 10..1000, 10 realizations, anisotropic
  Sigma (alpha=1.8, r=0.3), and the three sample-correlation families (exponential, nearest-neighbour,
  power-law).
- ``leakage-proof-ts-q2.json`` -- the coupled (kappa, kappa_tilde) renormalisation, Theorem IV.2's
  deterministic-equivalent R_out, and Theorem VI.1 / Eq. (44).

Everything here is computed with the frozen helpers of ``nbs_common`` (solve_renormalized,
corrgcv_rep, power_law_spectrum, power_law_target_weights, toeplitz_exponential) plus the two K-family
generators this module adds. ``run()`` writes ``leakage_study/figures/fig-{a1,a2,a3,d2}.{csv,meta.json,png}`` and returns
the DATA dict the notebook's plot cells and ``ledger_rows`` consume.

Choices the sources do not state (all "derived here", recorded in each meta.json["config"]):
- seeded feature/noise streams, 10 seeds per point;
- nearest-neighbour amplitude b = min(2*exp(-1/xi), 0.999) and power-law exponent chi = 1/(xi*ln 2):
  each family's lag-1 correlation is matched to the exponential family's exp(-1/xi) at that regime
  (the NN cap is the positive-definiteness limit of a unit-diagonal tridiagonal Toeplitz matrix);
- every K is normalised to Tr(K)/T = 1 (all three families already have unit diagonal);
- a1/a2 use the paper's matched noise (K' = K, q2 assumption 2); a3's training noise is i.i.d. as the
  plan's sampler specifies.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

import nbs_common as nc

# This venv's OpenBLAS spins its 12 threads on every small LAPACK call and a 100x100 solve costs
# ~160 ms instead of ~0.2 ms (measured; a 1000x100 SVD 150 ms instead of 7 ms). Two threads is the
# measured optimum, so run() caps BLAS threads while it computes and restores them afterwards, which
# keeps the rest of the kernel's timing unchanged.
def _blas_context():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=2)
    except Exception:  # pragma: no cover - threadpoolctl is optional; correctness never depends on it
        import contextlib
        return contextlib.nullcontext()

EXEC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXEC_DIR.parent
FIG_DIR = EXEC_DIR / "figures"
SPEC_DIR = PROJECT_ROOT / "viz" / "spec"

# Exact hex matching the mock render scripts , one colour per series.
SERIES_COLORS = {
    "GCV1": "#2a78d6",
    "GCV2": "#eb6834",
    "Carmack": "#1baf7a",
    "CorrGCV": "#eda100",
    "latent": "#0b0b0b",
    "theory": "#e34948",
}
_SERIES_KEY = {"gcv1": "GCV1", "gcv2": "GCV2", "carmack": "Carmack",
               "corrgcv": "CorrGCV", "latent": "latent", "theory": "theory"}
DEFAULT_PALETTE = {"ink": "#20252b", "muted": "#8b949e", "grid": "#c9ced6", "band": "#898781"}

N_FEATURES = 100
ALPHA_EXP = 1.8
R_EXP = 0.3
SIGMA_EPS = 0.05
SEEDS_PER_POINT = 10
T_GRID = [10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000]
REGIMES = {
    "weak": {"xi": 1e-2, "lambda": 1e-2, "sigma_eps": 0.05},
    "strong": {"xi": 1e2, "lambda": 1e-4, "sigma_eps": 0.05},
}
FAMILIES = ("exponential", "nearest_neighbor", "power_law")
ESTIMATORS = ("gcv1", "gcv2", "carmack", "corrgcv")

A2_T = 150
A2_LOG10_XI = np.linspace(-2.5, 2.0, 13)
A2_LOG10_LAMBDA = np.linspace(-6.0, -1.0, 13)
A3_T = 100
A3_XI = 1e2
A3_LAMBDA = 1e-4
A3_TAUS = list(range(1, 11)) + [12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
A3_TEST_DRAWS = 64

# fig-d2 (): the same T grid as a1, weak regime only, and the two recorded series.
D2_T_GRID = list(T_GRID)
D2_SEEDS = SEEDS_PER_POINT
D2_SERIES = ("carmack_estimate", "bracket")

RUN_IDS = {"a1": "nb1-r03a-a1", "a2": "nb1-r03a-a2", "a3": "nb1-r03a-a3",
           "d2": "nb1-r4a-d2"}
SECTIONS = {"a1": 6, "a2": 7, "a3": 8, "d2": 8}

EIG_SIGMA = nc.power_law_spectrum(N_FEATURES, ALPHA_EXP)
SQRT_EIG = np.sqrt(EIG_SIGMA)
WBAR = nc.power_law_target_weights(N_FEATURES, ALPHA_EXP, R_EXP)


# ---------------------------------------------------------------------------
# config / hashing / seeds
# ---------------------------------------------------------------------------


def config() -> dict:
    return {
        "module": "r03_plan_a",
        "plan": "round-03/plan-a",
        "n_features": N_FEATURES,
        "alpha_exp": ALPHA_EXP,
        "r_exp": R_EXP,
        "seeds_per_point": SEEDS_PER_POINT,
        "feature_seed0": 3100,
        "noise_seed0": 9100,
        "a1": {
            "T_grid": list(T_GRID),
            "regimes": {k: dict(v) for k, v in REGIMES.items()},
            "k_families": list(FAMILIES),
            "nearest_neighbor_b": "min(2*exp(-1/xi), 0.999) -- lag-1 matched, PD cap (derived here)",
            "power_law_chi": "1/(xi*ln 2) -- lag-1 matched (derived here)",
            "k_normalisation": "Tr(K)/T = 1",
            "noise": "matched K'=K (q2 assumption 2)",
        },
        "a2": {
            "T": A2_T,
            "N": N_FEATURES,
            "log10_xi": [float(v) for v in A2_LOG10_XI],
            "log10_lambda": [float(v) for v in A2_LOG10_LAMBDA],
            "alpha_exp": ALPHA_EXP,
            "r_exp": R_EXP,
            "sigma_eps": SIGMA_EPS,
            "noise": "matched K'=K (q2 assumption 2)",
            "ratio": "mean_over_seeds |gcv1-latent|/latent divided by mean_over_seeds |corrgcv-latent|/latent",
        },
        "a3": {
            "T": A3_T,
            "N": N_FEATURES,
            "xi": A3_XI,
            "lambda": A3_LAMBDA,
            "sigma_eps": SIGMA_EPS,
            "taus": list(A3_TAUS),
            "test_draws_per_seed": A3_TEST_DRAWS,
            "test_point": "x = X^T alpha + sqrt(1-rho) Sigma^{1/2} z (Thm VI.1 Eq. 43)",
            "risk": "((wbar - w_hat)^T x)^2 + sigma_eps^2",
            "alpha_exp": ALPHA_EXP,
            "r_exp": R_EXP,
        },
        "d2": {
            "regime": "weak",
            "xi": REGIMES["weak"]["xi"],
            "lambda": REGIMES["weak"]["lambda"],
            "sigma_eps": REGIMES["weak"]["sigma_eps"],
            "N": N_FEATURES,
            "T_grid": list(D2_T_GRID),
            "k_families": list(FAMILIES),
            "series": list(D2_SERIES),
            "seeds_per_point": D2_SEEDS,
            "bracket_formula": "1 - (df2*dtf2)/(df1*dtf1)/(1 - dtf2/dtf1) (Carmack GCCV, unsquared)",
            "carmack": "S^2 * bracket^2 * R_in (S = kappa/lambda, population df-form bracket)",
            "bracket_per_seed": (
                "feature-side df1/df2 re-estimated from the realized covariance diagonal via the "
                "exact trace identity E[X^T X / T] = Sigma (unbiased at the renormalized kappa); "
                "K-side dtf1/dtf2 from the design's known K -- so the recorded seed spread is "
                "finite-sample estimation error, while the exact bracket is deterministic"),
            "alpha_exp": ALPHA_EXP,
            "r_exp": R_EXP,
        },
    }


def config_hash() -> str:
    return hashlib.sha256(json.dumps(config(), sort_keys=True).encode()).hexdigest()


def _seed(stream: str, *parts) -> int:
    tag = "r03a|%s|%s" % (stream, "|".join(str(p) for p in parts))
    return int(nc.sha256_text(tag)[:14], 16)


# ---------------------------------------------------------------------------
# K families and the deterministic-equivalent theory curve
# ---------------------------------------------------------------------------


def family_params(family: str, xi: float) -> dict:
    """Per-family correlation parameter, lag-1 matched to exp(-1/xi) (derived here)."""
    if family == "exponential":
        return {"xi": float(xi)}
    if family == "nearest_neighbor":
        return {"b": float(min(2.0 * math.exp(-1.0 / xi), 0.999))}
    if family == "power_law":
        return {"chi": float(1.0 / (xi * math.log(2.0)))}
    raise ValueError(family)


def family_k(family: str, T: int, xi: float) -> np.ndarray:
    """The q3 sample-correlation families, normalised to Tr(K)/T = 1."""
    if family == "exponential":
        K = nc.toeplitz_exponential(T, xi)
    elif family == "nearest_neighbor":
        b = family_params(family, xi)["b"]
        K = np.eye(T)
        idx = np.arange(T - 1)
        K[idx, idx + 1] = b / 2.0
        K[idx + 1, idx] = b / 2.0
    elif family == "power_law":
        chi = family_params(family, xi)["chi"]
        d = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
        K = (1.0 + d) ** (-chi)
    else:
        raise ValueError(family)
    return K * (T / float(np.trace(K)))


def factor(K: np.ndarray):
    """(L, ev) with L L^T = K (Cholesky) and ev = eigenvalues of K.

    Same matrix-Gaussian law as nbs_common.matrix_gaussian_sigma's symmetric square root (any
    square root of K gives the same distribution) at a fraction of the cost. A matrix whose
    off-diagonals vanish below 1e-30 is the identity to float64 precision (the weak regime's
    xi=1e-2 makes every family's off-diagonal exp(-100) ~ 1e-44), so its spectrum is exactly 1.
    """
    off = K - np.diag(np.diag(K))
    if float(np.max(np.abs(off))) < 1e-30:
        return np.eye(K.shape[0]), np.ones(K.shape[0])
    return np.linalg.cholesky(K), np.linalg.eigvalsh(K)


def gaussian_design(T: int, L: np.ndarray, seed: int) -> np.ndarray:
    """X = K^{1/2} Z Sigma^{1/2}, rows carrying covariance K, features covariance Sigma."""
    Z = np.random.default_rng(seed).standard_normal((T, N_FEATURES))
    return (L @ Z) * SQRT_EIG


def theory_risk(ident: dict, sigma_eps: float = SIGMA_EPS) -> float:
    """Theorem IV.2's deterministic-equivalent R_out from the solved (kappa, kappa_tilde):

        R_g   ~ kappa^2/(1-gamma) * wbar^T Sigma (Sigma+kappa)^-2 wbar + gamma/(1-gamma) sigma_eps^2
        R_out = R_g + sigma_eps^2,      gamma = (df2/df1) * (ndf2/ndf1)
    """
    kappa = ident["kappa"]
    gamma = (ident["df2"] / ident["df1"]) * (ident["ndf2"] / ident["ndf1"])
    bias = kappa ** 2 * float(np.sum(WBAR ** 2 * EIG_SIGMA / (EIG_SIGMA + kappa) ** 2))
    return (bias / (1.0 - gamma) + (gamma / (1.0 - gamma)) * float(sigma_eps) ** 2
            + float(sigma_eps) ** 2)


# ---------------------------------------------------------------------------
# figure A1 -- risk vs T, 2 regimes x 3 K families x 6 series
# ---------------------------------------------------------------------------


def rows_a1() -> tuple:
    t0 = time.perf_counter()
    rows = []
    for regime, spec in REGIMES.items():
        lam, sig = spec["lambda"], spec["sigma_eps"]
        for family in FAMILIES:
            for T in T_GRID:
                K = family_k(family, T, spec["xi"])
                L, ev_K = factor(K)
                ident = nc.solve_renormalized(lam, N_FEATURES / T, EIG_SIGMA, ev_K)
                theory = theory_risk(ident, sig)
                est = {k: [] for k in ESTIMATORS}
                latent = []
                for s in range(SEEDS_PER_POINT):
                    base = _seed("a1", regime, family, T, s)
                    X = gaussian_design(T, L, base)
                    eps = L @ np.random.default_rng(base + 1).standard_normal(T)
                    y = X @ WBAR + sig * eps
                    rep = nc.corrgcv_rep(X, y, lam, ident, sig, WBAR, EIG_SIGMA)
                    for k in ESTIMATORS:
                        est[k].append(rep[k])
                    latent.append(rep["latent_risk"])
                for k in ESTIMATORS:
                    mean, se = _mean_se(est[k])
                    rows.append({"regime": regime, "k_family": family, "T": T,
                                 "series": k, "mean": mean, "se": se})
                mean, se = _mean_se(latent)
                rows.append({"regime": regime, "k_family": family, "T": T,
                             "series": "latent", "mean": mean, "se": se})
                rows.append({"regime": regime, "k_family": family, "T": T,
                             "series": "theory", "mean": theory, "se": 0.0})
    return rows, time.perf_counter() - t0


def _mean_se(values) -> tuple:
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


# ---------------------------------------------------------------------------
# figure A2 -- the (log10 xi, log10 lambda) advantage region
# ---------------------------------------------------------------------------


def rows_a2() -> tuple:
    t0 = time.perf_counter()
    rows = []
    sig = SIGMA_EPS
    for lxi in A2_LOG10_XI:
        xi = float(10.0 ** lxi)
        K = nc.toeplitz_exponential(A2_T, xi)
        L, ev_K = factor(K)
        for llam in A2_LOG10_LAMBDA:
            lam = float(10.0 ** llam)
            ident = nc.solve_renormalized(lam, N_FEATURES / A2_T, EIG_SIGMA, ev_K)
            errs = {k: [] for k in ("gcv1", "corrgcv")}
            for s in range(SEEDS_PER_POINT):
                base = _seed("a2", round(lxi, 6), round(llam, 6), s)
                X = gaussian_design(A2_T, L, base)
                eps = L @ np.random.default_rng(base + 1).standard_normal(A2_T)
                y = X @ WBAR + sig * eps
                rep = nc.corrgcv_rep(X, y, lam, ident, sig, WBAR, EIG_SIGMA)
                for k in errs:
                    errs[k].append(abs(rep[k] - rep["latent_risk"]) / rep["latent_risk"])
            e1 = float(np.mean(errs["gcv1"]))
            ec = float(np.mean(errs["corrgcv"]))
            rows.append({"log10_xi": float(lxi), "log10_lambda": float(llam),
                         "log10_ratio": float(math.log10(e1 / ec))})
    return rows, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# figure A3 -- Theorem VI.1's near-horizon test-point optimism
# ---------------------------------------------------------------------------


def rows_a3() -> tuple:
    t0 = time.perf_counter()
    T, sig = A3_T, SIGMA_EPS
    K = nc.toeplitz_exponential(T, A3_XI)
    L, ev_K = factor(K)
    ident = nc.solve_renormalized(A3_LAMBDA, N_FEATURES / T, EIG_SIGMA, ev_K)
    kappa_tilde = ident["kappa_tilde"]
    eye = np.eye(T)
    train = []
    for s in range(SEEDS_PER_POINT):
        base = _seed("a3", s)
        X = gaussian_design(T, L, base)
        rng = np.random.default_rng(base + 1)
        y = X @ WBAR + sig * rng.standard_normal(T)
        w_hat = np.linalg.solve(X.T @ X / T + A3_LAMBDA * np.eye(N_FEATURES), X.T @ y / T)
        dw = WBAR - w_hat
        train.append({"X": X, "dw": dw,
                      "z_k": rng.standard_normal((A3_TEST_DRAWS, N_FEATURES)),
                      "z_0": rng.standard_normal((A3_TEST_DRAWS, N_FEATURES))})
    rows = []
    t_idx = np.arange(1, T + 1)
    for tau in A3_TAUS:
        k = np.exp(-np.abs(T + tau - t_idx) / A3_XI)
        alpha = np.linalg.solve(K, k)
        rho = float(k @ alpha)
        w = np.linalg.solve(K + kappa_tilde * eye, alpha)
        w = np.linalg.solve(K + kappa_tilde * eye, w)
        last = kappa_tilde ** 2 * float((K @ alpha) @ w)
        ratios = []
        for tr in train:
            ua = float((tr["X"] @ tr["dw"]) @ alpha)
            sk = math.sqrt(max(0.0, 1.0 - rho))
            rk = float(np.mean((ua + sk * (tr["z_k"] @ (SQRT_EIG * tr["dw"]))) ** 2)) + sig ** 2
            r0 = float(np.mean((tr["z_0"] @ (SQRT_EIG * tr["dw"])) ** 2)) + sig ** 2
            ratios.append(rk / r0)
        mean, se = _mean_se(ratios)
        rows.append({"tau": int(tau), "rho": rho, "eq44_factor": 1.0 - rho + last,
                     "lower_band": 1.0 - rho, "upper_band": 1.0,
                     "empirical_mean": mean, "empirical_se": se})
    return rows, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# figure D2 -- is the Carmack dip the GCCV bracket's zero crossing? ()
# ---------------------------------------------------------------------------


def d2_bracket_realized(X, ident, sigma_eigenvalues=EIG_SIGMA):
    """Per-realization value of the paper's Carmack GCCV bracket at the renormalized ridge kappa.

    ``1 - (df2*dtf2)/(df1*dtf1)/(1 - dtf2/dtf1)``.  The K-side terms ``dtf1, dtf2`` are the
    design's known sample-correlation degrees of freedom (ev_K), as everywhere else in this
    notebook.  The feature-side terms are functionals of Sigma at kappa; their per-realization
    estimates use the exact trace identity ``E[X^T X / T] = Sigma`` (unit-diagonal K):
    ``df1_hat = N^-1 sum_i Shat_ii / (ev_i + kappa)`` and
    ``df2_hat = df1_hat - kappa * N^-1 sum_i Shat_ii / (ev_i + kappa)^2``,
    both exactly unbiased for ``df1``/``df2``.  The mean over seeds therefore reproduces the
    deterministic df-form bracket while the seed spread is genuine finite-sample estimation
    error -- the exact bracket is deterministic in ``ident`` and would have se = 0.  The exact
    value is exposed per realization as ``nbs_common.corrgcv_rep(...)["bracket"]``.
    """
    kap = float(ident["kappa"])
    ev = np.asarray(sigma_eigenvalues, float)
    shat_ii = np.mean(np.asarray(X, float) ** 2, axis=0)
    df1_hat = float(np.mean(shat_ii / (ev + kap)))
    g_hat = float(np.mean(shat_ii / (ev + kap) ** 2))
    df2_hat = df1_hat - kap * g_hat
    n1, n2 = float(ident["ndf1"]), float(ident["ndf2"])
    return 1.0 - (df2_hat * n2) / (df1_hat * n1) * 1.0 / (1.0 - n2 / n1)


def d2_exact_bracket(ident):
    """Deterministic df-form bracket straight from the solved renormalized ident."""
    d1, d2, n1, n2 = ident["df1"], ident["df2"], ident["ndf1"], ident["ndf2"]
    return 1.0 - (d2 * n2) / (d1 * n1) * 1.0 / (1.0 - n2 / n1)


def rows_d2() -> tuple:
    """Weak-regime rows for fig-d2: mean +/- se over 10 seeds, one row per (family, T, series)."""
    t0 = time.perf_counter()
    spec = REGIMES["weak"]
    lam, sig, xi = spec["lambda"], spec["sigma_eps"], spec["xi"]
    rows, detail = [], {}
    for family in FAMILIES:
        for T in D2_T_GRID:
            K = family_k(family, T, xi)
            L, ev_K = factor(K)
            ident = nc.solve_renormalized(lam, N_FEATURES / T, EIG_SIGMA, ev_K)
            carmack, bracket = [], []
            for s in range(D2_SEEDS):
                base = _seed("d2", family, T, s)
                X = gaussian_design(T, L, base)
                eps = L @ np.random.default_rng(base + 1).standard_normal(T)
                y = X @ WBAR + sig * eps
                rep = nc.corrgcv_rep(X, y, lam, ident, sig, WBAR, EIG_SIGMA)
                carmack.append(float(rep["carmack"]))
                bracket.append(d2_bracket_realized(X, ident))
            for series, vals in (("carmack_estimate", carmack), ("bracket", bracket)):
                mean, se = _mean_se(vals)
                rows.append({"regime": "weak", "k_family": family, "T": int(T),
                             "series": series, "mean": mean, "se": se})
            detail[(family, int(T))] = {
                "kappa": float(ident["kappa"]), "kappa_tilde": float(ident["kappa_tilde"]),
                "exact_bracket": float(d2_exact_bracket(ident)),
                "carmack_seeds": carmack, "bracket_seeds": bracket,
            }
    return rows, detail, time.perf_counter() - t0


def _first_sign_change_T(points):
    """Log-T-interpolated first sign-change location of a (T, value) series (None if none)."""
    for (t0_, v0), (t1_, v1) in zip(points, points[1:]):
        if v0 == 0.0:
            return float(t0_)
        if (v0 < 0) != (v1 < 0):
            frac = -v0 / (v1 - v0)
            return float(math.exp(math.log(t0_) + frac * (math.log(t1_) - math.log(t0_))))
    return None


def d2_stats(rows, detail) -> dict:
    """Dip location vs bracket zero crossing per K family, from the recorded series only."""
    out, n_sign = {}, 0
    for family in FAMILIES:
        carmack = sorted((r["T"], r["mean"]) for r in rows
                         if r["k_family"] == family and r["series"] == "carmack_estimate")
        bracket = sorted((r["T"], r["mean"]) for r in rows
                         if r["k_family"] == family and r["series"] == "bracket")
        dip_T = min(carmack, key=lambda p: p[1])[0]
        dip_mean = min(v for _, v in carmack)
        zero_T = _first_sign_change_T(bracket)
        exact = [detail[(family, T)]["exact_bracket"] for T in D2_T_GRID]
        exact_zero = _first_sign_change_T(list(zip(D2_T_GRID, exact)))
        if zero_T is not None:
            n_sign += 1
        out[family] = {"dip_T": int(dip_T), "dip_mean": float(dip_mean),
                       "zero_T": None if zero_T is None else float(zero_T),
                       "exact_zero_T": None if exact_zero is None else float(exact_zero),
                       "log10_ratio_dip_vs_zero": (None if zero_T is None
                                                   else float(math.log10(dip_T / zero_T)))}
    return {"families": out, "n_families_sign_change": n_sign}



# ---------------------------------------------------------------------------
# checks (shared by the notebook cells and any standalone verifier)
# ---------------------------------------------------------------------------


def check_c38(rows) -> dict:
    panels = {}
    worst = {}
    ok = True
    for regime in REGIMES:
        for family in FAMILIES:
            ts = sorted({r["T"] for r in rows if r["regime"] == regime and r["k_family"] == family})
            grid_ok = len(ts) >= 8 and ts[0] <= 10 and ts[-1] >= 1000
            min_se = min(float(r["se"]) for r in rows
                         if r["regime"] == regime and r["k_family"] == family
                         and r["series"] != "theory")
            panels[(regime, family)] = {"n_T": len(ts), "min_T": ts[0], "max_T": ts[-1]}
            worst[(regime, family)] = min_se
            ok = ok and grid_ok and min_se > 0.0
    return {"ok": bool(ok), "panels": panels, "min_se_overall": float(min(worst.values()))}


def check_c39_strong(rows) -> dict:
    panel = {(r["T"], r["series"]): float(r["mean"]) for r in rows
             if r["regime"] == "strong" and r["k_family"] == "exponential"}
    ts = sorted({T for (T, s) in panel if s == "latent" and T >= 100})
    detail = {}
    ok = True
    worst = 0.0
    for T in ts:
        lat = panel[(T, "latent")]
        errs = {k: abs(panel[(T, k)] - lat) / lat for k in ("gcv1", "gcv2", "corrgcv")}
        ratio = errs["corrgcv"] / max(errs["gcv1"], errs["gcv2"])
        worst = max(worst, ratio)
        detail[T] = errs
        ok = ok and errs["corrgcv"] < errs["gcv1"] and errs["corrgcv"] < errs["gcv2"]
    return {"ok": bool(ok), "n_T_ge_100": len(ts), "worst_ratio": float(worst), "detail": detail}


def check_c39_weak(rows) -> dict:
    detail = {}
    ok = True
    for family in FAMILIES:
        ts = sorted({r["T"] for r in rows if r["regime"] == "weak" and r["k_family"] == family})

        def spread(T):
            vals = [float(r["mean"]) for r in rows
                    if r["regime"] == "weak" and r["k_family"] == family
                    and r["series"] in ESTIMATORS and r["T"] == T]
            return max(vals) - min(vals)

        lo, hi = spread(ts[0]), spread(ts[-1])
        detail[family] = {"spread_T_min": lo, "spread_T_max": hi}
        ok = ok and hi < lo
    return {"ok": bool(ok), "detail": detail}


def check_c40(rows) -> dict:
    gaps = []
    per_panel = {}
    for regime in REGIMES:
        for family in FAMILIES:
            latent = {r["T"]: float(r["mean"]) for r in rows
                      if r["regime"] == regime and r["k_family"] == family and r["series"] == "latent"}
            theory = {r["T"]: float(r["mean"]) for r in rows
                      if r["regime"] == regime and r["k_family"] == family and r["series"] == "theory"}
            panel = [abs(theory[T] - latent[T]) / latent[T] for T in sorted(latent)]
            per_panel[(regime, family)] = float(np.median(panel))
            gaps.extend(panel)
    med = float(np.median(gaps))
    return {"ok": bool(med <= 0.25), "median_gap": med, "n_points": len(gaps),
            "max_gap": float(np.max(gaps)), "per_panel_median": per_panel}


def check_c41(rows) -> dict:
    xis = sorted({float(r["log10_xi"]) for r in rows})
    lams = sorted({float(r["log10_lambda"]) for r in rows})
    ratios = np.array([float(r["log10_ratio"]) for r in rows])
    frac = float(np.mean(ratios >= math.log10(2.0)))
    return {"ok": bool(len(xis) >= 6 and len(lams) >= 6 and np.all(np.isfinite(ratios))),
            "n_xi": len(xis), "n_lambda": len(lams), "advantage_fraction": frac,
            "min": float(ratios.min()), "max": float(ratios.max())}


def check_c42_band(rows) -> dict:
    eps = 1e-9
    worst = max(max(float(r["lower_band"]) - float(r["eq44_factor"]),
                    float(r["eq44_factor"]) - float(r["upper_band"])) for r in rows)
    return {"ok": bool(worst <= eps), "max_excursion": float(worst), "n_tau": len(rows)}


def check_c42_empirical(rows) -> dict:
    tol = 3.0 * max(float(r["empirical_se"]) for r in rows)
    worst = max(max(float(r["lower_band"]) - float(r["empirical_mean"]),
                    float(r["empirical_mean"]) - float(r["upper_band"])) for r in rows)
    vals = np.array([float(r["empirical_mean"]) for r in rows])
    spread = float(vals.max() - vals.min())
    return {"ok": bool(worst <= tol and spread > 0.05), "tol": float(tol),
            "max_excursion": float(worst), "spread": spread}


def _read_rows_d2() -> list:
    with open(FIG_DIR / "fig-d2.csv", newline="") as fh:
        return [{"regime": r["regime"], "k_family": r["k_family"], "T": float(r["T"]),
                 "series": r["series"], "mean": float(r["mean"]), "se": float(r["se"])}
                for r in csv.DictReader(fh)]


def check_d2(data=None) -> dict:
    """C53 well-formedness recomputed from the written fig-d2.csv (never from DATA).

    Both series must share a >= 8-point T grid per K family, both se > 0 (seeds ran), the
    Carmack mean must stay positive for its log axis, and the bracket's mean must change sign
    across the grid so the dip-vs-zero comparison is well defined.
    """
    rows = _read_rows_d2()
    weak = [r for r in rows if r["regime"] == "weak"]
    present = {r["k_family"] for r in weak}
    families = [f for f in FAMILIES if f in present]
    detail, ok, n_sign = {}, True, 0
    min_carmack, worst_ratio = float("inf"), 0.0
    for fam in families:
        carmack = sorted((r["T"], r["mean"], r["se"]) for r in weak
                         if r["k_family"] == fam and r["series"] == "carmack_estimate")
        bracket = sorted((r["T"], r["mean"], r["se"]) for r in weak
                         if r["k_family"] == fam and r["series"] == "bracket")
        same_grid = (len(carmack) >= 8 and len(bracket) >= 8
                     and [t for t, _, _ in carmack] == [t for t, _, _ in bracket])
        carmack_ok = all(math.isfinite(m) and m > 0.0 and se > 0.0 for _, m, se in carmack)
        bracket_ok = all(math.isfinite(m) and se > 0.0 for _, m, se in bracket)
        dip_T = min(carmack, key=lambda p: p[1])[0]
        zero_T = _first_sign_change_T([(t, m) for t, m, _ in bracket])
        if zero_T is not None:
            n_sign += 1
            worst_ratio = max(worst_ratio, abs(math.log10(dip_T / zero_T)))
        min_carmack = min(min_carmack, min(m for _, m, _ in carmack))
        detail[fam] = {"dip_T": int(dip_T), "zero_T": None if zero_T is None else float(zero_T),
                       "n_T_carmack": len(carmack), "n_T_bracket": len(bracket),
                       "log10_dip_over_zero": (None if zero_T is None
                                               else float(math.log10(dip_T / zero_T)))}
        ok = ok and same_grid and carmack_ok and bracket_ok and zero_T is not None
    return {"ok": bool(ok and len(families) >= 3), "n_families": len(families),
            "n_families_sign_change": n_sign, "families": detail,
            "min_carmack_mean": float(min_carmack), "max_abs_log10_ratio": float(worst_ratio)}


def check_d2_files() -> dict:
    """fig-d2 triple integrity: files, meta keys, caption verbatim, PNG width, ledger run_id."""
    csv_p = FIG_DIR / "fig-d2.csv"
    meta_p = FIG_DIR / "fig-d2.meta.json"
    png_p = FIG_DIR / "fig-d2.png"
    entry = {"csv": csv_p.exists(), "meta": meta_p.exists(), "png": png_p.exists()}
    if entry["meta"]:
        meta = json.loads(meta_p.read_text())
        entry["keys_ok"] = {"run_id", "seeds", "config", "caption", "how_to_read",
                            "plot_type"} <= set(meta)
        entry["how_to_read_ok"] = meta.get("how_to_read") == how_to_read("d2")
        entry["caption_ok"] = str(meta.get("caption", "")).startswith("Figure d2:")
        entry["plot_type"] = meta.get("plot_type")
        entry["run_id"] = meta.get("run_id")
    if entry["png"]:
        head = png_p.read_bytes()
        entry["png_ok"] = head[:8] == b"\x89PNG\r\n\x1a\n"
        entry["png_width"] = int.from_bytes(head[16:20], "big")
    ids = {r["run_id"] for r in nc.read_ledger()}
    entry["ledger_ok"] = entry.get("run_id") in ids
    mock_p = PROJECT_ROOT / "viz" / "mock" / "fig-d2.csv"
    entry["mock_differs"] = bool(entry["csv"] and mock_p.exists()
                                 and nc.sha256_file(csv_p) != nc.sha256_file(mock_p))
    ok = all(entry.get(k, False) for k in ("csv", "meta", "png", "keys_ok", "how_to_read_ok",
                                           "caption_ok", "png_ok", "ledger_ok", "mock_differs"))
    ok = ok and entry.get("png_width", 0) >= 1200 and entry.get("plot_type") == "ribbon"
    return {"ok": bool(ok), "detail": entry}


def figure_files_status() -> dict:
    """Artifact integrity for the three figure triples (files, meta keys, caption verbatim)."""
    status = {}
    ok = True
    for fig_id in ("a1", "a2", "a3"):
        csv_p = FIG_DIR / f"fig-{fig_id}.csv"
        meta_p = FIG_DIR / f"fig-{fig_id}.meta.json"
        png_p = FIG_DIR / f"fig-{fig_id}.png"
        entry = {"csv": csv_p.exists(), "meta": meta_p.exists(), "png": png_p.exists()}
        if entry["meta"]:
            meta = json.loads(meta_p.read_text())
            entry["keys_ok"] = {"run_id", "seeds", "config", "caption", "how_to_read",
                                "plot_type"} <= set(meta)
            entry["how_to_read_ok"] = meta.get("how_to_read") == how_to_read(fig_id)
            entry["plot_type"] = meta.get("plot_type")
        if entry["png"]:
            head = png_p.read_bytes()
            entry["png_ok"] = head[:8] == b"\x89PNG\r\n\x1a\n"
            entry["png_width"] = int.from_bytes(head[16:20], "big")
        status[fig_id] = entry
        ok = ok and all(entry.get(k, False) for k in ("csv", "meta", "png", "keys_ok",
                                                      "how_to_read_ok", "png_ok"))
        ok = ok and entry.get("png_width", 0) >= 1200
    return {"ok": bool(ok), "detail": status}


def check_ledger(rows=None) -> dict:
    ids = {RUN_IDS["a1"], RUN_IDS["a2"], RUN_IDS["a3"], RUN_IDS["d2"]}
    by_id = {}
    for r in nc.read_ledger():
        if r["run_id"] in ids:
            by_id[r["run_id"]] = r
    entries = list(by_id.values())
    claims = {c for r in entries for c in r["claim_ids"]}
    ok = (len(entries) == 4
          and claims >= {"C38", "C39", "C40", "C41", "C42", "C53"}
          and all(re.fullmatch(r"[0-9a-f]{64}", r["data_hash"]) for r in entries)
          and all(r["external_spend_usd"] == 0.0 for r in entries)
          and all(r["fold_boundaries"] is None for r in entries)
          and all(isinstance(r["metric"], dict) and r["metric"] for r in entries))
    return {"ok": bool(ok), "n_rows": len(entries),
            "claims": sorted(claims), "data_hash": entries[0]["data_hash"] if entries else None}


# ---------------------------------------------------------------------------
# figure IO
# ---------------------------------------------------------------------------

_A1_COLUMNS = ("regime", "k_family", "T", "series", "mean", "se")
_A2_COLUMNS = ("log10_xi", "log10_lambda", "log10_ratio")
_A3_COLUMNS = ("tau", "rho", "eq44_factor", "lower_band", "upper_band",
               "empirical_mean", "empirical_se")


def _write_csv(path: Path, columns, rows, formatters) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(columns))
        for row in rows:
            writer.writerow([formatters.get(c, _fmt_any)(row[c]) for c in columns])
    return nc.sha256_file(path)


def _fmt_any(value) -> str:
    if isinstance(value, str):
        return value
    return "%.10g" % float(value)


def _fmt(value) -> str:
    return "%.10g" % float(value)


def _fmt_int(value) -> str:
    return "%d" % int(value)


def _fmt6(value) -> str:
    return "%.6f" % float(value)


def spec_text(fig_id: str) -> str:
    return (SPEC_DIR / f"fig-{fig_id}.md").read_text()


def how_to_read(fig_id: str) -> str:
    m = re.search(r"\*\*How to read this chart:\*\*\s*(.+)", spec_text(fig_id), re.S)
    if not m:
        raise RuntimeError(f"the figure description has no 'How to read this chart' caption")
    return m.group(1).strip()


def spec_plot_type(fig_id: str) -> str:
    m = re.search(r"^plot_type:\s*(.+)$", spec_text(fig_id), re.M)
    if not m:
        raise RuntimeError(f"the figure description has no plot_type")
    return m.group(1).strip()


_A1_CAPTION = ("Figure a1: estimator risk vs. T across regimes and K families -- GCV1, GCV2, Carmack, "
               "CorrGCV and the latent out-of-sample risk with +-1.96*SE ribbons over 10 seeds, plus the "
               "dashed deterministic-equivalent theory curve (Thm IV.2 / Eq. 30).")
_A2_CAPTION = ("Figure a2: CorrGCV's advantage region over the (log10 xi, log10 lambda) plane -- colour "
               "is log10(err_GCV1 / err_CorrGCV), diverging around 0, contour at log10 2.")
_A3_CAPTION = ("Figure a3: near-horizon test-point optimism (Theorem VI.1) -- empirical "
               "R^k_out / R^{k=0}_out with +-1.96*SE ribbons over 10 seeds against the dashed Eq. (44) "
               "factor and the admissible band [1 - rho(tau), 1].")
_D2_CAPTION = ("Figure d2: the Carmack GCCV estimate (log y) and the paper's unsquared bracket "
               "1 - (df2*dtf2)/(df1*dtf1)/(1 - dtf2/dtf1) (linear y, dashed zero line) against a shared "
               "log T axis, weak regime, one line +-1.96*SE ribbon per K family over 10 seeds -- the "
               "dip in the upper panel is compared with the bracket's zero crossing in the lower panel.")

_CAPTIONS = {"a1": _A1_CAPTION, "a2": _A2_CAPTION, "a3": _A3_CAPTION, "d2": _D2_CAPTION}


def _meta(fig_id: str, cfg: dict) -> dict:
    return {
        "run_id": RUN_IDS[fig_id],
        "seeds": SEEDS_PER_POINT,
        "config": cfg,
        "caption": _CAPTIONS[fig_id],
        "how_to_read": how_to_read(fig_id),
        "plot_type": spec_plot_type(fig_id),
    }


def _save_fig(fig, path: Path, dpi: float = 200.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)


def run() -> dict:
    """Compute every figure, write the three CSV/meta/PNG triples, return DATA."""
    with _blas_context():
        return _run_impl()


def _run_impl() -> dict:
    t0 = time.perf_counter()
    rows_1, w1 = rows_a1()
    rows_2, w2 = rows_a2()
    rows_3, w3 = rows_a3()
    rows_d, d2_detail, wd2 = rows_d2()
    cfg = config()
    d2_rows = sorted(rows_d, key=lambda r: (FAMILIES.index(r["k_family"]), r["T"], D2_SERIES.index(r["series"])))
    data = {
        "seeds": SEEDS_PER_POINT,
        "config": cfg,
        "config_hash": config_hash(),
        "run_ids": dict(RUN_IDS),
        "a1": {"rows": rows_1, "meta": _meta("a1", cfg["a1"])},
        "a2": {"rows": rows_2, "meta": _meta("a2", cfg["a2"])},
        "a3": {"rows": rows_3, "meta": _meta("a3", cfg["a3"])},
        "d2": {"rows": d2_rows, "detail": d2_detail, "meta": _meta("d2", cfg["d2"]),
               "stats": d2_stats(d2_rows, d2_detail)},
        "wall_times": {"a1": w1, "a2": w2, "a3": w3, "d2": wd2},
    }
    _write_csv(FIG_DIR / "fig-a1.csv", _A1_COLUMNS, rows_1,
               {"T": _fmt_int, "mean": _fmt, "se": _fmt})
    _write_csv(FIG_DIR / "fig-a2.csv", _A2_COLUMNS, rows_2, {"log10_ratio": _fmt6})
    _write_csv(FIG_DIR / "fig-a3.csv", _A3_COLUMNS, rows_3, {"tau": _fmt_int})
    _write_csv(FIG_DIR / "fig-d2.csv", _A1_COLUMNS, d2_rows, {"T": _fmt_int, "mean": _fmt, "se": _fmt})
    for fig_id, render in (("a1", render_fig_a1), ("a2", render_fig_a2), ("a3", render_fig_a3),
                           ("d2", render_fig_d2)):
        (FIG_DIR / f"fig-{fig_id}.meta.json").write_text(json.dumps(_meta(fig_id, cfg[fig_id]), indent=1))
        # a2 is a single-column 4.6in figure: 200 dpi would be 920 px, so it keeps the mock's 320 dpi
        _save_fig(render(data, DEFAULT_PALETTE, show=False), FIG_DIR / f"fig-{fig_id}.png",
                  dpi=320.0 if fig_id == "a2" else 200.0)
    data["wall_time_s"] = time.perf_counter() - t0
    return data


def ledger_rows(data) -> list:
    """Four ledger rows (one per figure) covering C38-C42 and C53, zero external spend."""
    cfg_hash = data["config_hash"]
    c38 = check_c38(data["a1"]["rows"])
    c39s = check_c39_strong(data["a1"]["rows"])
    c39w = check_c39_weak(data["a1"]["rows"])
    c40 = check_c40(data["a1"]["rows"])
    c41 = check_c41(data["a2"]["rows"])
    c42b = check_c42_band(data["a3"]["rows"])
    c42e = check_c42_empirical(data["a3"]["rows"])
    c53 = check_d2(data)
    wt = data["wall_times"]
    return [
        nc.ledger_row(RUN_IDS["a1"], "01", SECTIONS["a1"], {"name": "corrgcv-risk-vs-T-grid"},
                      cfg_hash, None, "corrgcv-risk-curve",
                      {"T_grid": T_GRID, "regimes": list(REGIMES), "k_families": list(FAMILIES),
                       "seeds": SEEDS_PER_POINT, "noise": "matched K'=K"}, 0,
                      {"median_theory_gap": c40["median_gap"],
                       "strong_corrgcv_worst_err_ratio": c39s["worst_ratio"],
                       "min_se_over_panels": c38["min_se_overall"],
                       "min_weak_spread_ratio": min(v["spread_T_max"] / v["spread_T_min"]
                                                    for v in c39w["detail"].values())},
                      wt["a1"], ["C38", "C39", "C40"]),
        nc.ledger_row(RUN_IDS["a2"], "01", SECTIONS["a2"], {"name": "corrgcv-advantage-region"},
                      cfg_hash, None, "corrgcv-phase-diagram",
                      {"T": A2_T, "N": N_FEATURES, "log10_xi": [float(v) for v in A2_LOG10_XI],
                       "log10_lambda": [float(v) for v in A2_LOG10_LAMBDA], "seeds": SEEDS_PER_POINT,
                       "noise": "matched K'=K"}, 0,
                      {"advantage_fraction": c41["advantage_fraction"],
                       "min_log10_ratio": c41["min"], "max_log10_ratio": c41["max"]},
                      wt["a2"], ["C41"]),
        nc.ledger_row(RUN_IDS["a3"], "01", SECTIONS["a3"], {"name": "theorem-vi1-horizon-optimism"},
                      cfg_hash, None, "theorem-vi1-eq44",
                      {"T": A3_T, "N": N_FEATURES, "xi": A3_XI, "lambda": A3_LAMBDA,
                       "taus": A3_TAUS, "seeds": SEEDS_PER_POINT,
                       "test_draws_per_seed": A3_TEST_DRAWS}, 0,
                      {"eq44_max_excursion": c42b["max_excursion"],
                       "empirical_max_excursion": c42e["max_excursion"],
                       "empirical_spread": c42e["spread"]},
                      wt["a3"], ["C42"]),
        nc.ledger_row(RUN_IDS["d2"], "01", SECTIONS["d2"], {"name": "carmack-dip-vs-gccv-bracket"},
                      cfg_hash, None, "corrgcv-carmack-bracket",
                      {"regime": "weak", "xi": REGIMES["weak"]["xi"],
                       "lambda": REGIMES["weak"]["lambda"], "sigma_eps": REGIMES["weak"]["sigma_eps"],
                       "T_grid": list(D2_T_GRID), "k_families": list(FAMILIES),
                       "series": list(D2_SERIES), "seeds": D2_SEEDS}, 0,
                      {"n_families_sign_change": float(c53["n_families_sign_change"]),
                       "max_abs_log10_dip_over_zero": float(c53["max_abs_log10_ratio"]),
                       "min_carmack_mean": float(c53["min_carmack_mean"])},
                      wt["d2"], ["C53"]),
    ]


# ---------------------------------------------------------------------------
# rendering (plot-only; palette supplies ink/grid)
# ---------------------------------------------------------------------------


def _pal(palette, key) -> str:
    palette = palette or {}
    return palette.get(key, DEFAULT_PALETTE[key])


def _style():
    import matplotlib
    if "inline" not in matplotlib.get_backend().lower() and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")  # headless scripts must not fall into a half-installed GUI backend
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
    return plt


SERIES_LABEL = {
    "gcv1": r"$\mathrm{GCV}_1$", "gcv2": r"$\mathrm{GCV}_2$", "carmack": "Carmack",
    "corrgcv": "CorrGCV", "latent": r"$R_{\mathrm{out}}$ (latent)",
    "theory": r"theory (deterministic eq., Thm IV.2)",
}
FAMILY_LABEL = {"exponential": "exponential", "nearest_neighbor": "nearest-neighbour",
                "power_law": "power-law"}


def render_fig_a1(data, palette=None, show=True):
    """Inline redraw of figure a1. ``show=False`` is for run()'s PNG-only path: a
    figure shown from inside a compute cell would land in that cell's output and break the
    notebook's plot-only-cell contract."""
    plt = _style()
    ink, muted = _pal(palette, "ink"), _pal(palette, "muted")
    rows = data["a1"]["rows"]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), sharex=True)
    letters = iter("abcdef")
    for i, regime in enumerate(("weak", "strong")):
        for j, family in enumerate(FAMILIES):
            ax = axes[i, j]
            cell = [r for r in rows if r["regime"] == regime and r["k_family"] == family]
            for s in ("gcv1", "gcv2", "carmack", "corrgcv", "latent"):
                pts = sorted([r for r in cell if r["series"] == s], key=lambda r: r["T"])
                xs = [p["T"] for p in pts]
                ys = [p["mean"] for p in pts]
                se = [p["se"] for p in pts]
                col = SERIES_COLORS[_SERIES_KEY[s]]
                ax.plot(xs, ys, color=col, lw=1.4, label=SERIES_LABEL[s] if (i, j) == (0, 0) else None)
                ax.fill_between(xs, [max(y - 1.96 * e, 1e-12) for y, e in zip(ys, se)],
                                [y + 1.96 * e for y, e in zip(ys, se)], color=col, alpha=0.18,
                                linewidth=0)
            th = sorted([r for r in cell if r["series"] == "theory"], key=lambda r: r["T"])
            ax.plot([p["T"] for p in th], [p["mean"] for p in th],
                    color=SERIES_COLORS["theory"], lw=1.2, ls="--",
                    label=SERIES_LABEL["theory"] if (i, j) == (0, 0) else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Panel letters sit outside the data region (above the axes, left-aligned): drawn
            # inside the axes they overlapped the risk curves in an earlier render (per review notes).
            letter = next(letters)
            if i == 0:
                ax.set_title("(%s) %s" % (letter, FAMILY_LABEL[family]), fontsize=9, color=ink,
                             loc="left")
            else:
                ax.set_title("(%s)" % letter, fontsize=9, color=ink, loc="left")
            ax.grid(True, which="major", color=_pal(palette, "grid"), alpha=0.35, linewidth=0.5)
            if i == 1:
                ax.set_xlabel(r"$T$ (samples)", color=ink)
            if j == 0:
                ax.set_ylabel(r"$\hat R_{\mathrm{out}}$" + "\n(%s corr.)" % regime, color=ink)
            ax.tick_params(colors=ink)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(muted)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fig A1: estimator risk vs. $T$ across regimes and $K$ families", fontsize=10, color=ink)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    if show:
        plt.show()
    return fig


def _a2_matrix(rows):
    ux = sorted({r["log10_xi"] for r in rows})
    uy = sorted({r["log10_lambda"] for r in rows})
    Z = np.full((len(uy), len(ux)), np.nan)
    xi_idx = {v: i for i, v in enumerate(ux)}
    yi_idx = {v: i for i, v in enumerate(uy)}
    for r in rows:
        Z[yi_idx[r["log10_lambda"]], xi_idx[r["log10_xi"]]] = r["log10_ratio"]
    return np.array(ux), np.array(uy), Z


def render_fig_a2(data, palette=None, show=True):
    """Inline redraw of figure a2. ``show=False`` is for run()'s PNG-only path: a
    figure shown from inside a compute cell would land in that cell's output and break the
    notebook's plot-only-cell contract."""
    plt = _style()
    ink, muted = _pal(palette, "ink"), _pal(palette, "muted")
    ux, uy, Z = _a2_matrix(data["a2"]["rows"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vmax = float(np.nanmax(np.abs(Z)))
    im = ax.pcolormesh(ux, uy, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    cs = ax.contour(ux, uy, Z, levels=[math.log10(2.0)], colors=ink, linewidths=1.2)
    ax.clabel(cs, fmt={math.log10(2.0): r"$\log_{10}2$"}, fontsize=7)
    ax.text(0.04, 0.93, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top",
            color=ink)
    ax.set_xlabel(r"$\log_{10}\xi$", color=ink)
    ax.set_ylabel(r"$\log_{10}\lambda$", color=ink)
    ax.tick_params(colors=ink)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\mathrm{err}_{\mathrm{GCV}_1}/\mathrm{err}_{\mathrm{CorrGCV}})$",
                   fontsize=8, color=ink)
    cbar.ax.tick_params(colors=ink)
    ax.set_title("Fig A2: CorrGCV advantage region", fontsize=9, color=ink)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def render_fig_a3(data, palette=None, show=True):
    """Inline redraw of figure a3. ``show=False`` is for run()'s PNG-only path: a
    figure shown from inside a compute cell would land in that cell's output and break the
    notebook's plot-only-cell contract."""
    plt = _style()
    ink, muted = _pal(palette, "ink"), _pal(palette, "muted")
    rows = sorted(data["a3"]["rows"], key=lambda r: r["tau"])
    tau = [r["tau"] for r in rows]
    emp = [r["empirical_mean"] for r in rows]
    se = [r["empirical_se"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.fill_between(tau, [r["lower_band"] for r in rows], [r["upper_band"] for r in rows],
                    color=_pal(palette, "band"), alpha=0.18, linewidth=0,
                    label=r"admissible band $[1-\rho(\tau),\,1]$ (Thm VI.1)")
    ax.plot(tau, [r["eq44_factor"] for r in rows], color=SERIES_COLORS["theory"], lw=1.4, ls="--",
            label="Eq. (44) theory factor")
    ax.plot(tau, emp, color=SERIES_COLORS["GCV1"], lw=1.6,
            label=r"empirical $R^k_{\mathrm{out}}/R^{k=0}_{\mathrm{out}}$")
    ax.fill_between(tau, [e - 1.96 * s for e, s in zip(emp, se)], [e + 1.96 * s for e, s in zip(emp, se)],
                    color=SERIES_COLORS["GCV1"], alpha=0.20, linewidth=0)
    ax.text(0.02, 0.93, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top",
            color=ink)
    ax.set_xlabel(r"forecast lag $\tau$ (samples)", color=ink)
    ax.set_ylabel(r"$R^k_{\mathrm{out}}/R^{k=0}_{\mathrm{out}}$", color=ink)
    ax.tick_params(colors=ink)
    ax.grid(True, which="major", color=_pal(palette, "grid"), alpha=0.35, linewidth=0.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Fig A3: near-horizon test-point optimism (Thm VI.1)", fontsize=10, color=ink)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


D2_FAMILY_COLORS = {"exponential": "#2a78d6", "nearest_neighbor": "#eb6834",
                    "power_law": "#3f9b5c"}


def render_fig_d2(data, palette=None, show=True):
    """Inline redraw of figure d2: Carmack estimate (log y) over the GCCV bracket (linear y).

    Same fixed family colours as the reference figure (the figure description2.py); panel letters sit
    outside the data region above each panel.
    """
    plt = _style()
    ink, muted, grid = _pal(palette, "ink"), _pal(palette, "muted"), _pal(palette, "grid")
    rows = data["d2"]["rows"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                                         gridspec_kw={"height_ratios": (1.2, 1)})
    panels = (("carmack_estimate", ax_top, True, "(a)", "Carmack estimate"),
              ("bracket", ax_bot, False, "(b)", "bracket value"))
    for series, ax, logy, letter, ylabel in panels:
        for fam in FAMILIES:
            pts = sorted((r for r in rows if r["series"] == series and r["k_family"] == fam),
                         key=lambda r: r["T"])
            xs = [p["T"] for p in pts]
            ys = [p["mean"] for p in pts]
            ses = [p["se"] for p in pts]
            ax.plot(xs, ys, color=D2_FAMILY_COLORS[fam], lw=1.6,
                    label=FAMILY_LABEL[fam] if series == "carmack_estimate" else None)
            ax.fill_between(xs, [y - 1.96 * s for y, s in zip(ys, ses)],
                            [y + 1.96 * s for y, s in zip(ys, ses)],
                            color=D2_FAMILY_COLORS[fam], alpha=0.18, linewidth=0)
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color=ink, ls="--", lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(letter, fontsize=9, loc="left", color=ink)
        ax.grid(True, which="major", color=grid, alpha=0.35, linewidth=0.5)
        ax.tick_params(colors=ink)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
    ax_top.set_xscale("log")
    ax_top.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax_bot.set_xlabel(r"$T$ (samples)")
    fig.suptitle("Fig D2: Carmack estimate vs GCCV bracket, weak regime", fontsize=10, color=ink)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# notebook cells
# ---------------------------------------------------------------------------


def metric_watch(mid: str, target: float, achieved: float, direction: str) -> str:
    """Print one METRIC-WATCH line (same format as nb1's own helper) and return its status."""
    target, achieved = float(target), float(achieved)
    ok = (achieved <= target) if direction == "le" else (achieved >= target)
    status = "MET" if ok else "MISSED"
    print("METRIC-WATCH %s target=%.10g achieved=%.10g dir=%s status=%s"
          % (mid, target, achieved, direction, status))
    return status


_CLAIM_A1 = r"""### Fig A1 -- does each GCV-family estimator track $R_{\mathrm{out}}$ as $T$ grows?


**Claim.** For each of the three $K$ families and both Fig.-1 regimes, the four estimators
($\mathrm{GCV}_1$, $\mathrm{GCV}_2$, Carmack, CorrGCV) and the latent $R_{\mathrm{out}}$ are computed on an
11-point log-spaced $T$ grid spanning $[10, 1000]$, 10 seeds per point, with mean and SE.
**Claim.** Strong-correlation, exponential-$K$ panel: at every $T \ge 100$ CorrGCV's relative error
to the latent risk is below both $\mathrm{GCV}_1$'s and $\mathrm{GCV}_2$'s; weak-correlation panels: the
four estimators' spread at the largest $T$ is below the spread at the smallest $T$ in every family.
**Claim.** The deterministic-equivalent theory curve (Thm IV.2) overlays the empirical latent risk
with a median relative gap $\le 25\%$ over every (regime, family, $T$) point -- the plan's own falsifier
threshold ("the theory curve misses the empirical latent risk by > 25% median").

Falsifier look: in the strong-exponential panel the CorrGCV ribbon overlaps GCV$_1$'s at some $T \ge 100$;
in the weak panels the four estimator ribbons visibly fail to collapse onto one curve. Theory curves are
the paper's matched-noise setting ($K' = K$); the nearest-neighbour amplitude and power-law exponent are
**derived here** by lag-1 matching to the exponential family, and the NN cap $b \le 0.999$ is the
positive-definiteness limit of a unit-diagonal tridiagonal Toeplitz matrix."""

_CLAIM_A2 = r"""### Fig A2 -- where does CorrGCV's advantage over ordinary GCV live?


**Claim (C41, hypothesis).** Over a `13 x 13` log-spaced `(xi, lambda)` grid the region where CorrGCV
beats ordinary GCV$_1$ by at least `2x` is *measured* -- colour is
`log10(err_GCV1 / err_CorrGCV)` at fixed `T = 150`, `N = 100`, matched noise `K' = K`, 10 seeds per
cell. Execution settles the region's shape; this notebook only records it and never tunes the grid to a
target shape.

Falsifier look: a uniform colour field -- `log10_ratio` not varying meaningfully across the grid, which
would falsify the premise that CorrGCV's advantage is localised rather than everywhere-or-nowhere."""

_CLAIM_A3 = r"""### Fig A3 -- how optimistic is a near-horizon test point?


**Claim.** For forecast lags `tau = 1 ... 50` after the training window under exponential `K`
(`xi = 10^2`, `N = 100`, `T = 100`, `lambda = 10^-4`), the Eq. (44) factor
`1 - rho + kappa_tilde^2 alpha^T K (K + kappa_tilde)^-2 alpha` lies in `[1 - rho, 1]` at every lag, and
the empirical ratio `R^k_out / R^{k=0}_out` over 10 seeds (test point drawn as
`x = X^T alpha + sqrt(1 - rho) Sigma^{1/2} z`, `k = 0` point `Sigma^{1/2} z'`, risk
`((wbar - w_hat)^T x)^2 + sigma_eps^2`) is reported against that bound and against the Eq. (44) curve.

Falsifier look: the empirical ratio lies outside `[1 - rho, 1]` at any `tau`, or the curve is flat at 1
(no horizon optimism) -- either would falsify C42 / Theorem VI.1's applicability here. The training
noise here is i.i.d. (`K' = I_T`), not the paper's matched `K' = K`: the plan's sampler specifies it,
and the paper's Section IV.C already records that mismatch costs the CorrGCV guarantee, so the reported
theory-vs-empirical gap is part of the result, not a tuning knob."""

_CELL_A1 = r'''import corrgcv_study

DATA = corrgcv_study.run()
_plan_rows = corrgcv_study.ledger_rows(DATA)
try:
    _ledger_rows = list(_ledger_rows) + _plan_rows
except NameError:  # nb1 section 5 owns this list; tolerate a different integration point
    _ledger_rows = list(_plan_rows)
nbs_common.merge_ledger("01", _ledger_rows)

_rows = DATA["a1"]["rows"]
_c38 = corrgcv_study.check_c38(_rows)
_c39s = corrgcv_study.check_c39_strong(_rows)
_c39w = corrgcv_study.check_c39_weak(_rows)
_c40 = corrgcv_study.check_c40(_rows)

print("fig-a1: %d rows | %d T points %s | %d seeds/point | wall %.1fs"
      % (len(_rows), len(corrgcv_study.T_GRID), corrgcv_study.T_GRID, DATA["seeds"], DATA["wall_times"]["a1"]))
for _regime in ("weak", "strong"):
    print("  %-6s regime:" % _regime, " ".join(
        "%s>T[%g..%g]x%d" % (_fam, _p["min_T"], _p["max_T"], _p["n_T"])
        for (_r, _fam), _p in _c38["panels"].items() if _r == _regime))
print("  C39 strong/exponential worst err(CorrGCV)/max(err GCV1, err GCV2) over T>=100: %.4f"
      % _c39s["worst_ratio"])
for _fam, _d in _c39w["detail"].items():
    print("  C39 weak/%-16s spread T=%g: %.4g -> T=%g: %.4g"
          % (_fam, min(corrgcv_study.T_GRID), _d["spread_T_min"], max(corrgcv_study.T_GRID), _d["spread_T_max"]))
print("  C40 theory vs latent median relative gap over %d points: %.4f (per-panel medians %s)"
      % (_c40["n_points"], _c40["median_gap"],
         {("%s/%s" % k): round(v, 4) for k, v in _c40["per_panel_median"].items()}))

SC.append(nbs_common.self_check(30, _c38["ok"],
    note="C38: 3 families x 2 regimes x %d T points in [10,1000], min se over non-theory series %.3g > 0"
         % (len(corrgcv_study.T_GRID), _c38["min_se_overall"])))
SC.append(nbs_common.self_check(31, _c39s["ok"],
    note="C39: CorrGCV rel.err < GCV1 and GCV2 at all %d T>=100 points (worst ratio %.4f)"
         % (_c39s["n_T_ge_100"], _c39s["worst_ratio"])))
SC.append(nbs_common.self_check(32, _c39w["ok"],
    note="C39: weak-regime estimator spread shrinks from T=10 to T=1000 in all three K families"))
SC.append(nbs_common.self_check(33, _c40["ok"],
    note="C40: median |theory-latent|/latent %.4f <= 0.25 (plan A falsifier threshold)" % _c40["median_gap"]))
corrgcv_study.metric_watch("NB1-R3A-A1-THEORY-MEDIAN-GAP", 0.25, _c40["median_gap"], "le")
corrgcv_study.metric_watch("NB1-R3A-A1-STRONG-ERR-RATIO", 1.0, _c39s["worst_ratio"], "le")'''

_CELL_A2 = r'''_rows = DATA["a2"]["rows"]
_c41 = corrgcv_study.check_c41(_rows)
_files = corrgcv_study.figure_files_status()

print("fig-a2: %d rows on a %dx%d (log10 xi, log10 lambda) grid | wall %.1fs"
      % (len(_rows), _c41["n_xi"], _c41["n_lambda"], DATA["wall_times"]["a2"]))
print("  C41 [hypothesis, recorded not asserted]: %.1f%% of the grid has CorrGCV beating GCV1 by >= 2x"
      " (min log10_ratio %.3g, max %.3g)" % (100.0 * _c41["advantage_fraction"], _c41["min"], _c41["max"]))
print("  figure triples: %s" % {_f: {_k: _v for _k, _v in _d.items() if _k in ("csv", "meta", "png",
      "how_to_read_ok", "plot_type", "png_width")} for _f, _d in _files["detail"].items()})

SC.append(nbs_common.self_check(34, _c41["ok"],
    note="C41: %dx%d grid, all log10_ratio finite; advantage fraction %.3f recorded (shape not asserted)"
         % (_c41["n_xi"], _c41["n_lambda"], _c41["advantage_fraction"])))
SC.append(nbs_common.self_check(35, _files["ok"],
    note="fig-a1/a2/a3 CSV+meta+PNG written by run(), meta.how_to_read verbatim, PNG >= 1200 px"))
corrgcv_study.metric_watch("NB1-R3A-A2-ADVANTAGE-FRACTION", 0.0, _c41["advantage_fraction"], "ge")'''

_CELL_A3 = r'''_rows = DATA["a3"]["rows"]
_c42b = corrgcv_study.check_c42_band(_rows)
_c42e = corrgcv_study.check_c42_empirical(_rows)
_ledger = corrgcv_study.check_ledger()
_bands = [(r["tau"], r["lower_band"], r["eq44_factor"], r["upper_band"], r["empirical_mean"],
           r["empirical_se"]) for r in _rows]

print("fig-a3: %d lags tau=%s | wall %.1fs"
      % (len(_rows), [r["tau"] for r in _rows], DATA["wall_times"]["a3"]))
print("  tau / rho / Eq.(44) factor / [1-rho, 1] / empirical mean +- SE")
for _tau, _lo, _fac, _hi, _emp, _se in _bands[:4] + _bands[-3:]:
    print("   %3d  %.4f  %.4f  [%.4f, %.4f]  %.4f +- %.4f" % (_tau, 1.0 - _lo, _fac, _lo, _hi, _emp, _se))
print("  Eq.(44) band excursion %.3g (1e-9 slack); empirical max excursion %.4f vs 3*max(SE) %.4f;"
      " spread across tau %.4f" % (_c42b["max_excursion"], _c42e["max_excursion"], _c42e["tol"],
                                   _c42e["spread"]))
print("  ledger rows %s with claims %s (data_hash %s)"
      % (_ledger["n_rows"], _ledger["claims"], _ledger["data_hash"]))

SC.append(nbs_common.self_check(36, _c42b["ok"],
    note="C42: Eq.(44) factor in [1-rho, 1] at all %d lags (max excursion %.3g)" % (_c42b["n_tau"],
         _c42b["max_excursion"])))
SC.append(nbs_common.self_check(37, _c42e["ok"],
    note="C42: empirical ratio inside band +- 3*max(SE) and spread %.4f > 0.05 (not flat at 1)"
         % _c42e["spread"]))
SC.append(nbs_common.self_check(38, _ledger["ok"],
    note="C38-C42+C53 ledger rows merged (notebook 01), 64-hex data_hash, zero external spend"))
corrgcv_study.metric_watch("NB1-R3A-A3-EQ44-BAND-EXCESS", 1e-9, _c42b["max_excursion"], "le")
corrgcv_study.metric_watch("NB1-R3A-A3-EMP-SPREAD", 0.05, _c42e["spread"], "ge")'''

_CLAIM_D2 = r"""### Fig D2 -- is the Carmack dip the GCCV bracket's zero crossing? (C53, hypothesis)


**Claim (C53, hypothesis).** Over the shared 11-point log `T` grid spanning `[10, 1000]`, weak regime,
one line per K family and 10 seeds per point: the striking dip in the Carmack estimate sits at the `T`
where the paper's *unsquared* bracket `1 - (df2*dft2)/(df1*dft1)/(1 - dft2/dft1)` changes sign. The bracket's
K-side degrees of freedom are the design's known `K`; its feature-side `df1`/`df2` are re-estimated per
realization from the realized covariance diagonal via the exact trace identity `E[X^T X / T] = Sigma`
(unbiased at the renormalised `kappa`), so the recorded ribbon is genuine finite-sample estimation
error. The exact df-form bracket is deterministic in the solved `ident` and is exposed per realization
as `nbs_common.corrgcv_rep(...)["bracket"]`; its zero crossing is printed beside the estimated one.

Falsifier look: the Carmack dip sitting at a clearly different `T` than the bracket's zero crossing, or
the bracket never changing sign on the grid -- either would make the visual dip unaccounted for by the
formula. This is a *hypothesis*: the comparison is measured and printed, never asserted to a tolerance.

In this regime every family's off-diagonals are at `exp(-100)` or below, so the three known-`K`
spectra are the identity to float64 precision and the three lines nearly coincide (their visible
spread is the seed noise only) -- the same collapse fig-a1's weak row shows; the recorded rows are
still one per family, never a merged line."""

_CELL_D2 = r'''_d2 = corrgcv_study.check_d2(DATA)
_d2f = corrgcv_study.check_d2_files()
_s = DATA["d2"]["stats"]

print("fig-d2: %d rows | %d families x %d T points x %d series | %d seeds/point | wall %.1fs"
      % (len(DATA["d2"]["rows"]), _d2["n_families"], len(corrgcv_study.D2_T_GRID),
         len(corrgcv_study.D2_SERIES), corrgcv_study.D2_SEEDS, DATA["wall_times"]["d2"]))
for _fam, _f in _s["families"].items():
    print("  %-16s Carmack dip at T=%-5d (mean %.3g) | bracket zero at T=%.2f "
          "(exact df-form zero at T=%.2f) | log10 dip/zero %+.3f"
          % (_fam, _f["dip_T"], _f["dip_mean"], _f["zero_T"], _f["exact_zero_T"],
             _f["log10_ratio_dip_vs_zero"]))
print("  bracket means at T=%d -> T=%d: %s"
      % (corrgcv_study.D2_T_GRID[0], corrgcv_study.D2_T_GRID[-1], ", ".join(
          "%s %+.3f -> %+.3f" % (_fam, [r["mean"] for r in DATA["d2"]["rows"]
                                        if r["series"] == "bracket" and r["k_family"] == _fam][0],
                                 [r["mean"] for r in DATA["d2"]["rows"]
                                  if r["series"] == "bracket" and r["k_family"] == _fam][-1])
          for _fam in corrgcv_study.FAMILIES)))

SC.append(nbs_common.self_check(63, _d2["n_families_sign_change"] == _d2["n_families"] >= 3,
    note="C53: the bracket's recorded mean changes sign on the grid in every one of the %d K families "
         "(dip-vs-zero comparison well defined)" % _d2["n_families"]))
SC.append(nbs_common.self_check(64, _d2["ok"],
    note="C53: both series on the same >=8-point T grid in all families, se > 0 for both (seeds ran), "
         "Carmack mean strictly positive (log axis); worst |log10(dip/zero)| = %.3f"
         % _d2["max_abs_log10_ratio"]))
SC.append(nbs_common.self_check(65, _d2f["ok"],
    note="fig-d2 CSV+meta+PNG written, meta.how_to_read verbatim and plot_type=ribbon, PNG >= 1200 px, "
         "run_id resolves in the ledger, CSV differs from the reference figure"))
corrgcv_study.metric_watch("NB1-R4D-D2-BRACKET-SIGNCHANGE", 3.0, _d2["n_families_sign_change"], "ge")
corrgcv_study.metric_watch("NB1-R4D-D2-DIP-ZERO-LOG10", 0.5, _d2["max_abs_log10_ratio"], "le")'''


def section_cells() -> list:
    """Notebook cells for this study, in order: claim md / compute / how-to-read md / plot, per figure."""
    cells = []
    for fig_id, claim, compute in (("a1", _CLAIM_A1, _CELL_A1), ("a2", _CLAIM_A2, _CELL_A2),
                                   ("a3", _CLAIM_A3, _CELL_A3), ("d2", _CLAIM_D2, _CELL_D2)):
        tags = ["plan-a", "fig-%s" % fig_id]
        cells.append({"type": "md", "source": claim, "tags": list(tags)})
        cells.append({"type": "code", "source": compute, "tags": list(tags + ["self-check"])})
        cells.append({"type": "md", "source": "**How to read this chart:** " + how_to_read(fig_id),
                      "tags": list(tags)})
        cells.append({"type": "code",
                      "source": "corrgcv_study.render_fig_%s(DATA, PALETTE)" % fig_id,
                      "tags": list(tags + ["plot-only"])})
    return cells
