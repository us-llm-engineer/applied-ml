"""C7/C8: a constructed analytic DKW null and the alpha-trimming behaviour.

P3's object is the two-sided L1 variability statement
``||F_bar - G0||_1 <= alpha + |S| (gamma + delta_b + delta_c)`` with failure
probability ``eps <= 2 exp(-2 N delta_c^2) + 2 M exp(-2 N delta_b^2)``.
The null is built analytically -- the candidate CDF is *exactly* the mixture
``(1 - alpha) G0 + alpha H`` on a published grid -- so the test is a statement
about the sampling noise alone, not about two arbitrary empirical CDFs.
"""
from __future__ import annotations

import math

import numpy as np

from exec.common import rate_object, sha256_json

SUPPORT_LENGTH = 1.0
GRID_SIZE = 1001
BASE_SHAPE = 8.0
CONTAMINANT_LO = 0.8
CONTAMINANT_HI = 1.0

ALPHA = 0.4
GAMMA = 0.01
DELTA_B = 0.014
DELTA_C = 0.014
M_PIECES = 10
N_SAMPLES = 20000
DKW_SEEDS = tuple(range(600, 632))
MONTE_CARLO_TOLERANCE = 0.02

TRIM_SAMPLES = 2000
TRIM_ALPHA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TRIM_SEED = 20260921
CONTAMINANT_TAIL_LO = 0.95

CLEAN_SAMPLES = 4000
CLEAN_SEED = 101
CONTAMINATION_LEVELS = (0.05, 0.15, 0.30)
CONTAMINATION_SEEDS = (201, 202, 203)
BAND_LOW = 0.005
BAND_HIGH = 0.995
NEAR_ZERO_TOLERANCE = 0.02
ESTIMATE_MONOTONIC_TOLERANCE = 0.02

SPEC = {
    "support_length": SUPPORT_LENGTH,
    "base_cdf": "Beta(1, 8) on [0, 1]: G0(x) = 1 - (1 - x)^8",
    "contaminant_cdf": f"Uniform on [{CONTAMINANT_LO}, {CONTAMINANT_HI}]",
    "alpha": ALPHA,
    "gamma": GAMMA,
    "delta_b": DELTA_B,
    "delta_c": DELTA_C,
    "M": M_PIECES,
    "N": N_SAMPLES,
    "dkw_seeds": list(DKW_SEEDS),
    "trimming": {
        "samples": TRIM_SAMPLES,
        "alpha_grid": list(TRIM_ALPHA_GRID),
        "seed": TRIM_SEED,
        "contaminant_tail_lo": CONTAMINANT_TAIL_LO,
    },
    "contamination_estimation": {
        "clean_samples": CLEAN_SAMPLES,
        "levels": list(CONTAMINATION_LEVELS),
        "band": [BAND_LOW, BAND_HIGH],
    },
}
CONFIG = {
    "stage": "foundations-nb1",
    "family": "variability",
    "null_rule": "candidate_cdf = (1 - alpha) * base_cdf + alpha * contaminant_cdf, exact on the published grid",
    "discrepancy_rule": "trapezoid integral of |empirical_cdf - base_cdf| on the published grid",
    "violation_rule": "l1_discrepancy > eq32_rhs",
    "alpha_estimator": "outside-band fraction corrected by the base distribution's own outside mass",
}


def _grid() -> np.ndarray:
    return np.linspace(0.0, SUPPORT_LENGTH, GRID_SIZE)


def _base_cdf(x: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - x / SUPPORT_LENGTH) ** BASE_SHAPE


def _contaminant_cdf(x: np.ndarray) -> np.ndarray:
    scaled = (x - CONTAMINANT_LO) / (CONTAMINANT_HI - CONTAMINANT_LO)
    return np.clip(scaled, 0.0, 1.0)


def _mixture_cdf(x: np.ndarray, alpha: float, tail_lo: float = CONTAMINANT_LO) -> np.ndarray:
    return (1.0 - alpha) * _base_cdf(x) + alpha * _contaminant_cdf(x)


def _empirical_cdf(samples: np.ndarray, grid: np.ndarray, divisor: float | None = None) -> np.ndarray:
    ordered = np.sort(samples)
    counts = np.searchsorted(ordered, grid, side="right")
    return counts / float(len(samples) if divisor is None else divisor)


def _l1_discrepancy(samples: np.ndarray, grid: np.ndarray, divisor: float | None = None) -> float:
    empirical = _empirical_cdf(samples, grid, divisor=divisor)
    return float(np.trapezoid(np.abs(empirical - _base_cdf(grid)), grid))


def _sample_candidate(rng: np.random.Generator, size: int, alpha: float,
                      tail_lo: float = CONTAMINANT_LO) -> np.ndarray:
    """Exact mixture sampling: base via its inverse CDF, contaminant via its inverse CDF."""
    uniform = rng.random(size)
    from_contaminant = uniform < alpha
    if alpha <= 0.0:
        from_contaminant = np.zeros(size, dtype=bool)
    elif alpha >= 1.0:
        from_contaminant = np.ones(size, dtype=bool)
    base_uniform = np.where(from_contaminant, uniform, (uniform - alpha) / max(1.0 - alpha, 1e-300))
    contaminant_uniform = np.where(from_contaminant, uniform / max(alpha, 1e-300), uniform)
    base_draw = 1.0 - (1.0 - np.clip(base_uniform, 0.0, 1.0)) ** (1.0 / BASE_SHAPE)
    tail_draw = tail_lo + (CONTAMINANT_HI - tail_lo) * np.clip(contaminant_uniform, 0.0, 1.0)
    return np.where(from_contaminant, tail_draw, base_draw)


def dkw_experiment() -> dict:
    grid = _grid()
    base = _base_cdf(grid)
    contaminant = _contaminant_cdf(grid)
    candidate = (1.0 - ALPHA) * base + ALPHA * contaminant
    rhs = ALPHA + SUPPORT_LENGTH * (GAMMA + DELTA_B + DELTA_C)
    epsilon = 2.0 * math.exp(-2.0 * N_SAMPLES * DELTA_C ** 2) + 2.0 * M_PIECES * math.exp(
        -2.0 * N_SAMPLES * DELTA_B ** 2
    )
    trials = []
    for seed in DKW_SEEDS:
        rng = np.random.default_rng(seed)
        samples = _sample_candidate(rng, N_SAMPLES, ALPHA)
        discrepancy = _l1_discrepancy(samples, grid)
        trials.append(
            {
                "seed": int(seed),
                "l1_discrepancy": discrepancy,
                "violated_eq32": bool(discrepancy > rhs),
            }
        )
    frequency = float(np.mean([trial["violated_eq32"] for trial in trials]))
    return {
        "N": int(N_SAMPLES),
        "M": int(M_PIECES),
        "alpha": float(ALPHA),
        "gamma": float(GAMMA),
        "delta_b": float(DELTA_B),
        "delta_c": float(DELTA_C),
        "support_length": float(SUPPORT_LENGTH),
        "cdf_grid": grid.tolist(),
        "base_cdf": base.tolist(),
        "contaminant_cdf": contaminant.tolist(),
        "candidate_cdf": candidate.tolist(),
        "eq32_rhs": float(rhs),
        "eq33_epsilon_bound": float(epsilon),
        "trials": trials,
        "empirical_violation_frequency": frequency,
        "monte_carlo_tolerance": float(MONTE_CARLO_TOLERANCE),
        "mean_l1_discrepancy": float(np.mean([trial["l1_discrepancy"] for trial in trials])),
    }


def _trimmed_discrepancy(samples: np.ndarray, alpha: float, grid: np.ndarray, base_median: float) -> float:
    """L1 gap after discarding the alpha-fraction of samples furthest from the base median.

    The retained sample is a proper probability sample, so its empirical CDF is
    renormalised by the number of retained points; the discarded mass is not
    re-weighted or redistributed.
    """
    if alpha <= 0.0:
        return _l1_discrepancy(samples, grid)
    keep = int(round((1.0 - alpha) * len(samples)))
    order = np.argsort(np.abs(samples - base_median))
    retained = samples[order[:keep]]
    return _l1_discrepancy(retained, grid)


def _excess_discrepancy(samples: np.ndarray, alpha: float, grid: np.ndarray) -> float:
    """Deadband-excess form: integral of max(|F_emp - G0| - alpha, 0)."""
    empirical = _empirical_cdf(samples, grid)
    excess = np.maximum(np.abs(empirical - _base_cdf(grid)) - alpha, 0.0)
    return float(np.trapezoid(excess, grid))


def _band_alpha_estimate(samples: np.ndarray, grid: np.ndarray) -> float:
    """Outside-band fraction corrected by the base distribution's own outside mass."""
    low = float(SUPPORT_LENGTH * (1.0 - (1.0 - BAND_LOW) ** (1.0 / BASE_SHAPE)))
    high = float(SUPPORT_LENGTH * (1.0 - (1.0 - BAND_HIGH) ** (1.0 / BASE_SHAPE)))
    outside = float(np.mean((samples < low) | (samples > high)))
    base_outside_mass = (1.0 - BAND_HIGH) + BAND_LOW
    return float(np.clip((outside - base_outside_mass) / (1.0 - base_outside_mass), 0.0, 1.0))


def trimming_experiment() -> dict:
    grid = _grid()
    base_median = float(SUPPORT_LENGTH * (1.0 - 0.5 ** (1.0 / BASE_SHAPE)))
    rng = np.random.default_rng(TRIM_SEED)
    contaminated = _sample_candidate(rng, TRIM_SAMPLES, 0.30, tail_lo=CONTAMINANT_TAIL_LO)
    # The published curve uses the alpha-excess (deadband) definition: it is non-increasing
    # in alpha by construction.  The literal "discard the alpha-fraction furthest from the
    # base median without renormalising" variant was measured first and is NOT non-increasing
    # on this sample (discarding mass without renormalising leaves a top-end deficit that
    # grows with alpha), so the sanctioned fallback is published and both are reported.
    curve = [
        {"alpha": float(alpha), "discrepancy": _excess_discrepancy(contaminated, alpha, grid)}
        for alpha in TRIM_ALPHA_GRID
    ]
    trimmed_curve = [
        {"alpha": float(alpha), "discrepancy": _trimmed_discrepancy(contaminated, alpha, grid, base_median)}
        for alpha in TRIM_ALPHA_GRID
    ]
    clean_rng = np.random.default_rng(CLEAN_SEED)
    clean_sample = _sample_candidate(clean_rng, CLEAN_SAMPLES, 0.0, tail_lo=CONTAMINANT_TAIL_LO)
    cases = []
    for level, seed in zip(CONTAMINATION_LEVELS, CONTAMINATION_SEEDS):
        case_rng = np.random.default_rng(seed)
        sample = _sample_candidate(case_rng, CLEAN_SAMPLES, level, tail_lo=CONTAMINANT_TAIL_LO)
        cases.append(
            {
                "known_contamination": float(level),
                "estimated_alpha": _band_alpha_estimate(sample, grid),
                "seed": int(seed),
            }
        )
    return {
        "discrepancy_definition": (
            "alpha-excess (deadband) discrepancy: trapezoid integral of "
            "max(|empirical_cdf - base_cdf| - alpha, 0) on the published grid, computed on one "
            "fixed contaminated sample (2000 points, 30% tail contamination)"
        ),
        "discrepancy_by_alpha": curve,
        "trimmed_renormalised_by_alpha": trimmed_curve,
        "trimmed_definition": (
            "literal variant, reported for comparison: remove the alpha-fraction of samples "
            "furthest from the base median, renormalise the retained empirical CDF by the "
            "retained count"
        ),
        "uncontaminated_candidate": {
            "known_contamination": 0.0,
            "estimated_alpha": _band_alpha_estimate(clean_sample, grid),
            "near_zero_tolerance": float(NEAR_ZERO_TOLERANCE),
            "seed": int(CLEAN_SEED),
            "samples": int(CLEAN_SAMPLES),
        },
        "known_contamination_cases": cases,
        "estimate_monotonic_tolerance": float(ESTIMATE_MONOTONIC_TOLERANCE),
        "band": [float(BAND_LOW), float(BAND_HIGH)],
        "contaminant_tail": [float(CONTAMINANT_TAIL_LO), float(CONTAMINANT_HI)],
    }


def run() -> dict:
    return {"dkw": dkw_experiment(), "trimming": trimming_experiment()}


# ---------------------------------------------------------------------------
# Round 02 increment (C17/C18): literal trim vs alpha-excess, measured per seed
# ---------------------------------------------------------------------------

TRIM_R2_SEEDS = (301, 302, 303)
TRIM_R2_ALPHA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TRIM_R2_SAMPLES = 2000
TRIM_R2_CONTAMINATION = 0.30
TRIM_R2_TAIL_LO = 0.95


def trimming_sensitivity() -> dict:
    """Per-seed alpha-excess and literal renormalised-trim curves on a common alpha grid.

    Both are *measured* diagnostics of this project, not the paper's finite-sample
    theorem.  Alpha-excess is non-increasing by construction (a deadband can only
    shrink as it widens); the literal variant removes the alpha-fraction of samples
    furthest from the base median and renormalises the retained empirical CDF, so it
    is free to be non-monotone -- the published flag records what the seeds measured.
    """
    grid = _grid()
    base_median = float(SUPPORT_LENGTH * (1.0 - 0.5 ** (1.0 / BASE_SHAPE)))
    runs = []
    for seed in TRIM_R2_SEEDS:
        rng = np.random.default_rng(int(seed))
        sample = _sample_candidate(rng, TRIM_R2_SAMPLES, TRIM_R2_CONTAMINATION, tail_lo=TRIM_R2_TAIL_LO)
        alpha_excess = [_excess_discrepancy(sample, alpha, grid) for alpha in TRIM_R2_ALPHA_GRID]
        literal_trim = [_trimmed_discrepancy(sample, alpha, grid, base_median) for alpha in TRIM_R2_ALPHA_GRID]
        retained_at_max_alpha = int(round((1.0 - max(TRIM_R2_ALPHA_GRID)) * len(sample)))
        runs.append(
            {
                "seed": int(seed),
                "sample_count": int(len(sample)),
                "contamination": float(TRIM_R2_CONTAMINATION),
                "support_length": float(SUPPORT_LENGTH),
                "alpha_grid": [float(alpha) for alpha in TRIM_R2_ALPHA_GRID],
                "curves": {
                    "alpha_excess": [float(value) for value in alpha_excess],
                    "literal_trim": [float(value) for value in literal_trim],
                },
                "literal_retained_count": retained_at_max_alpha,
                "literal_retained_fraction": float(1.0 - max(TRIM_R2_ALPHA_GRID)),
            }
        )
    literal_nonmonotone = False
    for run in runs:
        values = np.asarray(run["curves"]["literal_trim"], dtype=float)
        literal_nonmonotone = literal_nonmonotone or bool(np.any(np.diff(values) > 1e-12))
    return {
        "runs": runs,
        "alpha_excess_is_project_diagnostic": True,
        "literal_is_project_diagnostic": True,
        "literal_nonmonotone_observed": bool(literal_nonmonotone),
        "alpha_excess_definition": (
            "deadband-excess discrepancy: trapezoid integral of max(|F_emp - G0| - alpha, 0) "
            "on the published grid; non-increasing in alpha by construction"
        ),
        "literal_definition": (
            "literal renormalised trimming: discard the alpha-fraction of samples furthest from "
            "the base median and renormalise the retained empirical CDF by the retained count; "
            "separately measured and free to be non-monotone"
        ),
        "contaminant_tail": [float(TRIM_R2_TAIL_LO), float(CONTAMINANT_HI)],
        "note": (
            "both curves are project diagnostics of this synthetic null, not the paper's "
            "finite-sample theorem, and neither is a production guarantee"
        ),
    }


# ---------------------------------------------------------------------------
# Round 03 increment (C24-C26): bounded-null coverage grid with Wilson intervals
# ---------------------------------------------------------------------------

R03_COVERAGE_N = (200, 800)
R03_COVERAGE_CONTAMINATION = (0.0, 0.15, 0.30)
R03_COVERAGE_SEEDS = (401, 402, 403, 404, 405)
R03_ALPHA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
R03_DEFINITIONS = ("alpha_excess", "literal_trim")


def round03_coverage_audit() -> dict:
    """The frozen N x contamination x seed bounded-null grid, with per-cell trimming curves."""
    grid = _grid()
    base_median = float(SUPPORT_LENGTH * (1.0 - 0.5 ** (1.0 / BASE_SHAPE)))
    configurations = {}
    coverage_rows = []
    trimming_rows = []
    for n_samples in R03_COVERAGE_N:
        for contamination in R03_COVERAGE_CONTAMINATION:
            for seed in R03_COVERAGE_SEEDS:
                run_id = f"r03-coverage-N{n_samples}-c{int(round(contamination * 100)):02d}-s{seed}"
                config = {
                    "N": int(n_samples),
                    "contamination": float(contamination),
                    "support_length": float(SUPPORT_LENGTH),
                    "M": int(M_PIECES),
                    "gamma": float(GAMMA),
                    "delta_b": float(DELTA_B),
                    "delta_c": float(DELTA_C),
                    "seed": int(seed),
                    "run_id": run_id,
                    "alpha_grid": [float(alpha) for alpha in R03_ALPHA_GRID],
                }
                config_hash = sha256_json(config)
                configurations[config_hash] = config
                rng = np.random.default_rng(int(seed))
                sample = _sample_candidate(rng, int(n_samples), float(contamination))
                discrepancy = _l1_discrepancy(sample, grid)
                eq32 = float(contamination) + SUPPORT_LENGTH * (GAMMA + DELTA_B + DELTA_C)
                eq33 = 2.0 * math.exp(-2.0 * n_samples * DELTA_C ** 2) + 2.0 * M_PIECES * math.exp(
                    -2.0 * n_samples * DELTA_B ** 2
                )
                observed = bool(discrepancy > eq32)
                coverage_rows.append(
                    {
                        "N": int(n_samples),
                        "contamination": float(contamination),
                        "seed": int(seed),
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "support_length": float(SUPPORT_LENGTH),
                        "bounded_support": True,
                        "analytical_null": "bounded-support-contamination-mixture",
                        "M": int(M_PIECES),
                        "gamma": float(GAMMA),
                        "delta_b": float(DELTA_B),
                        "delta_c": float(DELTA_C),
                        "alpha": float(contamination),
                        "eq32_rhs": float(eq32),
                        "eq33_epsilon_bound": float(eq33),
                        "l1_discrepancy": float(discrepancy),
                        "observed_violation": observed,
                        "accepted": bool(not observed),
                    }
                )
                for alpha in R03_ALPHA_GRID:
                    values = {
                        "alpha_excess": _excess_discrepancy(sample, alpha, grid),
                        "literal_trim": _trimmed_discrepancy(sample, alpha, grid, base_median),
                    }
                    retained = int(round((1.0 - alpha) * n_samples))
                    for definition in R03_DEFINITIONS:
                        trimming_rows.append(
                            {
                                "N": int(n_samples),
                                "contamination": float(contamination),
                                "seed": int(seed),
                                "run_id": run_id,
                                "config_hash": config_hash,
                                "support_length": float(SUPPORT_LENGTH),
                                "bounded_support": True,
                                "analytical_null": "bounded-support-contamination-mixture",
                                "M": int(M_PIECES),
                                "gamma": float(GAMMA),
                                "delta_b": float(DELTA_B),
                                "delta_c": float(DELTA_C),
                                "alpha": float(alpha),
                                "definition": definition,
                                "discrepancy": float(values[definition]),
                                "retained_count": retained,
                            }
                        )
    rates = []
    for n_samples in R03_COVERAGE_N:
        for contamination in R03_COVERAGE_CONTAMINATION:
            subset = [
                row
                for row in coverage_rows
                if row["N"] == n_samples and row["contamination"] == contamination
            ]
            failures = int(sum(bool(row["observed_violation"]) for row in subset))
            rates.append(
                {
                    "N": int(n_samples),
                    "contamination": float(contamination),
                    **rate_object(failures, len(subset)),
                }
            )
    return {
        "configurations": configurations,
        "coverage_rows": coverage_rows,
        "coverage_failure_rates": rates,
        "trimming_rows": trimming_rows,
        "alpha_grid": [float(alpha) for alpha in R03_ALPHA_GRID],
        "definitions": list(R03_DEFINITIONS),
        "bounded_support": True,
        "support_length": float(SUPPORT_LENGTH),
        "analytical_null": "bounded-support-contamination-mixture",
        "note": (
            "a bounded synthetic-null audit of the analytical Eq. 32/33 statement at small N; it "
            "is not the paper's CNN ensemble-size rule and not a production guarantee"
        ),
    }


# ---------------------------------------------------------------------------
# Round 04 increment (Query 10): the paper's own reported ensemble-size table
# (Table I, Section V-A2) published next to a classical-KS-style crossover
# computed on this project's OWN bounded null, so the "30 vs ~100" contrast
# is shown as a paper result alongside a project measurement, never merged.
# ---------------------------------------------------------------------------

R04_PAPER_TABLE_I = (
    {"m_ens": 3, "pct_alpha_hat_le_0p05": 77.20},
    {"m_ens": 5, "pct_alpha_hat_le_0p05": 89.20},
    {"m_ens": 10, "pct_alpha_hat_le_0p05": 96.80},
    {"m_ens": 20, "pct_alpha_hat_le_0p05": 99.40},
    {"m_ens": 30, "pct_alpha_hat_le_0p05": 100.00},
    {"m_ens": 70, "pct_alpha_hat_le_0p05": 100.00},
    {"m_ens": 100, "pct_alpha_hat_le_0p05": 100.00},
    {"m_ens": 150, "pct_alpha_hat_le_0p05": 100.00},
    {"m_ens": 200, "pct_alpha_hat_le_0p05": 100.00},
)
R04_N_GRID = (10, 30, 50, 100, 150, 200)
R04_KS_SEED = 20260922
R04_KS_TRIALS = 80
R04_KS_ALPHA_LEVEL = 0.05


def round04_ks_vs_alpha_sample_size(n_grid=R04_N_GRID, alpha_level: float = R04_KS_ALPHA_LEVEL) -> dict:
    """A classical two-sample KS rejection-rate crossover on this project's own bounded null.

    The paper (Table I, Section V-A2) reports that untrimmed comparisons need
    roughly M_ens=100 to reliably fall inside the L_infty band, while alpha-
    trimming reaches the same reliability from M_ens=30. That table is
    reproduced verbatim above as paper-reported context. What is computed
    here is a *different*, project-owned measurement: at what sample size N
    does a classical two-sample KS test between two draws from the SAME base
    distribution stop rejecting at the nominal alpha_level, on this file's
    own analytic null (see ``dkw_experiment``). The two curves are reported
    side by side, never merged into one number.
    """
    from scipy import stats as scipy_stats

    grid = _grid()
    rejection_rows = []
    for n in n_grid:
        rejected = 0
        for trial in range(R04_KS_TRIALS):
            rng = np.random.default_rng(R04_KS_SEED + 1000 * n + trial)
            sample_a = _sample_candidate(rng, int(n), 0.0)
            sample_b = _sample_candidate(rng, int(n), 0.0)
            result = scipy_stats.ks_2samp(sample_a, sample_b)
            rejected += int(result.pvalue < alpha_level)
        rejection_rows.append(
            {
                "N": int(n),
                "trials": int(R04_KS_TRIALS),
                "rejections": int(rejected),
                "rejection_rate": float(rejected) / float(R04_KS_TRIALS),
                "nominal_alpha": float(alpha_level),
            }
        )
    assert grid is not None  # the shared grid module-state is exercised for parity with dkw_experiment
    return {
        "paper_table_i": [dict(row) for row in R04_PAPER_TABLE_I],
        "paper_crossover_note": (
            "Table I (Section V-A2): untrimmed comparisons first reach 100% of ensembles at "
            "alpha_hat<=0.05 by M_ens=30; a classical (untrimmed) L_infty comparison needs M_ens>=100 "
            "to consistently fall inside the same band (Section III-A, Page 4)"
        ),
        "project_ks_rejection_by_n": rejection_rows,
        "project_measurement_note": (
            "a separate, project-owned measurement: classical two-sample KS rejection rate at "
            f"alpha={alpha_level} on this file's own same-distribution null, not the paper's CNN "
            "ensemble comparison -- the two curves are reported side by side, not merged"
        ),
        "alpha_level": float(alpha_level),
        "source_query": 10,
    }
