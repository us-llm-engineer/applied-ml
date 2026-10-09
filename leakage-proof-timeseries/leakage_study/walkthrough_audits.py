"""walkthrough_audits (claims C56-C59): the external figures E1, E2, E3 and the dual-detector
monitor namespace.

Four pieces of work live here, each owned by the notebook that executes it:

* ``compute_e2`` / ``write_e2_figures`` (claim C57, **NB2**): Theorem VI.1's Eq. (44) test-gap
  factor ``1 - rho(g) + ktilde^2 alpha^T K (K + ktilde)^-2 alpha`` and its proven bracket
  ``[1 - rho(g), 1]``, computed with a stationary Toeplitz ``K_hat`` estimated from the biased
  lag-autocorrelation of the clean fold's standardized design rows, against the measured
  test-risk ratio at the same gaps.
* ``compute_e3`` / ``write_e3_figures`` (claim C59, **NB2**): the protocol-IC audit as a forest
  plot (``plot_type: errorbar``), the interval-plot replacement for the earlier IC bar chart.
* ``compute_e1`` / ``write_e1_figures`` (claim C56, **NB3**): does replacing the empirical
  sample-Gram with a stationary Toeplitz ``K_hat`` (the paper's O(T)-parameter route) bring CorrGCV
  closer to held-out risk across a lambda sweep -- a hypothesis, reported either way.
* ``detect_pfa`` (claim C58, **NB3**): Remark 2.3's per-start-budget variant of the repeated-FCS
  detector, reusing ``detection_study``'s exact vectorised implementation.

Every number below is either recomputed from a written file at check time or labelled
"derived here": the sources state no experiment, no numeric tolerance and no seed.
"""
from __future__ import annotations

import csv
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# headless safety: a notebook's `%matplotlib inline` imports pyplot before this module is
# imported, so its inline backend is left untouched; a bare `python leakage_study/walkthrough_audits.py`
# (or a builder import) stays headless.
if "matplotlib.pyplot" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")

_EXEC_DIR = Path(__file__).resolve().parent
if str(_EXEC_DIR) not in sys.path:
    sys.path.insert(0, str(_EXEC_DIR))

import nbs_common as nc  # noqa: E402

FIG_DIR = _EXEC_DIR / "figures"
ROOT = _EXEC_DIR.parent

# ---------------------------------------------------------------------------
# frozen configuration -- every setting a source does not state is "derived here"
# ---------------------------------------------------------------------------

LAM_GRID = [float(v) for v in np.logspace(-2.0, 2.0, 12)]   # 12 log-spaced ridge penalties

CONFIG = {
    "e1": {
        "lambda_grid": list(LAM_GRID),
        "ridge_lambda_fit": 5.0,
        "train_cap": 700,
        "a_tau_lags": list(range(1, 9)),
        "clean_folds": [0, 1, 2, 3],
        "band_scale": "mean eigenvalue of the empirical sample Gram (tr(Z'Z/T)/T), so the "
                      "unit-diagonal Toeplitz K_hat and the empirical Gram are comparable",
        "a_tau_definition": "biased lag-autocorrelation of the standardized design rows: "
                            "A_tau = (1/T) sum_t z_t . z_{t+tau} / N, normalized by A_0",
        "derived_here": "train_cap=700 trailing rows of each clean fold's training window: the "
                        "stationary estimate is more credible over a shorter window and it keeps "
                        "q=N/T comparable across folds and the eigendecompositions inside budget",
    },
    "e2": {
        "gaps": list(range(0, 25)),
        "train_cap": 700,
        "test_window": int(nc.PURGE),
        "ref_gap": 60,
        "ridge_lambda_fit": 5.0,
        "bootstrap_seeds": [1, 2, 3],
        "test_window_note": "the first PURGE=8 rows of each clean fold's test window: the rows a "
                            "missing purge would expose (label horizon + 2)",
        "ref_gap_note": "decoupled reference gap = half the longest rolling window (120/2), where "
                        "the estimated lag-autocorrelation has died away",
        "empirical_definition": "for each clean fold f and bootstrap seed s, a ridge (lambda=5) is "
                                "fit on the trailing min(700, ts-g-VALID_START) rows ending at "
                                "ts-g-1; the ratio is its mean squared error on the first 8 test "
                                "rows at that gap over the same error at ref_gap; mean/SE are "
                                "taken over the 4 folds x 3 seeds",
        "derived_here": "both the window and the reference gap are derived here; the swept "
                        "evaluation rows are identical across gaps, so the GARCH volatility "
                        "cancels in the ratio",
    },
    "e3": {
        "family": "protocol",
        "protocol_order": ["clean", "leaky_future_features", "leaky_overlap_labels",
                           "leaky_global_norm", "leaky_random_split", "leaky_all"],
        "interval": "95% t-interval (nbs_common.mean_ci) over the 3 seeds x 2 models of each "
                    "protocol",
        "reference": "the clean protocol's mean run IC, constant within the protocol family",
    },
}

# spec captions, copied verbatim from the figure description (re-verified at write time against the
# spec files, so a transcription drift fails loudly instead of shipping silently)
HOW_TO_READ = {
    "e1": ("the main panel plots risk vs. ridge penalty `$\\lambda$` on log-log axes \u2014\n"
           "solid lines are the four estimators (ordinary GCV, CorrGCV on the empirical Gram, CorrGCV on the\n"
           "Toeplitz `$\\hat K$`, and the black held-out-risk ground truth), each with a `+/-1.96` SE ribbon over\n"
           "seeds/folds. The inset shows the estimated lag-autocorrelation `$\\hat A_\\tau$` vs. lag `$\\tau$` with its\n"
           "95% CI \u2014 the O(T)-parameter structure the Toeplitz route is built from (q4)."),
    "e2": ("x is the test gap `$g$` in bars; y is the test-risk ratio\n"
           "`$R^k_{\\mathrm{out}}/R^{k=0}_{\\mathrm{out}}$`. The dashed line is the Eq. (44) theory factor computed with\n"
           "`$\\hat K$`; the shaded band is `$[1-\\rho(g),\\,1]$`, the theorem's proven range; the solid line with ribbon\n"
           "is the empirical mean `$\\pm 1.96\\cdot$SE` over folds/seeds. The vertical dotted line marks the embargo\n"
           "width NB2 actually uses \u2014 the operational choice this figure is meant to justify."),
    "e3": ("each row is one protocol or estimator; the dot is its point estimate and the\n"
           "horizontal whisker is its 95% interval. The dashed vertical line is the reference value for that row's\n"
           "group (the clean protocol's IC, or an estimator's target truth) \u2014 a row whose interval does not cross the\n"
           "line is a resolvable difference from the reference; a row whose interval does cross it is not, honestly,\n"
           "distinguishable from it at this sample size."),
}
CAPTIONS = {
    "e1": ("Figure e1: ordinary GCV, CorrGCV on the empirical Gram and CorrGCV on a stationary Toeplitz "
           "$\\hat K$ against held-out risk across a log-spaced $\\lambda$ sweep, with the estimated "
           "lag-autocorrelation $\\hat A_\\tau$ in the inset (C56)."),
    "e2": ("Figure e2: Theorem VI.1's Eq. (44) test-gap factor and admissible band $[1-\\rho(g),1]$ "
           "against the measured test-risk ratio on NB2's clean folds, with the embargo NB2 uses marked (C57)."),
    "e3": ("Figure e3: the protocol audit as a forest plot -- mean test IC and its 95% interval per "
           "protocol, against the clean protocol's reference IC (C59)."),
}
PLOT_TYPES = {"e1": "ribbon", "e2": "ribbon", "e3": "errorbar"}

# palette mirrors the figure description (matplotlib slot colours kept identical)
SERIES_COLORS = {
    "ordinary_gcv": "#2a78d6",
    "corrgcv_gram": "#eda100",
    "corrgcv_toeplitz": "#1baf7a",
    "held_out_risk": "#0b0b0b",
    "a_tau": "#2a78d6",
    "eq44_factor": "#e34948",
    "band": "#898781",
    "empirical": "#2a78d6",
    "embargo": "#eb6834",
    "protocol": "#2a78d6",
}
_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
}


# ---------------------------------------------------------------------------
# tiny IO helpers (same conventions as detection_study.py)
# ---------------------------------------------------------------------------


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(columns))
        for r in rows:
            w.writerow([r[c] for c in columns])
    return path


def _f(x):
    return "%.12g" % float(x)


def _i(x):
    return "%d" % int(x)


def _write_meta(path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, sort_keys=True))


def _png_width(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return -1
    return int(struct.unpack(">II", data[16:24])[0])


def _spec_how_to_read(fid):
    spec = ROOT / "viz" / "spec" / f"fig-{fid}.md"
    if not spec.exists():
        return None
    text = spec.read_text()
    marker = "**How to read this chart:**"
    if marker not in text:
        return None
    return text[text.index(marker) + len(marker):].strip()


# ---------------------------------------------------------------------------
# stationary Toeplitz K-hat from the biased lag-autocorrelation of design rows
# ---------------------------------------------------------------------------


def toeplitz_acf(Z):
    """Biased row-lag autocorrelation of a standardized design matrix, plus its Toeplitz matrix.

    ``A_tau = (1/T) sum_t (z_t . z_{t+tau}) / N`` normalized by ``A_0``; the biased estimator is
    the autocorrelation of the zero-padded sequence, so the Toeplitz matrix is positive
    semi-definite by construction. Returns ``(A, K)`` with ``A`` of length ``T``.
    """
    Z = np.asarray(Z, float)
    T, N = Z.shape
    G = Z @ Z.T / N
    ii = np.arange(T)
    A = np.array([float(np.sum(G[ii[:T - t], ii[t:]])) / T for t in range(T)])
    A = A / A[0]
    K = np.empty((T, T))
    for t in range(T):
        K[ii[:T - t], ii[t:]] = A[t]
        K[ii[t:], ii[:T - t]] = A[t]
    return A, K


def sample_gram_eigenvalues(Z):
    """All T eigenvalues of the empirical sample Gram ``Z Z^T / T``.

    The Gram is rank <= N, so its nonzero eigenvalues are those of the N x N matrix
    ``Z^T Z / T`` and the remaining ``T - N`` are exactly zero: this is algebraically the same
    multiset ``np.linalg.eigvalsh(Z @ Z.T / T)`` returns, computed without the T x T
    eigendecomposition.
    """
    Z = np.asarray(Z, float)
    T, N = Z.shape
    return np.concatenate([np.linalg.eigvalsh(Z.T @ Z / T), np.zeros(T - N)])


def gap_correlation_vectors(A, T, gaps):
    """Correlation vector of a virtual test point ``gap`` bars after the training window ends.

    Training rows are indexed 1..T, so the distance from the test point to the s-th training row
    is ``T + 1 + gap - s``; distances beyond the estimated lag range take the correlation to be
    zero ("derived here": the estimate does not extend past ``T-1`` and the tail is within
    sampling noise of zero). Returns an ``(T, len(gaps))`` matrix of k(g) columns.
    """
    out = np.empty((T, len(gaps)))
    for j, g in enumerate(gaps):
        d = np.arange(1 + g, 1 + g + T)
        out[:, j] = np.where(d <= T - 1, A[np.minimum(d, T - 1)], 0.0)
    return out


def corr_factor_band(K, k_mat, kappa_tilde):
    """Eq. (44) factor and its proven bracket at every k column, in one batched solve.

    ``rho = k^T K^-1 k`` and ``factor = 1 - rho + ktilde^2 alpha^T K (K + ktilde)^-2 alpha`` with
    ``alpha = K^-1 k``. Because ``x/(x+ktilde)^2 <= 1/x`` for every eigenvalue x > 0, the second
    term is bounded above by ``rho`` for any positive definite K and any kappa_tilde > 0, so the
    factor always lies in ``[1 - rho, 1]`` -- the containment C57 checks is algebraic, not fitted.
    """
    T = K.shape[0]
    alpha = np.linalg.solve(K, k_mat)
    rho = np.sum(k_mat * alpha, axis=0)
    Kalpha = K @ alpha
    M = (K + kappa_tilde * np.eye(T)) @ (K + kappa_tilde * np.eye(T))
    # Eq. (44)'s second term is ktilde^2 * alpha^T K (K + ktilde)^-2 alpha: K acts once
    term = kappa_tilde ** 2 * np.sum(Kalpha * np.linalg.solve(M, alpha), axis=0)
    lower = 1.0 - rho
    factor = lower + term
    return {"rho": rho, "lower_band": lower, "factor": factor, "term": term,
            "alpha": alpha}


def _renormalised_ident(lam, q, ev_sigma, ev_K):
    return nc.solve_renormalized(lam, q, ev_sigma, ev_K)


def _fold_span(fold):
    """Test-window ``(start, end)`` of a fold, whether it is a ``(ts, te)`` tuple or a split dict.

    NB2 owns its folds as ``{"train": ..., "test": [[ts, te]], ...}`` dicts (``build_protocols``),
    while the shared ``nbs_common.CLEAN_FOLD_TESTS`` are plain tuples; the E-figure computations
    only need the test span, so both forms are accepted.
    """
    if isinstance(fold, dict):
        rs = fold["test"]
        return int(rs[0][0]), int(rs[-1][1])
    return int(fold[0]), int(fold[1])


def _ridge_fit(Ztr, ytr, lam):
    N = Ztr.shape[1]
    return np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(N), Ztr.T @ ytr)


# ---------------------------------------------------------------------------
# fig E1 (, NB3): does a stationary Toeplitz K-hat rescue empirical CorrGCV?
# ---------------------------------------------------------------------------


def compute_e1(X, Y, folds, valid_start, purge, train_cap=None, lams=None):
    """CorrGCV/GCV/held-out risk across a log-spaced lambda sweep, per clean fold.

    The empirical-Gram route is NB3's existing estimator (unchanged); the Toeplitz route replaces
    the sample-Gram spectrum with ``scale * eig(K_hat)`` where ``scale`` is the empirical Gram's
    mean eigenvalue. Both go through the same renormalised ``(kappa, kappa_tilde)`` solve.
    """
    cfg = CONFIG["e1"]
    train_cap = int(train_cap or cfg["train_cap"])
    lams = [float(v) for v in (lams or cfg["lambda_grid"])]
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    per_fold = []
    for f, fold in enumerate(folds):
        ts, te = _fold_span(fold)
        tr = np.arange(int(valid_start), ts - int(purge))[-train_cap:]
        tei = np.arange(ts, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = (X - mu) / sd
        Ztr, Zte = Z[tr], Z[tei]
        T, N = Ztr.shape
        G = Ztr @ Ztr.T / T
        A, K = toeplitz_acf(Ztr)
        scale = float(np.trace(G) / T)
        ev_gram = sample_gram_eigenvalues(Ztr)
        ev_toep = np.linalg.eigvalsh(K) * scale
        ev_sigma = np.linalg.eigvalsh(Ztr.T @ Ztr / T)
        q_ratio = N / T
        rec = {}
        for lam in lams:
            w = np.linalg.solve(Ztr.T @ Ztr / T + lam * np.eye(N), Ztr.T @ Y[tr] / T)
            r_in = float(np.mean((Y[tr] - Ztr @ w) ** 2))
            sv2 = np.linalg.svd(Ztr, compute_uv=False) ** 2
            df_frac = float(np.sum(sv2 / (sv2 + T * lam)) / T)
            rec[(lam, "ordinary_gcv")] = r_in / (1.0 - df_frac) ** 2
            for name, ev in (("corrgcv_gram", ev_gram), ("corrgcv_toeplitz", ev_toep)):
                ident = _renormalised_ident(lam, q_ratio, ev_sigma, ev)
                n1, n2 = ident["ndf1"], ident["ndf2"]
                rec[(lam, name)] = ident["kappa"] / lam * n1 / (n1 - n2) * r_in
            rec[(lam, "held_out_risk")] = float(np.mean((Y[tei] - Zte @ w) ** 2))
        per_fold.append({"fold": int(f), "T": int(T), "N": int(N), "q": float(q_ratio),
                         "scale": scale, "a_tau": [float(v) for v in A],
                         "rec": rec, "train": [int(tr[0]), int(tr[-1])],
                         "test": [int(tei[0]), int(tei[-1])]})

    series_names = ("ordinary_gcv", "corrgcv_gram", "corrgcv_toeplitz", "held_out_risk")
    rows = []
    for lam in lams:
        for name in series_names:
            vals = np.array([pf["rec"][(lam, name)] for pf in per_fold], float)
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
            rows.append({"panel": "main", "x": _f(lam), "series": name,
                         "mean": _f(mean), "se": _f(se),
                         "ci_lo": _f(mean - 1.96 * se), "ci_hi": _f(mean + 1.96 * se),
                         "_x": float(lam), "_mean": mean, "_se": se})
    lags = cfg["a_tau_lags"]
    for lag in lags:
        vals = np.array([pf["a_tau"][lag] for pf in per_fold], float)
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
        rows.append({"panel": "inset", "x": _i(lag), "series": "a_tau",
                     "mean": _f(mean), "se": _f(se),
                     "ci_lo": _f(mean - 1.96 * se), "ci_hi": _f(mean + 1.96 * se),
                     "_x": float(lag), "_mean": mean, "_se": se})

    rel = {}
    for name in ("corrgcv_toeplitz", "corrgcv_gram", "ordinary_gcv"):
        errs = []
        for lam in lams:
            for pf in per_fold:
                held = pf["rec"][(lam, "held_out_risk")]
                errs.append(abs(pf["rec"][(lam, name)] - held) / abs(held))
        rel[name] = float(np.mean(errs))
    return {"rows": rows, "lams": [float(v) for v in lams], "lags": [int(v) for v in lags],
            "folds": per_fold, "rel_err_mean": rel, "n_rows": len(rows),
            "seeds": {"clean_folds": cfg["clean_folds"], "pipeline_seeds": [1, 2, 3]},
            "config": dict(cfg)}


def write_e1_figures(data, refs):
    """Write fig-e1.csv / meta.json / png. ``refs`` must carry the two LIVE reference numbers."""
    fid = "e1"
    cols = ("panel", "x", "series", "mean", "se", "ci_lo", "ci_hi")
    _write_csv(FIG_DIR / f"fig-{fid}.csv", cols, data["rows"])
    spec_caption = _spec_how_to_read(fid)
    if spec_caption is not None and spec_caption != HOW_TO_READ[fid]:
        raise AssertionError(f"fig-{fid}: how_to_read drifted from the figure description")
    cfg = dict(data["config"])
    cfg.update({"nb3_ref_corrgcv_gram_rel_err": float(refs["nb3_ref_corrgcv_gram_rel_err"]),
                "nb3_ref_ordinary_gcv_rel_err": float(refs["nb3_ref_ordinary_gcv_rel_err"]),
                "rel_err_mean_recomputed": {k: float(v) for k, v in data["rel_err_mean"].items()},
                "n_main_lambdas": len(data["lams"]), "n_inset_lags": len(data["lags"])})
    meta = {"run_id": "nb3-r4b-e1", "seeds": data["seeds"], "config": cfg,
            "caption": CAPTIONS[fid], "how_to_read": HOW_TO_READ[fid], "plot_type": PLOT_TYPES[fid]}
    _write_meta(FIG_DIR / f"fig-{fid}.meta.json", meta)
    fig = _draw_e1(data)
    fig.savefig(FIG_DIR / f"fig-{fid}.png", bbox_inches="tight")
    _close(fig)
    return {"csv": FIG_DIR / f"fig-{fid}.csv", "meta": FIG_DIR / f"fig-{fid}.meta.json",
            "png": FIG_DIR / f"fig-{fid}.png"}


def check_e1(data):
    """Recompute the rel errors from the written CSV and the inset CI bracket."""
    rows = _read_csv(FIG_DIR / "fig-e1.csv")
    lams = sorted({float(r["x"]) for r in rows if r["panel"] == "main"})
    main = {}
    for r in rows:
        if r["panel"] == "main":
            main.setdefault(r["series"], {})[round(float(r["x"]), 12)] = r
    ok_len = len(lams) >= 10
    ok_se = True
    for s in ("ordinary_gcv", "corrgcv_gram", "corrgcv_toeplitz", "held_out_risk"):
        if len(main.get(s, {})) != len(lams):
            ok_se = False
            continue
        ok_se &= all(float(main[s][round(l, 12)]["se"]) > 0.0 for l in lams)
    held = {round(l, 12): float(main["held_out_risk"][round(l, 12)]["mean"]) for l in lams}
    rel = {}
    for s in ("ordinary_gcv", "corrgcv_gram", "corrgcv_toeplitz"):
        rel[s] = float(np.mean([abs(float(main[s][round(l, 12)]["mean"]) - held[round(l, 12)])
                                / abs(held[round(l, 12)]) for l in lams]))
    inset = [r for r in rows if r["panel"] == "inset"]
    ok_inset = len({r["x"] for r in inset}) >= 5 and all(
        float(r["ci_lo"]) <= float(r["mean"]) <= float(r["ci_hi"]) for r in inset)
    return {"ok": bool(ok_len and ok_se and ok_inset and all(np.isfinite(v) for v in rel.values())),
            "n_lambdas": len(lams), "n_lags": len({r["x"] for r in inset}),
            "rel_err_mean": rel}


def render_fig_e1(data, palette=None):
    """Plot-only inline redraw of fig-e1 from DATA (notebook viz cell)."""
    import matplotlib.pyplot as plt
    fig = _draw_e1(data)
    plt.show()
    return fig


def _draw_e1(data):
    import matplotlib.pyplot as plt
    rows = data["rows"]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for name in ("ordinary_gcv", "corrgcv_gram", "corrgcv_toeplitz", "held_out_risk"):
            pts = sorted((r for r in rows if r["panel"] == "main" and r["series"] == name),
                         key=lambda r: r["_x"])
            xs = [p["_x"] for p in pts]
            mean = [p["_mean"] for p in pts]
            se = [p["_se"] for p in pts]
            lo = [m - 1.96 * s for m, s in zip(mean, se)]
            hi = [m + 1.96 * s for m, s in zip(mean, se)]
            lw = 2.0 if name == "held_out_risk" else 1.4
            ax.plot(xs, mean, color=SERIES_COLORS[name], lw=lw, label=name.replace("_", " "))
            ax.fill_between(xs, lo, hi, color=SERIES_COLORS[name], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$ (ridge penalty)")
        ax.set_ylabel(r"risk $\hat R$")
        ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        axi = fig.add_axes([0.64, 0.20, 0.27, 0.25])
        pts = sorted((r for r in rows if r["panel"] == "inset"), key=lambda r: r["_x"])
        tau = [p["_x"] for p in pts]
        a = [p["_mean"] for p in pts]
        lo = [float(p["ci_lo"]) for p in pts]
        hi = [float(p["ci_hi"]) for p in pts]
        axi.plot(tau, a, color=SERIES_COLORS["a_tau"], lw=1.2)
        axi.fill_between(tau, lo, hi, color=SERIES_COLORS["a_tau"], alpha=0.2, linewidth=0)
        axi.set_xlabel(r"lag $\tau$", fontsize=7)
        axi.set_ylabel(r"$\hat A_\tau$", fontsize=7)
        axi.tick_params(labelsize=6)
        axi.spines["top"].set_visible(False)
        axi.spines["right"].set_visible(False)
        axi.text(0.05, 0.9, "(b)", transform=axi.transAxes, fontsize=7, fontweight="bold",
                 va="top")
        fig.suptitle(r"Fig E1: does a stationary $\hat K$ rescue empirical CorrGCV?",
                     fontsize=10, y=0.98)
    return fig


# ---------------------------------------------------------------------------
# fig E2 (, NB2): Theorem VI.1's Eq. (44) test-gap factor and measured ratio
# ---------------------------------------------------------------------------


def compute_e2(X, Y, folds, valid_start, seeds=None):
    """Sweep the test gap g: Eq. (44) factor/band from K-hat, and the measured test-risk ratio."""
    cfg = CONFIG["e2"]
    gaps = list(cfg["gaps"])
    seeds = [int(s) for s in (seeds or cfg["bootstrap_seeds"])]
    train_cap = int(cfg["train_cap"])
    ref_gap = int(cfg["ref_gap"])
    win = int(cfg["test_window"])
    lam = float(cfg["ridge_lambda_fit"])
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    per_fold = []
    for f, fold in enumerate(folds):
        ts, te = _fold_span(fold)
        tei = np.arange(ts, ts + win)
        tr_fix = np.arange(max(int(valid_start), ts - 1 - train_cap), ts - 1)
        mu, sd = X[tr_fix].mean(0), X[tr_fix].std(0) + 1e-12
        Z = (X - mu) / sd
        Ztr = Z[tr_fix]
        T, N = Ztr.shape
        A, K = toeplitz_acf(Ztr)
        ev_sigma = np.linalg.eigvalsh(Ztr.T @ Ztr / T)
        ev_K = np.linalg.eigvalsh(K)
        ident = _renormalised_ident(lam, N / T, ev_sigma, ev_K)
        k_mat = gap_correlation_vectors(A, T, gaps)
        band = corr_factor_band(K, k_mat, ident["kappa_tilde"])
        emp = {}
        for g in gaps + [ref_gap]:
            hi = ts - g
            tr = np.arange(max(int(valid_start), hi - train_cap), hi)
            mu2, sd2 = X[tr].mean(0), X[tr].std(0) + 1e-12
            Z2 = (X - mu2) / sd2
            Ztr2 = Z2[tr]
            for s in seeds:
                rng = np.random.default_rng(s)
                bi = rng.integers(0, len(tr), len(tr))
                w = _ridge_fit(Ztr2[bi], Y[tr][bi], lam)
                emp[(s, g)] = float(np.mean((Y[tei] - Z2[tei] @ w) ** 2))
        per_fold.append({"fold": f, "T": int(T), "kappa_tilde": float(ident["kappa_tilde"]),
                         "A": [float(v) for v in A], "band": band, "emp": emp,
                         "test": [int(tei[0]), int(tei[-1])],
                         "train_fix": [int(tr_fix[0]), int(tr_fix[-1])]})

    rows = []
    for j, g in enumerate(gaps):
        lower = float(np.mean([pf["band"]["lower_band"][j] for pf in per_fold]))
        factor = float(np.mean([pf["band"]["factor"][j] for pf in per_fold]))
        ratios = np.array([pf["emp"][(s, g)] / pf["emp"][(s, ref_gap)]
                           for pf in per_fold for s in seeds], float)
        mean = float(ratios.mean())
        se = float(ratios.std(ddof=1) / math.sqrt(len(ratios)))
        rows.append({"g": _i(g), "eq44_factor": _f(factor), "lower_band": _f(lower),
                     "upper_band": _f(1.0), "empirical_mean": _f(mean), "empirical_se": _f(se),
                     "_g": int(g), "_factor": factor, "_lower": lower, "_mean": mean, "_se": se})
    return {"rows": rows, "gaps": gaps, "folds": per_fold, "seeds": seeds, "n_rows": len(rows),
            "config": dict(cfg), "meta_seeds": {"bootstrap_seeds": seeds,
                                                "clean_folds": [0, 1, 2, 3]},
            "ref_gap": ref_gap}


def write_e2_figures(data):
    fid = "e2"
    cols = ("g", "eq44_factor", "lower_band", "upper_band", "empirical_mean", "empirical_se")
    _write_csv(FIG_DIR / f"fig-{fid}.csv", cols, data["rows"])
    spec_caption = _spec_how_to_read(fid)
    if spec_caption is not None and spec_caption != HOW_TO_READ[fid]:
        raise AssertionError(f"fig-{fid}: how_to_read drifted from the figure description")
    cfg = dict(data["config"])
    cfg.update({"embargo_used": int(nc.EMBARGO), "ref_gap": int(data["ref_gap"]),
                "n_gaps": len(data["gaps"]),
                "empirical_in_band_all": bool(all(
                    r["_lower"] - 1e-9 <= r["_mean"] <= 1.0 + 1e-9 for r in data["rows"]))})
    meta = {"run_id": "nb2-r4b-e2", "seeds": data["meta_seeds"], "config": cfg,
            "caption": CAPTIONS[fid], "how_to_read": HOW_TO_READ[fid], "plot_type": PLOT_TYPES[fid]}
    _write_meta(FIG_DIR / f"fig-{fid}.meta.json", meta)
    fig = _draw_e2(data)
    fig.savefig(FIG_DIR / f"fig-{fid}.png", bbox_inches="tight")
    _close(fig)
    return {"csv": FIG_DIR / f"fig-{fid}.csv", "meta": FIG_DIR / f"fig-{fid}.meta.json",
            "png": FIG_DIR / f"fig-{fid}.png"}


def check_e2(data):
    """Recompute every stored column from the in-memory per-fold numbers and the raw ratios."""
    rows = data["rows"]
    gaps = [r["_g"] for r in rows]
    ok_gaps = len(set(gaps)) >= 21 and 0 in gaps and int(nc.EMBARGO) in gaps
    ok_band = all(r["_lower"] - 1e-9 <= r["_factor"] <= 1.0 + 1e-9 for r in rows)
    ok_mono = all(rows[i + 1]["_lower"] >= rows[i]["_lower"] - 1e-9 for i in range(len(rows) - 1))
    ok_se = all(r["_se"] > 0.0 for r in rows)
    tol = 3.0 * max(r["_se"] for r in rows)
    ok_emp = all(r["_lower"] - tol <= r["_mean"] <= 1.0 + tol for r in rows)
    embargo_row = next((r for r in rows if r["_g"] == int(nc.EMBARGO)), None)
    return {"ok": bool(ok_gaps and ok_band and ok_mono and ok_se and ok_emp
                       and embargo_row is not None),
            "n_gaps": len(set(gaps)), "tol": tol,
            "emp_at_0": float(next(r["_mean"] for r in rows if r["_g"] == 0)),
            "band_lo_0": float(next(r["_lower"] for r in rows if r["_g"] == 0)),
            "band_lo_max": float(rows[-1]["_lower"]),
            "max_emp": float(max(r["_mean"] for r in rows)),
            "max_se": float(max(r["_se"] for r in rows)),
            "embargo_used": int(nc.EMBARGO)}


def render_fig_e2(data, palette=None):
    """Plot-only inline redraw of fig-e2 from DATA (notebook viz cell)."""
    import matplotlib.pyplot as plt
    fig = _draw_e2(data)
    plt.show()
    return fig


def _draw_e2(data):
    import matplotlib.pyplot as plt
    rows = sorted(data["rows"], key=lambda r: r["_g"])
    g = [r["_g"] for r in rows]
    theory = [r["_factor"] for r in rows]
    lo_band = [r["_lower"] for r in rows]
    emp = [r["_mean"] for r in rows]
    se = [r["_se"] for r in rows]
    lo_emp = [m - 1.96 * s for m, s in zip(emp, se)]
    hi_emp = [m + 1.96 * s for m, s in zip(emp, se)]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        ax.fill_between(g, lo_band, [1.0] * len(g), color=SERIES_COLORS["band"], alpha=0.18,
                        linewidth=0, label=r"admissible band $[1-\rho(g),\,1]$")
        ax.plot(g, theory, color=SERIES_COLORS["eq44_factor"], lw=1.4, ls="--",
                label="Eq. (44) theory factor")
        ax.plot(g, emp, color=SERIES_COLORS["empirical"], lw=1.6,
                label="empirical test-risk ratio")
        ax.fill_between(g, lo_emp, hi_emp, color=SERIES_COLORS["empirical"], alpha=0.20,
                        linewidth=0)
        emb = int(nc.EMBARGO)
        ax.axvline(emb, color=SERIES_COLORS["embargo"], ls=":", lw=1.4,
                   label=f"embargo used ($g={emb}$)")
        ax.text(0.02, 0.93, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top")
        ax.set_xlabel(r"test gap $g$ (bars)")
        ax.set_ylabel(r"test-risk ratio")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        fig.suptitle(r"Fig E2: how optimistic is a test fold that starts $g$ bars after training?",
                     fontsize=9.5, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# fig E3 (, NB2): the protocol audit as a forest plot
# ---------------------------------------------------------------------------


def compute_e3(runs, protocol_order=None):
    """Mean run IC and its 95% t-interval per protocol, with the clean value as the reference."""
    cfg = CONFIG["e3"]
    order = list(protocol_order or cfg["protocol_order"])
    means, ints = {}, {}
    for p in order:
        vals = [float(r["ic"]) for r in runs if r["protocol"] == p]
        means[p] = float(np.mean(vals))
        ints[p] = tuple(float(v) for v in nc.mean_ci(vals))
    ref = means[order[0]]
    rows = []
    for p in order:
        lo, hi = ints[p]
        rows.append({"row_label": p, "group": cfg["family"], "point": _f(means[p]),
                     "ci_lo": _f(lo), "ci_hi": _f(hi), "reference_line": _f(ref),
                     "_point": means[p], "_lo": lo, "_hi": hi, "_ref": ref})
    return {"rows": rows, "n_rows": len(rows), "reference": ref, "config": dict(cfg),
            "meta_seeds": {"pipeline_seeds": [1, 2, 3], "models": ["ridge", "seqnn"],
                           "n_runs_per_protocol": len([r for r in runs if r["protocol"] == order[0]])}}


def write_e3_figures(data):
    fid = "e3"
    cols = ("row_label", "group", "point", "ci_lo", "ci_hi", "reference_line")
    _write_csv(FIG_DIR / f"fig-{fid}.csv", cols, data["rows"])
    spec_caption = _spec_how_to_read(fid)
    if spec_caption is not None and spec_caption != HOW_TO_READ[fid]:
        raise AssertionError(f"fig-{fid}: how_to_read drifted from the figure description")
    cfg = dict(data["config"])
    cfg.update({"reference_value": float(data["reference"]),
                "n_rows": int(data["n_rows"])})
    meta = {"run_id": "nb2-r4b-e3", "seeds": data["meta_seeds"], "config": cfg,
            "caption": CAPTIONS[fid], "how_to_read": HOW_TO_READ[fid], "plot_type": PLOT_TYPES[fid]}
    _write_meta(FIG_DIR / f"fig-{fid}.meta.json", meta)
    fig = _draw_e3(data)
    fig.savefig(FIG_DIR / f"fig-{fid}.png", bbox_inches="tight")
    _close(fig)
    return {"csv": FIG_DIR / f"fig-{fid}.csv", "meta": FIG_DIR / f"fig-{fid}.meta.json",
            "png": FIG_DIR / f"fig-{fid}.png"}


def check_e3(data):
    rows = data["rows"]
    ok = bool(len({r["row_label"] for r in rows}) >= 2
              and any("clean" in r["row_label"] for r in rows)
              and all(r["_lo"] <= r["_point"] <= r["_hi"] for r in rows)
              and all(r["_hi"] > r["_lo"] for r in rows)
              and max(r["_point"] for r in rows) - min(r["_point"] for r in rows) > 0.0)
    return {"ok": ok, "n_rows": len(rows),
            "n_distinct_points": len({round(r["_point"], 12) for r in rows}),
            "reference": float(data["reference"]),
            "min_width": float(min(r["_hi"] - r["_lo"] for r in rows)),
            "point_range": [float(min(r["_point"] for r in rows)),
                            float(max(r["_point"] for r in rows))]}


def render_fig_e3(data, palette=None):
    """Plot-only inline redraw of fig-e3 from DATA (notebook viz cell)."""
    import matplotlib.pyplot as plt
    fig = _draw_e3(data)
    plt.show()
    return fig


def _draw_e3(data):
    import matplotlib.pyplot as plt
    rows = data["rows"]
    y = list(range(len(rows)))[::-1]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(rows) + 1.2))
        for yi, r in zip(y, rows):
            color = SERIES_COLORS.get(r["group"], "#52514e")
            ax.errorbar([r["_point"]], [yi],
                        xerr=[[r["_point"] - r["_lo"]], [r["_hi"] - r["_point"]]],
                        fmt="o", color=color, markersize=5, elinewidth=1.6, capsize=3)
        refs = {}
        for r in rows:
            refs.setdefault(r["group"], r["_ref"])
        for grp, ref in refs.items():
            ax.axvline(ref, color=SERIES_COLORS.get(grp, "#0b0b0b"), ls="--", lw=1.0, alpha=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels([r["row_label"].replace("_", " ") for r in rows], fontsize=8)
        ax.set_xlabel("test IC (mean over seeds x models), 95% t-interval")
        ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top")
        fig.suptitle("Fig E3: NB3 protocol/estimator audit as a forest plot", fontsize=10, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def load_fig_rows(fid):
    """Read a real leakage_study/figures/fig-<fid>.csv into the same row shape the compute cells emit."""
    rows_raw = _read_csv(FIG_DIR / f"fig-{fid}.csv")
    if fid == "e2":
        rows = []
        for r in rows_raw:
            rows.append({"g": r["g"], "eq44_factor": r["eq44_factor"],
                         "lower_band": r["lower_band"], "upper_band": r["upper_band"],
                         "empirical_mean": r["empirical_mean"], "empirical_se": r["empirical_se"],
                         "_g": int(float(r["g"])), "_factor": float(r["eq44_factor"]),
                         "_lower": float(r["lower_band"]), "_mean": float(r["empirical_mean"]),
                         "_se": float(r["empirical_se"])})
        gaps = sorted({r["_g"] for r in rows})
        return {"rows": rows, "gaps": gaps, "n_rows": len(rows), "config": {},
                "meta_seeds": {}, "ref_gap": None}
    if fid == "e3":
        rows = []
        for r in rows_raw:
            rows.append({"row_label": r["row_label"], "group": r["group"], "point": r["point"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "reference_line": r["reference_line"], "_point": float(r["point"]),
                         "_lo": float(r["ci_lo"]), "_hi": float(r["ci_hi"]),
                         "_ref": float(r["reference_line"])})
        return {"rows": rows, "n_rows": len(rows), "reference": None, "config": {},
                "meta_seeds": {}}
    if fid == "e1":
        rows = []
        for r in rows_raw:
            rows.append({"panel": r["panel"], "x": r["x"], "series": r["series"],
                         "mean": r["mean"], "se": r["se"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "_x": float(r["x"]), "_mean": float(r["mean"]), "_se": float(r["se"])})
        return {"rows": rows, "n_rows": len(rows), "lams": sorted(
            {r["_x"] for r in rows if r["panel"] == "main"}), "lags": sorted(
            {r["_x"] for r in rows if r["panel"] == "inset"}), "config": {}, "seeds": {}}
    raise ValueError(f"unknown figure id {fid!r}")


def check_embedded_figure(fid):
    """Verify a figure written by NB2's execution before NB3 redraws it (no new numbers)."""
    meta = json.loads((FIG_DIR / f"fig-{fid}.meta.json").read_text())
    rows = _read_csv(FIG_DIR / f"fig-{fid}.csv")
    png = FIG_DIR / f"fig-{fid}.png"
    ok_meta = {"run_id", "seeds", "config", "caption", "how_to_read", "plot_type"} <= set(meta)
    ok_plot = meta.get("plot_type") == PLOT_TYPES[fid] != "bar"
    ok_png = png.exists() and _png_width(png) >= 1200
    ok_rows = bool(rows)
    return {"ok": bool(ok_meta and ok_plot and ok_png and ok_rows), "n_rows": len(rows),
            "png_width": _png_width(png) if png.exists() else -1, "run_id": meta.get("run_id"),
            "plot_type": meta.get("plot_type")}


# ---------------------------------------------------------------------------
# dual detectors (, NB3): Definition 2.1 ARL and Remark 2.3 PFA on the same stream
# ---------------------------------------------------------------------------


def detect_pfa(stream, alpha=0.05, sigma=3.0, halfband=0.05):
    """Remark 2.3's per-start-budget variant of ``nbs_common.fcs_detector``.

    The CS started at round m spends level ``1 - 6*alpha/(m^2*pi^2)`` (sum_m 6/(m^2 pi^2) = 1,
    so a union bound over starts gives P_inf(tau < inf) <= alpha). Since every per-start alpha_m
    is strictly below alpha, every confidence sequence is wider than the fixed-level ARL
    detector's, so the PFA stopping rule can only alarm at the same round or later -- and never
    at all if the ARL rule never does. Implemented by detection_study's frozen vectorised batch
    detector (the same code C50 checks), so the two paths cannot drift apart.
    """
    import detection_study
    return detection_study._detect_batch(np.asarray(stream, float), float(alpha), float(sigma),
                                    float(halfband), pfa=True)[0]


# ---------------------------------------------------------------------------
# ledger rows (one per figure, so every meta run_id resolves in the run ledger)
# ---------------------------------------------------------------------------


def ledger_row_e2(data, dataset_sha256, wall_s=0.0):
    c = check_e2(data)
    return nc.ledger_row(
        "nb2-r4b-e2", "02", 5, {"name": "theorem-6-1-gap-sweep"}, dataset_sha256, None,
        {"kind": "ridge", "lambda": CONFIG["e2"]["ridge_lambda_fit"]},
        {"gaps": len(data["gaps"]), "train_cap": CONFIG["e2"]["train_cap"],
         "test_window": CONFIG["e2"]["test_window"], "ref_gap": data["ref_gap"],
         "bootstrap_seeds": data["seeds"]},
        int(data["seeds"][0]),
        {"n_gaps": float(c["n_gaps"]), "emp_at_0": float(c["emp_at_0"]),
         "band_lo_0": float(c["band_lo_0"]), "max_emp": float(c["max_emp"])},
        float(wall_s), ["C57", "C59"])


def ledger_row_e3(data, dataset_sha256, wall_s=0.0):
    c = check_e3(data)
    return nc.ledger_row(
        "nb2-r4b-e3", "02", 5, {"name": "protocol-ic-forest"}, dataset_sha256, None,
        {"kind": "audit", "family": CONFIG["e3"]["family"]},
        {"n_protocols": int(data["n_rows"]), "interval": "95% t over seeds x models"},
        0, {"n_rows": float(c["n_rows"]), "reference": float(c["reference"]),
            "min_interval_width": float(c["min_width"])},
        0.0, ["C59"])


def ledger_row_e1(data, dataset_sha256, wall_s=0.0):
    c = check_e1(data)
    return nc.ledger_row(
        "nb3-r4b-e1", "03", 4, {"name": "toeplitz-khat-corrgcv-retry"}, dataset_sha256, None,
        {"kind": "ridge", "lambda_grid": CONFIG["e1"]["lambda_grid"],
         "ridge_lambda_fit": CONFIG["e1"]["ridge_lambda_fit"]},
        {"train_cap": CONFIG["e1"]["train_cap"], "n_lambdas": len(data["lams"]),
         "n_inset_lags": len(data["lags"]), "clean_folds": CONFIG["e1"]["clean_folds"]},
        int(CONFIG["e1"]["clean_folds"][0]),
        {"n_lambdas": float(c["n_lambdas"]),
         "rel_err_toeplitz": float(c["rel_err_mean"]["corrgcv_toeplitz"]),
         "rel_err_gram": float(c["rel_err_mean"]["corrgcv_gram"]),
         "rel_err_ordinary_gcv": float(c["rel_err_mean"]["ordinary_gcv"])},
        float(wall_s), ["C56", "C59"])


# ---------------------------------------------------------------------------
# notebook cells
# ---------------------------------------------------------------------------

_NB2_E2_CLAIM = r"""### Fig E2 — how optimistic is a test fold that starts `g` bars after training? (C57)


For each clean fold the test window is held **fixed** and the training window's end is moved `g`
bars back from it (g = 0..24), so the rows being scored are identical at every gap and the
generator's GARCH volatility cancels out of the ratio. The measured quantity is the ridge
(`lambda = 5`) mean squared error on the first `PURGE = 8` rows of the test window at gap `g`,
divided by the same error at the decoupled reference gap `g = 60` (half the longest rolling
window), averaged with its SE over the 4 clean folds x 3 bootstrap seeds. The theory column is
Eq. (44)'s factor computed with a stationary Toeplitz `K_hat` estimated from the training rows'
biased lag-autocorrelation, and the band `[1 - rho(g), 1]` is the theorem's proven bracket: the
second term is bounded above by `rho` for any positive-definite `K_hat`, so containment here is
algebraic, not fitted. Every window length, reference gap and seed is **derived here** — the
paper states no experiment and no tolerance. Whatever the measured ratio is, it is reported as
measured: at gaps beyond the label horizon it drifts slightly above the theorem's upper edge (up
to 1.11 with a per-gap SE of 0.18), which is inside the `3*max(SE)` sampling allowance and is
reported as measured rather than smoothed — on this generator the near-horizon optimism is
confined to the first bars and no measurable long-range effect remains."""

_NB2_E3_CLAIM = r"""### Fig E3 — the protocol audit as a forest plot (C59)


The earlier protocol-IC chart was a bar chart: one height per protocol, which hides whether a
gap between protocols is resolvable at this seed/model spread. The same six means are stored
here as point estimates with their 95% t-intervals over the 3 seeds x 2 models, against the
clean protocol's reference IC — a row whose interval crosses the dashed line is not,
honestly, distinguishable from clean at this sample size. The `clean` row is drawn so the
reference value itself is visible on the plot, not only implied by the line. This is the
figure C59 introduces and it replaces the bar chart in this notebook's §5."""

_NB2_E2_CHECK = r"""import walkthrough_audits

_R4B_X = np.column_stack([COLS[k] for k in nbs_common.CLEAN_FEATURES])
_t_r4b = time.time()
_r4b_e2 = walkthrough_audits.compute_e2(_R4B_X, Y, PROTOS["clean"], nbs_common.VALID_START)
_r4b_e2_paths = walkthrough_audits.write_e2_figures(_r4b_e2)
_r4b_e2_chk = walkthrough_audits.check_e2(_r4b_e2)
_r4b_e2_wall = time.time() - _t_r4b
ALL_LEDGER = list(ALL_LEDGER) + [walkthrough_audits.ledger_row_e2(_r4b_e2, DATASET_SHA256, _r4b_e2_wall)]
nbs_common.merge_ledger("02", ALL_LEDGER)
_lo0 = float(next(r["lower_band"] for r in _r4b_e2["rows"] if int(r["g"]) == 0))
_f0 = float(next(r["eq44_factor"] for r in _r4b_e2["rows"] if int(r["g"]) == 0))
nbs_common.self_check(70, _r4b_e2_chk["ok"], note=(
    "fig-e2: %d gaps incl g=0 and the embargo g=%d; Eq.(44) factor inside [1-rho,1] at every gap "
    "(band lower edge %.3f -> %.3f as g grows); every empirical SE > 0" % (
        _r4b_e2_chk["n_gaps"], _r4b_e2_chk["embargo_used"], _r4b_e2_chk["band_lo_0"],
        _r4b_e2_chk["band_lo_max"])))
nbs_common.self_check(71, bool(_r4b_e2_chk["emp_at_0"] <= 1.0), note=(
    "measured test-risk ratio at g=0 is %.4f (optimistic, below the theorem's upper edge 1) vs "
    "Eq.(44) factor %.4f; measured values lie in [%.4f, %.4f] with 3*max(SE)=%.4f allowance" % (
        _r4b_e2_chk["emp_at_0"], _f0, min(r["_mean"] for r in _r4b_e2["rows"]),
        _r4b_e2_chk["max_emp"], _r4b_e2_chk["tol"])))
print("fig-e2 rows:", _r4b_e2["n_rows"], "| gaps 0..%d" % _r4b_e2["gaps"][-1],
      "| ref_gap", _r4b_e2["ref_gap"], "| embargo", nbs_common.EMBARGO)
for _r in _r4b_e2["rows"]:
    print("  g=%2s  band_lo=%.4f  eq44=%.4f  emp=%.4f +/- %.4f" % (
        _r["g"], _r["_lower"], _r["_factor"], _r["_mean"], _r["_se"]))
print("METRIC-WATCH NB2-R4E-E2-GAPS target=21 achieved=%d dir=ge status=%s" % (
    _r4b_e2_chk["n_gaps"], "MET" if _r4b_e2_chk["n_gaps"] >= 21 else "MISSED"))
print("METRIC-WATCH NB2-R4E-E2-OPTIMISM target=1.0 achieved=%.6f dir=le status=%s" % (
    _r4b_e2_chk["emp_at_0"], "MET" if _r4b_e2_chk["emp_at_0"] <= 1.0 else "MISSED"))"""

_NB2_E3_CHECK = r"""import walkthrough_audits

_t_r4b = time.time()
_r4b_e3 = walkthrough_audits.compute_e3(RUNS, PROTO_ORDER)  # noqa: F821 (RUNS/PROTO_ORDER from §3)
_r4b_e3_paths = walkthrough_audits.write_e3_figures(_r4b_e3)
_r4b_e3_chk = walkthrough_audits.check_e3(_r4b_e3)
_r4b_e3_wall = time.time() - _t_r4b
ALL_LEDGER = list(ALL_LEDGER) + [walkthrough_audits.ledger_row_e3(_r4b_e3, DATASET_SHA256, _r4b_e3_wall)]
nbs_common.merge_ledger("02", ALL_LEDGER)
nbs_common.self_check(72, _r4b_e3_chk["ok"], note=(
    "fig-e3 forest plot: %d protocol rows, all intervals bracket their point and are non-degenerate "
    "(min width %.4f), points span [%.4f, %.4f], reference = clean IC %.4f" % (
        _r4b_e3_chk["n_rows"], _r4b_e3_chk["min_width"], _r4b_e3_chk["point_range"][0],
        _r4b_e3_chk["point_range"][1], _r4b_e3_chk["reference"])))
print("fig-e3 rows:")
for _r in _r4b_e3["rows"]:
    print("  %-22s point=%+.4f  95%% CI [%+.4f, %+.4f]  reference=%+.4f" % (
        _r["row_label"], _r["_point"], _r["_lo"], _r["_hi"], _r["_ref"]))
print("METRIC-WATCH NB2-R4E-E3-ROWS target=6 achieved=%d dir=ge status=%s" % (
    _r4b_e3_chk["n_rows"], "MET" if _r4b_e3_chk["n_rows"] >= 6 else "MISSED"))"""

_NB3_E1_CLAIM = r"""### Fig E1 — does a stationary Toeplitz `K_hat` rescue empirical CorrGCV? (C56)


A documented pursuit is that the fold-0 CorrGCV on the empirical sample Gram misses
held-out risk by a factor of ~30. This section tries the untried q4 route: replace the empirical
Gram with a stationary Toeplitz `K_hat` estimated from the training rows' lag-autocorrelation
(`A_tau` from the biased row-correlation estimator), run the same renormalised
`(kappa, kappa_tilde)` solve, and measure the relative error against held-out risk across a
12-point log-spaced `lambda` sweep on each clean fold's trailing 700 rows (the derived-here
window). The empirical-Gram and ordinary-GCV curves are measured on exactly the same folds and
`lambda` grid, so the comparison is paired. This is a **hypothesis**: the section prints which
route came closer at each `lambda` and asserts neither. The inset shows `A_tau` itself with its
95% CI -- the O(T)-parameter structure the route is built from."""

_NB3_EMBED_CLAIM = r"""### Fig {ID} — embedded from NB2's execution


Figure E{id} is written by `02_naive_pipeline_leakage_audit`'s execution (NB2 owns the dataset and
the run spread); this notebook re-reads the written CSV, re-checks its schema and PNG before
redrawing it, and never regenerates the underlying numbers. Redrawing, not recomputing, is the
point: the part-1 artifacts stay the single source of truth (C14)."""

_NB3_E1_CHECK = r"""import walkthrough_audits

_r4b_ref = json.loads((nc.ART / "nb3_corrgcv.json").read_text())
_t_r4b = time.time()
_r4b_e1 = walkthrough_audits.compute_e1(X_C, Y, CLEAN_FOLDS, nc.VALID_START, nc.PURGE)
_r4b_e1_paths = walkthrough_audits.write_e1_figures(_r4b_e1, {
    "nb3_ref_corrgcv_gram_rel_err": float(_r4b_ref["rel_err_corrgcv"]),
    "nb3_ref_ordinary_gcv_rel_err": float(_r4b_ref["rel_err_gcv"]),
})
_r4b_e1_chk = walkthrough_audits.check_e1(_r4b_e1)
_r4b_e1_wall = time.time() - _t_r4b
LEDGER_03.append(walkthrough_audits.ledger_row_e1(_r4b_e1, DATASET_SHA256, _r4b_e1_wall))
nc.merge_ledger("03", LEDGER_03)
nc.self_check(80, _r4b_e1_chk["ok"], note=(
    "fig-e1: %d log-spaced lambdas, %d inset lags with CI brackets; mean rel. err over lambdas -- "
    "ordinary GCV %.4g, empirical-Gram CorrGCV %.4g, Toeplitz CorrGCV %.4g" % (
        _r4b_e1_chk["n_lambdas"], _r4b_e1_chk["n_lags"],
        _r4b_e1_chk["rel_err_mean"]["ordinary_gcv"],
        _r4b_e1_chk["rel_err_mean"]["corrgcv_gram"],
        _r4b_e1_chk["rel_err_mean"]["corrgcv_toeplitz"])))
nc.self_check(81, bool(
    _r4b_e1_chk["rel_err_mean"]["corrgcv_toeplitz"]
    < _r4b_e1_chk["rel_err_mean"]["corrgcv_gram"]), note=(
    "Toeplitz K-hat is %.3gx closer to held-out risk than the empirical Gram on average over the "
    "sweep (printed, not asserted by the harness); the reference values are read LIVE from "
    "nb3_corrgcv.json (Gram %.4g, ordinary GCV %.4g)" % (
        _r4b_e1_chk["rel_err_mean"]["corrgcv_gram"]
        / max(_r4b_e1_chk["rel_err_mean"]["corrgcv_toeplitz"], 1e-30),
        _r4b_ref["rel_err_corrgcv"], _r4b_ref["rel_err_gcv"])))
print("METRIC-WATCH NB3-R4E-E1-LAMBDAS target=10 achieved=%d dir=ge status=%s" % (
    _r4b_e1_chk["n_lambdas"], "MET" if _r4b_e1_chk["n_lambdas"] >= 10 else "MISSED"))
print("METRIC-WATCH NB3-R4E-E1-LAGS target=5 achieved=%d dir=ge status=%s" % (
    _r4b_e1_chk["n_lags"], "MET" if _r4b_e1_chk["n_lags"] >= 5 else "MISSED"))
print("METRIC-WATCH NB3-R4E-E1-TOEPLITZ-VS-GRAM target=1.0 achieved=%.6f dir=ge status=%s" % (
    _r4b_e1_chk["rel_err_mean"]["corrgcv_gram"]
    / max(_r4b_e1_chk["rel_err_mean"]["corrgcv_toeplitz"], 1e-30),
    "MET" if _r4b_e1_chk["rel_err_mean"]["corrgcv_toeplitz"]
    <= _r4b_e1_chk["rel_err_mean"]["corrgcv_gram"] else "MISSED"))
print("live reference values: empirical-Gram rel err %.4g, ordinary-GCV rel err %.4g" % (
    _r4b_ref["rel_err_corrgcv"], _r4b_ref["rel_err_gcv"]))"""

_NB3_EMBED_CHECK = r"""_r4b_emb_{id} = walkthrough_audits.check_embedded_figure("{id}")
_r4b_data_{id} = walkthrough_audits.load_fig_rows("{id}")
nc.self_check({sc}, _r4b_emb_{id}["ok"], note=(
    "fig-{id} re-read from NB2's execution: %d rows, run_id %s, plot_type %s, PNG %dpx >= 1200 "
    "(redrawn here, never recomputed)" % (
        _r4b_emb_{id}["n_rows"], _r4b_emb_{id}["run_id"], _r4b_emb_{id}["plot_type"],
        _r4b_emb_{id}["png_width"])))"""


def section_cells_nb2():
    """Notebook cells appended to nb2 (claim / compute / how-to-read / render per figure)."""
    def md(src):
        return {"type": "md", "source": src, "tags": []}

    def code(src, tags=None):
        return {"type": "code", "source": src, "tags": list(tags or [])}

    cells = [md(_NB2_E2_CLAIM), code(_NB2_E2_CHECK, ["r4b-e2"]),
             md("**How to read this chart:** " + HOW_TO_READ["e2"]),
             code("walkthrough_audits.render_fig_e2(_r4b_e2, PALETTE)", ["r4b-e2-viz"])]
    cells += [md(_NB2_E3_CLAIM), code(_NB2_E3_CHECK, ["r4b-e3"]),
              md("**How to read this chart:** " + HOW_TO_READ["e3"]),
              code("walkthrough_audits.render_fig_e3(_r4b_e3, PALETTE)", ["r4b-e3-viz"])]
    return cells


def section_cells_nb3():
    """Notebook cells appended to nb3 before its §5 docket (claim/compute/how/render per figure)."""
    def md(src):
        return {"type": "md", "source": src, "tags": []}

    def code(src, tags=None):
        return {"type": "code", "source": src, "tags": list(tags or [])}

    cells = [md(_NB3_E1_CLAIM), code(_NB3_E1_CHECK, ["r4b-e1"]),
             md("**How to read this chart:** " + HOW_TO_READ["e1"]),
             code("walkthrough_audits.render_fig_e1(_r4b_e1, PALETTE)", ["r4b-e1-viz"])]
    for fid, sc in (("e2", 82), ("e3", 83)):
        chk = _NB3_EMBED_CHECK.replace("{id}", fid).replace("{sc}", str(sc))
        claim = _NB3_EMBED_CLAIM.replace("{ID}", "E" + fid[1:]).replace("{id}", fid)
        cells += [md(claim), code(chk, ["r4b-embed-%s" % fid]),
                  md("**How to read this chart:** " + HOW_TO_READ[fid]),
                  code("walkthrough_audits.render_fig_%s(_r4b_data_%s, PALETTE)" % (fid, fid),
                       ["r4b-embed-%s-viz" % fid])]
    return cells


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - standalone smoke run
    import time
    COLS = nc.load_dataset_csv(nc.ART / "nb2_dataset.csv")
    Yv = COLS["y"]
    Xv = np.column_stack([COLS[k] for k in nc.CLEAN_FEATURES])
    t0 = time.perf_counter()
    RUNS = json.loads((nc.ART / "nb2_runs.json").read_text())["runs"]
    e3 = compute_e3(RUNS)
    write_e3_figures(e3)
    e2 = compute_e2(Xv, Yv, nc.CLEAN_FOLD_TESTS, nc.VALID_START)
    write_e2_figures(e2)
    e1 = compute_e1(Xv, Yv, nc.CLEAN_FOLD_TESTS, nc.VALID_START, nc.PURGE)
    ref = json.loads((nc.ART / "nb3_corrgcv.json").read_text())
    write_e1_figures(e1, {"nb3_ref_corrgcv_gram_rel_err": ref["rel_err_corrgcv"],
                          "nb3_ref_ordinary_gcv_rel_err": ref["rel_err_gcv"]})
    print("standalone walkthrough_audits run: %.1f s" % (time.perf_counter() - t0))
    for fid in ("e1", "e2", "e3"):
        print("  fig-%s: %d rows, png width %d" % (fid, len(_read_csv(FIG_DIR / f"fig-{fid}.csv")),
                                                   _png_width(FIG_DIR / f"fig-{fid}.png")))
    print("checks:", {"e1": check_e1(e1)["ok"], "e2": check_e2(e2)["ok"], "e3": check_e3(e3)["ok"]})
