"""Shared helpers for the leakage-proof time-series notebooks.

This is a synthetic-data research study and a starting point for further work — not production
trading code, and not evidence of live performance.
Everything here is deterministic given its seed; notebooks write artifacts under
``exec/artifacts/`` and the run ledger under ``exec/artifacts/run_ledger.jsonl``.

Only numpy / stdlib are imported at module import time; torch is imported inside
the sequence-model helper. No network, no LLM clients, no API keys.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXEC_DIR = Path(__file__).resolve().parent
ART = EXEC_DIR / "artifacts"
NB_DIR = EXEC_DIR.parent
LEDGER_PATH = ART / "run_ledger.jsonl"
EXTERNAL_SPEND_USD = 0.0

# ----------------------------------------------------------------------------
# IO / ledger / hashing
# ----------------------------------------------------------------------------


def ensure_dirs() -> None:
    ART.mkdir(parents=True, exist_ok=True)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(name: str, obj) -> Path:
    ensure_dirs()
    p = ART / name
    p.write_text(json.dumps(obj, indent=1, default=_json_default))
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_ledger() -> list:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]


def merge_ledger(notebook: str, rows: list) -> None:
    """Replace this notebook's rows, keep the other notebooks' rows, write sorted.

    A file lock makes concurrent notebook executions safe.
    """
    ensure_dirs()
    import tempfile
    lock_path = Path(tempfile.gettempdir()) / f"nbs-common-ledger-{notebook}.lock"
    with open(lock_path, "w") as lock:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        keep = [r for r in read_ledger() if r["notebook"] != notebook]
        all_rows = sorted(keep + list(rows), key=lambda r: (r["notebook"], r["run_id"]))
        LEDGER_PATH.write_text("".join(json.dumps(r, default=_json_default) + "\n" for r in all_rows))
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass


def ledger_row(run_id, notebook, section, protocol, data_hash, fold_boundaries, model,
               hyperparameters, seed, metric, wall_time_s, claim_ids):
    return {
        "run_id": run_id,
        "notebook": notebook,
        "section": int(section),
        "protocol": protocol if isinstance(protocol, dict) else {"name": str(protocol)},
        "data_hash": data_hash,
        "fold_boundaries": fold_boundaries,
        "model": str(model),
        "hyperparameters": hyperparameters,
        "seed": int(seed),
        "metric": {k: float(v) for k, v in metric.items()},
        "wall_time_s": float(wall_time_s),
        "claim_ids": list(claim_ids),
        "external_spend_usd": EXTERNAL_SPEND_USD,
    }


# ----------------------------------------------------------------------------
# self-checks / metrics / intervals
# ----------------------------------------------------------------------------


def self_check(i: int, ok: bool, note: str = "") -> dict:
    status = "MET" if ok else "NOT MET"
    print(f"SELF-CHECK {int(i)} [{status}]" + (f" {note}" if note else ""))
    return {"id": int(i), "status": status}


def sc_entry(i: int, claim_id: str, ok: bool) -> dict:
    return {"id": int(i), "claim_id": claim_id, "status": "MET" if ok else "NOT MET"}


def ic(pred, y) -> float:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    if pred.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def sharpe(pred, y) -> float:
    pnl = np.sign(np.asarray(pred, float)) * np.asarray(y, float)
    return float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0


def ic_ci(r: float, n_eff: float, z: float = 1.96) -> tuple:
    """Overlap-aware Fisher-z interval; n_eff = n_predictions / mean_label_length."""
    n_eff = max(float(n_eff), 4.0)
    se = z / math.sqrt(n_eff - 3.0)
    lo = math.tanh(math.atanh(max(min(r, 0.999999), -0.999999)) - se)
    hi = math.tanh(math.atanh(max(min(r, 0.999999), -0.999999)) + se)
    return float(lo), float(hi)


def mean_ci(values, z: float = 1.96) -> tuple:
    v = np.asarray(values, float)
    se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean() - z * se), float(v.mean() + z * se)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ----------------------------------------------------------------------------
# Synthetic market data (nb2 owns generation; nb3 reuses the artifacts)
# ----------------------------------------------------------------------------

LABEL_HORIZON = 6
N_BARS = 3200
REGIME_BOUNDS = [0, 420, 820, 1200, 1600, 2000, 2450, 2850, N_BARS]
REGIME_KINDS = ["weak_edge", "weak_edge", "zero_edge", "regime_edge",
                "weak_edge", "zero_edge", "weak_edge", "weak_edge"]
REGIME_TARGET_IC = [0.15, 0.09, 0.0, -0.13, 0.17, 0.0, 0.12, 0.07]
CLEAN_WINDOWS = (5, 10, 20, 40, 80, 120)
LEAK_OFFSET = 3
FEATURE_OFFSETS = {**{f"f_ma{w}": -(w - 1) for w in CLEAN_WINDOWS},
                   **{f"f_sd{w}": -(w - 1) for w in CLEAN_WINDOWS},
                   "f_mom": 0, "f_leak3": LEAK_OFFSET}
GENERATOR_SEED = 47


def _rolling(x, w):
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.concatenate([[0.0], x]))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def make_features(ret):
    """Clean features are available at decision time (offset <= 0); f_leak3 looks ahead."""
    f = {"f_mom": ret.copy()}
    for w in CLEAN_WINDOWS:
        f[f"f_ma{w}"] = _rolling(ret, w)
        f[f"f_sd{w}"] = np.sqrt(np.clip(_rolling(ret ** 2, w), 0.0, None))
    lk = np.zeros(len(ret))
    for t in range(len(ret)):
        lk[t] = ret[t + 1:t + 1 + LEAK_OFFSET].sum()  # truncated at the series end
    f["f_leak3"] = lk
    return f


def make_dataset(seed: int = GENERATOR_SEED):
    """Matrix-free synthetic market: AR(1) latent edge + GARCH volatility + regimes."""
    n = N_BARS
    h = LABEL_HORIZON
    rng = np.random.default_rng(seed)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + rng.standard_normal()
    u = (u - u.mean()) / u.std()
    omega, a, b = 0.05, 0.12, 0.82
    s2 = np.empty(n)
    s2[0] = omega / (1 - a - b)
    z = rng.standard_normal(n)
    e = np.zeros(n)
    for t in range(n):
        if t > 0:
            s2[t] = omega + a * e[t - 1] ** 2 + b * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    W = np.array([u[t + 1:t + 1 + h].sum() for t in range(n)])
    E6 = np.array([e[t + 1:t + 1 + h].sum() for t in range(n)])
    A = float(np.cov(u, W)[0, 1])
    B = float(W.var())
    C = float(E6.var())
    beta = np.zeros(n)
    for i in range(len(REGIME_BOUNDS) - 1):
        rho = REGIME_TARGET_IC[i]
        if rho != 0.0:
            beta[REGIME_BOUNDS[i]:REGIME_BOUNDS[i + 1]] = rho * math.sqrt(C) / math.sqrt(A * A - rho * rho * B)
    ret = beta * u + e
    y = np.array([ret[t + 1:t + 1 + h].sum() for t in range(n)])
    regime = np.zeros(n, dtype=int)
    edge_segment = np.zeros(n, dtype=int)
    ambiguous = np.zeros(n, dtype=int)
    for i in range(len(REGIME_BOUNDS) - 1):
        s, e_ = REGIME_BOUNDS[i], REGIME_BOUNDS[i + 1]
        regime[s:e_] = i
        edge_segment[s:e_] = i
    for bnd in REGIME_BOUNDS[1:-1]:
        lo, hi = max(0, bnd - h), min(n, bnd)
        ambiguous[lo:hi] = 1  # the label window straddles a regime boundary
    label_start = np.arange(1, n + 1)
    label_end = np.arange(0, n) + h
    label_complete = (label_end <= n - 1).astype(int)
    feats = make_features(ret)
    return {
        "seed": seed, "n": n, "h": h, "ret": ret, "y": y, "latent_signal": u, "beta": beta,
        "regime": regime, "edge_segment": edge_segment, "ambiguous": ambiguous,
        "label_start": label_start, "label_end": label_end, "label_complete": label_complete,
        "features": feats,
    }


def dataset_columns(ds) -> dict:
    cols = {
        "t": np.arange(ds["n"]),
        "feature_available_at": np.arange(ds["n"]),
        "label_start": ds["label_start"],
        "label_end": ds["label_end"],
        "label_complete": ds["label_complete"],
        "y": ds["y"],
        "ret": ds["ret"],
        "regime": ds["regime"],
        "edge_segment": ds["edge_segment"],
        "latent_signal": ds["latent_signal"],
        "ambiguous": ds["ambiguous"],
    }
    cols.update(ds["features"])
    return cols


def write_dataset_csv(ds, path) -> str:
    """Deterministic CSV (fixed float formatting) so identical runs hash identically."""
    import csv
    cols = dataset_columns(ds)
    order = ["t", "timestamp", "feature_available_at", "label_start", "label_end", "label_complete",
             "y", "ret", "regime", "edge_segment", "latent_signal", "ambiguous"] + list(ds["features"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(order)
        for i in range(ds["n"]):
            row = []
            for k in order:
                if k == "timestamp":
                    row.append(f"2015-01-01T{i // 6:04d}:{i % 6:02d}:00")
                elif k in ("t", "feature_available_at", "label_start", "label_end", "label_complete",
                           "regime", "edge_segment", "ambiguous"):
                    row.append(int(cols[k][i]))
                else:
                    row.append(f"{float(cols[k][i]):.10g}")
            w.writerow(row)
    return sha256_file(path)


def profile(ds) -> dict:
    regimes = []
    for i in range(len(REGIME_BOUNDS) - 1):
        regimes.append({"id": i, "start": int(REGIME_BOUNDS[i]), "end": int(REGIME_BOUNDS[i + 1]),
                        "kind": REGIME_KINDS[i], "true_ic": float(REGIME_TARGET_IC[i])})
    feats = {k: {"kind": ("leak_future" if k == "f_leak3" else "clean"),
                 "offset": int(FEATURE_OFFSETS[k])} for k in ds["features"]}
    return {"regimes": regimes, "features": feats}


# ----------------------------------------------------------------------------
# Protocols / folds
# ----------------------------------------------------------------------------

CLEAN_FOLD_TESTS = [(600, 1250), (1250, 1900), (1900, 2550), (2550, N_BARS - LABEL_HORIZON)]
PURGE = LABEL_HORIZON + 2
EMBARGO = 2
NOMINAL_CUT = int(0.7 * N_BARS)
VALID_START = max(CLEAN_WINDOWS)          # rolling-window warm-up
VALID_END = N_BARS - LABEL_HORIZON        # incomplete-label rows excluded


def _valid_rows(ds, cols_needed=()):
    n, h = ds["n"], ds["h"]
    vm = np.zeros(n, bool)
    vm[VALID_START:n - h] = True
    for k in cols_needed:
        vm &= ~np.isnan(ds["features"][k])
    return vm


def _ranges_from_idx(idx):
    idx = np.sort(np.asarray(idx, int))
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    ends = np.concatenate([idx[cuts] + 1, [idx[-1] + 1]])
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def _fold(train, test, norm_fit, purge, embargo, fold):
    return {"fold": int(fold), "train": _ranges_from_idx(train), "test": _ranges_from_idx(test),
            "norm_fit": _ranges_from_idx(norm_fit), "purge": int(purge), "embargo": int(embargo)}


def _walk_forward(purge, norm_all=False):
    folds = []
    for i, (ts, te) in enumerate(CLEAN_FOLD_TESTS):
        tr = np.arange(VALID_START, ts - purge)
        fit = np.arange(VALID_START, VALID_END) if norm_all else tr
        folds.append(_fold(tr, np.arange(ts, te), fit, purge, EMBARGO, i))
    return folds


def _random_split(seed, frac=0.3):
    rng = np.random.default_rng(seed)
    idx = VALID_START + rng.permutation(VALID_END - VALID_START)
    nte = int(round(frac * len(idx)))
    te = np.sort(idx[:nte])
    tr = np.sort(idx[nte:])
    return _fold(tr, te, tr, 0, 0, 0)


def _block_random(seed, nblocks=40, frac=0.3):
    rng = np.random.default_rng(seed)
    span = VALID_END - VALID_START
    block = span // nblocks
    order = rng.permutation(nblocks)
    test_blocks = set(order[:int(round(frac * nblocks))].tolist())
    tr, te = [], []
    for b in range(nblocks):
        idx = np.arange(VALID_START + b * block,
                        VALID_START + span if b == nblocks - 1 else VALID_START + (b + 1) * block)
        (te if b in test_blocks else tr).extend(idx.tolist())
    return _fold(np.array(tr), np.array(te), np.array(tr), 0, 0, 0)


def build_protocols(ds) -> dict:
    """Six protocols on identical data; every train/test row has a complete label."""
    clean = _walk_forward(PURGE)
    return {
        "clean": clean,
        "leaky_future_features": _walk_forward(PURGE),
        "leaky_overlap_labels": [_random_split(s) for s in (5, 6, 7)],
        "leaky_global_norm": _walk_forward(PURGE, norm_all=True),
        "leaky_random_split": [_block_random(s) for s in (11, 12, 13)],
        "leaky_all": [_random_split(s) for s in (21, 22, 23)],
    }


PROTOCOL_FEATURES = {
    "clean": "clean",
    "leaky_future_features": "leak",
    "leaky_overlap_labels": "clean",
    "leaky_global_norm": "clean",
    "leaky_random_split": "clean",
    "leaky_all": "leak",
}
CLEAN_FEATURES = [f"f_ma{w}" for w in CLEAN_WINDOWS] + [f"f_sd{w}" for w in CLEAN_WINDOWS] + ["f_mom"]
LEAK_FEATURES = CLEAN_FEATURES + ["f_leak3"]
RANDOM_SPLIT_PROTOCOLS = ("leaky_overlap_labels", "leaky_random_split", "leaky_all")


def fold_boundaries_for(protocol, folds):
    """Ledger fold envelopes; random/interleaved protocols record their nominal cut."""
    if protocol in RANDOM_SPLIT_PROTOCOLS:
        return [[0, NOMINAL_CUT, NOMINAL_CUT, N_BARS - 1] for _ in folds]
    out = []
    for f in folds:
        out.append([int(f["train"][0][0]), int(f["train"][-1][1]),
                    int(f["test"][0][0]), int(f["test"][-1][1])])
    return out


def load_dataset_csv(path):
    """Read nb2_dataset.csv into a column dict (floats where numeric)."""
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    return out


# ----------------------------------------------------------------------------
# ridge / walk-forward runs
# ----------------------------------------------------------------------------


def ridge_fit_predict(Xtr, ytr, Xte, lam=5.0):
    p = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(p), Xtr.T @ ytr)
    return Xte @ w


def standardise(Xtr, Xte, Xall=None, global_norm=False):
    src = Xall if (global_norm and Xall is not None) else Xtr
    m = np.nanmean(src, 0)
    s = np.nanstd(src, 0) + 1e-12
    return (Xtr - m) / s, (Xte - m) / s


# ----------------------------------------------------------------------------
# CorrGCV (Atanasov, Zavatone-Veth & Pehlevan, arXiv:2408.04607, Eq. 30)
# ----------------------------------------------------------------------------


def _solve_kappa(lam, q):
    b = 1.0 - q - lam
    return (-b + math.sqrt(b * b + 4.0 * lam)) / 2.0


def _solve_kappa_tilde(K, target):
    ev = np.linalg.eigvalsh(K)
    lo, hi = 1e-12, 1e8
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if float(np.mean(ev / (ev + mid))) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def toeplitz_corr(T, rho):
    idx = np.arange(T)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def matrix_gaussian(T, N, K, seed):
    rng = np.random.default_rng(seed)
    w, V = np.linalg.eigh(K)
    Ks = V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T
    Z = rng.standard_normal((T, N))
    return Ks @ Z


def corrgcv_estimates(X, y, lam, K, sigma_eps, w_bar):
    """Round-1 compatibility wrapper: isotropic Sigma = I, sample correlation K."""
    N = X.shape[1]
    ev_sigma = np.ones(N)
    ev_K = np.linalg.eigvalsh(K)
    ident = solve_renormalized(lam, N / X.shape[0], ev_sigma, ev_K)
    rep = corrgcv_rep(X, y, lam, ident, sigma_eps, w_bar, ev_sigma)
    return {**rep, "gcv": rep["gcv1"], "df_frac": rep["df_frac_hat"],
            "kappa": ident["kappa"], "kappa_tilde": ident["kappa_tilde"], "S": ident["kappa"] / lam,
            "ndf1": ident["ndf1"], "ndf2": ident["ndf2"]}


# ---------------------------------------------------------------------------
# Round 2 -- CorrGCV C20-C26 (Atanasov, Zavatone-Veth & Pehlevan, arXiv:2408.04607)
# ---------------------------------------------------------------------------


def power_law_spectrum(N, alpha_exp):
    """Sigma eigenvalues lambda_k ~ k^-alpha (q3 Fig. 1 anisotropic spectrum)."""
    ks = np.arange(1, N + 1, dtype=float)
    return ks ** (-float(alpha_exp))


def power_law_target_weights(N, alpha_exp, r_exp):
    """Target weights with lambda_k * wbar_k^2 ~ k^-(2*alpha*r+1) (q3 source exponent r)."""
    ks = np.arange(1, N + 1, dtype=float)
    eig = ks ** (-float(alpha_exp))
    return np.sqrt(ks ** (-(2.0 * float(alpha_exp) * float(r_exp) + 1.0)) / eig)


def toeplitz_exponential(T, xi):
    """Exponential sample-correlation family K_ts = exp(-|t-s|/xi) (q3)."""
    idx = np.arange(T)
    return np.exp(-np.abs(idx[:, None] - idx[None, :]) / float(xi))


def matrix_gaussian_sigma(T, N, K, sigma_eigenvalues, seed):
    """X = K^{1/2} Z Sigma^{1/2} with diagonal Sigma (q2's matrix-Gaussian design)."""
    rng = np.random.default_rng(seed)
    w, V = np.linalg.eigh(K)
    Ks = V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T
    return Ks @ rng.standard_normal((T, N)) @ np.diag(np.sqrt(np.asarray(sigma_eigenvalues, float)))


def _df1_df2(ev, kappa):
    return (float(np.mean(ev / (ev + kappa))), float(np.mean(ev ** 2 / (ev + kappa) ** 2)))


def _ndf1_ndf2(ev, kappa_tilde):
    return (float(np.mean(ev / (ev + kappa_tilde))),
            float(np.mean(ev ** 2 / (ev + kappa_tilde) ** 2)))


def solve_renormalized(lam, q, ev_sigma, ev_K, iters=200):
    """Solve q2's coupled system for the renormalized (kappa, kappa_tilde).

    q2: "The renormalized ridge is kappa = lambda S(df_1) ... By matrix duality,
    q df_1 ~= df1_tilde and kappa kappa_tilde / lambda = 1 / df1_tilde."

    Duality gives kappa_tilde(kappa) = lam / (kappa * q * df1(kappa)); substituting into
    the subordination identity leaves one scalar equation in kappa, solved by bisection.
    With K = I this reduces exactly to S = 1/(1 - q*df1) (the metamorphic guard, C21).
    """
    lam, q = float(lam), float(q)
    ev_sigma = np.asarray(ev_sigma, float)
    ev_K = np.asarray(ev_K, float)

    def F(kappa):
        d1, _ = _df1_df2(ev_sigma, kappa)
        kt = lam / (kappa * q * d1)
        n1, _ = _ndf1_ndf2(ev_K, kt)
        return n1 - q * d1

    a, b = 1e-10, 1e10
    fa, fb = F(a), F(b)
    if fa * fb > 0.0:
        xs = np.logspace(-10, 10, 400)
        vals = np.array([F(x) for x in xs])
        idx = np.nonzero(np.diff(np.sign(vals)) != 0)[0]
        if not len(idx):
            raise RuntimeError("solve_renormalized: no bracket for kappa")
        a, b = xs[idx[0]], xs[idx[0] + 1]
    for _ in range(iters):
        m = math.sqrt(a * b)
        if F(m) < 0.0:
            a = m
        else:
            b = m
    kappa = math.sqrt(a * b)
    df1, df2 = _df1_df2(ev_sigma, kappa)
    kappa_tilde = lam / (kappa * q * df1)
    ndf1, ndf2 = _ndf1_ndf2(ev_K, kappa_tilde)
    sub = abs(kappa * kappa_tilde / lam - 1.0 / ndf1) / (1.0 / ndf1)
    dual = abs(q * df1 - ndf1) / ndf1
    return {"kappa": float(kappa), "kappa_tilde": float(kappa_tilde), "df1": df1, "df2": df2,
            "ndf1": ndf1, "ndf2": ndf2, "subordination_residual": float(sub),
            "duality_residual": float(dual)}


def corrgcv_rep(X, y, lam, ident, sigma_eps, w_bar, sigma_eigenvalues):
    """One realization: R_in, empirical df, and q3's four estimators against latent risk.

    ``bracket`` is q3's Carmack GCCV bracket ``1 - (df2*dtf2)/(df1*dtf1)/(1 - dtf2/dtf1)``
    exposed unsquared and signed (round 4's C53 records it as its own quantity); ``carmack``
    is the same term squared, so the two never drift apart.
    """
    T, N = X.shape
    q = N / T
    S_hat = X.T @ X / T
    w_hat = np.linalg.solve(S_hat + lam * np.eye(N), X.T @ y / T)
    resid = y - X @ w_hat
    r_in = float(resid @ resid / T)
    dw = np.asarray(w_bar, float) - w_hat
    latent_risk = float(dw @ (np.asarray(sigma_eigenvalues, float) * dw) + sigma_eps ** 2)
    sv2 = np.linalg.svd(X, compute_uv=False) ** 2
    df_frac_hat = float(np.sum(sv2 / (sv2 + T * lam)) / T)
    S = ident["kappa"] / lam
    d1, d2, n1, n2 = ident["df1"], ident["df2"], ident["ndf1"], ident["ndf2"]
    bracket = 1.0 - (d2 * n2) / (d1 * n1) * 1.0 / (1.0 - n2 / n1)
    return {
        "in_sample_risk": r_in,
        "latent_risk": latent_risk,
        "df_frac_hat": df_frac_hat,
        "gcv1": r_in / (1.0 - df_frac_hat) ** 2,
        "gcv2": S ** 2 * r_in,
        "carmack": S ** 2 * bracket ** 2 * r_in,
        "bracket": float(bracket),
        "corrgcv": S * n1 / (n1 - n2) * r_in,
    }


# ---------------------------------------------------------------------------
# Round 2 -- post-selection Sharpe family (Pav, arXiv:2606.01650) C27-C30
# ---------------------------------------------------------------------------

EULER_GAMMA = 0.5772156649015329


def _phi_inv(p):
    import statistics
    return statistics.NormalDist(0.0, 1.0).inv_cdf(p)


def js_hat(observed, n, k, sel):
    """q5's James-Stein: s_v = (1 - ((k-2)/n)/||zeta_hat - zeta_bar 1||^2)_+."""
    grand = float(np.mean(observed))
    denom = float(np.sum((np.asarray(observed, float) - grand) ** 2))
    if denom <= 0.0:
        return grand
    sv = max(0.0, 1.0 - ((k - 2.0) / n) / denom)
    return float(grand + sv * (observed[sel] - grand))


def expected_max_hat(observed, n, k, sel):
    """q5 Eq. 5: the expected-maximum debiasing for k equal Sharpes."""
    term = ((1.0 - EULER_GAMMA) * _phi_inv(1.0 - 1.0 / k)
            + EULER_GAMMA * _phi_inv(1.0 - 1.0 / (k * math.e)))
    return float(observed[sel] - term / math.sqrt(n))


def sure_hat(observed, n, k, sel):
    """SURE soft-thresholding of the selected estimate toward the grand mean."""
    grand = float(np.mean(observed))
    thr = math.sqrt(2.0 * math.log(k) / n)
    delta = observed[sel] - grand
    return float(grand + math.copysign(max(0.0, abs(delta) - thr), delta))


def selection_regime(layout, k, n, rho, n_reps=500, seed0=7000):
    """Fixed population zeta vector; correlated noise zeta_hat ~ N(zeta, R/n), R = (1-rho)I + rho*11'."""
    zeta = np.zeros(int(k))
    if layout == "all_equal":
        zeta[:] = 0.3
    elif layout == "one_good":
        zeta[0] = 0.6
        zeta[1:] = 0.1
    elif layout == "two_good":
        zeta[:2] = 0.6
        zeta[2:] = 0.1
    else:
        raise ValueError(f"unknown layout {layout!r}")
    R = (1.0 - rho) * np.eye(k) + rho * np.ones((k, k))
    L = np.linalg.cholesky(R)
    reps = []
    for s in range(n_reps):
        rng = np.random.default_rng(seed0 + s)
        observed = zeta + (L @ rng.standard_normal(int(k))) / math.sqrt(n)
        sel = int(np.argmax(observed))
        reps.append({
            "seed": int(seed0 + s),
            "observed": [float(v) for v in observed],
            "true": [float(v) for v in zeta],
            "sel": sel,
            "naive": float(observed[sel]),
            "expected_max": expected_max_hat(observed, n, k, sel),
            "js": js_hat(observed, n, k, sel),
            "sure": sure_hat(observed, n, k, sel),
            "true_selected": float(zeta[sel]),
        })
    summary = {}
    for name in ("naive", "expected_max", "js", "sure"):
        est = np.array([r[name] for r in reps], float)
        tru = np.array([r["true_selected"] for r in reps], float)
        summary[f"bias_{name}"] = float(np.mean(est - tru))
        summary[f"rmse_{name}"] = float(np.sqrt(np.mean((est - tru) ** 2)))
    return {"layout": layout, "k": int(k), "n": int(n),
            "corr_structure": {"kind": "equicorrelated", "rho": float(rho)},
            "reps": reps, "summary": summary}


# ---------------------------------------------------------------------------
# Round 2 -- worst-case in-band null stream for an informative ARL check (C34)
# ---------------------------------------------------------------------------


def two_point_stream(seed, horizon=500, halfband=0.05):
    """Worst-case in-band stream: two-point at the band edges (variance equals the
    Hoeffding proxy), the most informative no-change stream for a width-envelope CS."""
    rng = np.random.default_rng(seed)
    return 0.5 + halfband * np.where(rng.random(horizon) < 0.5, -1.0, 1.0)


# ----------------------------------------------------------------------------
# Repeated forward confidence sequence (Shekhar & Ramdas, arXiv:2309.09111, Def 2.1)
# ----------------------------------------------------------------------------


def fcs_widths(horizon, alpha=0.05, sigma=3.0, halfband=0.05):
    """Hoeffding-mixture forward-CS half-widths for a stream in an in-control band +/- halfband."""
    v = (2.0 * halfband) ** 2 / 4.0
    N = np.arange(1, horizon + 1, dtype=float)
    return np.sqrt((2.0 * (1.0 + N * sigma ** 2 * v) / (N ** 2 * sigma ** 2))
                   * (math.log(1.0 / alpha) + 0.5 * np.log(1.0 + N * sigma ** 2 * v)))


def fcs_detector(x, alpha=0.05, sigma=3.0, halfband=0.05, trace=False):
    """Pure function of the observed stream. Returns alarm time (1-based) or None.

    Each forward confidence sequence is intersected with its own past (the nested
    form of Definition 2.1 / Remark 1.4 in Shekhar & Ramdas, arXiv:2309.09111).
    """
    x = np.asarray(x, float)
    H = len(x)
    w = fcs_widths(H, alpha, sigma, halfband)
    cs = np.cumsum(np.concatenate([[0.0], x]))
    cur_lo = np.full(H + 1, -np.inf)
    cur_hi = np.full(H + 1, np.inf)
    steps = []
    alarm = None
    for t in range(1, H + 1):
        starts = np.arange(1, t + 1)
        n = t - starts + 1
        mean = (cs[t] - cs[starts - 1]) / n
        raw_lo = mean - w[n - 1]
        raw_hi = mean + w[n - 1]
        if t > 1:
            cur_lo[1:t] = np.maximum(cur_lo[1:t], raw_lo[:-1])
            cur_hi[1:t] = np.minimum(cur_hi[1:t], raw_hi[:-1])
        cur_lo[t] = raw_lo[-1]
        cur_hi[t] = raw_hi[-1]
        L, U = float(cur_lo[1:t + 1].max()), float(cur_hi[1:t + 1].min())
        empty = bool(L > U)
        if trace:
            steps.append({"t": int(t), "x": float(x[t - 1]),
                          "active": [{"start": int(s), "lo": float(cur_lo[s]), "hi": float(cur_hi[s])}
                                     for s in starts],
                          "empty": empty, "lo": None if empty else L, "hi": None if empty else U})
        if empty:
            alarm = t
            break
    if trace:
        return {"alarm_t": alarm, "steps": steps}
    return alarm


# ----------------------------------------------------------------------------
# convenience: seeded null / change streams for the detector demo
# ----------------------------------------------------------------------------


def null_stream(seed, horizon=200, halfband=0.05):
    rng = np.random.default_rng(seed)
    return 0.5 + halfband * (2.0 * rng.random(horizon) - 1.0)


def change_stream(seed, delta, change_time=50, horizon=250, halfband=0.05):
    x = null_stream(seed, horizon, halfband)
    x[change_time:] = x[change_time:] + delta
    return x


# ---------------------------------------------------------------------------
# Figure embedding (round 4, C55/C59): a section's plot cell renders the figure with the
# notebook's PALETTE, shows it inline, and archives exactly that render as the publication PNG.
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402

_VIZ_CALL = _re.compile(r"\s*(corrgcv_study|selection_study|detection_study|walkthrough_audits)\.render_fig_([a-z]\d)\((.+), PALETTE\)\s*", _re.S)


def embed_viz_source(src: str) -> str:
    """Rewrite a plan module's plot cell so the notebook shows and archives the same figure."""
    m = _VIZ_CALL.fullmatch(src)
    if not m:
        return src
    mod, fid, data = m.groups()
    return (f"_fig = {mod}.render_fig_{fid}({data}, PALETTE)  # inline render, this notebook's PALETTE\n"
            f"from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA  # bare canvas after plt.show(), attach one to measure\n"
            f"_bb = _fig.get_tightbbox(_FCA(_fig).get_renderer())  # tight bbox in inches, used to size the archived PNG\n"
            f'_fig.savefig("../figures/fig-{fid}.png", dpi=max(200, 1300 / (_bb.width + 0.2)), bbox_inches="tight")  # archive what is shown')
