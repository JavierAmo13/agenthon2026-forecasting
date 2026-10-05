
"""Residual law: from OOF residuals to a marginal distribution per task.

G_i = (1 - eta) F_emp + eta F_t   — a small Student-t mix over the empirical
residual distribution so the worst observed residual is not a hard ceiling.
eta=0 recovers the pure empirical marginal. The residual mean is deducted
once (emp is centered) and restored once at generation.

Scale calibration is temporal: recent squared residuals vs their long-run
variance, clipped — quiet folds narrow sigma, agitated ones widen it.

Text enters at exactly one point (plan 3.5): sigma *= exp(clip(g(z), a, b))
where g is a tiny calibrator fit on historical (text state, |resid|) pairs.
With insufficient admissible pairs the calibrator abstains (factor = 1.0);
it can widen OR narrow — widening is not assumed to be an improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fit_residual_model(resid: pd.Series, eta: float = 0.15,
                       recent: int = 63) -> dict:
    """Estimate the residual law from OOF residuals.

    Returns {mu, sigma, nu, emp, eta, scale_recent}. ``emp`` is the sorted,
    mean-centered residual vector; ``nu`` via moments on excess kurtosis;
    ``scale_recent`` a bounded temporal recalibration.
    """
    r = resid.dropna().to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        sd = float(np.std(r)) if r.size else 1.0
        return {"mu": 0.0, "sigma": max(sd, 1e-8), "nu": np.inf,
                "emp": np.zeros(0), "eta": 0.0, "scale_recent": 1.0,
                "n": int(r.size)}
    mu = float(np.mean(r))
    sd = float(np.std(r))
    iqr = float(np.subtract(*np.percentile(r, [75, 25])))
    robust = iqr / 1.349 if iqr > 0 else sd
    sigma = max(0.5 * (sd + robust), 1e-8)
    k = float(stats.kurtosis(r, fisher=True))
    nu = np.inf
    if k > 1.0:  # heavy tails -> Student-t component worth having
        nu = float(np.clip(6.0 / k + 4.0, 3.0, 60.0))
    # temporal recalibration: recent residual energy vs full-sample variance
    tail = r[-recent:]
    scale_recent = float(np.clip(np.sqrt(np.mean(tail**2) / max(sd**2, 1e-16)),
                                 0.7, 1.4)) if tail.size >= 10 else 1.0
    emp = np.sort(r - mu)  # centered once; restored once downstream
    return {"mu": mu, "sigma": sigma, "nu": nu, "emp": emp,
            "eta": float(np.clip(eta, 0.0, 0.5)) if emp.size >= 40 else 0.0,
            "scale_recent": scale_recent, "n": int(r.size)}


def inv_marginal(model: dict, u: np.ndarray) -> np.ndarray:
    """Quantile function of G_i at u in (0,1) — mixture in quantile space:
    (1-eta) F_emp^{-1}(u) + eta F_t^{-1}(u), with mu restored exactly once."""
    u = np.clip(np.asarray(u, dtype=float), 1e-6, 1.0 - 1e-6)
    mu, sigma, nu = model["mu"], model["sigma"], model.get("nu", np.inf)
    emp, eta = model.get("emp", np.zeros(0)), model.get("eta", 0.0)
    if emp.size >= 20:
        grid = (np.arange(emp.size) + 0.5) / emp.size
        q_emp = np.interp(u, grid, emp)          # centered quantiles
    else:
        q_emp = stats.norm.ppf(u) * sigma
    if eta > 0:
        if np.isfinite(nu) and nu > 2.0:
            q_t = stats.t.ppf(u, nu) * np.sqrt((nu - 2.0) / nu) * sigma
        else:
            q_t = stats.norm.ppf(u) * sigma
        q = (1.0 - eta) * q_emp + eta * q_t
    else:
        q = q_emp
    return mu + q


def sample_residuals(model: dict, shape, rng: np.random.Generator,
                     u: np.ndarray | None = None) -> np.ndarray:
    """Monte-Carlo draws of the marginal. ``u`` lets the joint layer reuse
    copula uniforms so dependence is set by the copula, not by resampling."""
    if u is None:
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=shape)
    return inv_marginal(model, u)


# --- text-conditioned sigma (the single text entry point) -------------------

def fit_text_calibrator(z_frame: pd.DataFrame, abs_resid: pd.Series,
                        sigma_base: float, min_pairs: int = 30) -> dict | None:
    """Least squares of log(|r|/sigma_base) on [unc, surp, persist].

    Only fitted when enough admissible (text state, outcome) pairs exist —
    otherwise the caller abstains. Returns {"coef": ..., "n": ...}.
    """
    if z_frame is None or z_frame.empty:
        return None
    zz = z_frame[["text_unc", "text_surp", "text_persist"]].copy()
    yy = np.log(abs_resid.clip(lower=1e-12) / max(sigma_base, 1e-12))
    df = zz.join(yy.rename("target")).dropna()
    if len(df) < min_pairs:
        return None
    Xd = np.column_stack([np.ones(len(df)), df[["text_unc", "text_surp",
                                                "text_persist"]].to_numpy()])
    coef, *_ = np.linalg.lstsq(Xd, df["target"].to_numpy(), rcond=None)
    return {"coef": coef, "n": int(len(df))}


def text_sigma_factor(z_now: dict | None, cal: dict | None,
                      a: float = -0.3, b: float = 0.6) -> float:
    """sigma multiplier = exp(clip(g(z), a, b)); abstains to 1.0."""
    if not cal or not z_now:
        return 1.0
    z = np.array([1.0, z_now.get("unc", np.nan), z_now.get("surp", np.nan),
                  z_now.get("pers", np.nan)])
    if not np.isfinite(z[1:]).all():
        return 1.0
    g = float(z @ cal["coef"])
    return float(np.exp(np.clip(g, a, b)))
