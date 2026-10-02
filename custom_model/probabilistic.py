"""Turn point forecasts into distributions.

Residual model: sigma (and optionally Student-t nu) estimated from
walk-forward residuals; the empirical standardized residual vector is kept
as `emp` so downstream draws can use the true error distribution (asymmetric
marginals) instead of a parametric shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class PersistenceModel:
    """future = current level of the asset's own series (target-type aware)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceModel":
        self.mu_ = float(np.nanmean(y)) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), getattr(self, "mu_", 0.0))


def fit_residual_model(resid: pd.Series) -> dict:
    """Estimate sigma and tail shape from OOF residuals.

    Student-t nu via method of moments on the excess kurtosis when the tails
    are heavy enough to need it; otherwise Normal. A robust sigma (IQR-based)
    is blended with the plain std.
    """
    r = resid.dropna().to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        sd = float(np.std(r)) if r.size else 1.0
        return {"mu": 0.0, "sigma": max(sd, 1e-8), "nu": np.inf, "emp": None}
    mu = float(np.mean(r))
    sd = float(np.std(r))
    iqr = float(np.subtract(*np.percentile(r, [75, 25])))
    robust = iqr / 1.349 if iqr > 0 else sd
    sigma = max(0.5 * (sd + robust), 1e-8)
    k = float(stats.kurtosis(r, fisher=True))
    nu = np.inf
    if k > 1.0:  # excess kurtosis beyond ~1 -> worth heavier tails
        nu = float(np.clip(6.0 / k + 4.0, 3.0, 60.0))
    # Standardized OOF residuals (mean 0, std 1): the empirical marginal.
    # Draws sample its quantile function -> real asymmetry and kurtosis.
    emp = np.sort((r - r.mean()) / sd) if r.size >= 30 and sd > 0 else None
    return {"mu": mu, "sigma": sigma, "nu": nu, "emp": emp}


def sample_residuals(model: dict, shape: tuple[int, ...], rng: np.random.Generator,
                     z: np.ndarray | None = None) -> np.ndarray:
    """Draw standardized residual noise of the requested shape.

    If ``z`` is given it must be standard normal and is reused (for joint
    draws); otherwise fresh noise is drawn. ``nu`` < inf applies a shared
    chi-square scaling so the heavy tail is common to the whole draw.
    """
    nu = model.get("nu", np.inf)
    sigma = model.get("sigma", 1.0)
    mu = model.get("mu", 0.0)
    emp = model.get("emp")
    if emp is not None:
        # bootstrap the real residual distribution (asymmetric, fat-tailed)
        return mu + sigma * rng.choice(emp, size=shape)
    if z is None:
        z = rng.standard_normal(shape)
    if np.isfinite(nu):
        chi = rng.chisquare(nu, size=shape[:1] + (1,) * (len(shape) - 1) if len(shape) > 1 else shape)
        t_scale = np.sqrt(nu / chi)
        return mu + sigma * z * t_scale * np.sqrt((nu - 2.0) / nu) if nu > 2 else mu + sigma * z * t_scale
    return mu + sigma * z