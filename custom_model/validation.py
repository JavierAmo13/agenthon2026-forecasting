
"""Local evaluation metrics — explicit proxies, never the official score.

Used for member weighting (OOF CRPS proxy) and for the post-calibration
evaluation block when backtesting. Local normalizations do not match the
official scorer's hidden references; treat every number as a proxy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def crps_samples(draws: np.ndarray, y: float, max_draws: int = 500) -> float:
    """Ensemble CRPS: E|X - y| - 0.5 E|X - X'| on a subsample."""
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(y):
        return np.nan
    if x.size > max_draws:
        x = x[np.linspace(0, x.size - 1, max_draws).astype(int)]
    t1 = np.mean(np.abs(x - y))
    t2 = np.mean(np.abs(x[:, None] - x[None, :]))
    return float(t1 - 0.5 * t2)


def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """Closed-form Gaussian CRPS — cheap proxy for member scoring."""
    sigma = max(float(sigma), 1e-8)
    z = (y - mu) / sigma
    return float(sigma * (z * (2 * stats.norm.cdf(z) - 1)
                          + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


def pinball(draws: np.ndarray, y: float, q: float) -> float:
    """Pinball loss of the empirical q-quantile of draws at level q."""
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(y):
        return np.nan
    qv = np.quantile(x, q)
    d = y - qv
    return float(np.maximum(q * d, (q - 1) * d))


def coverage(draws: np.ndarray, y: float, alpha: float = 0.9) -> float:
    """1.0 if y falls inside the central alpha-interval of draws."""
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(y):
        return np.nan
    lo, hi = np.quantile(x, [(1 - alpha) / 2, 1 - (1 - alpha) / 2])
    return float(lo <= y <= hi)


def variogram(draws_mat: np.ndarray, y_vec: np.ndarray, p: float = 0.5,
              max_pairs: int = 200, seed: int = 0) -> float:
    """Variogram-style dependence check across tasks: |draw_i - draw_j|^p
    averaged over draws and pairs vs the same power of the observation."""
    rng = np.random.default_rng(seed)
    k = draws_mat.shape[1]
    if k < 2:
        return np.nan
    i, j = np.triu_indices(k, 1)
    if len(i) > max_pairs:
        sel = rng.choice(len(i), max_pairs, replace=False)
        i, j = i[sel], j[sel]
    pred = np.mean(np.abs(draws_mat[:, i] - draws_mat[:, j]) ** p)
    obs = np.mean(np.abs(y_vec[i] - y_vec[j]) ** p)
    return float(pred - obs)
