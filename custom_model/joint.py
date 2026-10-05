
"""Joint scenario generation across the (asset x horizon) grid.

Student-t copula on rank-estimated correlation:
  * Residuals are aligned by origin; the common sample size is logged.
  * R comes from Kendall tau -> sin(pi/2 * tau), the right latent parameter
    for an elliptical copula, then shrunk toward the equicorrelation base:
        R* = (1 - lam) R_hat + lam R_equi
    Both are projected to nearest-PSD before Cholesky.
  * Copula nu is a JOINT tail-dependence parameter, kept separate from the
    marginal nu's (plan correction: nu of the copula does not change G_i).
  * Draws: U ~ copula -> G_i^{-1}(U) per task, so marginals are preserved
    exactly even when nu is heavy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .probabilistic import inv_marginal


def _nearest_psd_corr(c: np.ndarray) -> np.ndarray:
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def rank_corr(resid_frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Kendall-tau correlation mapped to the copula's latent Pearson-like
    parameter. Returns (R, min common pairwise sample)."""
    tau = resid_frame.corr(method="kendall").to_numpy(dtype=float)
    common = int(resid_frame.notna().astype(int).T.dot(
        resid_frame.notna().astype(int)).to_numpy().min())
    R = np.sin(np.pi / 2.0 * np.nan_to_num(tau, nan=0.0))
    np.fill_diagonal(R, 1.0)
    return R, common


def shrink_corr(R: np.ndarray, lam: float) -> np.ndarray:
    """R* = (1-lam) R + lam R_equi, R_equi = constant-correlation base with
    the same average off-diagonal (keeps the global co-movement level)."""
    k = R.shape[0]
    if k <= 1:
        return R
    off = R[~np.eye(k, dtype=bool)]
    equi = np.full((k, k), float(np.clip(np.nanmean(off), -0.9, 0.9)))
    np.fill_diagonal(equi, 1.0)
    out = (1.0 - lam) * R + lam * equi
    return _nearest_psd_corr(out)


def copula_uniforms(R: np.ndarray, nu: float, n_draws: int,
                    rng: np.random.Generator) -> np.ndarray:
    """U ~ t_nu copula (or Gaussian when nu=inf) with latent corr R."""
    k = R.shape[0]
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        L = np.eye(k)
    z = rng.standard_normal((n_draws, k)) @ L.T
    if np.isfinite(nu) and nu > 2.0:
        chi = rng.chisquare(nu, size=(n_draws, 1))
        z = z * np.sqrt(nu / chi)
        return sstats.t.cdf(z, df=nu)
    return sstats.norm.cdf(z)


def joint_samples(
    tasks: list[tuple[str, int]],
    mus: np.ndarray,
    resid_models: list[dict],
    resid_frame: pd.DataFrame | None,
    n_draws: int,
    seed: int = 0,
    sigma_mult: np.ndarray | None = None,
    lam: float = 0.35,
    nu_copula: float | None = None,
) -> np.ndarray:
    """Draw correlated samples for all tasks at once. [n_draws, n_tasks].

    sigma_mult[i] bundles the single-point adjustments (regime numeric scale,
    text factor) applied to task i's dispersion — each applied exactly once.
    """
    rng = np.random.default_rng(seed)
    k = len(tasks)
    mus = np.asarray(mus, dtype=float)
    mult = np.ones(k) if sigma_mult is None else np.asarray(sigma_mult, float)

    if resid_frame is not None and resid_frame.shape[1] == k:
        R_hat, common = rank_corr(resid_frame)
        eff_lam = lam if common >= 20 else max(lam, 0.8)  # thin overlap -> trust the base
        R = shrink_corr(R_hat, eff_lam)
    else:
        R = np.eye(k)

    if nu_copula is None:
        nus = [m.get("nu", np.inf) for m in resid_models]
        nu_copula = float(np.nanmin(nus)) if nus else np.inf

    U = copula_uniforms(R, nu_copula, n_draws, rng)
    out = np.empty((n_draws, k), dtype=float)
    for j, m in enumerate(resid_models):
        q = inv_marginal(m, U[:, j])
        out[:, j] = mus[j] + m.get("mu", 0.0) + mult[j] * (q - m.get("mu", 0.0))
    return out


def to_output_tensor(samples: np.ndarray, assets: list[str], horizons: list[int],
                     tasks: list[tuple[str, int]]) -> np.ndarray:
    """[n_draws, n_tasks] -> [n_draws, n_assets, n_horizons] in declared order."""
    n_draws = samples.shape[0]
    out = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    idx = {t: i for i, t in enumerate(tasks)}
    for ai, a in enumerate(assets):
        for hi, h in enumerate(horizons):
            out[:, ai, hi] = samples[:, idx[(a, h)]]
    return out
