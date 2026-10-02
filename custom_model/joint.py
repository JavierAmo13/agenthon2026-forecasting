"""Joint scenario generation across the (asset x horizon) grid.

The composite puts 0.3 on the joint variogram: independent per-asset draws
leave that score on the table. Dependence is a Student-t copula: correlated
normals divided by one shared chi-square factor per draw -> tail co-movement
(assets crash *together*, which a Gaussian copula understates). The copula's
nu comes from the pooled excess kurtosis of standardized OOF residuals.

Each task's marginal uses the empirical quantile function of its own OOF
residuals (real asymmetry) when >=30 residuals exist, else the parametric
Student-t/Normal inverse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st


def _nearest_psd_corr(c: np.ndarray) -> np.ndarray:
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def _copula_nu(resid_frame: pd.DataFrame | None) -> float:
    """Degrees of freedom for the shared copula shock, from the pooled excess
    kurtosis of standardized OOF residuals. inf -> Gaussian copula."""
    if resid_frame is None:
        return np.inf
    r = resid_frame.to_numpy(dtype=float)
    sd = np.nanstd(r, axis=0)
    keep = np.isfinite(sd) & (sd > 0)
    if keep.sum() == 0:
        return np.inf
    z = (r[:, keep] - np.nanmean(r[:, keep], axis=0)) / sd[keep]
    z = z[np.isfinite(z)]
    if z.size < 200:
        return np.inf
    ex = float(np.mean(z ** 4)) - 3.0  # z already standardized
    return float(np.clip(6.0 / ex + 4.0, 3.0, 30.0)) if ex > 1e-6 else np.inf


def _task_quantile(model: dict, u: np.ndarray) -> np.ndarray:
    """Copula uniforms -> standardized shocks for one task: empirical
    quantiles when available, else the parametric Student-t/Normal inverse
    rescaled to unit variance."""
    emp = model.get("emp")
    if emp is not None:
        return np.quantile(emp, u)
    nu = model.get("nu", np.inf)
    if np.isfinite(nu) and nu > 2.0:
        return _st.t.ppf(u, df=nu) * np.sqrt((nu - 2.0) / nu)
    return _st.norm.ppf(u)


def joint_samples(
    tasks: list[tuple[str, int]],
    mus: np.ndarray,
    resid_models: list[dict],
    resid_frame: pd.DataFrame | None,
    n_draws: int,
    seed: int = 0,
    regime_scale: float = 1.0,
) -> np.ndarray:
    """Draw correlated samples for all tasks at once.

    Parameters
    ----------
    tasks : [(asset, horizon), ...] in output order
    mus : point prediction per task at the as-of
    resid_models : fitted dict per task {mu, sigma, nu, emp}
    resid_frame : OOF residuals, columns = task index, aligned on dates
    Returns samples [n_draws, n_tasks]; caller reshapes to [n_draws, n_assets, n_horizons].
    """
    rng = np.random.default_rng(seed)
    k = len(tasks)
    mus = np.asarray(mus, dtype=float)
    sigma = np.array([max(m.get("sigma", 1.0), 1e-8) * regime_scale for m in resid_models])
    bias = np.array([m.get("mu", 0.0) for m in resid_models])
    nu_c = _copula_nu(resid_frame)

    if resid_frame is not None and resid_frame.shape[1] == k:
        corr = _nearest_psd_corr(resid_frame.corr().to_numpy(dtype=float))
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            chol = np.eye(k)
    else:
        chol = np.eye(k)

    z = rng.standard_normal((n_draws, k)) @ chol.T
    if np.isfinite(nu_c) and nu_c > 2.0:
        # one shared divisor per draw: the t-copula tail co-dependence
        chi = rng.chisquare(nu_c, size=(n_draws, 1))
        z = z * np.sqrt((nu_c - 2.0) / chi)
        U = np.clip(_st.t.cdf(z * np.sqrt(nu_c / (nu_c - 2.0)), df=nu_c),
                    1e-6, 1 - 1e-6)
    else:
        U = np.clip(_st.norm.cdf(z), 1e-6, 1 - 1e-6)
    zq = np.empty_like(U)
    for i in range(k):
        zq[:, i] = _task_quantile(resid_models[i], U[:, i])
    return mus[None, :] + bias[None, :] + zq * sigma[None, :]


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