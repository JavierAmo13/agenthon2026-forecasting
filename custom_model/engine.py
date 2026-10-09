
"""Joint innovations engine — the forecast backbone.

Per the organizers' solver playbook, the winning recipe on the internal
blind-solve evaluation was a scenario-weighted mixture over a
bootstrap/regime engine. This implements exactly that backbone:

  * Per-asset step series (level diffs / return rows), holes dropped.
  * Vol-standardized innovations  z_t = (x_t - m_t) / sigma_t  with sigma_t
    an EWMA of squared steps: the empirical SHAPE (fat tails, skew) survives
    while dispersion is re-anchored to the current regime sigma_now.
  * Stationary block bootstrap on ONE shared date index: the same resampled
    calendar day drives every asset, so empirical cross-asset and cross-
    horizon dependence — including joint tail co-movement — is preserved
    without fitting a copula.
  * Recency-weighted block starts (60% recent-window + 40% full-history
    blend), plus an optional crisis-day boost that concentrates starts on
    dates where max|z| was extreme — fattening joint tails when the text or
    regime says stress.
  * Paths accumulate per asset; every cell reads off the same path, so
    cross-horizon covariance min(s_i, s_j)*Sigma is reproduced by
    construction rather than estimated.

Determinism: seed is a fixed function of the unit id (see forecast.py); the
harness's fresh QFBENCH_SEED on rerun does not change the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd

from . import steps as st

# ---------------------------------------------------------------- tuning
RECENCY_HALFLIFE_D = 750      # daily: ~3y half-life for block starts
RECENCY_HALFLIFE_M = 36       # monthly: ~3y in monthly obs
RECENCY_MIX = 0.60            # share of recency weight vs uniform
BLOCK_LEN_D = 8               # mean stationary block length, daily
BLOCK_LEN_M = 2               # monthly series: blocks of ~2 obs
TAIL_BOOST = 0.8              # extra start weight on max|z|>2.5 days
Z_CLIP = 8.0                  # safety cap on standardized shocks
SIGMA_FLOOR_Q = 1.0           # sigma_now floor vs full-sample sigma
MIN_COMMON = 40               # min aligned dates for the joint pool


@dataclass
class AssetModel:
    asset: str
    anchor: float
    z: pd.Series            # standardized innovations, date-indexed
    sigma_now: float        # current-regime per-step sd
    drift_step: float       # per-step drift to add on the path
    cadence: str            # 'daily' | 'monthly'
    notes: dict = field(default_factory=dict)
    sigma_pool: np.ndarray | None = None
    # When set (transfer assets), each draw samples its per-step sigma
    # from the asset's own historical vol-regime distribution, biased to
    # the stressed regime. This is a scale mixture over the asset's own
    # history: honest dispersion for an asset whose recent regime is
    # withheld, rather than a point estimate of a calm endpoint.


def _build_asset_model(asset: str, s: pd.Series, bundle,
                       *, drift_shrink: float = 0.7) -> AssetModel:
    """z path, current sigma and per-step drift for one target asset."""
    cad = st.cadence(s)
    x = st.clean_steps(s, bundle.target_type)
    if len(x) < 8:
        sd = float(s.diff().std()) if len(s) > 3 else 1.0
        anchor = 0.0 if bundle.target_type == "log_return" else float(s.iloc[-1])
        return AssetModel(asset, anchor, pd.Series(dtype=float),
                          max(sd, 1e-8), 0.0, cad, {"fallback": "short"})

    m = st.rolling_drift(x)
    sig = st.ewma_vol(x)
    z = ((x - m.fillna(median_or_zero(x))) / sig).clip(-Z_CLIP, Z_CLIP).dropna()
    sigma_full = float(x.std()) if len(x) > 5 else float(sig.iloc[-1])
    sigma_now = float(sig.iloc[-1])
    lo = float(os.environ.get("VFLOOR", str(SIGMA_FLOOR_Q)))
    sigma_now = float(np.clip(sigma_now,
                              lo * sigma_full, 3.0 * sigma_full))
    if bundle.target_type == "log_return":
        mu = float(x.mean())
    else:
        # shrink the raw mean: noisy level drifts, nonzero factor premia
        mu = drift_shrink * float(x.mean())
        mu = float(np.clip(mu, -0.8 * sigma_full, 0.8 * sigma_full))
    anchor = 0.0 if bundle.target_type == "log_return" else float(s.iloc[-1])
    return AssetModel(asset, anchor, z, sigma_now, mu, cad,
                      {"sigma_now": sigma_now, "sigma_full": sigma_full,
                       "n_steps": len(x),
                       "target_type": bundle.target_type})


def median_or_zero(x: pd.Series) -> float:
    v = float(x.median()) if len(x) else 0.0
    return v if np.isfinite(v) else 0.0


def _start_weights(common_index: pd.DatetimeIndex, zmax: pd.Series,
                   cadence: str, tail_boost: float) -> np.ndarray:
    """Recency+uniform blend, boosted on joint-stress days."""
    T = len(common_index)
    hl = RECENCY_HALFLIFE_M if cadence == "monthly" else RECENCY_HALFLIFE_D
    age = np.arange(T - 1, -1, -1, dtype=float)          # 0 = newest
    rec = 0.5 ** (age / hl)
    w = RECENCY_MIX * rec / rec.sum() + (1 - RECENCY_MIX) / T
    if tail_boost > 0:
        crisis = (zmax > 2.5).to_numpy(dtype=float)
        w = w * (1.0 + tail_boost * crisis)
        w = w / w.sum()
    return w


def _sample_blocks(rng: np.random.Generator, w: np.ndarray, T: int,
                   n_draws: int, S: int, L: float) -> np.ndarray:
    """[n_draws, S] matrix of row indexes into the shared z frame —
    stationary block bootstrap: with prob 1/L open a new block at a
    weighted start, else continue the previous row."""
    idx = np.empty((n_draws, S), dtype=np.int64)
    geo = rng.random((n_draws, S))
    starts = rng.choice(T, size=(n_draws, S), p=w)
    cur = starts[:, 0]
    idx[:, 0] = cur
    for k in range(1, S):
        cont = geo[:, k] < 1.0 - 1.0 / L
        cur = np.where(cont, (cur + 1) % T, starts[:, k])
        idx[:, k] = cur
    return idx


def simulate(models: list[AssetModel], step_counts: dict[tuple[str, int], int],
             horizons: list[int], n_draws: int, seed: int,
             *, tail_boost: float = TAIL_BOOST) -> np.ndarray:
    """[n_draws, n_assets, n_horizons] joint draws in the card's grid.

    models must be in the declared asset order; step_counts gives each
    cell's observation steps.
    """
    rng = np.random.default_rng(seed)
    A = len(models)
    assets = [m.asset for m in models]
    S = max(step_counts[(a, h)] for a in assets for h in horizons)
    cadence = models[0].cadence if models else "daily"

    # shared z frame on the intersection of dates (empirical joint law)
    zlist = [m.z.rename(m.asset) for m in models]
    Zf = pd.concat(zlist, axis=1, join="inner").dropna()
    if len(Zf) < MIN_COMMON:
        # sparse overlap -> widen the frame on union w/ per-asset mean-impute
        Zf = pd.concat(zlist, axis=1, join="outer")
        Zf = Zf.fillna(0.0).dropna(how="all")
    Z = Zf.to_numpy(dtype=float)
    T = len(Z)
    if T < 5:
        Z = rng.standard_normal((200, A))
        T = 200

    zmax = pd.Series(np.abs(Z).max(axis=1), index=Zf.index)
    w = _start_weights(Zf.index, zmax, cadence, tail_boost)
    L = BLOCK_LEN_M if cadence == "monthly" else BLOCK_LEN_D
    pick = _sample_blocks(rng, w, T, n_draws, S, L)

    sig = np.array([m.sigma_now for m in models])
    mu = np.array([m.drift_step for m in models])
    # per-draw sigma matrix: transfer assets draw their regime scale from
    # their own early-window sigma path (size-biased toward stress)
    sig_d = np.tile(sig, (n_draws, 1))
    for ai, m in enumerate(models):
        if m.sigma_pool is not None and len(m.sigma_pool) > 30:
            pool = np.asarray(m.sigma_pool, dtype=float)
            w = pool / pool.sum()
            sig_d[:, ai] = rng.choice(pool, size=n_draws, p=w)
    innov = Z[pick] * sig_d[:, None, :] + mu[None, None, :]   # [d, S, A]
    # log_return cells accumulate log(1+r), matching the realized target
    is_lr = np.array([m.notes.get("target_type") == "log_return"
                      for m in models])
    innov[:, :, is_lr] = np.log1p(np.clip(innov[:, :, is_lr], -0.999999, None))
    cum = np.cumsum(innov, axis=1)

    out = np.empty((n_draws, A, len(horizons)))
    for ai, m in enumerate(models):
        for hi, h in enumerate(horizons):
            s = step_counts[(m.asset, h)]
            out[:, ai, hi] = m.anchor + cum[:, s - 1, ai]
    return out


def build_models(bundle, assets: list[str] | None = None) -> list[AssetModel]:
    """AssetModel for every target asset that has a series."""
    out = []
    for a in (assets or bundle.target_assets):
        s = bundle.series(a)
        if s is None or len(s) < 2:
            continue
        out.append(_build_asset_model(a, s, bundle))
    return out
