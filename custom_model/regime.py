
"""Volatility regime: scale factor for predictive dispersion.

Rule-based: recent realized vol vs its long-run distribution, with windows
in OBSERVATIONS matched to the target frequency (21/252 daily, 3/36
monthly). Guards: minimum history, nonzero denominators, non-flat series.

v2 returns the component split for the adjustments log: numeric scale,
text adjustment and stress stay separate so each effect is auditable
(plan 3.7 — text enters at exactly one multiplier point).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_scale(bundle, *, lookback: int | None = None,
                 ref: int | None = None) -> dict:
    """{scale, ratio, n_assets, components} — scale in [0.8, 2.0]."""
    opy = 12 if bundle.target_frequency == "monthly" else 252
    lookback = lookback or max(3, opy // 12)
    ref = ref or opy
    ratios = []
    for a in bundle.available_assets():
        try:
            s = bundle.asset_series(a).dropna()
        except KeyError:
            continue
        if len(s) < max(ref, lookback + 10):
            continue
        r = np.log(s).diff() if (s > 0).all() else s.diff()
        short = float(r.iloc[-lookback:].std())
        long = float(r.iloc[-ref:].std())
        if np.isfinite(short) and np.isfinite(long) and long > 1e-12:
            ratios.append(short / long)
    if not ratios:
        return {"scale": 1.0, "ratio": np.nan, "n_assets": 0,
                "components": {"numeric": 1.0, "text": 1.0, "stress": 1.0}}
    q = float(np.median(ratios))
    # Map: 0.6->0.85, 1.0->1.0, 1.5->1.3, 2.5->1.7 (piecewise-linear, clamped).
    scale = np.interp(q, [0.4, 0.8, 1.2, 1.8, 3.0], [0.8, 0.9, 1.15, 1.5, 1.9])
    scale = float(np.clip(scale, 0.8, 2.0))
    return {"scale": scale, "ratio": q, "n_assets": len(ratios),
            "components": {"numeric": scale, "text": 1.0, "stress": 1.0}}
