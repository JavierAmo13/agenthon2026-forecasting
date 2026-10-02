"""Volatility regime: scale factor for predictive dispersion.

Rule-based for now: recent realized vol of the target assets vs its own
long-run distribution. NORMAL ~1.0, HIGH_VOL ~1.25, STRESS ~1.5, SHOCK ~1.8.
The factor widens residual sigma when the market is unusually agitated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_scale(bundle, *, lookback: int = 21, ref: int = 252) -> float:
    """Ratio of short-run to long-run vol across target assets, mapped to a
    dispersion multiplier in [0.8, 2.0]."""
    ratios = []
    for a in bundle.target_assets:
        try:
            s = bundle.asset_series(a)
        except KeyError:
            continue
        s = s.dropna()
        if len(s) < ref:
            continue
        r = np.log(s).diff() if (s > 0).all() else s.diff()
        short = float(r.iloc[-lookback:].std())
        long = float(r.iloc[-ref:].std())
        if np.isfinite(short) and np.isfinite(long) and long > 0:
            ratios.append(short / long)
    if not ratios:
        return 1.0
    q = float(np.median(ratios))
    # Map: 0.6->0.85, 1.0->1.0, 1.5->1.3, 2.5->1.7 (piecewise-linear, clamped).
    scale = np.interp(q, [0.4, 0.8, 1.2, 1.8, 3.0], [0.8, 0.9, 1.15, 1.5, 1.9])
    return float(np.clip(scale, 0.8, 2.0))