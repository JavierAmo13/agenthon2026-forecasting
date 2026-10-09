
"""Transfer assets: an early window plus one row at the as-of.

The F2/F3 transfer cards ship the target's early history (e.g. BRL
1995-2003) and a single observation at the as-of date, with the middle
years withheld. The anchor is therefore correct (the as-of row); what the
early window cannot give is TODAY's volatility regime - a 2003-vintage
sigma badly under-covers a 2013 taper tantrum.

Fix: rescale the transfer asset's early-window sigma by how much the
rest of the panel's vol regime moved between the end of the early window
and the as-of. The ratio of cross-asset EWMA medians is the observable
that carries the regime change. It is honest (panel-derived, no text)
and bounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import steps as st


def detect_transfer(bundle, series: pd.Series, gap_days: int = 900) -> bool:
    """True when the asset's last-obs gap before the as-of row is huge -
    the withheld-middle signature of a transfer card."""
    if series is None or len(series) < 3:
        return False
    gaps = series.index.to_series().diff().dt.days
    return bool(gaps.iloc[-1] > gap_days) if len(gaps) > 1 else False


def basket_regime_ratio(bundle, end_date: pd.Timestamp,
                        exclude: set[str]) -> float:
    """Median over context assets of sigma_now / sigma_at(end_date).

    Clipped to [0.5, 3.0]: a bounded regime read, not an amplifier.
    """
    ratios = []
    for a, s in bundle.all_series().items():
        if a in exclude or s is None or len(s) < 120:
            continue
        # pick the series' own natural step: returns panels ship per-step
        # returns (median |x| << 1), everything else diffs
        x = s.astype(float)
        x = x if float(x.abs().median()) < 0.5 else x.diff()
        x = x.dropna()
        if len(x) < 120:
            continue
        sig = st.ewma_vol(x)
        past = sig[sig.index <= end_date]
        now = sig.iloc[-1]
        if len(past) and np.isfinite(now) and past.iloc[-1] > 0:
            ratios.append(float(now / past.iloc[-1]))
    if not ratios:
        return 1.0
    return float(np.clip(np.median(ratios), 0.5, 3.0))


def basket_sigma_now(bundle, exclude: set[str]) -> float:
    """Median current EWMA per-step sigma across context series.

    The playbook's sanctioned width for a transfer asset: the related
    panel's vol scaled by a beta. The basket's CURRENT sigma is the only
    honest in-sample read of today's regime; the early window alone is
    blind to it (a pegged CNY window carries no crisis memory at all).
    """
    vals = []
    for a, s in bundle.all_series().items():
        if a in exclude or s is None or len(s) < 60:
            continue
        x = s.astype(float)
        lvl = abs(float(x.iloc[-1]))
        if float(x.abs().median()) < 0.5:
            steps = x          # returns-style panel: already relative-ish
        else:
            steps = x.diff() / lvl if lvl > 0 else x.diff()
        steps = steps.dropna()
        if len(steps) < 60:
            continue
        vals.append(float(st.ewma_vol(steps).iloc[-1]))
    return float(np.median(vals)) if vals else 0.0


def stressed_sigma_floor(bundle, series: pd.Series) -> float:
    """75th percentile of the EWMA sigma path over the asset's OWN early
    window - the dispersion it showed when it was stressed, not the calm
    reading at the window's end. The early windows contain the asset's
    real crises (BRL 1999 deval, INR 1997-98, CNY regime changes); a
    transfer card asks for the NEW regime, so the calm-endpoint sigma
    under-covers by construction."""
    x = st.clean_steps(series, bundle.target_type)
    if len(x) < 30:
        return 0.0
    sig = st.ewma_vol(x)
    return float(np.percentile(sig.dropna(), 75))
