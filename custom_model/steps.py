
"""Step series, gap rule, horizon->step resolution, EWMA vol regime.

Everything here works on one asset's own date-indexed series. Mirrors
M0-BASELINE S3 conventions so our floors and the divisor share the world:

  * level      -> first differences, with the hole rule of S3.3 applied
                  on the trailing window's spacing
  * log_return -> the row itself (panels already ship per-step returns)
  * monthly    -> cadence detected from spacing; horizon keys resolved to
                  monthly steps via observation_periods (named-month rule)
                  or the steps-1..5 override without sealed dates
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EWMA_LAM = 0.94          # per-observation decay (RiskMetrics)
_EWMA_LAM_M = 0.80        # monthly series: ~18 obs effective window


def step_series(s: pd.Series, target_type: str) -> pd.Series:
    """Per-observation step series of an asset (drop NaN, keep dates)."""
    s = s.dropna().sort_index()
    if target_type == "log_return":
        return s.astype(float)
    return s.diff()


def gap_mask(index: pd.DatetimeIndex, window: int = 300) -> pd.Series:
    """True where the step ending at index[i] spans a data hole.

    M0 S3.3: hole = interval > max(10 x median spacing of the trailing
    window, 5 days). Median computed over the trailing `window` rows'
    intervals - on a transfer card the early-window median, so the decade
    bridge is dropped while old observations survive.
    """
    if len(index) < 3:
        return pd.Series(False, index=index)
    iv = pd.Series(index, index=index).diff().dt.days
    med = float(iv.iloc[-window:].median())
    if not np.isfinite(med) or med <= 0:
        return pd.Series(False, index=index)
    return iv > max(med * 10.0, 5.0)


def clean_steps(s: pd.Series, target_type: str,
                window: int = 300) -> pd.Series:
    """Step series with hole-bridging steps dropped."""
    st = step_series(s, target_type)
    drop = gap_mask(s.index, window=window)
    drop = drop.reindex(st.index, fill_value=False)
    return st[~drop]


def cadence(s: pd.Series) -> str:
    """'monthly' when the median spacing exceeds ~20 days else 'daily'."""
    if len(s) < 5:
        return "daily"
    med = float(pd.Series(s.index).diff().dt.days.median())
    return "monthly" if med > 20 else "daily"


def ewma_vol(x: pd.Series, lam: float | None = None) -> pd.Series:
    """EWMA of squared steps -> per-observation sigma path (in step units).
    Floor: 30% of the full-sample sd so standardized shocks never explode."""
    x = x.dropna()
    if len(x) < 3:
        sd = float(x.std()) if len(x) else 1.0
        return pd.Series(max(sd, 1e-8), index=x.index)
    if lam is None:
        lam = _EWMA_LAM_M if cadence(x) == "monthly" else _EWMA_LAM
    v = (x.to_numpy(dtype=float)) ** 2
    out = np.empty(len(v))
    acc = v[0]
    wsum = 1.0
    for i in range(len(v)):
        acc = lam * acc + (1 - lam) * v[i]
        wsum = lam * wsum + (1 - lam)
        out[i] = acc / wsum
    sig = np.sqrt(out)
    floor = 0.3 * float(np.nanstd(x.to_numpy()))
    sig = np.maximum(sig, max(floor, 1e-12))
    return pd.Series(sig, index=x.index)


def rolling_drift(x: pd.Series, window: int = 252) -> pd.Series:
    """Slow drift estimate per observation (trailing mean)."""
    return x.rolling(min(window, max(len(x) // 3, 10)), min_periods=10).mean()


def monthly_steps(anchor: pd.Timestamp, period: str) -> int:
    y0, m0 = anchor.year, anchor.month
    y1, m1 = int(period[:4]), int(period[5:7])
    return 12 * (y1 - y0) + (m1 - m0)


def resolve_steps(bundle) -> dict[tuple[str, int], int]:
    """{(asset, h): steps in panel observations}.

    Daily panels: the declared key IS the business-day count.
    Monthly cards with observation_periods: months from the anchor
    observation's month to the named month (named-month rule). Without a
    mapping, month count inferred as round(h/21) is the honest fallback -
    it is what a participant can compute without sealed target dates, and
    beats taking 140 as months by a factor of ~7.
    """
    out: dict[tuple[str, int], int] = {}
    periods = bundle.observation_periods
    for a in bundle.target_assets:
        s = bundle.series(a)
        cad = cadence(s) if s is not None else "daily"
        for hi, h in enumerate(bundle.horizons):
            if (bundle.target_frequency == "monthly" or cad == "monthly"):
                if periods and s is not None and len(s):
                    st = monthly_steps(s.index[-1], str(periods[hi]))
                    out[(a, h)] = max(st, 1)
                else:
                    # keys look like business days on every monthly card
                    out[(a, h)] = max(1, int(round(h / 21.0)))
            else:
                out[(a, h)] = int(h)
    return out


def obs_period(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return index.to_period("M")
