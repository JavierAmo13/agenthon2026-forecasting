
"""Target construction + label availability.

Per (asset, horizon) one supervised target:
  level      -> value at t + h observations
  log_return -> sum of log(1 + r) over the next h observations
Monthly-frequency targets map horizon keys to steps via the unit's
observation_periods (resolved from the last available observation), keeping
the original horizon key. The publication date is never inferred from a
monthly label; availability of a monthly label is the end of its period.

v2 adds the availability contract: for every origin t we return
``avail(t, h)``, the date of the last observation the label consumes.
A training row is admissible only when avail <= cutoff — this is what makes
--asof overrides and historical pseudo-origins safe: labels realized after
the cutoff read as unseen, not as usable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ASSET_COLS = ("asset", "asset_id")


def _find_series(bundle, asset: str) -> pd.Series:
    # Full series (not truncated at as-of): labels need future values; the
    # availability contract below is what keeps post-cutoff labels out of train.
    return bundle.asset_series(asset, upto_asof=False)


def _forward_logret(r: pd.Series, h: int) -> pd.Series:
    lr = np.log1p(r.astype(float))
    return lr.rolling(h, min_periods=h).sum().shift(-h)


def _daily(bundle, asset: str, h: int, target_type: str) -> tuple[pd.Series, pd.Series]:
    """(y, avail): y(t) consumes observations up to and including index[t+h]."""
    s = _find_series(bundle, asset)
    if target_type == "log_return":
        y = _forward_logret(s, h)
    else:
        y = s.astype(float).shift(-h)
    avail = s.index.to_series().shift(-h)
    avail.index = s.index
    return y, avail


def _monthly_steps(bundle) -> dict[tuple[str, int], tuple[pd.Series, pd.Series, int]]:
    """{(asset, h): (y, avail, steps)} — avail = end of the target period."""
    spec_tgt = bundle.spec.get("targets", {})
    periods = spec_tgt.get("observation_periods") or bundle.observation_periods
    horizons = bundle.horizons
    cutoff = pd.Timestamp(bundle.asof)
    out: dict[tuple[str, int], tuple[pd.Series, pd.Series, int]] = {}
    for asset in bundle.available_assets():
        s = _find_series(bundle, asset)
        s = s[s.index <= cutoff]
        sm = s.copy()
        sm.index = pd.DatetimeIndex(sm.index).to_period("M")
        sm = sm[~sm.index.duplicated(keep="last")]
        anchor_p = sm.index[-1]
        per_idx = sm.index.to_series()
        for hi, h in enumerate(horizons):
            if periods:
                tgt_p = pd.Period(periods[hi], freq="M")
                steps = int(tgt_p.ordinal - anchor_p.ordinal)
            else:
                steps = h  # fallback: treat key as month count
            y = sm.shift(-steps)
            tgt_per = per_idx.shift(-steps)
            ts = pd.PeriodIndex(tgt_per.values, freq="M").to_timestamp(how="end")
            avail = pd.Series(pd.DatetimeIndex(ts).normalize(), index=sm.index)
            out[(asset, h)] = (y, avail, steps)
    return out


def horizon_step_map(bundle) -> dict[tuple[str, int], int]:
    """{(asset, h): observation steps} — daily: the key itself; monthly:
    steps resolved via observation_periods (opaque keys stay opaque)."""
    if bundle.target_frequency == "monthly":
        return {k: steps for k, (_y, _a, steps) in _monthly_steps(bundle).items()}
    return {(a, h): h for a in bundle.available_assets() for h in bundle.horizons}


def build_target(bundle, asset: str, horizon: int) -> tuple[pd.Series, pd.Series]:
    """(y, avail) for origin dates t. Trailing `horizon` rows are NaN."""
    if bundle.target_frequency == "monthly":
        y, avail, _ = _monthly_steps(bundle)[(asset, horizon)]
        return y, avail
    return _daily(bundle, asset, horizon, bundle.target_type)


def build_all_targets(bundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(y_frame, avail_frame), columns ``<asset>__h<horizon>``."""
    cols, avails = {}, {}
    if bundle.target_frequency == "monthly":
        for (asset, h), (series, avail, _steps) in _monthly_steps(bundle).items():
            cols[f"{asset}__h{h}"] = series
            avails[f"{asset}__h{h}"] = avail
        y_df, a_df = pd.DataFrame(cols), pd.DataFrame(avails)
        y_df.index = y_df.index.to_timestamp()
        a_df.index = a_df.index.to_timestamp()
        return y_df, a_df
    for asset in bundle.available_assets():
        for h in bundle.horizons:
            y, avail = _daily(bundle, asset, h, bundle.target_type)
            cols[f"{asset}__h{h}"] = y
            avails[f"{asset}__h{h}"] = avail
    return pd.DataFrame(cols), pd.DataFrame(avails)
