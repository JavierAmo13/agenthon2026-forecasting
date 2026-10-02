"""Target construction: define what "the future value" means.

Per (asset, horizon) one supervised target:
  level      -> value at t + h observations (shift -h on the asset's own index)
  log_return -> sum of log(1 + r) over the next h observations
Monthly-frequency targets map horizon keys to monthly steps supplied by the
unit's observation_periods (resolved via the toolkit helper when available).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ASSET_COLS = ("asset", "asset_id")


def _find_series(bundle, asset: str) -> pd.Series:
    return bundle.asset_series(asset, upto_asof=False)


def _forward_logret(r: pd.Series, h: int) -> pd.Series:
    lr = np.log1p(r.astype(float))
    return lr.rolling(h, min_periods=h).sum().shift(-h)


def _daily_target(s: pd.Series, h: int, target_type: str) -> pd.Series:
    if target_type == "log_return":
        return _forward_logret(s, h)
    return s.astype(float).shift(-h)


def monthly_targets(bundle) -> dict[tuple[str, int], tuple[pd.Series, int]]:
    """Monthly-frequency units -> {(asset, horizon_key): (target_series, n_month_steps)}.

    The target series is indexed by month-period timestamps; its values are the
    asset's level `steps` months ahead, where steps come from the unit's
    explicit observation_periods (preferred) or the toolkit helper.
    """
    spec_tgt = bundle.spec.get("targets", {})
    periods = spec_tgt.get("observation_periods")
    horizons = bundle.horizons
    out: dict[tuple[str, int], tuple[pd.Series, int]] = {}
    for asset in bundle.target_assets:
        s = _find_series(bundle, asset)
        s = s[s.index <= pd.Timestamp(bundle.asof)]
        sm = s.copy()
        sm.index = pd.DatetimeIndex(sm.index).to_period("M")
        sm = sm[~sm.index.duplicated(keep="last")]
        anchor_p = sm.index[-1]
        for hi, h in enumerate(horizons):
            if periods:
                tgt_p = pd.Period(periods[hi], freq="M")
                steps = int(tgt_p.ordinal - anchor_p.ordinal)
            else:
                steps = h  # fallback: treat key as month count
            out[(asset, h)] = (sm.shift(-steps), steps)
    return out


def build_target(bundle, asset: str, horizon: int) -> pd.Series:
    """y(t): the realized target for origin date t. Last `horizon` rows are NaN."""
    if bundle.target_frequency == "monthly":
        return monthly_targets(bundle)[(asset, horizon)][0]
    s = _find_series(bundle, asset)
    return _daily_target(s, horizon, bundle.target_type)


def build_all_targets(bundle) -> pd.DataFrame:
    """Columns ``<asset>__h<horizon>`` for every declared combination."""
    cols = {}
    if bundle.target_frequency == "monthly":
        for (asset, h), (series, _steps) in monthly_targets(bundle).items():
            cols[f"{asset}__h{h}"] = series
        df = pd.DataFrame(cols)
        df.index = df.index.to_timestamp()
        return df
    for asset in bundle.target_assets:
        s = _find_series(bundle, asset)
        for h in bundle.horizons:
            cols[f"{asset}__h{h}"] = _daily_target(s, h, bundle.target_type)
    return pd.DataFrame(cols)