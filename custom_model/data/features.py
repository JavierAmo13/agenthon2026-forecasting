"""Feature engineering: raw panels -> wide feature matrix.

Per asset: level, diffs, log-returns (positive series), rolling means,
z-scores, momentum, volatility. Cross-sectional: spreads / log-ratios
between assets plus cross-mean/std/min/max/zscore.

Gap-aware: differences spanning holes >10x median step are dropped
(em_transfer panels ship an early window + a single as-of row).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ASSET_COLS = ("asset", "asset_id")

FAST = (1, 5, 21)
SLOW = (5, 21, 63)
VOLS = (5, 21, 63)
ZS = (21, 63)


def _asset_col(df: pd.DataFrame) -> str | None:
    return next((c for c in _ASSET_COLS if c in df.columns), None)


def panel_kind(panel_id: str, wide: pd.DataFrame) -> str:
    """'return' (rows are periodic returns), 'price' (positive quotes -> log
    space makes sense), or 'level' (yields/macro -> plain diffs)."""
    name = panel_id.lower()
    if "factor" in name:
        return "return"
    if ("fx" in name or "em_" in name or "transfer" in name) and (wide > 0).all().all():
        return "price"
    if (wide > 0).all().all() and ("fx" in name or "em_" in name):
        return "price"
    return "level"


def _gap_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True where the step to the previous observation is a data hole."""
    if len(index) < 3:
        return pd.Series(False, index=index)
    step = pd.Series(index, index=index).diff().dt.days
    med = step.median()
    if not np.isfinite(med) or med <= 0:
        return pd.Series(False, index=index)
    return step > max(med * 10.0, 5.0)


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    col = _asset_col(df)
    wide = df.pivot_table(index="date", columns=col, values="value", aggfunc="last")
    return wide.sort_index()


def _per_asset_features(wide: pd.DataFrame, kind: str, prefix: str) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    gaps = _gap_mask(wide.index)
    for asset in wide.columns:
        s = wide[asset].astype(float)
        base = f"{prefix}{asset}__"
        feats[base + "level"] = s
        if kind == "return":
            for w in FAST:
                feats[base + f"ret_sum_{w}"] = s.rolling(w, min_periods=w).sum()
            for w in VOLS:
                feats[base + f"vol_{w}"] = s.rolling(w, min_periods=w).std()
            for w in SLOW:
                feats[base + f"mean_{w}"] = s.rolling(w, min_periods=w).mean()
            for w in ZS:
                m = s.rolling(w, min_periods=w).mean()
                sd = s.rolling(w, min_periods=w).std()
                feats[base + f"zscore_{w}"] = (s - m) / sd.replace(0, np.nan)
            continue
        chg = np.log(s) if kind == "price" else s
        d1 = chg.diff(1).where(~gaps)
        for w in FAST:
            d = chg.diff(w)
            if w == 1:
                d = d1
            feats[base + ("logret_" if kind == "price" else "diff_") + str(w)] = d
        for w in (5, 21):
            feats[base + f"momentum_{w}"] = chg.diff(w)
        for w in VOLS:
            feats[base + f"vol_{w}"] = d1.rolling(w, min_periods=w).std()
        for w in SLOW:
            feats[base + f"mean_{w}"] = s.rolling(w, min_periods=w).mean()
        for w in ZS:
            m = s.rolling(w, min_periods=w).mean()
            sd = s.rolling(w, min_periods=w).std()
            feats[base + f"zscore_{w}"] = (s - m) / sd.replace(0, np.nan)
    return pd.DataFrame(feats)


def _cross_features(wide: pd.DataFrame, kind: str, prefix: str) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    cols = list(wide.columns)
    xs_mean = wide.mean(axis=1)
    xs_std = wide.std(axis=1)
    feats[prefix + "xs__cross_mean"] = xs_mean
    feats[prefix + "xs__cross_std"] = xs_std
    feats[prefix + "xs__cross_min"] = wide.min(axis=1)
    feats[prefix + "xs__cross_max"] = wide.max(axis=1)
    for a in cols:
        feats[f"{prefix}xs__{a}__cross_zscore"] = (wide[a] - xs_mean) / xs_std.replace(0, np.nan)
    if len(cols) > 1 and len(cols) <= 12:
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if kind == "price":
                    feats[f"{prefix}xs__{a}_vs_{b}__logratio"] = np.log(wide[a] / wide[b])
                else:
                    feats[f"{prefix}xs__{a}_vs_{b}__spread"] = wide[a] - wide[b]
    return pd.DataFrame(feats)


def build_features(panels: dict[str, pd.DataFrame], asof: str | pd.Timestamp) -> pd.DataFrame:
    """All panels -> single wide feature frame indexed by date, truncated at as-of.

    Columns: ``<panel>__<asset>__<feature>`` and ``<panel>__xs__...``.
    Panels are aligned on the union of dates (outer join, sorted).
    """
    asof = pd.Timestamp(asof)
    blocks = []
    for pid, df in panels.items():
        if _asset_col(df) is None or "value" not in df.columns:
            continue
        wide = to_wide(df[df["date"] <= asof])
        if wide.empty:
            continue
        kind = panel_kind(pid, wide)
        blocks.append(_per_asset_features(wide, kind, f"{pid}__"))
        blocks.append(_cross_features(wide, kind, f"{pid}__"))
    if not blocks:
        return pd.DataFrame()
    feats = pd.concat(blocks, axis=1).sort_index()
    return feats.loc[:, ~feats.columns.duplicated()]