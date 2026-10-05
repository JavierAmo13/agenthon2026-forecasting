
"""Feature engineering: raw panels -> wide feature matrix.

Per asset: level, diffs/log-returns, rolling means/vols/z-scores, momentum,
relative vol (short vs long) and vol change. Cross-sectional stats stay
inside each panel — same panel means same units, so aggregates and pairwise
spreads never mix incompatible magnitudes.

v2 additions (plan 3.1):
  * ``__obs_age`` — days since the last real observation of the asset.
  * ``__is_miss`` — the cell was never observed on that calendar date.
    Filled data keeps its age: rellenar no equivale a observar de nuevo.
  * ``relvol``/``vol_chg`` — cheap relative-volatility measures.
  * ``xs__cross_range`` — same-panel max-min (curve-slope proxy when the
    panel holds comparable maturities).

Gap-aware: differences spanning holes >10x the median step are dropped.
Everything is truncated at as-of before any statistic is computed.
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


def _obs_age(s: pd.Series) -> pd.Series:
    """Days since the asset's last non-NaN observation on this calendar."""
    t = pd.Series(s.index, index=s.index)
    return (t - t.where(s.notna()).ffill()).dt.days


def _per_asset_features(wide: pd.DataFrame, kind: str, prefix: str) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    gaps = _gap_mask(wide.index)
    for asset in wide.columns:
        s = wide[asset].astype(float)
        base = f"{prefix}{asset}__"
        feats[base + "level"] = s
        vols: dict[int, pd.Series] = {}
        if kind == "return":
            for w in FAST:
                feats[base + f"ret_sum_{w}"] = s.rolling(w, min_periods=w).sum()
            for w in VOLS:
                vols[w] = s.rolling(w, min_periods=w).std()
                feats[base + f"vol_{w}"] = vols[w]
            for w in SLOW:
                feats[base + f"mean_{w}"] = s.rolling(w, min_periods=w).mean()
            for w in ZS:
                m = s.rolling(w, min_periods=w).mean()
                sd = s.rolling(w, min_periods=w).std()
                feats[base + f"zscore_{w}"] = (s - m) / sd.replace(0, np.nan)
        else:
            chg = np.log(s) if kind == "price" else s
            d1 = chg.diff(1).where(~gaps)
            for w in FAST:
                feats[base + ("logret_" if kind == "price" else "diff_") + str(w)] = d1 if w == 1 else chg.diff(w)
            for w in (5, 21):
                feats[base + f"momentum_{w}"] = chg.diff(w)
            for w in VOLS:
                vols[w] = d1.rolling(w, min_periods=w).std()
                feats[base + f"vol_{w}"] = vols[w]
            for w in SLOW:
                feats[base + f"mean_{w}"] = s.rolling(w, min_periods=w).mean()
            for w in ZS:
                m = s.rolling(w, min_periods=w).mean()
                sd = s.rolling(w, min_periods=w).std()
                feats[base + f"zscore_{w}"] = (s - m) / sd.replace(0, np.nan)
        # Relative vol and vol change: cheap regime hints, same units as vol.
        v5, v63 = vols.get(5), vols.get(63)
        if v5 is not None and v63 is not None:
            feats[base + "relvol_5_63"] = v5 / v63.replace(0, np.nan)
        v21 = vols.get(21)
        if v21 is not None and v63 is not None:
            feats[base + "vol_chg_21_63"] = v21 - v63
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
    feats[prefix + "xs__cross_range"] = wide.max(axis=1) - wide.min(axis=1)
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


def _staleness_on_union(wide: pd.DataFrame, prefix: str,
                        index: pd.DatetimeIndex) -> pd.DataFrame:
    """obs_age / is_miss recomputed on the union calendar: a panel whose
    calendar lacks a date must still report the age of its last obs."""
    feats: dict[str, pd.Series] = {}
    wr = wide.reindex(index)
    for a in wide.columns:
        s = wr[a]
        feats[f"{prefix}{a}__obs_age"] = _obs_age(s)
        feats[f"{prefix}{a}__is_miss"] = s.isna().astype(float)
    return pd.DataFrame(feats, index=index)


def build_features(panels: dict[str, pd.DataFrame], asof: str | pd.Timestamp) -> pd.DataFrame:
    """All panels -> single wide feature frame indexed by date, truncated at as-of.

    Columns: ``<panel>__<asset>__<feature>`` and ``<panel>__xs__...``.
    Panels are aligned on the union of dates (outer join, sorted).
    """
    asof = pd.Timestamp(asof)
    wides: dict[str, pd.DataFrame] = {}
    kinds: dict[str, str] = {}
    union: pd.DatetimeIndex = pd.DatetimeIndex([])
    for pid, df in panels.items():
        if _asset_col(df) is None or "value" not in df.columns:
            continue
        wide = to_wide(df[df["date"] <= asof])
        if wide.empty:
            continue
        wides[pid] = wide
        kinds[pid] = panel_kind(pid, wide)
        union = union.union(wide.index)
    if not wides:
        return pd.DataFrame()
    blocks = [_staleness_on_union(wide, f"{pid}__", union)
              for pid, wide in wides.items()]
    blocks += [b for pid, wide in wides.items()
               for b in (_per_asset_features(wide, kinds[pid], f"{pid}__"),
                         _cross_features(wide, kinds[pid], f"{pid}__"))]
    feats = pd.concat(blocks, axis=1).sort_index()
    return feats.loc[:, ~feats.columns.duplicated()]
