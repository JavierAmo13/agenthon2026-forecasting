
"""Dataset assembly: join feature matrix X with target y per (asset, horizon).

A row enters training only when its label is admissible at the cutoff:
y(t) exists AND avail(t) <= cutoff (the label's last consumed observation
was already published). Rows whose label is realized after the cutoff are
kept out of train — that is what makes --asof overrides leak-proof.

NaN policy: warmup rows where slow features cannot exist are dropped with
the y-NaN rows; X keeps NaN for models that handle it natively (HGB).
``feature_mask`` selects usable columns inside each train window: constant
or almost-empty columns are cut per fold, never globally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame          # features, index = origin dates
    y: pd.Series             # admissible target values aligned to X
    X_pred: pd.DataFrame     # last feature row -> prediction at as-of
    feature_names: list[str]
    asset: str
    horizon: int
    avail: pd.Series | None = None   # label availability date per origin

    def train_only(self) -> "Dataset":
        m = self.y.notna()
        return Dataset(self.X[m], self.y[m], self.X_pred, self.feature_names,
                       self.asset, self.horizon, self.avail[m] if self.avail is not None else None)


def feature_mask(X: pd.DataFrame, min_nonnull: float = 0.3) -> pd.Series:
    """Column selection fitted on a train slice only: keep columns that are
    populated enough and not constant there."""
    return (X.notna().mean() >= min_nonnull) & (X.nunique(dropna=True) > 1)


def build_dataset(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    avail: pd.DataFrame,
    asset: str,
    horizon: int,
    *,
    cutoff: str | pd.Timestamp,
    nan_policy: str = "drop_warmup",
    monthly: bool = False,
) -> Dataset:
    """X = features, y = targets[f'{asset}__h{horizon}'], filtered by availability."""
    col = f"{asset}__h{horizon}"
    if col not in targets.columns:
        raise KeyError(f"target column {col!r} missing")
    cutoff = pd.Timestamp(cutoff)
    y = targets[col]
    av = avail[col] if col in avail.columns else pd.Series(pd.NaT, index=y.index)

    has_avail = col in avail.columns
    if monthly:
        fm = features.copy()
        fm.index = pd.DatetimeIndex(fm.index).to_period("M")
        fm = fm.groupby(level=0).last()
        ym, am = y.copy(), av.copy()
        ym.index = pd.DatetimeIndex(ym.index).to_period("M")
        am.index = pd.DatetimeIndex(am.index).to_period("M")
        idx = fm.index.union(ym.dropna().index).sort_values()
        X, y, av = fm.reindex(idx), ym.reindex(idx), am.reindex(idx)
    else:
        idx = features.index.union(y.dropna().index).sort_values()
        X, y, av = features.reindex(idx), y.reindex(idx), av.reindex(idx)

    # Prediction rows: last feature row, at/near as-of by construction.
    X_pred = X.iloc[[-1]].copy()

    # Admissible train rows: label exists and was available by the cutoff.
    # avail values are timestamps (period-end for monthly) or NaT -> exclude.
    in_time = av.le(cutoff) if has_avail else pd.Series(True, index=y.index)
    train_mask = y.notna() & in_time.fillna(False)
    X_tr, y_tr = X[train_mask], y[train_mask]
    av_tr = av[train_mask]

    if nan_policy == "drop":
        good = X_tr.notna().all(axis=1)
        X_tr, y_tr, av_tr = X_tr[good], y_tr[good], av_tr[good]
    elif nan_policy == "impute":
        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med)
        X_pred = X_pred.fillna(med)

    return Dataset(X_tr, y_tr, X_pred, list(X.columns), asset, horizon, av_tr)


def build_all(features: pd.DataFrame, targets: pd.DataFrame, avail: pd.DataFrame,
              bundle, *, nan_policy: str = "drop_warmup") -> dict[tuple[str, int], Dataset]:
    """One Dataset per (asset, horizon) declared by the unit."""
    monthly = bundle.target_frequency == "monthly"
    out = {}
    for a in bundle.target_assets:
        for h in bundle.horizons:
            out[(a, h)] = build_dataset(features, targets, avail, a, h,
                                        cutoff=bundle.asof,
                                        nan_policy=nan_policy, monthly=monthly)
    return out
