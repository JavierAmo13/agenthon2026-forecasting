"""Dataset assembly: join feature matrix X with target y per (asset, horizon).

NaN policy (doc sect.21): warmup rows where slow features cannot exist are
dropped (structural, not leakage); tail rows where the target is not yet
realized are dropped for training but the last valid feature row is kept
for prediction. Optionally median-impute remaining holes (fit on train only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame          # features, index = origin dates
    y: pd.Series             # target values aligned to X
    X_pred: pd.DataFrame     # last-row features for prediction at as-of
    feature_names: list[str]
    asset: str
    horizon: int

    def train_only(self) -> "Dataset":
        m = self.y.notna()
        return Dataset(self.X[m], self.y[m], self.X_pred, self.feature_names,
                       self.asset, self.horizon)


def build_dataset(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    asset: str,
    horizon: int,
    *,
    nan_policy: str = "drop_warmup",
    monthly: bool = False,
) -> Dataset:
    """X = features (all), y = targets[f'{asset}__h{horizon}'].

    ``nan_policy``:
      * ``drop_warmup`` — drop rows where y is NaN, keep X NaN for models
        that handle it (HGB). For strict models use ``impute``.
      * ``impute`` — additionally median-impute X (median over training rows).
      * ``drop`` — drop rows with any NaN in X or y.

    ``monthly``: when True the feature frame is collapsed to month-ends
    (last available daily row per calendar month) and aligned to the
    monthly target index by period. Use for ``target_frequency == 'monthly'``
    units where the daily feature calendar never contains the month-start
    target timestamps.
    """
    col = f"{asset}__h{horizon}"
    if col not in targets.columns:
        raise KeyError(f"target column {col!r} missing")
    y = targets[col]
    if monthly:
        fm = features.copy()
        fm.index = pd.DatetimeIndex(fm.index).to_period("M")
        fm = fm.groupby(level=0).last()
        ym = y.copy()
        ym.index = pd.DatetimeIndex(ym.index).to_period("M")
        idx = fm.index.union(ym.dropna().index).sort_values()
        X = fm.reindex(idx)
        y = ym.reindex(idx)
    else:
        idx = features.index.union(y.dropna().index).sort_values()
        X = features.reindex(idx)
        y = y.reindex(idx)

    # Prediction rows: last feature row(s) whose origin is at/near as-of.
    X_pred = X.iloc[[-1]].copy()

    train_mask = y.notna()
    X_tr, y_tr = X[train_mask], y[train_mask]

    if nan_policy == "drop":
        good = X_tr.notna().all(axis=1)
        X_tr, y_tr = X_tr[good], y_tr[good]
    elif nan_policy == "impute":
        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med)
        X_pred = X_pred.fillna(med)

    return Dataset(X_tr, y_tr, X_pred, list(X.columns), asset, horizon)


def build_all(features: pd.DataFrame, targets: pd.DataFrame,
              bundle, *, nan_policy: str = "drop_warmup") -> dict[tuple[str, int], Dataset]:
    """One Dataset per (asset, horizon) declared by the unit."""
    monthly = bundle.target_frequency == "monthly"
    out = {}
    for a in bundle.target_assets:
        for h in bundle.horizons:
            out[(a, h)] = build_dataset(features, targets, a, h,
                                        nan_policy=nan_policy, monthly=monthly)
    return out