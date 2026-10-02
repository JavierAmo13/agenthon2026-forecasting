"""Point models: persistence, Ridge, gradient boosting.

XGBoost is not vendored offline; HistGradientBoostingRegressor is the
equivalent non-linear learner here (and handles NaN natively). Swap in
xgboost.XGBRegressor inside fit_model if the image ships it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class LatentRegressor:
    """Base model fed raw features + PCA latent state z_t (with lags).

    The PCA lives inside the estimator so walk_forward_residuals refits it on
    each fold's training window — never on data ahead of the fold origin.
    """

    def __init__(self, base: str = "hgb", n_components: int = 15,
                 lags: tuple[int, ...] = (1, 5, 21), seed: int = 0,
                 max_iter: int = 300):
        self.base = base
        self.n_components = n_components
        self.lags = tuple(lags)
        self.seed = seed
        self.max_iter = max_iter

    def _lagged(self, z: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in z.columns:
            cols[c] = z[c]
            for lag in self.lags:
                cols[f"{c}_lag{lag}"] = z[c].shift(lag)
        return pd.DataFrame(cols, index=z.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LatentRegressor":
        from .latent import LatentState
        self.latent_ = LatentState(self.n_components, self.lags)
        Z = self.latent_.fit_transform(X)
        z_cols = [c for c in Z.columns if "_lag" not in c]
        self.z_tail_ = Z[z_cols].tail(max(self.lags))
        self.mdl_ = fit_model(pd.concat([X, Z], axis=1), y, self.base,
                              self.seed, self.max_iter)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z_now = self.latent_.transform(X)
        hist = pd.concat([self.z_tail_, z_now])
        Zp = self._lagged(hist).iloc[-len(z_now):]
        Zp.index = z_now.index
        return predict(self.mdl_, pd.concat([X, Zp], axis=1))


def fit_model(X: pd.DataFrame, y: pd.Series, kind: str = "hgb", seed: int = 0,
              max_iter: int = 300):
    """Train one point model. Returns a fitted estimator-like object."""
    kind = kind.lower()
    if kind == "persistence":
        from .probabilistic import PersistenceModel
        return PersistenceModel().fit(X, y)
    if kind.endswith("+latent"):
        base = kind[: -len("+latent")] or "hgb"
        return LatentRegressor(base, seed=seed, max_iter=max_iter).fit(X, y)
    if kind == "ridge":
        med = X.median(numeric_only=True).fillna(0.0)  # all-NaN cols -> 0
        mdl = Ridge(alpha=1.0)
        mdl.fit(X.fillna(med), y)
        mdl._impute = med
        return mdl
    if kind in ("hgb", "xgboost", "gbm"):
        mdl = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=0.06,
            max_depth=6,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        mdl.fit(X, y)  # HGB accepts NaN natively
        return mdl
    raise ValueError(f"unknown model kind {kind!r}")


def predict(model, X: pd.DataFrame) -> np.ndarray:
    med = getattr(model, "_impute", None)
    if med is not None:
        X = X.fillna(med).fillna(0.0)
    return np.asarray(model.predict(X), dtype=float)


def walk_forward_residuals(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str = "hgb",
    *,
    n_folds: int = 4,
    min_train: int | None = None,
    embargo: int = 0,
    seed: int = 0,
    max_iter: int = 300,
) -> pd.Series:
    """Out-of-fold residuals y - yhat on the recent part of the sample.

    Time-ordered expanding-window splits; only folds past ``min_train``
    produce residuals. ``embargo`` rows at the end of each training window are
    dropped so targets realized inside the test window cannot leak into fit.
    Returns a Series aligned to X's index (NaN where no OOF prediction exists).
    """
    n = len(X)
    if min_train is None:
        min_train = max(int(n * 0.5), 60)
    resid = pd.Series(np.nan, index=X.index, dtype=float)
    if n <= min_train + 5:
        mdl = fit_model(X, y, kind, seed, max_iter=max_iter)
        return y - pd.Series(predict(mdl, X), index=X.index)
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        tr_lo, tr_hi = 0, max(0, edges[i] - embargo)
        te_lo, te_hi = edges[i], edges[i + 1]
        if te_lo <= tr_lo or te_hi <= te_lo:
            continue
        mdl = fit_model(X.iloc[tr_lo:tr_hi], y.iloc[tr_lo:tr_hi], kind, seed + i,
                        max_iter=max_iter)
        pred = predict(mdl, X.iloc[te_lo:te_hi])
        resid.iloc[te_lo:te_hi] = (y.iloc[te_lo:te_hi].to_numpy() - pred)
    return resid