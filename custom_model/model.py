
"""Point models: baselines, scaled Ridge, gradient boosting (+latent PCA).

v2 (plan 3.3):
  * Ridge is a chain median-impute -> StandardScaler -> Ridge, fitted only
    inside each train window; alpha picked on a temporal holdout tail.
  * Baselines are honest and separate: ``persistence`` = target mean,
    ``last`` = last observed level (level targets) or a shrunken drift
    (return targets). They are scored like any other member.
  * HGB early stopping uses the temporal tail of train via warm_start
    chunks — never an implicit random split.
  * ``+latent`` keeps the PCA inside the estimator, refit per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MeanModel:
    """Target-mean baseline (the historical 'persistence' of the doc)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanModel":
        self.mu_ = float(np.nanmean(y)) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), getattr(self, "mu_", 0.0))


class LastLevelModel:
    """Last-observation baseline. For level targets predicts the asset's own
    current level column; for return targets a drift shrunken to 50% of the
    in-sample mean (regularized, per plan 3.3)."""

    def __init__(self, asset: str, target_type: str = "level"):
        self.asset, self.target_type = asset, target_type

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LastLevelModel":
        self.drift_ = 0.5 * float(np.nanmean(y)) if len(y) else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_type == "log_return":
            return np.full(len(X), self.drift_)
        col = next((c for c in X.columns if c.endswith(f"__{self.asset}__level")
                    or c.endswith(f"{self.asset}__level")), None)
        if col is None:
            return np.full(len(X), self.drift_)
        return X[col].astype(float).to_numpy()


def _fit_ridge(X: pd.DataFrame, y: pd.Series, seed: int):
    """Median-impute -> scale -> Ridge; alpha grid validated on the last 20%
    of train (temporal, not random). Refit the winner on the whole window."""
    n = len(X)
    n_val = max(int(n * 0.2), 20)
    alphas = (0.3, 1.0, 3.0, 10.0)
    best_alpha = 1.0
    if n > n_val + 60:
        Xc, yc = X.iloc[:-n_val], y.iloc[:-n_val]
        Xv, yv = X.iloc[-n_val:], y.iloc[-n_val:]
        best = np.inf
        for al in alphas:
            m = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(), Ridge(alpha=al))
            m.fit(Xc, yc)
            sc = mean_squared_error(yv, m.predict(Xv))
            if sc < best:
                best, best_alpha = sc, al
    mdl = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler(), Ridge(alpha=best_alpha))
    mdl.fit(X, y)
    mdl._alpha = best_alpha
    return mdl


def _fit_hgb(X: pd.DataFrame, y: pd.Series, seed: int, max_iter: int):
    """HGB with temporal early stopping: grow trees in warm_start chunks on
    the core window, score the last-15% tail, refit full train at best_iter."""
    n = len(X)
    n_val = max(int(n * 0.15), 20)
    params = dict(learning_rate=0.06, max_depth=6, l2_regularization=1.0,
                  random_state=seed)
    if n <= n_val + 80:
        m = HistGradientBoostingRegressor(max_iter=max_iter, early_stopping=False,
                                          **params)
        m.fit(X, y)  # HGB accepts NaN natively
        m._best_iter = max_iter
        return m
    Xc, yc = X.iloc[:-n_val], y.iloc[:-n_val]
    Xv, yv = X.iloc[-n_val:], y.iloc[-n_val:]
    m = HistGradientBoostingRegressor(max_iter=1, warm_start=True,
                                      early_stopping=False, **params)
    best_iter, best_loss = max_iter, np.inf
    done = 0
    while done < max_iter:
        step = min(50, max_iter - done)
        m.set_params(max_iter=done + step)
        m.fit(Xc, yc)
        loss = mean_squared_error(yv, m.predict(Xv))
        if loss < best_loss - 1e-9:
            best_loss, best_iter = loss, done + step
        done += step
    final = HistGradientBoostingRegressor(max_iter=best_iter,
                                          early_stopping=False, **params)
    final.fit(X, y)
    final._best_iter = best_iter
    return final


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
              max_iter: int = 300, asset: str = "", target_type: str = "level"):
    """Train one point model. Returns a fitted estimator-like object."""
    kind = kind.lower()
    if kind == "persistence":
        return MeanModel().fit(X, y)
    if kind == "last":
        return LastLevelModel(asset, target_type).fit(X, y)
    if kind.endswith("+latent"):
        base = kind[: -len("+latent")] or "hgb"
        return LatentRegressor(base, seed=seed, max_iter=max_iter).fit(X, y)
    if kind == "ridge":
        return _fit_ridge(X, y, seed)
    if kind in ("hgb", "xgboost", "gbm"):
        return _fit_hgb(X, y, seed, max_iter)
    raise ValueError(f"unknown model kind {kind!r}")


def predict(model, X: pd.DataFrame) -> np.ndarray:
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
    asset: str = "",
    target_type: str = "level",
) -> pd.Series:
    """Out-of-fold residuals y - yhat on the recent part of the sample.

    Time-ordered expanding-window splits with an ``embargo`` of rows between
    train end and test start so targets realized inside the test window never
    enter fit. Feature selection runs inside each fold's train slice.
    Returns a Series aligned to X's index (NaN where no OOF prediction exists).
    """
    from .data.dataset import feature_mask

    n = len(X)
    if min_train is None:
        min_train = max(int(n * 0.5), 60)
    resid = pd.Series(np.nan, index=X.index, dtype=float)
    if n <= min_train + 5:
        m = feature_mask(X)
        mdl = fit_model(X.loc[:, m], y, kind, seed, max_iter=max_iter,
                        asset=asset, target_type=target_type)
        return y - pd.Series(predict(mdl, X.loc[:, m]), index=X.index)
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        tr_lo, tr_hi = 0, max(0, edges[i] - embargo)
        te_lo, te_hi = edges[i], edges[i + 1]
        if te_lo <= tr_lo or te_hi <= te_lo:
            continue
        m = feature_mask(X.iloc[tr_lo:tr_hi])
        mdl = fit_model(X.iloc[tr_lo:tr_hi].loc[:, m], y.iloc[tr_lo:tr_hi],
                        kind, seed + i, max_iter=max_iter,
                        asset=asset, target_type=target_type)
        pred = predict(mdl, X.iloc[te_lo:te_hi].loc[:, m])
        resid.iloc[te_lo:te_hi] = (y.iloc[te_lo:te_hi].to_numpy() - pred)
    return resid
