"""Latent market state: PCA compression of the feature block.

X -> standardized -> PCA -> z_t factors (plus their lags), appended to the
feature matrix so the model sees compressed cross-asset state, not only the
raw columns. Fit strictly on data <= the training window to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class LatentState:
    def __init__(self, n_components: int = 15, lags: tuple[int, ...] = (1, 5, 21)):
        self.n_components = n_components
        self.lags = lags

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X.copy()
        med = Xn.median(numeric_only=True)
        self.med_ = med
        Z = Xn.fillna(med).to_numpy(dtype=float)
        self.mu_ = np.nanmean(Z, axis=0)
        self.sd_ = np.nanstd(Z, axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Zs = np.nan_to_num((Z - self.mu_) / self.sd_, nan=0.0)
        k = int(min(self.n_components, Zs.shape[1], max(len(Zs) - 1, 1)))
        self.pca_ = PCA(n_components=k)
        comp = self.pca_.fit_transform(Zs)
        cols = {}
        for i in range(k):
            zt = pd.Series(comp[:, i], index=X.index, name=f"z{i}")
            cols[f"z{i}"] = zt
            for lag in self.lags:
                cols[f"z{i}_lag{lag}"] = zt.shift(lag)
        return pd.DataFrame(cols)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.fillna(self.med_).to_numpy(dtype=float)
        Zs = np.nan_to_num((Z - self.mu_) / self.sd_, nan=0.0)
        comp = self.pca_.transform(Zs)
        return pd.DataFrame(comp, index=X.index, columns=[f"z{i}" for i in range(comp.shape[1])])