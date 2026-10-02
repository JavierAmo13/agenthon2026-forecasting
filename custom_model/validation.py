"""Walk-forward validation with empirical CRPS.

CRPS of an empirical distribution from samples s_1..s_n at realized y:
    CRPS = mean|s - y| - 0.5 * mean|s - s'|
Random train/test splits are never used (time series).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crps_empirical(samples: np.ndarray, y: float) -> float:
    """Energy form: E|X-y| - 0.5 E|X-X'|. samples: 1-D array of draws."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan
    term1 = np.mean(np.abs(s - y))
    d = np.abs(s[:, None] - s[None, :])
    term2 = 0.5 * np.mean(d)
    return float(term1 - term2)


def walk_forward_splits(n: int, *, n_folds: int = 4, min_train_frac: float = 0.5,
                        embargo: int = 0):
    """Yield (train_idx, test_idx) expanding-window splits."""
    min_train = max(int(n * min_train_frac), 60)
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        te_lo, te_hi = edges[i], edges[i + 1]
        tr_hi = max(0, te_lo - embargo)
        if te_hi <= te_lo or tr_hi <= 30:
            continue
        yield np.arange(0, tr_hi), np.arange(te_lo, te_hi)