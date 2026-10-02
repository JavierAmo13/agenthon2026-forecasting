"""Model-layer tests on synthetic data (fast, no units needed).

Runnable both under pytest and directly: ``python test_model.py``.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def _toy(n=400, p=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, p)), index=idx,
                     columns=[f"f{i}" for i in range(p)])
    X.iloc[:10, 5] = np.nan  # warmup-style NaN
    y = X["f0"].shift(-21) + rng.normal(scale=0.1, size=n)
    y = y.iloc[:-21]
    return X.iloc[:-21], y


def test_fit_predict_finite():
    from custom_model.model import fit_model, predict

    for kind in ["persistence", "ridge", "hgb", "hgb+latent"]:
        X, y = _toy()
        m = fit_model(X, y, kind, seed=1, max_iter=60)
        p = predict(m, X.iloc[[-1]])
        assert p.shape == (1,), kind
        assert np.isfinite(p[0]), kind


def test_walk_forward_residuals_no_leak():
    """OOF residuals only exist on rows past min_train."""
    from custom_model.model import walk_forward_residuals

    for kind in ["ridge", "hgb+latent"]:
        X, y = _toy()
        r = walk_forward_residuals(X, y, kind, n_folds=2, min_train=200,
                                   embargo=21, seed=1, max_iter=60)
        assert r.iloc[:200].isna().all(), kind
        assert r.iloc[200:].notna().all(), kind


def test_joint_samples_shape_and_corr():
    from custom_model.joint import joint_samples

    rng = np.random.default_rng(0)
    k = 6
    resid = pd.DataFrame(rng.normal(size=(500, k)) @ rng.normal(size=(k, k)))
    mus = np.arange(k, dtype=float)
    models = [{"mu": 0.0, "sigma": 1.0, "nu": np.inf,
               "emp": np.sort(rng.normal(size=300))}] * k
    s = joint_samples(list(range(k)), mus, models, resid, 400, seed=0)
    assert s.shape == (400, k)
    assert np.isfinite(s).all()
    # correlated residuals -> correlated draws (weak check: not identity)
    c = np.corrcoef(s.T)
    assert np.abs(c[np.triu_indices(k, 1)]).mean() > 0.02


def test_ensemble_pooling_shape():
    from custom_model.ensemble import pool_draws, inverse_score_weights

    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 2, 1))
    b = rng.normal(1.0, size=(300, 2, 1))
    out = pool_draws([a, b], [0.5, 0.5], 400, seed=0)
    assert out.shape == (400, 2, 1)
    w = inverse_score_weights([1.0, 2.0])
    assert abs(w[0] - 2 / 3) < 1e-9 and abs(sum(w) - 1) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(vars().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all model tests passed")