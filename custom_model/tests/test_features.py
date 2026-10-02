"""Feature-building tests: shapes, naming, no-leakage at as-of.

Runnable both under pytest and directly: ``python test_features.py``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

UNITS = pathlib.Path(__file__).resolve().parents[2] / "units"
EXAMPLE = UNITS / "t2-F3-boj-ust-channel-2023"


def test_feature_matrix_shape_and_names():
    if not EXAMPLE.exists():
        return
    import pandas as pd

    from custom_model.data.loader import load_unit
    from custom_model.data.features import build_features

    b = load_unit(EXAMPLE)
    X = build_features(b.panels, b.asof)
    assert not X.empty
    assert X.index.max() <= pd.Timestamp(b.asof)
    cols = set(X.columns)
    # per-asset features exist for both panels
    assert any(c.startswith("g10_fx_daily__AUD__") for c in cols)
    assert any(c.startswith("rates_daily__UST_10Y__") for c in cols)
    # cross-sectional block present
    assert any("__xs__cross_mean" in c for c in cols)
    # feature families per the recipe
    assert "rates_daily__UST_10Y__level" in cols
    assert "rates_daily__UST_10Y__diff_21" in cols
    assert "g10_fx_daily__JPY__logret_1" in cols
    assert "g10_fx_daily__JPY__vol_63" in cols
    # duplicates removed
    assert len(X.columns) == len(set(X.columns))


def test_no_leakage_from_future():
    """Truncating history must not change features at earlier dates."""
    if not EXAMPLE.exists():
        return
    import numpy as np
    import pandas as pd

    from custom_model.data.loader import load_unit
    from custom_model.data.features import build_features

    b = load_unit(EXAMPLE)
    full = build_features(b.panels, b.asof)
    cut = pd.Timestamp(b.asof) - pd.Timedelta(days=400)
    part = build_features(b.panels, cut)
    common = part.index.intersection(full.index)
    assert len(common) > 100
    a = full.loc[common].astype(float)
    c = part.loc[common].astype(float)
    # rows identical apart from NaN/edge differences
    diff = (a - c).abs().max().max()
    assert np.isfinite(diff) and diff < 1e-8


if __name__ == "__main__":
    for name, fn in sorted(vars().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all feature tests passed")