
"""End-to-end smoke: synthetic bundle -> run_bundle -> correct draw shape,
no NaNs, no text branch. Uses only the cheap members so it runs in seconds."""

import numpy as np
import pandas as pd

from custom_model.data.loader import DataBundle
from custom_model.forecast import run_bundle


def _synthetic_bundle():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    frames = {}
    for pid, assets in (("fx", ["EURUSD", "USDJPY"]), ("yields", ["UST2Y"])):
        lvl = {a: 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx))))
               for a in assets}
        long = pd.DataFrame(lvl, index=idx).reset_index(names="date").melt(
            id_vars="date", var_name="asset", value_name="value")
        frames[pid] = long
    return DataBundle(
        unit_dir=None, card_id="synthetic", card_family="T2-F1",
        asof=str(idx[-1].date()), target_assets=["EURUSD", "USDJPY", "UST2Y"],
        horizons=[5, 21], target_type="level", target_frequency="daily",
        n_draws_min=200, panels=frames, panel_specs={}, texts=[],
        spec={}, card={})


def test_run_bundle_shapes_and_support():
    b = _synthetic_bundle()
    n = 200
    samples, stats = run_bundle(b, n_draws=n, seed=1, use_text=False,
                                ensemble=("persistence", "ridge"))
    assert samples.shape == (n, 3, 2)
    assert np.isfinite(samples).all()
    # level forecasts must sit near the last observed level, not at zero
    for ai, a in enumerate(b.target_assets):
        last = float(b.asset_series(a).iloc[-1])
        med = np.median(samples[:, ai, -1])
        assert abs(med - last) / last < 0.5
    assert not stats.get("fallback")


def test_missing_asset_routes_to_fallback():
    b = _synthetic_bundle()
    b.target_assets.append("GHOST")
    b.missing_assets = ["GHOST"]
    samples, stats = run_bundle(b, n_draws=200, seed=2, use_text=False,
                                ensemble=("persistence",))
    assert samples.shape == (200, 4, 2)
    assert stats["missing_assets"] == ["GHOST"]
