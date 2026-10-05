
"""Target contracts: log_return = sum of log(1+r); labels unavailable past
the cutoff never enter train (the --asof trap, fixed)."""

import numpy as np
import pandas as pd

from custom_model.data.loader import DataBundle, TextDoc
from custom_model.data.targets import build_target, build_all_targets
from custom_model.data.dataset import build_dataset


def _bundle(series_vals, asof="2024-12-31", target_type="level"):
    idx = pd.bdate_range("2024-01-01", periods=len(series_vals))
    df = pd.DataFrame({"date": idx, "asset": "A", "value": series_vals})
    return DataBundle(
        unit_dir=None, card_id="u", card_family="", asof=asof,
        target_assets=["A"], horizons=[5], target_type=target_type,
        target_frequency="daily", n_draws_min=200,
        panels={"p": df}, panel_specs={}, texts=[], spec={}, card={})


def test_log_return_target_is_forward_sum():
    r = [0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.01, 0.0, -0.005, 0.004]
    b = _bundle(r, target_type="log_return")
    y, avail = build_target(b, "A", 3)
    expect = np.log1p(pd.Series(r)).rolling(3).sum().shift(-3)
    np.testing.assert_allclose(y.to_numpy(), expect.to_numpy())
    # availability = date of the last consumed observation (t+h)
    assert str(avail.iloc[0].date()) == str(pd.bdate_range("2024-01-01", periods=11)[3].date())


def test_availability_contract_drops_future_labels():
    # series extends past asof: labels realized after cutoff must not train
    vals = np.linspace(100, 130, 80)
    b = _bundle(vals, asof="2024-03-15")          # mid-sample cutoff
    y_df, av_df = build_all_targets(b)
    feats = pd.DataFrame({"p__A__level": b.asset_series("A", upto_asof=False)},
                         index=b.asset_series("A", upto_asof=False).index)
    ds = build_dataset(feats, y_df, av_df, "A", 5, cutoff=b.asof)
    assert ds.avail.max() <= pd.Timestamp(b.asof)
    assert len(ds.y) < (pd.Timestamp(b.asof) - pd.Timestamp("2024-01-01")).days
