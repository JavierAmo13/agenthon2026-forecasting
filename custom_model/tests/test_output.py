
"""Output contract: the scorer reads exactly (draw, asset, horizon, value)
with draws as integers 0..n-1, one row per (draw, asset, horizon), all
values finite, and meta declaring assets/horizons in the declared order.
Exactly three deliverables in the output dir."""

import json
import tempfile
import pathlib

import numpy as np
import pandas as pd

from custom_model.forecast import run_bundle, _write_deliverables, OUT_COLS
from custom_model.tests.test_smoke import _synthetic_bundle


def _run(tmp: pathlib.Path):
    b = _synthetic_bundle()
    n = 150
    samples, stats = run_bundle(b, n_draws=n, seed=0, use_text=False,
                                ensemble=("persistence",))
    _write_deliverables(tmp / "forecast.parquet", samples, b, stats, "test")
    return b, n, tmp


def test_parquet_contract():
    with tempfile.TemporaryDirectory() as td:
        b, n, out = _run(pathlib.Path(td))
        df = pd.read_parquet(out / "forecast.parquet")
        assert tuple(df.columns) == OUT_COLS
        assert df["draw"].dtype.kind == "i"
        assert set(df["draw"].unique()) == set(range(n))
        assert not df.duplicated(["draw", "asset", "horizon"]).any()
        assert np.isfinite(df["value"]).all()
        # meta declares assets/horizons in the declared order
        meta = json.loads((out / "forecast_meta.json").read_text())
        assert meta["unit_id"] == b.card_id
        assert meta["asof"] == b.asof
        assert meta["representation"] == "samples"
        assert meta["asset_ids"] == b.target_assets
        assert meta["horizons"] == list(b.horizons)
        assert meta["n_draws"] == n
        assert meta["rationale"]["file"] == "forecast_rationale.md"
        # exactly three deliverables, no temporaries left behind
        assert sorted(p.name for p in out.iterdir()) == [
            "forecast.parquet", "forecast_meta.json", "forecast_rationale.md"]
