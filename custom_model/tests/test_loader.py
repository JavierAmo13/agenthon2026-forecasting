"""Smoke tests for data/loader.py on a real unit.

Runnable both under pytest and directly: ``python test_loader.py``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

UNITS = pathlib.Path(__file__).resolve().parents[2] / "units"
EXAMPLE = UNITS / "t2-F3-boj-ust-channel-2023"


def test_load_unit_fields():
    if not EXAMPLE.exists():
        return
    from custom_model.data.loader import load_unit

    b = load_unit(EXAMPLE)
    assert b.card_id == "t2-F3-boj-ust-channel-2023"
    assert b.target_assets == ["UST_10Y", "JPY"]
    assert b.horizons == [21, 63]
    assert b.target_type == "level"
    assert b.target_frequency == "daily"
    assert b.n_draws_min >= 200
    assert len(b.panels) == 2
    assert len(b.texts) == 9


def test_asset_series_respects_asof():
    if not EXAMPLE.exists():
        return
    import pandas as pd

    from custom_model.data.loader import load_unit

    b = load_unit(EXAMPLE)
    s = b.asset_series("UST_10Y")
    assert s.index.max() <= pd.Timestamp(b.asof)
    assert len(s) > 1000


def test_missing_asset_raises():
    if not EXAMPLE.exists():
        return
    from custom_model.data.loader import load_unit

    b = load_unit(EXAMPLE)
    try:
        b.asset_series("NOT_A_REAL_ASSET")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown asset")


if __name__ == "__main__":
    for name, fn in sorted(vars().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all loader tests passed")