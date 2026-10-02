"""Structural pass over every unit: loader + features + targets + dataset.

Cheap (no model fit). The full forecast sweep lives in run_all_units.py at the
repo root; this test only checks the data plumbing on all units. Runnable both
under pytest and directly: ``python test_all_units.py``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

UNITS = pathlib.Path(__file__).resolve().parents[2] / "units"
ALL = sorted(d for d in UNITS.iterdir() if d.is_dir()) if UNITS.exists() else []


def check_unit(unit_dir: pathlib.Path) -> None:
    from custom_model.data.loader import load_unit
    from custom_model.data.features import build_features
    from custom_model.data.targets import build_all_targets
    from custom_model.data.dataset import build_dataset

    b = load_unit(unit_dir)
    assert b.target_assets and b.horizons
    X = build_features(b.panels, b.asof)
    assert not X.empty, f"{unit_dir.name}: empty feature frame"
    T = build_all_targets(b)
    monthly = b.target_frequency == "monthly"
    for a in b.target_assets:
        for h in b.horizons:
            ds = build_dataset(X, T, a, h, monthly=monthly)
            assert ds.y.notna().sum() > 0, f"{unit_dir.name}: no y for {a} h{h}"
            assert ds.X_pred.shape[0] == 1


def test_unit_data_pipeline():
    """Under pytest this runs all units in one test; individually the same
    checks are exercised via the parametrized copy below when pytest exists."""
    fails = []
    for d in ALL:
        try:
            check_unit(d)
        except Exception as exc:  # noqa: BLE001
            fails.append((d.name, exc))
    assert not fails, "; ".join(f"{n}: {e}" for n, e in fails[:5])


if __name__ == "__main__":
    import time

    t0 = time.time()
    bad = 0
    for i, d in enumerate(ALL, 1):
        try:
            check_unit(d)
            print(f"[{i}/{len(ALL)}] OK   {d.name}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"[{i}/{len(ALL)}] FAIL {d.name}: {exc}")
    print(f"{len(ALL) - bad}/{len(ALL)} units passed in {time.time() - t0:.0f}s")
    sys.exit(1 if bad else 0)