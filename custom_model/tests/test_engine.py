
"""Engine invariants: path structure, joint dependence, regime scaling."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from custom_model.engine import AssetModel, simulate  # noqa: E402


def _mk(asset, anchor, sd, mu=0.0, n=1500, seed=0, cadence="daily"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    z = pd.Series(rng.standard_t(5, n) , index=idx)
    return AssetModel(asset=asset, anchor=anchor, z=z, sigma_now=sd,
                      drift_step=mu, cadence=cadence,
                      notes={"target_type": "level"})


def test_path_coherence():
    """Two horizons of one asset share the same path: corr(X_h1, X_h2)
    must approach the random-walk value sqrt(s1/s2), not be independent."""
    m = _mk("A", 0.0, 0.01)
    sc = {("A", 20): 20, ("A", 60): 60}
    out = simulate([m], sc, [20, 60], n_draws=4000, seed=7)
    x1, x2 = out[:, 0, 0], out[:, 0, 1]
    c = np.corrcoef(x1, x2)[0, 1]
    exp = np.sqrt(20 / 60)          # shared-path RW correlation
    assert 0.4 < c < 0.7, f"cross-horizon corr {c:.3f} vs RW {exp:.2f}"
    print(f"path coherence ok: corr={c:.3f} (RW expectation {exp:.2f})")


def test_joint_dependence():
    """Two assets sharing the same z-calendar co-move: same-index z's
    produce correlated draws."""
    rng = np.random.default_rng(3)
    n = 1500
    z_common = rng.standard_t(4, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    z1 = pd.Series(z_common * 0.9 + 0.1 * rng.standard_t(5, n), index=idx)
    z2 = pd.Series(z_common * 0.9 + 0.1 * rng.standard_t(5, n), index=idx)
    m1 = AssetModel("A", 0.0, z1, 0.01, 0.0, "daily",
                    {"target_type": "level"})
    m2 = AssetModel("B", 0.0, z2, 0.01, 0.0, "daily",
                    {"target_type": "level"})
    sc = {("A", 40): 40, ("B", 40): 40}
    out = simulate([m1, m2], sc, [40], n_draws=3000, seed=11)
    c = np.corrcoef(out[:, 0, 0], out[:, 1, 0])[0, 1]
    assert c > 0.5, f"joint corr {c:.3f} too weak for shared calendar"
    print(f"joint dependence ok: corr={c:.3f}")


def test_regime_scaling():
    """sigma_now scales the dispersion, not the anchor."""
    m = _mk("A", 5.0, 0.02)
    sc = {("A", 50): 50}
    out = simulate([m], sc, [50], n_draws=2000, seed=13)
    sd = float(np.std(out[:, 0, 0]))
    assert 0.05 < sd < 0.4, f"sd {sd} implausible for sigma_now=0.02,s=50"
    assert abs(np.median(out[:, 0, 0]) - 5.0) < 0.3 * sd
    print(f"regime scaling ok: sd={sd:.3f}, median={np.median(out[:,0,0]):.3f}")


if __name__ == "__main__":
    test_path_coherence()
    test_joint_dependence()
    test_regime_scaling()
    print("all engine tests pass")
