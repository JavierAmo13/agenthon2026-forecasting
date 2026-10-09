
"""End-to-end test with a synthetic House reply — exercises the full
text overlay path (prompt build -> parse -> apply -> deliverables)."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from custom_model import forecast as fm  # noqa: E402
from custom_model.load import load_unit  # noqa: E402
from custom_model import text            # noqa: E402


def _fake_reply(bundle):
    return {
        "assets": {a: {"drift_bp": 120.0, "vol_scale": 1.4, "skew": -0.2,
                       "why": "fomc-statement-2024-06-12: hawkish hold"}
                   for a in bundle.target_assets},
        "scenarios": [{"p": 0.2, "shifts_sd": {bundle.target_assets[0]: -1.0},
                       "vol": 1.5, "label": "hawkish surprise"}],
        "anchors": {},
    }


ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_overlay(tmp_path=None):
    unit = ROOT / "t2_repo/units/t2-EXAMPLE-ust-curve-1m"
    bundle = load_unit(unit)
    orig_chat, orig_avail = text.chat, text.available
    text.chat = lambda prompt, max_tokens=3500: (_fake_reply(bundle), "")
    text.available = lambda: True
    try:
        samples, stats = fm.run_bundle(bundle, n_draws=800, seed=1,
                                       use_text=True)
    finally:
        text.chat, text.available = orig_chat, orig_avail
    assert samples.shape == (800, 4, 1)
    assert np.isfinite(samples).all()
    print("skipped:", stats["text"].get("skipped"), "docs:",
          stats["text"].get("docs_used"))
    assert stats["text"]["applied"] is True
    d = stats["text"]["ledger"]
    assert d["UST_2Y"]["applied"]
    print("overlay ok — UST_2Y ledger:", d["UST_2Y"])
    print("scenarios:", d.get("_scenarios"))


if __name__ == "__main__":
    test_overlay()
