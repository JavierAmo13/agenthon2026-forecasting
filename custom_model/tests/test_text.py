"""Text pipeline tests: llm extraction cache/degradation + text_features.

Runnable both under pytest and directly: ``python test_text.py``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

UNITS = pathlib.Path(__file__).resolve().parents[2] / "units"
EXAMPLE = UNITS / "t2-F3-boj-ust-channel-2023"


def test_extract_doc_offline_returns_none():
    """Without MODEL_ENDPOINT the House client must degrade to None, not raise."""
    import os

    os.environ.pop("MODEL_ENDPOINT", None)
    os.environ.pop("MODEL_NAME", None)
    from custom_model.llm import extract_doc

    assert extract_doc("d1", "some text") is None


def test_text_features_causal_and_shaped():
    if not EXAMPLE.exists():
        return
    import pandas as pd

    from custom_model.data.loader import load_unit
    from custom_model.text_features import build_text_features

    b = load_unit(EXAMPLE)
    cal = pd.date_range("2023-01-01", b.asof, freq="B")
    fake = {
        doc.doc_id: {
            "sentiment": -0.3, "relevance": 0.8, "surprise": 0.5,
            "risk_intensity": 0.7, "direction": "hawkish",
            "event_type": "central_bank", "assets_mentioned": ["JPY", "UST_10Y"],
        }
        for doc in b.texts
    }
    tf = build_text_features(b.texts, fake, cal)
    assert not tf.empty
    assert tf.index.equals(cal)
    assert (tf.filter(like="_count").to_numpy() >= 0).all()
    # a doc timestamped after a row must not affect it
    dated = [pd.Timestamp(d.timestamp) for d in b.texts if d.timestamp]
    first_doc_ts = min(dated)
    before = tf.loc[tf.index < first_doc_ts]
    if len(before):
        assert (before.fillna(0).to_numpy() == 0).all()


def test_empty_extractions_empty_frame():
    import pandas as pd

    from custom_model.text_features import build_text_features

    idx = pd.date_range("2023-01-01", periods=50, freq="B")
    tf = build_text_features([], {}, idx)
    assert tf.empty


if __name__ == "__main__":
    for name, fn in sorted(vars().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all text tests passed")