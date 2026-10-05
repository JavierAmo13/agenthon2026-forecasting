
"""Text features are causal: an event dated t+10 cannot move rows <= t."""

import numpy as np
import pandas as pd

from custom_model.data.loader import TextDoc
from custom_model.text_features import build_text_features


def _doc(doc_id, ts, text="central bank raises rates unexpectedly"):
    return TextDoc(doc_id=doc_id, timestamp=ts, path=None, text=text)


def _extraction(doc_id, unc="high", direction="up", support="explicit"):
    return {doc_id: {"events": [{
        "doc_id": doc_id, "timestamp_supplied": "2024-01-10",
        "event_type": "central_bank", "evidence": "raises rates",
        "novelty": "new", "surprise_evidence": "unexpectedly",
        "uncertainty": unc, "persistence": "medium",
        "effects": [{"asset": "A", "direction": direction,
                     "channel": "rates", "support": support}]}]}}


def test_event_only_affer_availability_date():
    idx = pd.bdate_range("2024-01-01", periods=30)
    docs = [_doc("d1", "2024-01-10")]
    ex = _extraction("d1")
    feats, z = build_text_features(docs, ex, idx, ["A"])
    before = feats.loc[: "2024-01-09", "text_n_events"]
    after = feats.loc["2024-01-10":, "text_n_events"]
    assert (before == 0).all()
    assert (after > 0).all()
    # half-life decay: the weight must decrease monotonically post-event
    assert after.iloc[0] > after.iloc[-1]


def test_no_docs_means_zero_state():
    idx = pd.bdate_range("2024-01-01", periods=10)
    feats, z = build_text_features([], {}, idx, ["A"])
    assert (feats["text_n_events"] == 0).all()
    assert (feats["text_state"] == 0).all()
