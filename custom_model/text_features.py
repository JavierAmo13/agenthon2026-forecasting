"""Semantic outputs -> time-indexed numeric features.

Aggregates per-doc LLM extractions into per-date aggregates
(sentiment_mean/std, surprise_mean, risk_mean, event counts, per-asset
mention intensity), then reindexes onto the feature calendar, forward-filling
with a decay so a document influences the days after its timestamp only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_KEYS = ("sentiment", "relevance", "surprise", "risk_intensity")
EVENT_TYPES = ("central_bank", "macro_release", "geopolitics", "politics")


def docs_to_daily(texts, extractions: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for doc in texts:
        ex = extractions.get(doc.doc_id)
        if not ex or not doc.timestamp:
            continue
        row = {"date": pd.Timestamp(doc.timestamp)}
        for k in NUMERIC_KEYS:
            try:
                row[k] = float(ex.get(k, np.nan))
            except (TypeError, ValueError):
                row[k] = np.nan
        et = str(ex.get("event_type", "")).lower()
        for t in EVENT_TYPES:
            row[f"ev_{t}"] = 1.0 if et == t else 0.0
        mentioned = [str(a).upper() for a in (ex.get("assets_mentioned") or [])]
        row["_mentioned"] = mentioned
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date").sort_index()
    agg = pd.DataFrame(index=df.index)
    for k in NUMERIC_KEYS:
        agg[f"text_{k}_mean"] = df.groupby(level=0)[k].mean()
        agg[f"text_{k}_std"] = df.groupby(level=0)[k].std()
    for t in EVENT_TYPES:
        agg[f"text_{t}_count"] = df.groupby(level=0)[f"ev_{t}"].sum()
    agg["text_doc_count"] = df.groupby(level=0).size().astype(float)
    agg = agg.groupby(level=0).last()
    # Per-asset mention intensity
    mentioned_series = df["_mentioned"]
    all_assets = sorted({a for lst in mentioned_series for a in lst})
    for a in all_assets[:15]:
        agg[f"text_{a}_mentions"] = mentioned_series.apply(lambda lst: float(a in lst)).groupby(level=0).sum()
    return agg


def align_to_calendar(daily: pd.DataFrame, index: pd.DatetimeIndex,
                      decay_days: int = 21) -> pd.DataFrame:
    """Reindex daily text aggregates onto the feature calendar.

    Count/intensity features decay exponentially after the document date;
    means forward-fill until superseded. All strictly causal (t uses docs
    timestamped <= t).
    """
    if daily.empty:
        return pd.DataFrame(index=index)
    out = pd.DataFrame(index=index)
    for col in daily.columns:
        s = daily[col].reindex(daily.index.union(index)).sort_index()
        s = s.ffill()
        s = s.reindex(index)
        out[col] = s
    count_cols = [c for c in out.columns if c.endswith("_count") or c.endswith("_mentions")]
    decay = np.exp(-np.arange(len(index)) / max(decay_days, 1))
    for col in count_cols:
        vals = out[col].fillna(0).to_numpy()
        smoothed = np.zeros(len(index))
        run = 0.0
        for i, v in enumerate(vals):
            run = run * np.exp(-1.0 / decay_days) + v
            smoothed[i] = run
        out[col] = smoothed
    return out


def build_text_features(texts, extractions, index: pd.DatetimeIndex) -> pd.DataFrame:
    daily = docs_to_daily(texts, extractions)
    return align_to_calendar(daily, index)