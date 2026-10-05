
"""Validated event extractions -> causal time-indexed features + text state.

Each event carries a per-type half-life H_tipo:
    w_j(t) = relevance_j * 0.5 ** (age_j / H_tipo)
with relevance from effect support (explicit > inferred > insufficient), so
an event fades at the speed of its class — not at one uniform decay.

Outputs (strictly causal: an event only affects dates >= its doc's
availability date):
  text_n_events   distinct-event count, decayed
  text_unc        weighted uncertainty score
  text_surp       weighted surprise rate (only events WITH surprise_evidence)
  text_persist    weighted persistence score
  text_agree      direction agreement |sum w·d| / sum w·|d| over effects
  text_state      0 = no text, 1 = nothing decisive, 2 = disagreement, 3 = aligned
  text_dir_<A>    support-weighted direction in [-1, 1] per target asset
  text_w_<A>      total support weight behind that direction

The z_frame returned alongside is the compact text state per origin — what
the sigma calibrator fits on, without re-calling House.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = {"central_bank": 21.0, "macro_release": 10.0,
             "geopolitics": 42.0, "politics": 21.0, "other": 10.0}
UNCERT = {"low": 0.2, "medium": 0.5, "high": 0.8}
PERSIST = {"short": 0.2, "medium": 0.5, "long": 0.9}
DIR = {"up": 1.0, "down": -1.0, "unclear": 0.0}
SUPPORT_W = {"explicit": 1.0, "inferred": 0.6, "insufficient": 0.25}


def _flatten(texts, extractions: dict[str, dict],
             target_assets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, effects): one row per event / per (event, asset) effect."""
    ts_by_id = {d.doc_id: d.timestamp for d in texts}
    ev_rows, fx_rows = [], []
    for doc_id, payload in (extractions or {}).items():
        ts = ts_by_id.get(doc_id, "")
        if not ts:
            continue
        date = pd.Timestamp(str(ts)[:10])
        for ev in payload.get("events") or []:
            if str(ev.get("novelty", "")) == "duplicate":
                continue                    # same event told twice adds nothing
            effects = [e for e in ev.get("effects", [])
                       if e.get("asset") in target_assets]
            w0 = max([SUPPORT_W.get(e.get("support", "insufficient"), 0.25)
                      for e in effects], default=0.15)
            ev_rows.append({
                "date": date, "w0": w0,
                "hl": HALF_LIFE.get(str(ev.get("event_type", "other")), 10.0),
                "unc": UNCERT.get(str(ev.get("uncertainty", "unknown")), np.nan),
                "pers": PERSIST.get(str(ev.get("persistence", "unknown")), np.nan),
                "surp": 1.0 if ev.get("surprise_evidence") else 0.0,
            })
            for e in effects:
                fx_rows.append({"date": date, "asset": e["asset"], "w0": w0,
                                "hl": HALF_LIFE.get(str(ev.get("event_type", "other")), 10.0),
                                "d": DIR.get(e.get("direction", "unclear"), 0.0),
                                "sw": SUPPORT_W.get(e.get("support", "insufficient"), 0.25)})
    return pd.DataFrame(ev_rows), pd.DataFrame(fx_rows)


def _weights(df: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    """[n_rows, n_dates] w_j(t); zero before the event's availability date."""
    if df.empty:
        return np.zeros((0, len(index)))
    age = (index.to_numpy(dtype="datetime64[D]")[None, :]
           - df["date"].to_numpy(dtype="datetime64[D]")[:, None]).astype("timedelta64[D]").astype(float)
    hl = df["hl"].to_numpy()[:, None]
    return np.where(age >= 0, df["w0"].to_numpy()[:, None] * 0.5 ** (age / hl), 0.0)


def build_text_features(texts, extractions, index: pd.DatetimeIndex,
                        target_assets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features_on_calendar, z_frame). z_frame holds the compact state used
    by the sigma calibrator: text_unc / text_surp / text_persist / count."""
    events, effects = _flatten(texts, extractions, target_assets)
    W = _weights(events, index)                    # [n_events, n_dates]
    w = W.sum(axis=0)
    wsafe = np.where(w > 0, w, np.nan)

    def _wmean(col: str) -> np.ndarray:
        v = events[col].to_numpy() if not events.empty else np.zeros(0)
        ok = np.isfinite(v)
        num = (W[ok] * np.where(ok, v, 0.0)[ok][:, None]).sum(axis=0) if ok.any() else np.zeros(len(index))
        return num / wsafe

    z = pd.DataFrame(index=index)
    z["text_n_events"] = w
    z["text_unc"] = _wmean("unc")
    z["text_surp"] = _wmean("surp") if not events.empty else 0.0
    z["text_persist"] = _wmean("pers")
    z["text_surp"] = z["text_surp"].fillna(0.0)

    # direction agreement over all target-asset effects
    if not effects.empty:
        We = _weights(effects, index) * effects["sw"].to_numpy()[:, None]
        wd = (We * effects["d"].to_numpy()[:, None]).sum(axis=0)
        wa = (We * np.abs(effects["d"].to_numpy())[:, None]).sum(axis=0)
        z["text_agree"] = np.divide(np.abs(wd), wa, out=np.zeros(len(index)),
                                    where=wa > 0)
    else:
        z["text_agree"] = 0.0
        We = np.zeros((0, len(index)))

    state = np.zeros(len(index))
    state[w > 0] = 1
    state[(w > 0) & (z["text_agree"] > 0) & (z["text_agree"] < 0.5)] = 2
    state[(w > 0) & (z["text_agree"] >= 0.5)] = 3
    z["text_state"] = state

    feats = z.copy()
    for a in target_assets:
        if effects.empty:
            feats[f"text_dir_{a}"] = 0.0
            feats[f"text_w_{a}"] = 0.0
            continue
        sel = (effects["asset"] == a).to_numpy()
        if not sel.any():
            feats[f"text_dir_{a}"] = 0.0
            feats[f"text_w_{a}"] = 0.0
            continue
        Wa = We[sel]
        d_a = effects["d"].to_numpy()[sel]
        num = (Wa * d_a[:, None]).sum(axis=0)
        den = Wa.sum(axis=0)
        feats[f"text_dir_{a}"] = np.divide(np.clip(num, -1e15, 1e15), den,
                                           out=np.zeros(len(index)),
                                           where=den > 0).clip(-1, 1)
        feats[f"text_w_{a}"] = den
    return feats, z[["text_unc", "text_surp", "text_persist", "text_n_events"]]


def text_state_now(z_frame: pd.DataFrame) -> dict | None:
    """Compact z at the last calendar date — what the sigma calibrator sees."""
    if z_frame is None or z_frame.empty:
        return None
    last = z_frame.iloc[-1]
    if not np.isfinite(last.get("text_unc", np.nan)):
        return None
    return {"unc": float(last["text_unc"]), "surp": float(last["text_surp"]),
            "pers": float(last["text_persist"]),
            "n": float(last["text_n_events"])}
