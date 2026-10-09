
"""Apply validated text adjustments to the draw tensor.

Three knobs, per the solver playbook, each applied with hard clamps and
each logged line-by-line for the rationale:

  * drift   — center shift, in bp of |anchor| (level/yield) or absolute
              return bp (log_return); clamped to +-1.5 x cell sd so a stray
              model number cannot crater the card
  * vol     — dispersion multiplier about the median, [0.7, 1.6]
  * skew    — one-sided stretch in [-0.5, 0.5]: the cheap stand-in for a
              two-branch scenario mixture (widens one tail only)
  * scenarios — optional labelled branches: a per-draw assignment turns the
              draws into a real mixture distribution over the joint grid
"""

from __future__ import annotations

import math

import numpy as np

DRIFT_SD_CAP = 1.5
VOL_LO, VOL_HI = 0.7, 1.6
SKEW_CAP = 0.5


def _finite(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def apply_adjustments(draws: np.ndarray, assets: list[str], horizons: list[int],
                      anchors: dict, cell_sd: dict, reply: dict,
                      target_type: str,
                      vol_clamp: tuple[float, float] = (VOL_LO, VOL_HI),
                      drift_cap: float = DRIFT_SD_CAP) -> tuple[np.ndarray, dict]:
    """draws [n, A, H] -> adjusted; returns (draws, applied_ledger)."""
    n = draws.shape[0]
    ledger: dict[str, dict] = {}
    per = reply.get("assets") if isinstance(reply, dict) else None
    if not isinstance(per, dict):
        per = reply if isinstance(reply, dict) else {}

    for ai, a in enumerate(assets):
        spec = per.get(a)
        if not isinstance(spec, dict):
            ledger[a] = {"applied": False}
            continue
        anchor = abs(anchors.get(a, 0.0)) or 1.0
        scale = 1.0 if target_type == "log_return" else anchor
        shift = scale * _finite(spec.get("drift_bp")) / 1e4
        vol = min(max(_finite(spec.get("vol_scale"), 1.0), vol_clamp[0]),
                  vol_clamp[1])
        skew = min(max(_finite(spec.get("skew")), -SKEW_CAP), SKEW_CAP)
        notes = []
        for hi, h in enumerate(horizons):
            sd = float(cell_sd.get((a, h), 0.0)) or 1.0
            cap = drift_cap * sd
            s = max(-cap, min(cap, shift))
            col = draws[:, ai, hi]
            med = np.median(col)
            dev = col - med
            if skew != 0.0:
                pos = dev > 0
                dev = np.where(pos, dev * (1 + skew), dev * (1 - skew))
                notes.append(f"h{h}: skew {skew:+.2f}")
            draws[:, ai, hi] = med + dev * vol + s
        ledger[a] = {"applied": True, "drift_bp": _finite(spec.get("drift_bp")),
                     "vol_scale": vol, "skew": skew,
                     "why": str(spec.get("why", ""))[:300],
                     "notes": notes}

    # labelled scenario mixture (optional): assign draws to branches
    scs = reply.get("scenarios") if isinstance(reply, dict) else None
    if isinstance(scs, list) and scs:
        picks = np.zeros(n, dtype=int)   # 0 = base
        ledger["_scenarios"] = []
        lo = 0
        for si, sc in enumerate(scs[:3]):
            p = min(max(_finite(sc.get("p")), 0.0), 1.0)
            cnt = int(round(p * n))
            if cnt <= 0:
                continue
            picks[lo:lo + cnt] = si + 1
            lo += cnt
            sv = min(max(_finite(sc.get("vol"), 1.0), 0.8), 2.0)
            shifts = sc.get("shifts_sd") or {}
            rec = {"label": str(sc.get("label", f"s{si}"))[:80], "p": p,
                   "n_draws": cnt, "vol": sv, "shifts": {}}
            for ai, a in enumerate(assets):
                k = _finite((shifts or {}).get(a))
                if k == 0.0:
                    continue
                sel = picks == si + 1
                for hi, h in enumerate(horizons):
                    sd = float(cell_sd.get((a, h), 0.0)) or 1.0
                    col = draws[sel, ai, hi]
                    med = np.median(col)
                    draws[sel, ai, hi] = med + (col - med) * sv + k * sd
                rec["shifts"][a] = k
            ledger["_scenarios"].append(rec)
    return draws, ledger
