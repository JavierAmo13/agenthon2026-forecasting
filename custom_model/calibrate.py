
"""In-run conformal calibration of the engine's own output.

Re-run the same engine at past origins and compare its predicted quantiles
with the values that actually followed (all inside the panel's own history,
all <= asof). The per-cell PIT histogram tells us whether our distribution
is systematically off-center or mis-width — we then apply a bounded affine
correction per cell (shift + spread around the median), shrunk toward
identity as the sample thins.

This directly targets what the composite scores: marginal CRPS punishes
bias and wrong spread; pinball punishes wrong tail quantiles. A corrected
engine beats an uncorrected one whenever it has any systematic defect,
and the shrinkage/clamps keep it honest when the defect is noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import AssetModel, simulate


def _origins(n: int, step_max: int, n_target: int) -> list[int]:
    """Row positions of pseudo-asofs inside the series, spaced so each
    origin's label window does not overlap the next one's origin."""
    lo = max(n // 5, 40)
    hi = n - step_max - 1
    if hi <= lo:
        return []
    spacing = max(step_max + 2, 8)
    return list(range(lo, hi, spacing))[-n_target:]


def calibrate(bundle, models: list[AssetModel],
              step_counts: dict[tuple[str, int], int],
              seed: int, *, n_origins: int = 14, n_draws_cal: int = 600,
              tail_boost: float = 0.8) -> dict:
    """Per-cell affine corrections {cell: (shift_sd, scale)} + diagnostics.

    For each usable origin the SAME engine runs on the series truncated at
    the origin date; the realized continuation comes straight from the
    series itself — every point is inside the panel, so no leakage.
    """
    from .engine import _build_asset_model

    corr: dict[tuple[str, int], tuple[float, float]] = {}
    stats: dict[str, dict] = {}
    for m in models:
        s_full = bundle.series(m.asset)
        if s_full is None or len(s_full) < 60:
            continue
        smax = max(step_counts[(m.asset, h)] for h in bundle.horizons)
        origins = _origins(len(s_full), smax, n_origins)
        stats[m.asset] = {"origins": len(origins)}
        if len(origins) < 8:
            continue
        pits: dict[int, list[float]] = {h: [] for h in bundle.horizons}
        for oi in origins:
            cut_date = s_full.index[oi]
            mm = _build_asset_model(m.asset, s_full.iloc[:oi + 1], bundle)
            sc = {(m.asset, h): step_counts[(m.asset, h)] for h in bundle.horizons}
            sim = simulate([mm], sc, bundle.horizons, n_draws_cal,
                           seed + 7919 + oi, tail_boost=tail_boost)[:, 0, :]
            for hi, h in enumerate(bundle.horizons):
                sneed = step_counts[(m.asset, h)]
                end = oi + sneed
                if end >= len(s_full):
                    continue
                if bundle.target_type == "log_return":
                    r = s_full.iloc[oi + 1:end + 1].clip(lower=-0.999999)
                    y = float(np.log1p(r).sum())
                else:
                    y = float(s_full.iloc[end])
                if not np.isfinite(y):
                    continue
                pits[h].append(float(np.mean(sim[:, hi] <= y)))
        for hi, h in enumerate(bundle.horizons):
            ps = np.array(pits[h])
            if len(ps) < 8:
                corr[(m.asset, h)] = (0.0, 1.0)
                stats[m.asset][f"h{h}"] = {"n": len(ps), "note": "thin"}
                continue
            w = len(ps) / (len(ps) + 8.0)
            med = float(np.median(ps))
            iqr = float(np.subtract(*np.percentile(ps, [75, 25])))
            shift_sd = -(med - 0.5) * 2.0
            scale = iqr / 0.5
            scale = float(np.clip(1 + w * (scale - 1), 0.75, 1.45))
            shift_sd = float(np.clip(w * shift_sd, -0.45, 0.45))
            corr[(m.asset, h)] = (shift_sd, scale)
            stats[m.asset][f"h{h}"] = {"n": len(ps),
                                       "shift_sd": round(shift_sd, 3),
                                       "scale": round(scale, 3),
                                       "pit_med": round(med, 3)}
    return {"corr": corr, "stats": stats}


def apply_corrections(samples: np.ndarray, assets: list[str], horizons: list[int],
                      corr: dict) -> np.ndarray:
    """Affine per-cell correction around the draw median.

    cells flatten in card order: asset-major, horizon-minor — the scorer's
    own order. A per-coordinate monotone transform keeps each draw a valid
    joint scenario (the copula ordering is preserved).
    """
    d = len(assets) * len(horizons)
    mat = samples.reshape(samples.shape[0], d) if samples.ndim == 3 else samples
    for i, (a, h) in enumerate((a, h) for a in assets for h in horizons):
        shift_sd, scale = corr.get((a, h), (0.0, 1.0))
        if shift_sd == 0.0 and scale == 1.0:
            continue
        col = mat[:, i]
        med = np.median(col)
        sd = np.std(col)
        if not np.isfinite(sd) or sd <= 0:
            continue
        mat[:, i] = med + (col - med) * scale + shift_sd * sd
    return samples
