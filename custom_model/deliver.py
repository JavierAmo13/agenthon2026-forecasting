
"""Deliverables: atomic writes of exactly the three contract files.

    forecast.parquet       long format: draw, asset, horizon, value
                           exactly n_draws x n_cells rows (limits.py's
                           expect_exact_rows bound)
    forecast_meta.json     schema keys + our audit block
    forecast_rationale.md  numbered derivation with doc citations —
                           a review screen over the leaderboard, never
                           scored, so it must be honest and checkable
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pandas as pd

OUT_COLS = ("draw", "asset", "horizon", "value")


def write_parquet(samples: np.ndarray, assets: list[str],
                  horizons: list[int], out: pathlib.Path) -> None:
    n = samples.shape[0]
    df = pd.DataFrame({
        "draw": np.repeat(np.arange(n, dtype=np.int64),
                          len(assets) * len(horizons)),
        "asset": np.tile([a for a in assets for h in horizons], n),
        "horizon": np.tile(np.array([h for a in assets for h in horizons],
                                    dtype=np.int64), n),
        "value": samples.reshape(n, -1).ravel(),
    })[list(OUT_COLS)]
    if not np.isfinite(df["value"].to_numpy()).all():
        raise ValueError("non-finite values in forecast draws")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    for comp in ("zstd", "snappy", None):
        try:
            df.to_parquet(tmp, index=False, compression=comp)
            break
        except Exception:
            continue
    else:
        raise ValueError("cannot write forecast.parquet (no codec works)")
    os.replace(tmp, out)


def write_meta(out: pathlib.Path, bundle, n_draws: int, method: str,
               stats: dict) -> None:
    # g2 bind_metadata requires meta.asof == trusted_asof(card) EXACTLY, and
    # trusted_asof reads [forecast].asof BEFORE [provenance].data_cutoff —
    # the precedence order is contractual, not cosmetic.
    card_asof = str(bundle.card.get("forecast", {}).get("asof")
                    or bundle.card.get("provenance", {}).get("data_cutoff")
                    or bundle.asof)[:10]
    meta = {
        "unit_id": bundle.card_id,
        "asof": card_asof,
        "representation": "samples",
        "asset_ids": list(bundle.target_assets),
        "horizons": [int(h) for h in bundle.horizons],
        "n_draws": int(n_draws),
        "units": bundle.value_unit or "panel native units",
        "rationale": {"file": "forecast_rationale.md", "method": method},
        "reasoning_applied": bool(stats.get("text", {}).get("applied")),
        "reasoning_skipped_reason": stats.get("text", {}).get("skipped", ""),
        "custom": {"version": "4.0.0", "stats": stats},
    }
    if bundle.target_type in ("level", "log_return", "yield"):
        meta["target"] = bundle.target_type
    mp = pathlib.Path(out).parent / "forecast_meta.json"
    mp.write_text(json.dumps(meta, default=str, indent=1), encoding="utf-8")


def write_rationale(out: pathlib.Path, bundle, stats: dict, method: str) -> None:
    adj = stats.get("text", {}).get("ledger", {})
    cal = stats.get("calibration", {})
    lines = [
        f"# Forecast rationale — {bundle.card_id}",
        "",
        f"As-of {bundle.asof}. Target {bundle.target_type} "
        f"({bundle.target_frequency}). Assets: {', '.join(bundle.target_assets)}. "
        f"Horizons: {bundle.horizons}.",
        "",
        "## Method",
        method,
        "",
        "1. Anchor: last observed level per asset at the as-of (log_return: 0).",
        "2. Innovations: vol-standardized step series (EWMA sigma path),",
        "   block-bootstrapped on a shared calendar so joint and cross-horizon",
        "   dependence is empirical, recency-weighted 60/40 vs full history.",
        "3. Regime: paths rescaled to the current EWMA sigma per asset.",
        "4. Calibration: conformal PIT backtest at past origins; bounded",
        "   per-cell shift/scale corrections, shrunk toward identity.",
        "5. Text: House-model read of the dated corpus -> bounded drift_bp /",
        "   vol_scale / skew per asset, optional labelled scenario mixture.",
        "",
        "## Text adjustments",
        f"applied: {stats.get('text', {}).get('applied', False)} "
        f"({stats.get('text', {}).get('skipped', '')})",
    ]
    for a, rec in adj.items():
        if a == "_scenarios":
            for s in rec:
                lines.append(f"- scenario '{s['label']}' p={s['p']:.2f} "
                             f"n={s['n_draws']} vol={s['vol']} "
                             f"shifts={s['shifts']}")
        elif isinstance(rec, dict) and rec.get("applied"):
            lines.append(f"- {a}: drift {rec['drift_bp']:+.1f}bp "
                         f"vol x{rec['vol_scale']:.2f} skew {rec['skew']:+.2f} "
                         f"— {rec['why']}")
    lines += ["", "## Calibration"]
    for a, st in cal.get("stats", {}).items():
        lines.append(f"- {a}: {st}")
    lines += [
        "",
        "## Uncertainty",
        "Width is the honest EWMA-regime + fat-tailed empirical shape; the",
        "calibration layer only corrects measured miscalibration. Documents",
        "dated after the as-of were not read.",
        "",
        f"Fallback: {bool(stats.get('fallback'))} "
        f"{stats.get('reason', '')}",
    ]
    rp = pathlib.Path(out).parent / "forecast_rationale.md"
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(out: pathlib.Path, samples: np.ndarray, bundle,
              stats: dict, method: str) -> None:
    # g0 requires the output tree to hold EXACTLY the three contract files.
    # Purge leftover *.tmp (from a kill between write and os.replace on a
    # previous stage) before producing anything — a stray file refuses the
    # whole submission even when the parquet itself is valid.
    try:
        for stale in pathlib.Path(out).parent.glob("*.tmp"):
            stale.unlink()
    except OSError:
        pass
    # parquet is the scored artifact and must succeed; sidecars are
    # best-effort so a metadata serialization quirk can never eat the run.
    write_parquet(samples, bundle.target_assets, bundle.horizons, out)
    import sys, traceback
    for fn in (write_meta, write_rationale):
        try:
            if fn is write_meta:
                fn(out, bundle, samples.shape[0], method, stats)
            else:
                fn(out, bundle, stats, method)
        except Exception:
            traceback.print_exc(file=sys.stderr)
