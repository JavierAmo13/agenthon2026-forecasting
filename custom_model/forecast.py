
"""forecast verb — the Track-2 pipeline.

    forecast --panels /input/panels/ --text /input/text/ \
             --asof YYYY-MM-DD --out /output/forecast.parquet

Staged so that a valid submission exists as early as possible and every
later stage only rewrites it with a better forecast:

  stage 0  engine draws (regime-scaled joint bootstrap) -> deliverables
  stage 1  + transfer-asset regime rescale + conformal calibration
  stage 2  + House text overlay (drift / vol / skew / scenario mixture)

A crash or deadline at any point leaves the last good deliverables in
place. Seed = crc32(unit_id)^salt: deterministic per unit and stable
across the verification rerun (which changes QFBENCH_SEED on purpose).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

from .load import load_unit
from . import steps as st
from .engine import AssetModel, _build_asset_model, simulate
from .calibrate import calibrate, apply_corrections
from . import deliver, text, transfer, apply, selfcheck

DEFAULT_DRAWS = 3000
MAX_DRAWS = 20_000
TIME_BUDGET = 1100.0


def _find_unit(panels_arg: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(panels_arg)
    for cand in [p, p.parent, p.parent.parent]:
        if (cand / "card.toml").exists():
            return cand
    return p


def _unit_seed(card_id: str) -> int:
    return (zlib.crc32(card_id.encode()) & 0x7FFFFFFF) ^ 0x5EED


def _gaussian_floor(bundle, n_draws: int, seed: int) -> np.ndarray:
    """M0-faithful correlated Gaussian walk — the guaranteed floor."""
    rng = np.random.default_rng(seed ^ 0xF00D)
    assets, horizons = bundle.target_assets, bundle.horizons
    steps_map = st.resolve_steps(bundle)
    S = max(steps_map.values())
    diffs, last, mu = {}, {}, {}
    for a in assets:
        s = bundle.series(a)
        if s is None or len(s) < 2:
            diffs[a] = np.zeros(0); last[a] = 0.0; mu[a] = 0.0; continue
        x = st.clean_steps(s, bundle.target_type).iloc[-300:]
        diffs[a] = x.to_numpy(dtype=float)
        last[a] = 0.0 if bundle.target_type == "log_return" else float(s.iloc[-1])
        mu[a] = float(x.mean()) if len(x) else 0.0
    wide = np.full((S, len(assets)), 0.0)
    sd = np.array([max(float(np.std(diffs[a])), 1e-6) if len(diffs[a]) else 1.0
                   for a in assets])
    if len(assets) > 1:
        m = max(len(diffs[a]) for a in assets)
        X = np.full((m, len(assets)), np.nan)
        for i, a in enumerate(assets):
            X[:len(diffs[a]), i] = diffs[a]
        C = np.corrcoef(X[~np.isnan(X).any(axis=1)].T) if len(X) > 20 else np.eye(len(assets))
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        w, v = np.linalg.eigh(C)
        C = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
        dd = np.sqrt(np.diag(C))
        C = C / np.outer(dd, dd)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(len(assets))
    else:
        L = np.eye(len(assets))
    eps = rng.standard_normal((n_draws, S, len(assets))) @ L.T * sd
    cum = np.cumsum(eps, axis=1)
    out = np.empty((n_draws, len(assets), len(horizons)))
    for ai, a in enumerate(assets):
        for hi, h in enumerate(horizons):
            sc = max(1, steps_map.get((a, h), h))
            out[:, ai, hi] = last[a] + mu[a] * sc + cum[:, sc - 1, ai]
    return out


def _transfer_adjust(bundle, models: list[AssetModel], stats: dict) -> None:
    """Rescale sigma_now of withheld-middle assets by the basket's regime move."""
    stats.setdefault("transfer", {})
    for m in models:
        s = bundle.series(m.asset)
        if s is None or not transfer.detect_transfer(bundle, s):
            continue
        end = s.index[-2] if len(s) > 1 else s.index[-1]
        ratio = transfer.basket_regime_ratio(
            bundle, end, exclude=set(bundle.target_assets))
        old = m.sigma_now
        m.sigma_now = float(np.clip(m.sigma_now * ratio,
                                    m.sigma_now * 0.6, m.sigma_now * 3.0))
        stats["transfer"][m.asset] = {"early_end": str(end)[:10],
                                      "basket_ratio": round(ratio, 3),
                                      "sigma_old": round(old, 5),
                                      "sigma_new": round(m.sigma_now, 5)}


def _synthesize_missing(bundle, stats: dict) -> list[AssetModel]:
    """For an asset absent from every panel: honest wide-prior model.
    Anchor 0-free: use median |level| of present series or 1.0; sigma from
    the median per-step sd of context assets scaled x2."""
    out = []
    sds = []
    for a, s in bundle.all_series().items():
        if s is not None and len(s) > 30:
            sds.append(float(st.clean_steps(s, bundle.target_type).std()))
    base_sd = float(np.median(sds)) * 2.0 if sds else 1.0
    for a in bundle.missing_assets:
        out.append(AssetModel(
            asset=a,
            anchor=0.0 if bundle.target_type == "log_return" else 1.0,
            z=pd.Series(dtype=float),
            sigma_now=max(base_sd, 1e-6), drift_step=0.0,
            cadence="daily",
            notes={"synthetic": "absent_from_panels",
                   "target_type": bundle.target_type}))
        stats["transfer"][a] = {"synthetic": True}
    return out


def run_bundle(bundle, *, n_draws: int, seed: int,
               deadline: float | None = None,
               use_text: bool = True,
               family_prior: bool = True) -> tuple[np.ndarray, dict]:
    stats = {"unit": bundle.card_id, "asof": bundle.asof,
             "family": bundle.family, "fallback": False,
             "text": {"applied": False, "skipped": "not attempted"},
             "calibration": {}, "transfer": {}, "engine": {}}
    expired = lambda: deadline is not None and time.time() > deadline

    step_counts = st.resolve_steps(bundle)
    stats["engine"]["steps"] = {f"{a}__{h}": v for (a, h), v in step_counts.items()}

    # --- stage 0: engine draws, no overlay --------------------------------
    models = []
    for a in bundle.present_assets:
        s = bundle.series(a)
        models.append(_build_asset_model(a, s, bundle))
    models += _synthesize_missing(bundle, stats)
    ordered = {m.asset: m for m in models}
    models = [ordered[a] for a in bundle.target_assets if a in ordered]
    if not models:
        return _gaussian_floor(bundle, n_draws, seed), {**stats,
                                                      "fallback": True,
                                                      "reason": "no usable series"}
    tail_boost = 1.4 if str(bundle.family).endswith("F4") else 0.8
    samples = simulate(models, step_counts, bundle.horizons, n_draws, seed,
                       tail_boost=tail_boost)
    stats["engine"]["sigma_now"] = {m.asset: round(m.sigma_now, 6) for m in models}
    stats["engine"]["drift_step"] = {m.asset: round(m.drift_step, 8) for m in models}

    # --- stage 1: transfer regime rescale + conformal calibration ---------
    if not expired():
        try:
            _transfer_adjust(bundle, models, stats)
            samples = simulate(models, step_counts, bundle.horizons, n_draws,
                               seed, tail_boost=tail_boost)
        except Exception as e:
            stats["transfer"]["error"] = str(e)[:200]
    if not expired() and os.environ.get("NOCAL", "0") != "1":
        try:
            cal = calibrate(bundle, models, step_counts, seed,
                            tail_boost=tail_boost)
            stats["calibration"] = cal["stats"]
            samples = apply_corrections(samples, bundle.target_assets,
                                        bundle.horizons, cal["corr"])
        except Exception as e:
            stats["calibration"] = {"error": str(e)[:200]}

    # --- stage 2: House text overlay --------------------------------------
    if use_text and not expired():
        docs, skipped = text.pick_docs(bundle.texts, bundle.asof)
        stats["text"]["docs_used"] = len(docs)
        stats["text"]["docs_skipped"] = skipped
        if not docs:
            stats["text"]["skipped"] = "no admissible documents"
        elif not text.available():
            stats["text"]["skipped"] = "house endpoint unset"
        else:
            ctx = {}
            for m in models:
                smax = max(step_counts[(m.asset, h)] for h in bundle.horizons)
                s = bundle.series(m.asset)
                hist = "full"
                if s is not None and transfer.detect_transfer(bundle, s):
                    hist = f"early_window_to_{s.index[-2].date()}+asof_row"
                elif m.notes.get("synthetic"):
                    hist = "absent"
                ctx[m.asset] = {"anchor": m.anchor, "sigma_h": m.sigma_now *
                                float(np.sqrt(smax)), "steps": smax,
                                "history": hist}
            reply, reason = text.chat(text.build_prompt(bundle, docs, ctx))
            if reply is None:
                stats["text"]["skipped"] = reason
            else:
                cell_sd = {(a, h): float(np.std(samples[:, ai, hi]))
                           for ai, a in enumerate(bundle.target_assets)
                           for hi, h in enumerate(bundle.horizons)}
                anchors = {m.asset: m.anchor for m in models}
                samples, ledger = apply.apply_adjustments(
                    samples, bundle.target_assets, bundle.horizons,
                    anchors, cell_sd, reply, bundle.target_type)
                stats["text"]["applied"] = True
                stats["text"]["skipped"] = ""
                stats["text"]["ledger"] = ledger

    # family prior: a card self-declared F4 ("tail/shock from text") tells
    # the task itself that an extreme move sits in the horizon. When no
    # text read refined it, widen honestly and lean the downside —
    # documented in the rationale as the family-level prior, not a guess.
    if family_prior and not stats["text"].get("applied"):
        fam = str(bundle.family)
        pv, pk, label = 0.0, 0.0, ""
        if fam.endswith("F4"):
            pv = float(os.environ.get("F4_VOL", "2.2"))
            pk = float(os.environ.get("F4_SKEW", "-0.35"))
            label = "F4"
        elif fam.endswith("F2"):
            pv = float(os.environ.get("F2_VOL", "1.5"))
            pk = float(os.environ.get("F2_SKEW", "-0.15"))
            label = "F2"
        if label:
            cell_sd = {(a, h): float(np.std(samples[:, ai, hi]))
                       for ai, a in enumerate(bundle.target_assets)
                       for hi, h in enumerate(bundle.horizons)}
            reply = {"assets": {a: {"drift_bp": 0.0, "vol_scale": pv,
                                    "skew": pk,
                                    "why": f"{label} family prior (no text read)"}
                                for a in bundle.target_assets}}
            anchors = {m.asset: m.anchor for m in models}
            samples, ledger = apply.apply_adjustments(
                samples, bundle.target_assets, bundle.horizons,
                anchors, cell_sd, reply, bundle.target_type,
                vol_clamp=(0.8, 2.4), drift_cap=1.5)
            stats["text"]["ledger"] = ledger
            stats["text"]["family_prior"] = label
    return samples, stats


def main(argv: list[str] | None = None) -> int:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "4")
    ap = argparse.ArgumentParser(prog="forecast")
    ap.add_argument("--panels", type=pathlib.Path, required=True)
    ap.add_argument("--text", type=pathlib.Path, required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--card", type=pathlib.Path, default=None)
    ap.add_argument("--n-draws", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--time-budget", type=float, default=TIME_BUDGET)
    a = ap.parse_args(argv)

    unit_dir = _find_unit(a.panels)
    bundle = load_unit(unit_dir, asof=a.asof)
    n_draws = max(a.n_draws or DEFAULT_DRAWS, bundle.n_draws_min, 200)
    n_draws = min(n_draws, MAX_DRAWS)
    seed = a.seed if a.seed is not None else _unit_seed(bundle.card_id)
    deadline = time.time() + a.time_budget
    try:
        samples, stats = run_bundle(bundle, n_draws=n_draws, seed=seed,
                                    deadline=deadline, use_text=not a.no_text)
        errs = selfcheck.check_samples(samples, bundle.target_assets,
                                       bundle.horizons)
        if errs:
            raise ValueError("selfcheck: " + "; ".join(errs))
    except Exception as exc:
        print(f"pipeline failed ({type(exc).__name__}: {exc}); "
              f"emitting gaussian floor", file=sys.stderr)
        stats = {"unit": bundle.card_id, "asof": bundle.asof,
                 "fallback": True, "reason": f"pipeline error: {exc}"[:200],
                 "text": {"applied": False, "skipped": "pipeline error"},
                 "calibration": {}, "transfer": {}, "engine": {}}
        samples = _gaussian_floor(bundle, n_draws, seed)

    method = ("regime-scaled joint block bootstrap + conformal calibration"
              + (" + House text overlay" if stats.get("text", {}).get("applied")
                 else " (text skipped: "
                      + stats.get("text", {}).get("skipped", "n/a") + ")"))
    deliver.write_all(a.out, samples, bundle, stats, method)
    print(f"wrote {a.out}: {samples.shape[0]} draws x "
          f"{len(bundle.target_assets)} assets x {len(bundle.horizons)} horizons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
