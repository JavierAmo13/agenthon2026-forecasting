"""forecast.py — the orchestrator. Implements the `forecast` verb:

    forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD \
             --out /output/forecast.parquet

Flow per the master recipe:
    load_unit -> features -> targets -> dataset -> model -> probabilistic ->
    joint -> forecast.parquet (+ forecast_meta.json + forecast_rationale.md)

Every step degrades gracefully: a task whose model cannot fit falls back to a
persistence/gaussian marginal; a unit whose whole pipeline fails still emits a
schema-valid joint gaussian walk.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

from .data.loader import load_unit, DataBundle
from .data.features import build_features, panel_kind, to_wide
from .data.targets import build_all_targets, monthly_targets
from .data.dataset import build_dataset
from .model import fit_model, predict, walk_forward_residuals
from .probabilistic import fit_residual_model
from .joint import joint_samples, to_output_tensor, _nearest_psd_corr
from .regime import regime_scale

DEFAULT_DRAWS = 500
MAX_DRAWS = 20_000


# ---------------------------------------------------------------------------
# Statistical fallback (reference-grade gaussian walk, joint)
# ---------------------------------------------------------------------------

def _diffs_no_gaps(s: pd.Series) -> pd.Series:
    d = s.diff()
    idx = pd.DatetimeIndex(s.index)
    step = pd.Series(idx, index=idx).diff().dt.days
    if step.notna().sum() == 0:
        return d
    return d.where(step <= max(float(step.median()) * 10.0, 5.0))


def _horizon_steps(bundle) -> dict[tuple[str, int], float]:
    """Map each declared horizon key to the number of observation steps it
    means for this unit. Daily units: the key is the step count. Monthly
    units: the key is opaque — steps come from observation_periods."""
    if bundle.target_frequency != "monthly":
        return {(a, h): float(h) for a in bundle.target_assets
                for h in bundle.horizons}
    return {(a, h): float(steps)
            for (a, h), (_s, steps) in monthly_targets(bundle).items()}


def _gaussian_walk(bundle: DataBundle, n_draws: int, seed: int) -> np.ndarray:
    """Schema-valid joint gaussian random walk — the guaranteed floor."""
    rng = np.random.default_rng(seed)
    assets, horizons = bundle.target_assets, bundle.horizons
    steps_map = _horizon_steps(bundle)
    steps_by_asset = {}
    for a in assets:
        try:
            s = bundle.asset_series(a).dropna()
        except KeyError:
            s = pd.Series(dtype=float)
        steps_by_asset[a] = s
    if bundle.target_type == "log_return":
        # Panel rows are simple returns; the target is the cumulative sum of
        # log(1 + r) over the horizon, so per-step draws are log(1 + r) and the
        # walk starts at 0.
        diffs = {a: np.log1p(s[s > -1]).dropna() for a, s in steps_by_asset.items()}
        last = np.zeros(len(assets))
        drift = np.array([float(d.mean()) if len(d) else 0.0 for d in diffs.values()])
    else:
        diffs = {a: _diffs_no_gaps(s).dropna() for a, s in steps_by_asset.items()}
        last = np.array([float(s.iloc[-1]) if len(s) else 0.0 for s in steps_by_asset.values()])
        drift = np.zeros(len(assets))
    sd = np.array([max(float(d.std()), 1e-4) if len(d) else 1.0 for d in diffs.values()])
    aligned = pd.DataFrame(diffs).dropna()
    corr = _nearest_psd_corr(aligned.corr().to_numpy()) if len(aligned) >= 30 else np.eye(len(assets))
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        chol = np.eye(len(assets))
    out = np.empty((n_draws, len(assets), len(horizons)))
    for hi, h in enumerate(horizons):
        z = rng.standard_normal((n_draws, len(assets))) @ chol.T
        steps = np.array([steps_map.get((a, h), float(h)) for a in assets])
        out[:, :, hi] = (last + drift * steps)[None, :] + \
            z * (sd * np.sqrt(steps))[None, :]
    return out


# ---------------------------------------------------------------------------
# ML path
# ---------------------------------------------------------------------------

def _task_mu_sigma(bundle, features, targets, asset, horizon, model_kind, seed,
                   n_folds, min_train, max_iter=300, row_cap=None):
    """One (asset,horizon) task -> (mu_hat, resid_model, resid_series)."""
    ds = build_dataset(features, targets, asset, horizon, nan_policy="drop_warmup",
                       monthly=(bundle.target_frequency == "monthly"))
    if row_cap and len(ds.X) > row_cap:
        keep_idx = ds.X.index[-row_cap:]
        ds = type(ds)(ds.X.loc[keep_idx], ds.y.loc[keep_idx], ds.X_pred,
                      ds.feature_names, ds.asset, ds.horizon)
        min_train = max(int(row_cap * 0.5), 60)
    if len(ds.X) < 40:
        raise ValueError(f"too few training rows ({len(ds.X)})")
    mdl = fit_model(ds.X, ds.y, model_kind, seed, max_iter=max_iter)
    mu = float(predict(mdl, ds.X_pred)[0])
    embargo = horizon if bundle.target_frequency == "daily" else \
        monthly_targets(bundle).get((asset, horizon), (None, horizon))[1]
    resid = walk_forward_residuals(ds.X, ds.y, model_kind, n_folds=n_folds,
                                   min_train=min_train, embargo=int(embargo),
                                   seed=seed, max_iter=max_iter)
    return mu, fit_residual_model(resid), resid


def _member_crps(resid_model: dict, resid: pd.Series, seed: int = 0) -> float:
    """Raw validation score for one member's residual model on one task:
    mean over held-out origins of CRPS(draws ~ F, realized resid)."""
    from .probabilistic import sample_residuals
    from .validation import crps_empirical
    r = resid.dropna().to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return np.inf
    rng = np.random.default_rng(seed)
    draws = sample_residuals(resid_model, (200,), rng)
    return float(np.mean([crps_empirical(draws, ri) for ri in r]))


def _tasks_for_kind(bundle, features, targets, tasks, model_kind, seed,
                    n_folds, min_train, max_iter, stats, row_cap=None):
    """Fit every (asset,horizon) task for one model kind; per-task fallback."""
    task_stats = stats.setdefault("tasks", {})
    mus = np.empty(len(tasks))
    resid_models, resid_cols = [], {}
    for ti, (a, h) in enumerate(tasks):
        try:
            mu, rm, resid = _task_mu_sigma(
                bundle, features, targets, a, h, model_kind, seed + ti,
                n_folds=n_folds, min_train=min_train, max_iter=max_iter,
                row_cap=row_cap)
            resid_cols[ti] = resid
        except Exception as exc:  # noqa: BLE001 - per-task isolation
            mu, rm, resid = _fallback_task(bundle, a, h)
            resid_cols[ti] = resid
            task_stats[f"{a}__h{h}"] = {"fallback": True, "error": str(exc)[:200]}
        mus[ti] = mu
        resid_models.append(rm)
        task_stats.setdefault(f"{a}__h{h}", {}).update({"mu": mu, "sigma": rm["sigma"]})
    resid_frame = pd.DataFrame(resid_cols).dropna(how="all")
    if len(resid_frame) < 30:  # too few rows for a stable correlation estimate
        resid_frame = None
    return mus, resid_models, resid_frame


def run_bundle(bundle: DataBundle, *, n_draws: int, seed: int = 0,
               model_kind: str = "hgb", use_text: bool = True,
               ensemble: tuple[str, ...] | None = None) -> tuple[np.ndarray, dict]:
    """Full pipeline on a loaded unit. Returns [n_draws, n_assets, n_horizons] + stats.

    ``ensemble`` lists the member model kinds (e.g. ("hgb", "ridge",
    "persistence")); each produces its own joint draws, weighted by inverse
    walk-forward CRPS and pooled. ``None`` runs the single ``model_kind``.
    """
    tasks = [(a, h) for a in bundle.target_assets for h in bundle.horizons]
    stats: dict = {"unit": bundle.card_id, "asof": bundle.asof, "tasks": {},
                   "fallback": False}

    features = build_features(bundle.panels, bundle.asof)

    # --- optional text branch (House endpoint absent -> empty frame) ---
    if use_text and bundle.texts:
        try:
            from .llm import extract_corpus
            from .text_features import build_text_features
            ex = extract_corpus(bundle.texts)
            tf = build_text_features(bundle.texts, ex, features.index)
            if not tf.empty:
                features = pd.concat([features, tf], axis=1)
                stats["text_docs_used"] = len(ex)
        except Exception as exc:  # noqa: BLE001
            stats["text_error"] = str(exc)

    if features.empty:
        stats["fallback"] = True
        stats["reason"] = "no numeric features"
        return _gaussian_walk(bundle, n_draws, seed), stats

    targets = build_all_targets(bundle)
    n = len(features)
    min_train = max(int(n * 0.5), 60)
    # Compute budget scales down when the task grid is large (e.g. 10 assets x
    # 2 horizons -> 20 tasks x 5 fits) so wide units stay under the unit clock.
    n_tasks = len(tasks)
    if n_tasks <= 4:
        n_folds, max_iter, row_cap = 4, 300, None
    elif n_tasks <= 10:
        n_folds, max_iter, row_cap = 3, 200, 4000
    else:
        n_folds, max_iter, row_cap = 2, 150, 3000

    members = list(ensemble) if ensemble else [model_kind]
    scale = regime_scale(bundle) if bundle.target_frequency == "daily" else 1.0
    stats["regime_scale"] = scale

    member_samples, member_scores = [], []
    for mi, kind in enumerate(members):
        mus, resid_models, resid_frame = _tasks_for_kind(
            bundle, features, targets, tasks, kind, seed + 1000 * mi,
            n_folds, min_train, max_iter, stats if mi == 0 else
            stats.setdefault("member_tasks", {}).setdefault(kind, {}),
            row_cap=row_cap)
        flat = joint_samples(tasks, mus, resid_models, resid_frame, n_draws,
                             seed=seed + 1000 * mi, regime_scale=scale)
        samp = to_output_tensor(flat, bundle.target_assets, bundle.horizons, tasks)
        member_samples.append(samp)
        # raw per-task validation CRPS on this member's OOF residuals
        scores = []
        for ti in range(len(tasks)):
            rser = resid_frame[ti].dropna() if resid_frame is not None else pd.Series(dtype=float)
            scores.append(_member_crps(resid_models[ti], rser, seed + ti))
        member_scores.append(scores)

    if len(member_samples) > 1:
        # Normalise per task by the cross-member mean so wide-scale tasks
        # (e.g. JPY ~140) do not drown narrow ones (UST ~4), then average.
        M = np.array(member_scores, dtype=float)          # [members, tasks]
        finite = np.isfinite(M)
        denom = np.where(finite, M, np.nan)
        denom = np.nanmean(denom, axis=0)
        denom = np.where(np.isfinite(denom) & (denom > 0), denom, 1.0)
        norm = np.where(finite, M / denom[None, :], 10.0)  # missing score = bad
        member_scores = norm.mean(axis=1).tolist()
        from .ensemble import pool_draws, inverse_score_weights
        weights = inverse_score_weights(member_scores)
        stats["ensemble"] = {"members": members, "scores": member_scores,
                             "weights": weights}
        samples = pool_draws(member_samples, weights, n_draws, seed=seed)
    else:
        samples = member_samples[0]
    return samples, stats


def _fallback_task(bundle, asset, horizon):
    """Persistence centre + residual sigma from historical steps."""
    try:
        s = bundle.asset_series(asset).dropna()
    except KeyError:
        s = pd.Series(dtype=float)
    n_steps = _horizon_steps(bundle).get((asset, horizon), float(horizon))
    if bundle.target_type == "log_return":
        steps = np.log1p(s[s > -1])
        mu = float(steps.mean()) * n_steps if len(steps) else 0.0
        sd = float(steps.std()) * np.sqrt(n_steps) if len(steps) else 1.0
        resid = pd.Series(dtype=float)
    else:
        diffs = _diffs_no_gaps(s).dropna()
        mu = float(s.iloc[-1]) if len(s) else 0.0
        sd = float(diffs.std()) * np.sqrt(n_steps) if len(diffs) else 1.0
        resid = diffs - diffs.mean() if len(diffs) else pd.Series(dtype=float)
    return mu, {"mu": 0.0, "sigma": max(sd, 1e-8), "nu": np.inf}, resid


# ---------------------------------------------------------------------------
# CLI / deliverables
# ---------------------------------------------------------------------------

def _find_unit_dir(panels_arg: pathlib.Path) -> pathlib.Path:
    for cand in (panels_arg, panels_arg.parent):
        if (cand / "card.toml").exists():
            return cand
    raise SystemExit(f"card.toml not found near {panels_arg}")


def _write_deliverables(out_path: pathlib.Path, samples: np.ndarray,
                        bundle: DataBundle, n_draws: int, stats: dict,
                        method: str) -> None:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    assets, horizons = bundle.target_assets, bundle.horizons
    rows = (
        {"draw": d, "asset": a, "horizon": h, "value": float(samples[d, ai, hi])}
        for d in range(n_draws)
        for ai, a in enumerate(assets)
        for hi, h in enumerate(horizons)
    )
    df = pd.DataFrame(list(rows)).astype(
        {"draw": "int32", "asset": "string", "horizon": "int32", "value": "float64"})
    df.to_parquet(out_path, index=False)

    (out_dir / "forecast_meta.json").write_text(json.dumps({
        "unit_id": bundle.card_id,
        "asof": bundle.asof,
        "representation": "samples",
        "asset_ids": assets,
        "horizons": horizons,
        "n_draws": int(n_draws),
        "target": bundle.target_type,
        "rationale": {"file": "forecast_rationale.md", "method": method},
    }, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Forecast rationale — {bundle.card_id}",
        "",
        f"As of **{bundle.asof}**, joint distribution over {', '.join(assets)} "
        f"at horizon(s) {horizons} ({bundle.target_frequency} {bundle.target_type}). "
        f"{n_draws} draws.",
        "",
        "## Method",
        "",
        method,
        "",
        "## Per-task anchors",
        "",
        "| task | mu | sigma | note |",
        "|---|---|---|---|",
    ]
    for task, info in stats.get("tasks", {}).items():
        if isinstance(info, dict):
            lines.append(
                f"| {task} | {info.get('mu', float('nan')):.4f} | "
                f"{info.get('sigma', float('nan')):.4f} | "
                f"{'FALLBACK' if info.get('fallback') else 'model'} |")
    lines += [
        "",
        "## Dependence and text",
        "",
        f"Student-t copula on residual correlation across the (asset,horizon) grid; "
        f"regime scale {stats.get('regime_scale', 1.0):.2f}. "
        f"Text documents read: {stats.get('text_docs_used', 0)}.",
    ]
    (out_dir / "forecast_rationale.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast",
                                description="Track-2 custom model (ML + joint draws).")
    p.add_argument("--panels", type=pathlib.Path, required=True)
    p.add_argument("--text", type=pathlib.Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--card", type=pathlib.Path, default=None)
    p.add_argument("--n-draws", type=int, default=None)
    p.add_argument("--seed", type=int,
                 default=int(os.environ.get("QFBENCH_SEED", "0")))
    p.add_argument("--model", default="hgb",
                   choices=["hgb", "ridge", "persistence", "hgb+latent"])
    p.add_argument("--ensemble", default="hgb,ridge,persistence",
                   help="comma-separated member kinds for draw pooling; "
                        "'none' runs the single --model")
    p.add_argument("--no-text", action="store_true")
    a = p.parse_args(argv)

    unit_dir = _find_unit_dir(a.panels)
    bundle = load_unit(unit_dir)
    if a.asof and a.asof != bundle.asof:
        print(f"note: --asof {a.asof} overrides card-derived as-of "
              f"{bundle.asof}", file=sys.stderr)
        bundle.asof = a.asof
    # F4 tail-from-text cards need deeper tails: README recommends >= 1000 draws.
    family_floor = 1000 if str(bundle.card_family).upper() == "T2-F4" else DEFAULT_DRAWS
    floor = max(bundle.n_draws_min, family_floor, 200)
    n_draws = max(a.n_draws or floor, floor)
    if n_draws > MAX_DRAWS:
        raise SystemExit(f"--n-draws {n_draws} exceeds ceiling {MAX_DRAWS}")

    members = (None if a.ensemble.lower() in ("none", "") else
               tuple(s.strip() for s in a.ensemble.split(",")))
    warnings.simplefilter("ignore")
    try:
        samples, stats = run_bundle(bundle, n_draws=n_draws, seed=a.seed,
                                    model_kind=a.model, use_text=not a.no_text,
                                    ensemble=members)
    except Exception as exc:  # noqa: BLE001 - last-resort: always emit output
        print(f"pipeline failed ({type(exc).__name__}: {exc}); "
              f"emitting joint gaussian walk", file=sys.stderr)
        stats = {"unit": bundle.card_id, "asof": bundle.asof, "tasks": {},
                 "fallback": True, "reason": f"pipeline error: {exc}"[:200]}
        samples = _gaussian_walk(bundle, n_draws, a.seed)
    method = ("gradient-boosted point models + walk-forward residual "
              "distribution + Student-t copula joint draws"
              if not stats.get("fallback") else
              "joint gaussian random walk (pipeline fallback)")
    if members:
        w = stats.get("ensemble", {}).get("weights", [])
        method += (" | ensemble " + "+".join(members) +
                   " weights " + "/".join(f"{x:.2f}" for x in w))
    _write_deliverables(a.out, samples, bundle, n_draws, stats, method)
    print(f"wrote {a.out} ({n_draws} draws x {len(bundle.target_assets)} assets "
          f"x {len(bundle.horizons)} horizons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())