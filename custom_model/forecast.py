
"""forecast.py — the orchestrator. Implements the `forecast` verb:

    forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD \
             --out /output/forecast.parquet

Flow (plan 3.10 — valid forecast first, improvements gated by deadline):
    load_unit -> base gaussian walk (kept as the early valid forecast) ->
    features -> text events -> datasets with label availability ->
    per-member point model + OOF residuals -> residual law G_i ->
    regime + single-point text sigma adjust -> t-copula joint draws ->
    shrunk ensemble pooling -> atomic write of exactly 3 deliverables.

Every stage degrades gracefully and the internal deadline can hand back the
early base forecast — a pipeline abort never leaves the unit without output.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

from .data.loader import load_unit

DEFAULT_DRAWS = 1000      # plan: start at 1000 joint draws (parser: 200..20000)
MAX_DRAWS = 20_000
OUT_COLS = ("draw", "asset", "horizon", "value")   # canonical scorer schema

# Experimental-vs-reference switch: reference keeps v1 behaviour
# (pure empirical marginal, unshrunk correlation, unshrunk weights).
PRESETS = {
    "experimental": {"eta": 0.15, "lam": 0.35, "gamma": 0.25},
    "reference": {"eta": 0.0, "lam": 0.0, "gamma": 0.0},
}


# --- statistical fallback: correlated gaussian walk -------------------------

def _diffs_no_gaps(s: pd.Series) -> pd.Series:
    d = s.diff()
    idx = pd.DatetimeIndex(s.index)
    step = pd.Series(idx, index=idx).diff().dt.days
    if step.notna().sum() == 0:
        return d
    return d.where(step <= max(float(step.median()) * 10.0, 5.0))


def _gaussian_walk(bundle, n_draws: int, seed: int) -> np.ndarray:
    """Correlated gaussian random walk per asset — the guaranteed floor.
    Per-step innovations share one correlation matrix; horizons are
    cumulative sums of the same path, so draws are coherent across h."""
    from .data.targets import horizon_step_map

    rng = np.random.default_rng(seed)
    assets, horizons = bundle.target_assets, bundle.horizons
    steps_map = horizon_step_map(bundle)
    max_steps = max(int(v) for v in steps_map.values())

    diffs, last, drift = {}, {}, {}
    for a in assets:
        try:
            s = bundle.asset_series(a).dropna()
        except KeyError:
            s = pd.Series(dtype=float)
        if bundle.target_type == "log_return":
            d = np.log1p(s[s > -1]).dropna()
            last[a] = 0.0                       # target starts at 0
            drift[a] = 0.5 * float(d.mean()) if len(d) else 0.0
        else:
            d = _diffs_no_gaps(s).dropna()
            last[a] = float(s.iloc[-1]) if len(s) else 0.0
            drift[a] = 0.0
        diffs[a] = d

    wide = pd.DataFrame(diffs)
    sd = np.array([max(float(wide[a].std()), 1e-4) if a in wide else 1.0
                   for a in assets])
    if len(assets) > 1 and wide.notna().sum().min() >= 20:
        from .joint import _nearest_psd_corr
        C = _nearest_psd_corr(wide.corr().to_numpy(dtype=float))
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(len(assets))
    else:
        L = np.eye(len(assets))

    eps = rng.standard_normal((n_draws, max_steps, len(assets))) @ L.T
    eps = eps * sd[None, None, :]
    cum = np.cumsum(eps, axis=1)

    out = np.empty((n_draws, len(assets), len(horizons)))
    for ai, a in enumerate(assets):
        for hi, h in enumerate(horizons):
            st = max(1, int(steps_map.get((a, h), h)))
            out[:, ai, hi] = last[a] + drift[a] * st + cum[:, st - 1, ai]
    return out


def _find_unit_dir(panels_arg: pathlib.Path) -> pathlib.Path:
    """The CLI gets a panels dir; the unit root is whichever ancestor/child
    actually contains card.toml."""
    p = pathlib.Path(panels_arg)
    cands = [p, p.parent] + (list(p.iterdir()) if p.is_dir() else [])
    for cand in cands:
        if pathlib.Path(cand).is_dir() and (pathlib.Path(cand) / "card.toml").exists():
            return pathlib.Path(cand)
    return p


# --- one task: point model + residual law -----------------------------------

def _task_fit(bundle, ds, kind: str, steps: int, n_folds: int,
              min_train: int | None, seed: int, max_iter: int, eta: float):
    """Fit one (asset, horizon): returns (mu, resid_model, oof_resid)."""
    from .model import fit_model, predict, walk_forward_residuals
    from .probabilistic import fit_residual_model
    from .data.dataset import feature_mask

    resid = walk_forward_residuals(ds.X, ds.y, kind, n_folds=n_folds,
                                   min_train=min_train, embargo=max(steps, 1),
                                   seed=seed, max_iter=max_iter,
                                   asset=ds.asset, target_type=bundle.target_type)
    m = feature_mask(ds.X)
    mdl = fit_model(ds.X.loc[:, m], ds.y, kind, seed, max_iter=max_iter,
                    asset=ds.asset, target_type=bundle.target_type)
    mu = float(predict(mdl, ds.X_pred.loc[:, m])[0])
    rm = fit_residual_model(resid, eta=eta)
    return mu, rm, resid


def _fallback_task(bundle, asset: str, steps: int, eta: float):
    """Persistence mean + historical-diff residual law — always available."""
    from .probabilistic import fit_residual_model

    try:
        s = bundle.asset_series(asset).dropna()
    except KeyError:
        s = pd.Series(dtype=float)
    if bundle.target_type == "log_return":
        d = np.log1p(s[s > -1]).dropna()
        fwd = d.rolling(int(steps), min_periods=1).sum()
        mu = float(d.mean() * steps) if len(d) else 0.0
    else:
        d = _diffs_no_gaps(s).dropna()
        fwd = s - s.shift(int(steps))
        mu = float(s.iloc[-1]) if len(s) else 0.0
    resid = (fwd - fwd.shift(int(steps))).dropna() if len(fwd) > steps else d
    rm = fit_residual_model(resid, eta=eta)
    return mu, rm, resid


def _numeric_summary(bundle) -> str:
    """Brief numeric context for the optional synthesis call."""
    parts = []
    for a in bundle.available_assets()[:6]:
        try:
            s = bundle.asset_series(a).dropna()
            if len(s) > 30:
                chg = (s.iloc[-1] / s.iloc[-22] - 1) if (s > 0).all() else s.iloc[-1] - s.iloc[-22]
                parts.append(f"{a}: last={s.iloc[-1]:.4g}, 21-obs chg={chg:.4g}")
        except KeyError:
            continue
    return "; ".join(parts)


# --- the pipeline -------------------------------------------------------------

def run_bundle(bundle, *, n_draws: int, seed: int, model_kind: str = "hgb",
               use_text: bool = True,
               ensemble: tuple[str, ...] | None = ("hgb", "ridge", "persistence"),
               deadline: float | None = None, eta: float = 0.15,
               lam: float = 0.35, gamma: float = 0.25) -> tuple[np.ndarray, dict]:
    """Full pipeline for one unit. Returns (samples [n,a,h], stats)."""
    from .data.features import build_features
    from .data.targets import build_all_targets, horizon_step_map
    from .data.dataset import build_dataset
    from .probabilistic import fit_text_calibrator, text_sigma_factor
    from .joint import joint_samples, to_output_tensor
    from .regime import regime_scale
    from .ensemble import pool_draws, member_weights
    from .validation import crps_gaussian
    from . import llm, text_features as tf

    stats = {"unit": bundle.card_id, "asof": bundle.asof, "family": bundle.card_family,
             "tasks": {}, "adjustments": {}, "fallback": False}
    expired = lambda: deadline is not None and time.time() > deadline

    # Stage 0: a complete valid forecast exists before any budget is spent.
    base = _gaussian_walk(bundle, n_draws, seed + 7)
    if bundle.missing_assets:
        stats["missing_assets"] = list(bundle.missing_assets)

    # Stage 1: features (truncated at as-of) + optional text state.
    feats = build_features(bundle.panels, bundle.asof)
    if feats.empty:
        return base, {**stats, "fallback": True, "reason": "empty features"}
    text_ok = False
    z_frame = pd.DataFrame(index=feats.index)
    if use_text and bundle.texts:
        budget = llm.Budget()
        extractions, llm_stats = llm.extract_corpus(bundle.texts, bundle,
                                                  budget, deadline)
        stats["llm"] = llm_stats
        text_feats, z_frame = tf.build_text_features(
            bundle.texts, extractions, feats.index, bundle.target_assets)
        if not text_feats.empty:
            feats = feats.join(text_feats, how="left")
            text_ok = bool(stats["llm"].get("events", 0) > 0)
    if expired():
        return base, {**stats, "fallback": True, "reason": "deadline after text"}

    # Stage 2: targets with availability contract + per-task datasets.
    targets_df, avail_df = build_all_targets(bundle)
    steps_map = horizon_step_map(bundle)
    # Full declared grid: missing assets stay in the output and get the
    # explicit fallback route (never a silent proxy).
    tasks = [(a, h) for a in bundle.target_assets for h in bundle.horizons]
    datasets = {}
    for t in tasks:
        try:
            datasets[t] = build_dataset(feats, targets_df, avail_df, t[0], t[1],
                                        cutoff=bundle.asof,
                                        monthly=bundle.target_frequency == "monthly")
        except Exception:
            datasets[t] = None

    n_tasks = max(len(tasks), 1)
    n_folds, max_iter = ((4, 300) if n_tasks <= 4 else
                         (3, 200) if n_tasks <= 10 else (2, 150))
    row_cap = 4000 if n_tasks <= 10 else 3000

    # Stage 3: regime + per-member fits. sigma multiplier per task bundles
    # numeric regime, temporal recalibration and the single text adjustment.
    reg = regime_scale(bundle)
    stats["adjustments"]["regime"] = reg["components"]
    text_cal, z_now = None, None
    members = ensemble or (model_kind,)
    member_tensors, member_scores, member_names = [], [], []
    resid_frames: dict[str, pd.DataFrame] = {}

    for mi, kind in enumerate(members):
        if expired():
            break
        mseed = seed * 1009 + mi * 97 + 13
        mus, resid_models, resid_cols, mus_map = [], [], {}, {}
        for t in tasks:
            ds = datasets.get(t)
            ok = ds is not None and len(ds.X) >= 30 and ds.y.notna().sum() >= 30
            try:
                if ok and len(ds.X) > row_cap:
                    ds = type(ds)(ds.X.iloc[-row_cap:], ds.y.iloc[-row_cap:],
                                  ds.X_pred, ds.feature_names, ds.asset,
                                  ds.horizon,
                                  ds.avail.iloc[-row_cap:] if ds.avail is not None else None)
                steps = steps_map.get(t, int(t[1]))
                mu, rm, resid = (_task_fit(bundle, ds, kind, steps,
                                           n_folds, None, mseed, max_iter, eta)
                                 if ok else
                                 _fallback_task(bundle, t[0], steps, eta))
            except Exception:
                mu, rm, resid = _fallback_task(bundle, t[0],
                                               steps_map.get(t, int(t[1])), eta)
                ok = False
            mus.append(mu)
            resid_models.append(rm)
            resid_cols[t] = resid
            stats["tasks"][f"{t[0]}__h{t[1]}"] = {
                "mu": mu, "sigma": rm["sigma"], "nu": rm.get("nu"),
                "n_oof": rm.get("n", 0), "fallback": not ok}

        resid_frame = pd.DataFrame(resid_cols)
        resid_frames[kind] = resid_frame

        # text calibrator: pooled |resid| ~ text state, once per member,
        # fitted on historical origins only (no new House calls).
        # Abstention means sigma_final = sigma_base (factor 1.0) — the task
        # still forecasts — and the reason is logged. Limitation of this
        # implementation, not of the method: monthly units lack a validated
        # alignment between text dates, prediction origins and residuals.
        if not text_ok:
            text_reason = "no_events" if use_text else "text_disabled"
        elif bundle.target_frequency == "monthly":
            text_reason = "monthly_alignment_not_implemented"
        else:
            pooled = resid_frame.abs().mean(axis=1)
            z_hist = z_frame.reindex(pooled.index)
            text_cal = fit_text_calibrator(z_hist, pooled,
                                           sigma_base=float(np.nanmean(
                                               [m["sigma"] for m in resid_models])))
            z_now = tf.text_state_now(z_frame)
            text_reason = ("active" if (text_cal and z_now) else
                           "insufficient_pairs" if text_cal is None
                           else "no_text_state")
        txt_f = text_sigma_factor(z_now, text_cal)
        stats["adjustments"]["text_sigma"] = txt_f
        stats["adjustments"]["text_sigma_reason"] = text_reason
        mult = np.array([reg["scale"] * rm.get("scale_recent", 1.0) * txt_f
                         for rm in resid_models])

        draws = joint_samples(tasks, np.array(mus), resid_models, resid_frame,
                              n_draws, seed=mseed, sigma_mult=mult, lam=lam)
        tensor = to_output_tensor(draws, bundle.target_assets, bundle.horizons, tasks)
        member_tensors.append(tensor)
        member_names.append(kind)

        # member score: mean closed-form gaussian CRPS of each task's residual
        # law against its own OOF residuals, normalized by task scale (proxy).
        scs = []
        for j, t in enumerate(tasks):
            r = resid_frame[t].dropna().to_numpy()
            if r.size < 10:
                continue
            rm = resid_models[j]
            sd = rm["sigma"] * mult[j]
            scs.append(float(np.mean([crps_gaussian(rm["mu"], sd, ri)
                                      for ri in r[-200:]])))
        member_scores.append(float(np.mean(scs)) if scs else np.inf)

    if not member_tensors:
        return base, {**stats, "fallback": True, "reason": "deadline or no member fit"}

    weights = member_weights(member_scores, gamma)
    stats["ensemble"] = {"members": member_names, "weights": weights,
                         "scores": member_scores}
    samples = pool_draws(member_tensors, weights, n_draws, seed + 3)
    return samples, stats


# --- deliverables ------------------------------------------------------------

def _write_parquet_atomic(samples: np.ndarray, bundle, out: pathlib.Path) -> None:
    """Long format with the canonical schema OUT_COLS = (draw, asset,
    horizon, value). Atomic: tmp + os.replace."""
    assets, horizons = bundle.target_assets, bundle.horizons
    n_draws = samples.shape[0]
    df = pd.DataFrame({
        "draw": np.tile(np.arange(n_draws, dtype=np.int64),
                        len(assets) * len(horizons)),
        "asset": np.repeat(assets, len(horizons) * n_draws),
        "horizon": np.tile(np.repeat(horizons, n_draws), len(assets)),
        "value": samples.transpose(1, 2, 0).ravel(),
    })[list(OUT_COLS)]
    assert np.isfinite(df["value"]).all(), "non-finite draws in output"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)


def _write_deliverables(out: pathlib.Path, samples: np.ndarray, bundle,
                        stats: dict, method: str) -> None:
    """Exactly three files: forecast.parquet, forecast_meta.json,
    forecast_rationale.md. Temporaries are removed; nothing else is written
    to the output dir."""
    out = pathlib.Path(out)
    out_dir = out.parent
    _write_parquet_atomic(samples, bundle, out)

    meta = {"unit_id": bundle.card_id, "asof": bundle.asof,
            "family": bundle.card_family,
            "target_type": bundle.target_type,
            "target_frequency": bundle.target_frequency,
            "asset_ids": list(bundle.target_assets), "horizons": bundle.horizons,
            "representation": "samples",
            "rationale":{"file": "forecast_rationale.md", "method":"..."},
            "target": bundle.target_type,
            "n_draws": int(samples.shape[0]), "method": method,
            "custom": {"version": "2.0.0", "tasks": stats.get("tasks", {}),
                       "config": stats.get("config", {}),
                       "adjustments": stats.get("adjustments", {}),
                       "ensemble": stats.get("ensemble", {}),
                       "llm": stats.get("llm", {}),
                       "missing_assets": stats.get("missing_assets", []),
                       "fallback": stats.get("fallback", False)}}
    mp = out_dir / "forecast_meta.json"
    mtmp = mp.with_suffix(".tmp")
    mtmp.write_text(json.dumps(meta, default=str, indent=1), encoding="utf-8")
    os.replace(mtmp, mp)

    adj = stats.get("adjustments", {})
    reg = adj.get("regime", {})
    lines = [
        f"# Forecast rationale — {bundle.card_id}",
        "",
        f"Anchor (as-of): {bundle.asof} | family {bundle.card_family or 'n/a'} | "
        f"target {bundle.target_type}/{bundle.target_frequency}",
        f"Method: {method}",
        "",
        "## Data",
        f"Panels truncated at as-of; targets realize strictly after each origin "
        f"and rows whose label availability exceeds the cutoff are excluded "
        f"from training. Missing assets: "
        f"{', '.join(stats.get('missing_assets', [])) or 'none'}.",
        "",
        "## Adjustments applied",
        f"Regime numeric scale x{reg.get('numeric', 1.0):.2f} | text sigma "
        f"x{adj.get('text_sigma', 1.0):.2f} "
        f"(reason: {adj.get('text_sigma_reason', 'n/a')}) | "
        f"stress x{reg.get('stress', 1.0):.2f}",
        f"Text requests spent: {stats.get('llm', {}).get('requests', 0)} "
        f"(cap {stats.get('llm', {}).get('requests_max', 'n/a')}); "
        f"events used: {stats.get('llm', {}).get('events', 0)}.",
        "Note: the text->sigma calibrator is limited to daily-frequency "
        "units in this implementation — a limitation of the implementation, "
        "not of the method. Abstention leaves sigma unchanged; the forecast "
        "is always produced.",
        "",
        "## Component states",
        "eta/lam/gamma: see meta.custom.config (preset). "
        "text->sigma calibrator: " + adj.get('text_sigma_reason', 'n/a') + ". "
        "Per-task member weights: not implemented (would break joint "
        "scenarios). HMM regime: not implemented. Optional synthesis call: "
        "implemented, used only when budget remains.",
        "",
        "## Uncertainty",
        "Residual law per task: empirical marginal mixed with a Student-t "
        "tail (eta-bounded); joint dependence via a Student-t copula on "
        "rank-estimated, shrunk correlation. Falls back to a correlated "
        "gaussian walk if the model pipeline cannot run.",
        "",
        f"Fallback active: {bool(stats.get('fallback'))}"
        + (f" — {stats.get('reason')}" if stats.get("reason") else ""),
    ]
    rp = out_dir / "forecast_rationale.md"
    rtmp = rp.with_suffix(".tmp")
    rtmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(rtmp, rp)


def main(argv: list[str] | None = None) -> int:
    # one parallelism layer: pin BLAS threads, workers stay at our level
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "4")

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
                   choices=["hgb", "ridge", "persistence", "last", "hgb+latent"])
    p.add_argument("--ensemble", default="hgb,ridge,persistence",
                   help="comma-separated member kinds; 'none' runs only --model")
    p.add_argument("--time-budget", type=float, default=1200.0,
                   help="internal deadline in seconds (below the unit cap)")
    p.add_argument("--preset", default="experimental",
                   choices=sorted(PRESETS),
                   help="'reference' reproduces v1 behaviour; 'experimental' "
                        "enables the v2 candidates (eta, lam, gamma). "
                        "Explicit flags below override the preset.")
    p.add_argument("--eta", type=float, default=None,
                   help="Student-t mix weight in the marginal (0 = empirical)")
    p.add_argument("--lam", type=float, default=None,
                   help="correlation shrinkage toward equicorrelation base")
    p.add_argument("--gamma", type=float, default=None,
                   help="ensemble weight shrinkage toward uniform")
    p.add_argument("--no-text", action="store_true")
    a = p.parse_args(argv)

    preset = PRESETS[a.preset]
    a.eta = preset["eta"] if a.eta is None else a.eta
    a.lam = preset["lam"] if a.lam is None else a.lam
    a.gamma = preset["gamma"] if a.gamma is None else a.gamma

    unit_dir = _find_unit_dir(a.panels)
    bundle = load_unit(unit_dir)
    if a.asof and a.asof != bundle.asof:
        print(f"note: --asof {a.asof} overrides card-derived as-of "
              f"{bundle.asof}", file=sys.stderr)
        bundle.asof = a.asof

    floor = max(bundle.n_draws_min, DEFAULT_DRAWS, 200) #200
    n_draws = max(a.n_draws or floor, floor)
    if n_draws > MAX_DRAWS:
        raise SystemExit(f"--n-draws {n_draws} exceeds ceiling {MAX_DRAWS}")

    members = (None if a.ensemble.lower() in ("none", "") else
               tuple(s.strip() for s in a.ensemble.split(",")))
    warnings.simplefilter("ignore")
    deadline = time.time() + a.time_budget
    try:
        samples, stats = run_bundle(bundle, n_draws=n_draws, seed=a.seed,
                                    model_kind=a.model, use_text=not a.no_text,
                                    ensemble=members, deadline=deadline,
                                    eta=a.eta, lam=a.lam, gamma=a.gamma)
        stats["config"] = {"preset": a.preset, "eta": a.eta,
                           "lam": a.lam, "gamma": a.gamma}
    except Exception as exc:  # last-resort: always emit output
        print(f"pipeline failed ({type(exc).__name__}: {exc}); "
              f"emitting joint gaussian walk", file=sys.stderr)
        stats = {"unit": bundle.card_id, "asof": bundle.asof, "tasks": {},
                 "fallback": True, "reason": f"pipeline error: {exc}"[:200]}
        samples = _gaussian_walk(bundle, n_draws, a.seed)

    method = ("point models + OOF residual law + shrunk Student-t copula joint draws"
              if not stats.get("fallback") else
              "joint gaussian random walk (pipeline fallback)")
    if members and stats.get("ensemble"):
        w = stats["ensemble"].get("weights", [])
        method += (" | ensemble " + "+".join(stats["ensemble"]["members"]) +
                   " weights " + "/".join(f"{x:.2f}" for x in w))
    _write_deliverables(a.out, samples, bundle, stats, method)
    print(f"wrote {a.out} ({n_draws} draws x {len(bundle.target_assets)} assets "
          f"x {len(bundle.horizons)} horizons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
