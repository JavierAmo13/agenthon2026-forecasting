"""Run the custom_model pipeline over every unit and report pass/fail.

Two modes:
  --mode structural : loader -> features -> targets -> dataset only (fast;
                      catches missing assets, odd panels, bad frequencies)
  --mode forecast   : full run_bundle + write forecast.parquet + schema check

Usage:
  python run_all_units.py --mode structural
  python run_all_units.py --mode forecast --model ridge --workers 6
  python run_all_units.py --mode forecast --model hgb --units "F3|F4" --limit 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent))

UNITS_DIR = pathlib.Path("units")
OUT_DIR = pathlib.Path("out_units")


def _check_parquet(path: pathlib.Path, bundle) -> list[str]:
    """Schema-level checks on a written forecast.parquet."""
    import pandas as pd
    errs = []
    df = pd.read_parquet(path)
    need = {"draw", "asset", "horizon", "value"}
    if set(df.columns) != need:
        errs.append(f"columns {sorted(df.columns)} != {sorted(need)}")
        return errs
    if not df["value"].notna().all() or not df["value"].map(lambda v: v == v and abs(v) < 1e30).all():
        errs.append("non-finite or extreme values in 'value'")
    n_draws = df["draw"].nunique()
    if n_draws < max(200, bundle.n_draws_min):
        errs.append(f"n_draws {n_draws} < min {max(200, bundle.n_draws_min)}")
    want = {(a, h) for a in bundle.target_assets for h in bundle.horizons}
    per_draw = df.groupby("draw").apply(
        lambda g: set(zip(g["asset"], g["horizon"])) == want, include_groups=False)
    if not per_draw.all():
        errs.append("some draws lack the full asset x horizon grid")
    if set(df["asset"]) != set(bundle.target_assets):
        errs.append(f"assets {sorted(df['asset'])} != declared {sorted(bundle.target_assets)}")
    if set(df["horizon"]) != set(bundle.horizons):
        errs.append(f"horizons {sorted(df['horizon'])} != declared {sorted(bundle.horizons)}")
    return errs


def run_one(unit_dir: str, mode: str, model: str, n_draws: int | None,
            use_text: bool, write_out: bool, ensemble: tuple | None = None) -> dict:
    """Run one unit; return a result dict (never raises)."""
    import warnings
    warnings.simplefilter("ignore")
    t0 = time.time()
    u = pathlib.Path(unit_dir)
    res: dict = {"unit": u.name, "mode": mode, "model": model, "ok": True}
    try:
        from custom_model.data.loader import load_unit
        from custom_model.data.features import build_features
        from custom_model.data.targets import build_all_targets
        from custom_model.data.dataset import build_dataset

        bundle = load_unit(u)
        res.update(family=bundle.card_family, asof=bundle.asof,
                   freq=bundle.target_frequency, ttype=bundle.target_type,
                   n_assets=len(bundle.target_assets), horizons=bundle.horizons,
                   n_texts=len(bundle.texts),
                   panels={k: list(v.shape) for k, v in bundle.panels.items()})

        features = build_features(bundle.panels, bundle.asof)
        res["n_features"] = int(features.shape[1]) if not features.empty else 0
        res["n_rows"] = int(features.shape[0]) if not features.empty else 0
        targets = build_all_targets(bundle)
        res["n_target_cols"] = int(targets.shape[1])

        monthly = bundle.target_frequency == "monthly"
        for a in bundle.target_assets:
            for h in bundle.horizons:
                ds = build_dataset(features, targets, a, h, monthly=monthly)
                if len(ds.y) == 0 or ds.y.notna().sum() == 0:
                    raise ValueError(f"empty training set for {a}__h{h}")
        res["structural"] = "ok"

        if mode == "forecast":
            from custom_model.forecast import run_bundle, _write_deliverables
            n = n_draws or max(bundle.n_draws_min, 500, 200)
            samples, stats = run_bundle(bundle, n_draws=n, seed=0,
                                        model_kind=model, use_text=use_text,
                                        ensemble=ensemble)
            res["stats"] = stats
            if write_out:
                out = OUT_DIR / u.name / "forecast.parquet"
                _write_deliverables(out, samples, bundle, n, stats,
                                    f"run_all_units sweep ({model})")
                errs = _check_parquet(out, bundle)
                res["parquet"] = str(out)
                if errs:
                    res["schema_errors"] = errs
                    raise ValueError("; ".join(errs))
    except Exception as exc:  # noqa: BLE001
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
        res["trace"] = traceback.format_exc()[-3000:]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["structural", "forecast"], default="structural")
    ap.add_argument("--model", default="hgb",
                    choices=["hgb", "ridge", "persistence", "hgb+latent"])
    ap.add_argument("--ensemble", default="hgb,ridge,persistence",
                    help="member kinds for draw pooling; 'none' = single --model")
    ap.add_argument("--units", default=None, help="regex filter on unit dir names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-draws", type=int, default=None)
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-write", action="store_true",
                    help="forecast mode: run but do not write parquet outputs")
    ap.add_argument("--report", default="units_report.json")
    a = ap.parse_args()

    dirs = [d for d in sorted(UNITS_DIR.iterdir()) if d.is_dir()]
    if a.units:
        rx = re.compile(a.units)
        dirs = [d for d in dirs if rx.search(d.name)]
    if a.limit:
        dirs = dirs[: a.limit]
    members = (None if a.ensemble.lower() in ("none", "") else
               tuple(s.strip() for s in a.ensemble.split(",")))
    print(f"{len(dirs)} units | mode={a.mode} model={a.model} "
          f"ensemble={members} workers={a.workers}")

    results = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_one, str(d), a.mode, a.model, a.n_draws,
                              not a.no_text, not a.no_write, members): d.name
                    for d in dirs}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(dirs)}] {r['unit']}: "
                      f"{'OK' if r['ok'] else 'FAIL ' + str(r.get('error',''))[:140]}"
                      f" ({r['seconds']}s)", flush=True)
    else:
        for i, d in enumerate(dirs, 1):
            r = run_one(str(d), a.mode, a.model, a.n_draws,
                        not a.no_text, not a.no_write, members)
            results.append(r)
            print(f"[{i}/{len(dirs)}] {d.name}: "
                  f"{'OK' if r['ok'] else 'FAIL ' + str(r.get('error',''))[:140]}"
                  f" ({r['seconds']}s)", flush=True)

    pathlib.Path(a.report).write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    fails = [r for r in results if not r["ok"]]
    total_s = sum(r["seconds"] for r in results)
    print(f"\n{len(results) - len(fails)}/{len(results)} OK | "
          f"{len(fails)} FAIL | total {total_s:.0f}s -> {a.report}")
    for r in fails:
        print("  FAIL", r["unit"], "->", r.get("error", "")[:160])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())