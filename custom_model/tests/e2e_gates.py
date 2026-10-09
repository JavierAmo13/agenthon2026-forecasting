
"""E2E: run the verb on a unit, then replay the REAL gates g0-g3 via the
official scoring CLI (gates-only mode, no realized needed).

    py -3.13 v3_src/custom_model/tests/e2e_gates.py <unit_dir>

Exits 0 only when the official verifier says admissible.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def gates(out_dir: pathlib.Path, unit: pathlib.Path) -> dict:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = (f"{ROOT / 't2_repo'};{ROOT / 'common_repo' / 'common'};"
                         + env.get("PYTHONPATH", ""))
    p = subprocess.run(
        [sys.executable, "-m", "qfbench2_track_forecasting.scoring", "score",
         "--card", str(unit / "card.toml"),
         "--forecast", str(out_dir / "forecast.parquet")],
        capture_output=True, text=True, env=env, cwd=str(ROOT))
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"parse_error": p.stdout[-1500:], "stderr": p.stderr[-1500:]}


def run_unit(unit: pathlib.Path, out_dir: pathlib.Path) -> dict:
    unit, out_dir = pathlib.Path(unit), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "v3_src") + ";" + env.get("PYTHONPATH", "")
    import tomllib
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    asof = str(card.get("forecast", {}).get("asof")
               or card.get("provenance", {}).get("data_cutoff"))[:10]
    p = subprocess.run(
        [sys.executable, "-m", "custom_model.forecast",
         "--panels", str(unit / "panels" if (unit / "panels").is_dir() else unit),
         "--text", str(unit / "text"),
         "--asof", asof, "--out", str(out_dir / "forecast.parquet"),
         "--no-text"],
        capture_output=True, text=True, env=env, cwd=str(ROOT))
    ok = p.returncode == 0
    print(p.stdout.strip()[-400:])
    if p.stderr.strip():
        print("stderr:", p.stderr.strip()[-400:])
    if not ok:
        return {"forecast_failed": p.returncode}
    return gates(out_dir, unit)


if __name__ == "__main__":
    units = ([pathlib.Path(x) for x in sys.argv[1:]]
             if len(sys.argv) > 1
             else [ROOT / "t2_repo/units/t2-EXAMPLE-ust-curve-1m",
                   ROOT / "t2_repo/units/t2-F1-cpi-glidepath-2023",
                   ROOT / "t2_repo/units/t2-F3-em-transfer-joint-2013",
                   ROOT / "t2_repo/units/t2-F4-covid-mkt-2020"])
    for u in units:
        print("=" * 60, "\n", u.name)
        res = run_unit(u, ROOT / "v3_src/_runs" / u.name)
        print(json.dumps(res, indent=2)[:1200])
