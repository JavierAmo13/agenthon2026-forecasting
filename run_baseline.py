import json
import pathlib
import tomllib

import pandas as pd

from baselines.base import ForecastRequest
from baselines.theta_arima import ThetaARIMABaseline


UNIT = pathlib.Path("units/t2-EXAMPLE-ust-curve-1m")
OUT = UNIT / "output"

OUT.mkdir(exist_ok=True)

card = tomllib.loads((UNIT / "card.toml").read_text())

assets = card["targets"]["asset_ids"]
horizons = card["targets"]["horizons"]
target = card["targets"]["target_type"]

panel = pd.read_parquet(UNIT / "rates_daily.parquet")

request = ForecastRequest(
    panels={"rates_daily": panel},
    asof="2024-06-28",
    asset_ids=assets,
    horizons=horizons,
    n_draws=500,
)

result = ThetaARIMABaseline().forecast(request)

print("Implementation:", result.metadata.get("implementation"))
print("Samples:", result.samples.shape)

rows = []

for draw in range(result.samples.shape[0]):
    for asset_i, asset in enumerate(assets):
        for horizon_i, horizon in enumerate(horizons):
            rows.append({
                "draw": draw,
                "asset": asset,
                "horizon": horizon,
                "value": float(result.samples[draw, asset_i, horizon_i]),
            })

pd.DataFrame(rows).to_parquet(
    OUT / "forecast.parquet",
    index=False,
)

metadata = {
    "unit_id": card["task"]["id"],
    "asof": "2024-06-28",
    "representation": "samples",
    "asset_ids": assets,
    "horizons": horizons,
    "n_draws": 500,
    "target": target,
}

(OUT / "forecast_meta.json").write_text(
    json.dumps(metadata, indent=2)
)

(OUT / "forecast_rationale.md").write_text(
    "# Baseline forecast\n\n"
    "Statistical Theta/AutoARIMA baseline. "
    "The public baseline does not consume the text corpus.\n"
)

print("Created:")
print(OUT / "forecast.parquet")
print(OUT / "forecast_meta.json")
print(OUT / "forecast_rationale.md")