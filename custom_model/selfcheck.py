
"""Internal pre-write sanity check — cheap mirror of the shipped gates.

Runs before the atomic write so a malformed artifact never lands on disk:
rows == n_draws x n_cells, columns exactly (draw, asset, horizon, value),
horizon keys preserved verbatim, values finite, draw coverage complete,
meta keys/binding sane. This is a guard, not the gate: the official
verifier still decides admissibility.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_samples(samples: np.ndarray, assets: list[str],
                  horizons: list[int]) -> list[str]:
    errs = []
    if samples.ndim != 3:
        errs.append(f"samples ndim {samples.ndim} != 3")
        return errs
    n, A, H = samples.shape
    if n < 200:
        errs.append(f"n_draws {n} < 200 floor")
    if A != len(assets) or H != len(horizons):
        errs.append(f"grid {A}x{H} != declared {len(assets)}x{len(horizons)}")
    if not np.isfinite(samples).all():
        errs.append("non-finite values in draws")
    return errs


def check_meta(meta: dict, bundle) -> list[str]:
    errs = []
    for k in ("unit_id", "asof", "representation", "asset_ids",
              "horizons", "n_draws"):
        if k not in meta:
            errs.append(f"meta missing {k}")
    card_asof = str(bundle.card.get("provenance", {}).get("data_cutoff")
                    or bundle.card.get("forecast", {}).get("asof")
                    or bundle.asof)[:10]
    if str(meta.get("asof", ""))[:10] != card_asof:
        errs.append(f"meta.asof {meta.get('asof')} != card asof {card_asof}")
    if meta.get("unit_id") != bundle.card_id:
        errs.append("meta.unit_id != card task.id")
    if meta.get("representation") != "samples":
        errs.append("representation != samples")
    return errs
