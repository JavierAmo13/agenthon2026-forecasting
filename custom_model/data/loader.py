
"""Unit loader: enters a unit directory and returns a DataBundle.

Reads card.toml (authoritative targets), forecast_spec.json (optional),
all *.parquet panels (long format: date, asset, value, panel_id) and the
text/ corpus (corpus_index.json + *.txt).

v2: declared target assets are checked against the panels; missing ones
land on bundle.missing_assets (explicit transfer route downstream, never
a silent substitution). The document `timestamp` is kept as the contract
availability date — no finer semantics are inferred.
"""

from __future__ import annotations

import json
import pathlib
import tomllib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_ASSET_COLS = ("asset", "asset_id")


@dataclass
class TextDoc:
    doc_id: str
    timestamp: str      # availability date supplied by the contract
    path: pathlib.Path
    doc_type: str = ""
    text: str = ""


@dataclass
class DataBundle:
    unit_dir: pathlib.Path
    card_id: str
    card_family: str
    asof: str
    target_assets: list[str]
    horizons: list[int]
    target_type: str
    target_frequency: str
    n_draws_min: int
    panels: dict[str, pd.DataFrame]
    panel_specs: dict[str, Any]
    texts: list[TextDoc]
    spec: dict[str, Any] = field(default_factory=dict)
    card: dict[str, Any] = field(default_factory=dict)
    observation_periods: list[str] | None = None
    missing_assets: list[str] = field(default_factory=list)

    def asset_series(self, asset: str, upto_asof: bool = True) -> pd.Series:
        """History of one asset as a datetime-indexed Series, whichever panel holds it."""
        asof = pd.Timestamp(self.asof)
        for df in self.panels.values():
            col = next((c for c in _ASSET_COLS if c in df.columns), None)
            if col is None or "value" not in df.columns:
                continue
            sub = df[df[col].astype(str) == asset]
            if sub.empty:
                continue
            sub = sub.sort_values("date")
            if upto_asof:
                sub = sub[sub["date"] <= asof]
            if not sub.empty:
                return pd.Series(sub["value"].to_numpy(dtype=float),
                                 index=pd.DatetimeIndex(sub["date"]))
        raise KeyError(f"asset {asset!r} not present in any panel")

    def available_assets(self) -> list[str]:
        return [a for a in self.target_assets if a not in self.missing_assets]


def _load_spec(unit_dir: pathlib.Path) -> dict[str, Any]:
    p = unit_dir / "forecast_spec.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_card(unit_dir: pathlib.Path) -> dict[str, Any]:
    p = unit_dir / "card.toml"
    if not p.exists():
        raise FileNotFoundError(f"card.toml missing in {unit_dir}")
    return tomllib.loads(p.read_text(encoding="utf-8"))


def _load_panels(unit_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    files = sorted(unit_dir.glob("*.parquet"))
    if not files and (unit_dir / "panels").is_dir():
        files = sorted((unit_dir / "panels").glob("*.parquet"))
    panels = {}
    for f in files:
        df = pd.read_parquet(f)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10))
        panels[f.stem] = df
    return panels


def _load_texts(unit_dir: pathlib.Path) -> list[TextDoc]:
    text_dir = unit_dir / "text"
    if not text_dir.is_dir():
        return []
    index_path = text_dir / "corpus_index.json"
    docs: list[TextDoc] = []
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        for d in idx.get("documents", []):
            f = text_dir / d.get("file", "")
            docs.append(TextDoc(
                doc_id=d.get("doc_id", f.stem), timestamp=d.get("timestamp", ""),
                path=f, doc_type=d.get("doc_type", ""),
                text=f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""))
        return docs
    for f in sorted(text_dir.glob("*.txt")):
        docs.append(TextDoc(doc_id=f.stem, timestamp="", path=f,
                            text=f.read_text(encoding="utf-8", errors="replace")))
    return docs


def _resolve_asof(unit_dir: pathlib.Path, spec: dict, card: dict) -> str:
    idx = unit_dir / "text" / "corpus_index.json"
    if idx.exists():
        data = json.loads(idx.read_text(encoding="utf-8"))
        if data.get("asof"):
            return str(data["asof"])[:10]
    for cand in (card.get("provenance", {}).get("data_cutoff"),
                 card.get("text", {}).get("cutoff"), spec.get("asof")):
        if cand:
            return str(cand)[:10]
    ends = [p.get("end_date") for p in spec.get("panels", []) if p.get("end_date")]
    if ends:
        return max(ends)[:10]
    raise ValueError(f"cannot resolve as-of date for {unit_dir.name}")


def _present_assets(panels: dict[str, pd.DataFrame]) -> set[str]:
    out: set[str] = set()
    for df in panels.values():
        col = next((c for c in _ASSET_COLS if c in df.columns), None)
        if col is not None:
            out.update(df[col].astype(str).unique())
    return out


def load_unit(unit_dir: str | pathlib.Path) -> DataBundle:
    unit_dir = pathlib.Path(unit_dir)
    spec = _load_spec(unit_dir)
    card = _load_card(unit_dir)
    panels = _load_panels(unit_dir)
    texts = _load_texts(unit_dir)

    tgt = card.get("targets") or spec.get("targets") or {}
    spec_tgt = spec.get("targets", {})
    assets = list(tgt.get("asset_ids") or spec_tgt.get("asset_ids") or [])
    horizons = [int(h) for h in (tgt.get("horizons") or spec_tgt.get("horizons") or [])]
    if not assets or not horizons:
        raise ValueError(f"{unit_dir.name}: card/spec lacks target assets/horizons")

    n_draws_min = (spec.get("n_draws_min")
                   or card.get("scoring", {}).get("params", {}).get("n_draws_min")
                   or 200)
    panel_specs = {p.get("panel_id", k): p for k, p in
                   ((p.get("panel_id"), p) for p in spec.get("panels", []))} if spec.get("panels") else {}
    for pid, pmeta in (card.get("panels") or {}).items():
        if isinstance(pmeta, dict):
            panel_specs.setdefault(pid, pmeta)

    bundle = DataBundle(
        unit_dir=unit_dir,
        card_id=card.get("task", {}).get("id", spec.get("card_id", unit_dir.name)),
        card_family=spec.get("card_family") or card.get("metadata", {}).get("category", ""),
        asof=_resolve_asof(unit_dir, spec, card),
        target_assets=assets,
        horizons=horizons,
        target_type=tgt.get("target_type", spec_tgt.get("target_type", "level")),
        target_frequency=tgt.get("target_frequency")
            or card.get("metadata", {}).get("target_frequency", "daily"),
        n_draws_min=int(n_draws_min),
        panels=panels,
        panel_specs=panel_specs,
        texts=texts,
        spec=spec,
        card=card,
        observation_periods=spec_tgt.get("observation_periods"),
    )
    # Explicit transfer route: assets declared but absent from every panel are
    # flagged; downstream they get the statistical fallback, not a proxy made up on the fly.
    present = _present_assets(panels)
    bundle.missing_assets = [a for a in assets if a not in present]
    return bundle
