
"""Unit loading: card.toml + forecast_spec.json + panels + text corpus.

One DataBundle per unit. Reads only the unit directory handed to the verb.
The as-of is resolved with precedence: --asof CLI arg (what the harness and
gate g2 use), then corpus_index.json `asof`, then card [forecast].asof /
[provenance].data_cutoff. Panels are truncated at the as-of on read.

Assets are looked up across ALL shipped panels (sorted order). A target asset
absent from every panel lands on `missing_assets` — the transfer route
(engine + text) handles it explicitly rather than fabricating a proxy.
"""

from __future__ import annotations

import json
import pathlib
import tomllib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ASSET_COLS = ("asset", "asset_id")
ISO = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class TextDoc:
    doc_id: str
    timestamp: str
    path: pathlib.Path
    doc_type: str = ""
    source: str = ""
    text: str = ""


@dataclass
class DataBundle:
    unit_dir: pathlib.Path
    card: dict
    spec: dict
    card_id: str
    family: str
    asof: str
    target_assets: list[str]
    horizons: list[int]
    target_type: str
    target_frequency: str
    value_unit: str
    n_draws_min: int
    panels: dict[str, pd.DataFrame]
    texts: list[TextDoc]
    present_assets: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    observation_periods: list[str] | None = None

    def series(self, asset: str, upto_asof: bool = True) -> pd.Series | None:
        """Datetime-indexed value series of an asset, first panel (sorted
        order) that carries it — M0's own selection rule (S3.1)."""
        cut = pd.Timestamp(self.asof) if upto_asof else None
        for stem in sorted(self.panels):
            df = self.panels[stem]
            col = next((c for c in _ASSET_COLS if c in df.columns), None)
            if col is None or "value" not in df.columns:
                continue
            sub = df[df[col].astype(str) == asset]
            if sub.empty:
                continue
            sub = sub.sort_values("date")
            if cut is not None:
                sub = sub[sub["date"] <= cut]
            if sub.empty:
                continue
            return pd.Series(
                pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=float),
                index=pd.DatetimeIndex(pd.to_datetime(
                    sub["date"].astype(str).str.slice(0, 10))),
            ).groupby(level=0).last().sort_index()
        return None

    def all_series(self, upto_asof: bool = True) -> dict[str, pd.Series]:
        """Every asset in every panel -> series (for context/cross-asset use).
        Target assets keep the first-panel-wins rule."""
        out: dict[str, pd.Series] = {}
        for stem in sorted(self.panels):
            df = self.panels[stem]
            col = next((c for c in _ASSET_COLS if c in df.columns), None)
            if col is None or "value" not in df.columns:
                continue
            for a in df[col].astype(str).unique():
                if a in out:
                    continue
                s = self.series(a, upto_asof)
                if s is not None:
                    out[a] = s
        return out


def _load_panels(unit: pathlib.Path, asof: str | None) -> dict[str, pd.DataFrame]:
    files = sorted(unit.glob("*.parquet"))
    pd_dir = unit / "panels"
    if pd_dir.is_dir():
        files += sorted(pd_dir.glob("*.parquet"))
    out: dict[str, pd.DataFrame] = {}
    for f in files:
        df = pd.read_parquet(f)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10))
            if asof:
                df = df[df["date"] <= pd.Timestamp(asof)]
        out[f.stem] = df.reset_index(drop=True)
    return out


def _load_texts(unit: pathlib.Path) -> list[TextDoc]:
    td = unit / "text"
    if not td.is_dir():
        return []
    docs: list[TextDoc] = []
    idx_p = td / "corpus_index.json"
    if idx_p.exists():
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
        for d in idx.get("documents", []):
            f = td / d.get("file", "")
            docs.append(TextDoc(
                doc_id=str(d.get("doc_id", f.stem)),
                timestamp=str(d.get("timestamp", ""))[:10],
                path=f, doc_type=str(d.get("doc_type", "")),
                source=str(d.get("source", "")),
                text=f.read_text(encoding="utf-8", errors="replace")
                if f.is_file() else ""))
        return docs
    for f in sorted(td.glob("*.txt")):
        docs.append(TextDoc(doc_id=f.stem, timestamp="", path=f,
                            text=f.read_text(encoding="utf-8", errors="replace")))
    return docs


def _resolve_asof(unit: pathlib.Path, card: dict, spec: dict,
                  cli_asof: str | None) -> str:
    # Data reads are bounded by the EARLIEST declared cutoff: the CLI --asof
    # (harness-issued) and the card's own trusted as-of. min() means a card
    # can never make us read past the harness cutoff nor vice versa.
    cands: list[str] = []
    if cli_asof and ISO.match(cli_asof):
        cands.append(cli_asof[:10])
    for cand in (card.get("forecast", {}).get("asof"),
                 card.get("provenance", {}).get("data_cutoff"),
                 spec.get("asof")):
        v = str(cand)[:10]
        if cand and ISO.match(v):
            cands.append(v)
            break  # first card-level hit only: forecast.asof before data_cutoff
    if cands:
        return min(cands)
    idx_p = unit / "text" / "corpus_index.json"
    if idx_p.exists():
        try:
            v = str(json.loads(idx_p.read_text(encoding="utf-8")).get("asof", ""))[:10]
            if ISO.match(v):
                return v
        except Exception:
            pass
    raise ValueError(f"cannot resolve as-of date for {unit.name}")


def load_unit(unit_dir: str | pathlib.Path, asof: str | None = None) -> DataBundle:
    unit_dir = pathlib.Path(unit_dir)
    card = tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))
    spec_p = unit_dir / "forecast_spec.json"
    spec = json.loads(spec_p.read_text(encoding="utf-8")) if spec_p.exists() else {}
    resolved_asof = _resolve_asof(unit_dir, card, spec, asof)
    panels = _load_panels(unit_dir, resolved_asof)
    texts = _load_texts(unit_dir)

    tgt = card.get("targets") or spec.get("targets") or {}
    spec_tgt = spec.get("targets", {})
    assets = list(tgt.get("asset_ids") or spec_tgt.get("asset_ids") or [])
    horizons = [int(h) for h in (tgt.get("horizons") or spec_tgt.get("horizons") or [])]
    if not assets or not horizons:
        raise ValueError(f"{unit_dir.name}: no target assets/horizons declared")
    periods = tgt.get("observation_periods") or spec_tgt.get("observation_periods")

    bundle = DataBundle(
        unit_dir=unit_dir, card=card, spec=spec,
        card_id=card.get("task", {}).get("id", spec.get("card_id", unit_dir.name)),
        family=spec.get("card_family") or card.get("metadata", {}).get("category", ""),
        asof=resolved_asof,
        target_assets=assets, horizons=horizons,
        target_type=tgt.get("target_type", spec_tgt.get("target_type", "level")),
        target_frequency=tgt.get("target_frequency")
        or card.get("metadata", {}).get("target_frequency", "daily"),
        value_unit=str(tgt.get("value_unit", "")),
        n_draws_min=int(spec.get("n_draws_min")
                        or card.get("scoring", {}).get("params", {}).get("n_draws_min")
                        or 200),
        panels=panels, texts=texts,
        observation_periods=periods,
    )
    present = set()
    for df in panels.values():
        col = next((c for c in _ASSET_COLS if c in df.columns), None)
        if col is not None:
            present.update(df[col].astype(str).unique())
    bundle.missing_assets = [a for a in assets if a not in present]
    bundle.present_assets = [a for a in assets if a in present]
    return bundle
