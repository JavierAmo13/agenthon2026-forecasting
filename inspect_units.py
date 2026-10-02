from pathlib import Path
import json
import tomllib

ROOT = Path("units")

print("=" * 80)
print("INVENTARIO COMPLETO DE UNITS")
print("=" * 80)

units = [p for p in ROOT.iterdir() if p.is_dir()]

print(f"\nNúmero de unidades: {len(units)}\n")

for i, unit in enumerate(sorted(units), 1):

    print("\n" + "=" * 80)
    print(f"{i}. {unit.name}")
    print("=" * 80)

    # ---------------------------------------------------------
    # CARD
    # ---------------------------------------------------------
    card_path = unit / "card.toml"

    if card_path.exists():
        try:
            with open(card_path, "rb") as f:
                card = tomllib.load(f)

            task = card.get("task", {})
            targets = card.get("targets", {})
            panel = card.get("panel", {})
            text = card.get("text", {})

            print("\n[CARD]")

            print("  task_id:",
                  task.get("id", "N/A"))

            print("  title:",
                  task.get("title", "N/A"))

            print("  category:",
                  task.get("category", "N/A"))

            print("  panel:",
                  panel.get("id", panel.get("name", "N/A")))

            print("  assets:",
                  targets.get("asset_ids", "N/A"))

            print("  horizons:",
                  targets.get("horizon", "N/A"))

            print("  target_type:",
                  targets.get("target_type", "N/A"))

            print("  text source:",
                  text.get("source", "N/A"))

        except Exception as e:
            print("  ERROR leyendo card.toml:", e)

    # ---------------------------------------------------------
    # TODOS LOS ARCHIVOS
    # ---------------------------------------------------------
    files = sorted(
        p for p in unit.rglob("*")
        if p.is_file()
    )

    print("\n[ARCHIVOS]")

    for p in files:
        relative = p.relative_to(unit)
        print(f"  {relative}")

    # ---------------------------------------------------------
    # TEXTOS
    # ---------------------------------------------------------
    text_files = sorted(unit.rglob("*.txt"))

    print(f"\n[TEXTO] {len(text_files)} archivos .txt")

    for p in text_files:
        try:
            size = p.stat().st_size

            with open(
                p,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:
                content = f.read()

            print(
                f"  {p.relative_to(unit)} "
                f"| {size:,} bytes "
                f"| {len(content):,} chars"
            )

        except Exception as e:
            print(
                f"  {p.relative_to(unit)} "
                f"| ERROR: {e}"
            )

    # ---------------------------------------------------------
    # JSON
    # ---------------------------------------------------------
    json_files = sorted(unit.rglob("*.json"))

    print(f"\n[JSON] {len(json_files)} archivos")

    for p in json_files:

        try:
            with open(
                p,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:
                data = json.load(f)

            if isinstance(data, dict):
                keys = list(data.keys())[:15]
                print(
                    f"  {p.relative_to(unit)} "
                    f"| keys={keys}"
                )

            elif isinstance(data, list):
                print(
                    f"  {p.relative_to(unit)} "
                    f"| list[{len(data)}]"
                )

            else:
                print(
                    f"  {p.relative_to(unit)} "
                    f"| type={type(data).__name__}"
                )

        except Exception as e:
            print(
                f"  {p.relative_to(unit)} "
                f"| ERROR: {e}"
            )

    # ---------------------------------------------------------
    # PARQUET
    # ---------------------------------------------------------
    parquet_files = sorted(unit.rglob("*.parquet"))

    print(f"\n[PARQUET] {len(parquet_files)} archivos")

    for p in parquet_files:

        try:
            import pandas as pd

            df = pd.read_parquet(p)

            print(
                f"  {p.relative_to(unit)}"
            )
            print(
                f"      shape={df.shape}"
            )
            print(
                f"      columns={list(df.columns)}"
            )

            if "asset" in df.columns:
                print(
                    f"      assets={df['asset'].nunique()}"
                )

            if "date" in df.columns:
                print(
                    f"      dates={df['date'].min()} -> "
                    f"{df['date'].max()}"
                )

        except Exception as e:
            print(
                f"  {p.relative_to(unit)} "
                f"| ERROR: {e}"
            )

print("\n" + "=" * 80)
print("FIN DEL INVENTARIO")
print("=" * 80)