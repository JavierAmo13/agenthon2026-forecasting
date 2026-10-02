# Submission packaging — Track 2 custom model

Everything here is run from the repository root. Steps 3–5 need Docker and the
`qfbench2` CLI; on a Windows box without Docker, do them on the Linux dev box or
in CI.

## 1. Build

```bash
docker build -f submission/Dockerfile -t ghcr.io/<org>/t2-forecasting-agent:<tag> .
```

## 2. Smoke test (offline, mirrors the harness)

```bash
docker run --rm --network=none --cpus=4 --memory=16g \
  -v $(pwd)/units/t2-F3-boj-ust-channel-2023:/input:ro \
  -v $(pwd)/output:/output \
  ghcr.io/<org>/t2-forecasting-agent:<tag> \
  forecast --panels /input/panels --text /input/text \
           --asof 2023-07-21 --out /output/forecast.parquet
```

Check `output/` holds `forecast.parquet`, `forecast_meta.json`,
`forecast_rationale.md`. Then score if the toolkit is installed:

```bash
python scoring/scoring.py score \
  --card units/t2-F3-boj-ust-channel-2023/card.toml \
  --forecast output/forecast.parquet
```

## 3. Push and seal

Push the image, copy its digest into `submission.json` (`image.digest`,
`image.repository`, `image.registry`), set `team_id` (derived by
`qfbench2 submission alias`), then reseal:

```python
from qfbench2_common.contracts.descriptor import seal_descriptor_digest
# seal_descriptor_digest(descriptor) -> fills descriptor_digest
```

`pack` reseals automatically; hand-edited files need the toolkit call above.

## 4. Pack and upload

```bash
qfbench2 submission pack \
  --descriptor submission/submission.json \
  --team-number <N> --out submission.zip
```

Upload `submission.zip` on the Track-2 CodaBench page. Limits: 5 uploads/day,
20 total during Development; last runs must start by 20:00 UTC on 2026-10-12.

## Notes

- `models[]` declares the House model because `llm.py` calls it when
  `MODEL_ENDPOINT`/`MODEL_NAME`/`MODEL_TOKEN` are injected by the harness.
  Locally (no endpoint) the text branch degrades to numeric-only.
- If you ship a text-ablated variant too, submit it as a second upload — the
  comparison is only visible on the Development leaderboard.
- Keep `ARTIFACT_PROVENANCE.md`-style notes of what went into the image; see
  `docs/ARTIFACT-POLICY.md`.