# Cómo usar este proyecto — Track 2 Agenthon 2026

Este árbol lo genera `GENERAR_TODO.ipynb` (Run All). Genera:

- `custom_model/` — el agente de forecasting completo (ver DOCUMENTACION_COMPLETA.md)
- `run_all_units.py` — barrido de las 104 units con chequeo de schema
- `inspect_panels.py`, `survey_units.py`, `test_pipeline.py` — utilidades de exploración
- `submission/` — Dockerfile + descriptor + guía de empaquetado para la entrega

## 1. Dependencias

```
pip install numpy==2.3.3 pandas==2.3.2 pyarrow==22.0.0 scipy==1.16.3 scikit-learn==1.7.2
```

(Python >= 3.13. En Windows corporativo sin PyPI, las versiones pueden instalarse
desde un mirror interno o un venv ya preparado.)

## 2. Probar rápido

```
cd track2-forecasting-public
python custom_model/tests/test_model.py        # sintético, ~20 s
python custom_model/tests/test_loader.py
python custom_model/tests/test_features.py
python custom_model/tests/test_text.py
python custom_model/tests/test_all_units.py    # paseo estructural por las 104 units
python test_pipeline.py                        # pipeline datos en una unit
```

## 3. Generar un forecast

```
python -m custom_model.forecast --panels units/t2-F3-boj-ust-channel-2023/panels \
    --text units/t2-F3-boj-ust-channel-2023/text --asof 2023-07-21 \
    --out output/boj/forecast.parquet
```

Produce `forecast.parquet` + `forecast_meta.json` + `forecast_rationale.md`
(el contrato que exige el harness). Flags útiles: `--model ridge|hgb|persistence|hgb+latent`,
`--ensemble hgb,ridge,persistence` (default) | `--ensemble none`, `--no-text`, `--n-draws N`,
`--seed N` (o `QFBENCH_SEED`).

## 4. Barrido de las 104 units

```
python run_all_units.py --mode forecast --workers 6 --report units_report.json
```

(En esta máquina tardó ~2.5 h con 6 workers; en el runner del concurso hay 16 vCPU.
`--mode structural` solo valida la tubería de datos en ~6 min.)

## 5. Entregar

Los pasos con Docker y el descriptor `submission.json` están en
`submission/README.md`. Resumen: build de la imagen, push, sellar el digest del
descriptor con el toolkit `qfbench2_common`, `qfbench2 submission pack`, subir el
zip a CodaBench (máx. 5/día, 20 en fase Development; cierre 12-oct-2026 23:59 AoE).

## Notas

- Sin `MODEL_ENDPOINT`/`MODEL_NAME`/`MODEL_TOKEN` la rama de texto se degrada a
  numérico-solo: todo sigue funcionando offline.
- Si algo falla a mitad de una unit, el CLI emite igualmente un random walk
  gaussiano conjunto válido para schema (ver DOCUMENTACION_COMPLETA.md §7).