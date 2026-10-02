# Documentación completa — agente de forecasting Track 2

## 1. Qué es esto

Agenthon 2026 Track 2: dado un paquete de datos por unit (paneles parquet +
textos con timestamp + card.toml), producir **draws Monte Carlo** de la
distribución conjunta del futuro de los assets target:

```
forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD \
         --out /output/forecast.parquet
```

Salida obligatoria en `/output`: `forecast.parquet` (columnas `draw:int32,
asset:string, horizon:int32, value:float64`, cada draw con TODOS los
asset×horizon), `forecast_meta.json` (unit_id, asof, asset_ids, horizons,
n_draws, representation, target), `forecast_rationale.md` (texto no vacío).

Restricciones: cutoff de información en `asof`, red restringida (solo el
endpoint House vía proxy auditado), presupuesto de reloj por unit (~1800 s de
fallback si la card no fija timeout), seed vía `QFBENCH_SEED`.

## 2. Estructura del código

```
custom_model/
  data/
    loader.py        DataBundle: card+spec+paneles+textos; as-of resuelto de
                     corpus_index/panel/card; serie por asset truncada al as-of
    features.py      features por asset (nivel, diffs 1/5/21, logrets, vol
                     5/21/63, rangos, cross-sectional mean/std/min/max por
                     fecha). Todo truncado al as-of; sin leakage
    targets.py       y = valor a h pasos (daily: shift(-h); monthly: periodos
                     explícitos de observation_periods)
    dataset.py       X/y alineados + X_pred (última fila); monthly colapsa a
                     periodo (última obs mensual)
  model.py           fit_model: hgb (HistGradientBoosting), ridge, persistence;
                     walk_forward_residuals (folds expansivos + embargo);
                     LatentRegressor (PCA+z lags, refit por fold)
  latent.py          LatentState: PCA sobre features + z_t y lags
  probabilistic.py   resid_model: media+sigma+nu (Student-t si curtosis alta);
                     PersistenceModel; sample_residuals; winsorize
  joint.py           joint_samples: media + resid_oof; correlación gaussiana
                     (copula) sobre residuos; to_output_tensor
  regime.py          regime_scale: escala de vol por régimen (físicas: vol
                     reciente vs histórica)
  ensemble.py        pool_draws + inverse_score_weights
  llm.py             cliente House (POST $MODEL_ENDPOINT/v1/chat/completions,
                     budget 25 req, temperatura 0, seed 0, thinking off,
                     cache .cache/llm o $LLM_CACHE_DIR); offline -> None
  text_features.py   docs -> features diarios causales (sentiment, relevance,
                     sorpresa, riesgo, dirección hawkish/dovish, event_type,
                     assets mencionados); count + decay exponencial
  forecast.py        CLI/verbo forecast + orquestación + fallbacks + entregables
  tests/             test_loader/features/text/model/all_units (pytest o directo)
run_all_units.py     barrido 104 units: modo structural (barato) o forecast
submission/          Dockerfile, submission.json (12 campos C5 1.1.0), README
                     de pasos manuales, ARTIFACT_PROVENANCE.md
```

## 3. El algoritmo, paso a paso

Por unit, con tareas = asset × horizon:

1. **Carga.** `load_unit` lee card.toml + forecast_spec.json + paneles + textos;
   resuelve el as-of (corpus_index > spec > card) y la frecuencia/target_type.
2. **Features** (causales): por cada panel asset: nivel, diffs, logrets, vol
   rodante, rango, distancia a máximos; bloque cross-sectional por fecha.
   Todo estrictamente ≤ asof.
3. **Targets.** Daily: y_t = x_{t+h}. Monthly: los keys de horizonte son
   opacos; los pasos se derivan de `observation_periods` (ej. key 140 -> 8
   meses). Embargo del walk-forward = nº de pasos de observación, no la key.
4. **Dataset.** Alineado X/y por fecha (o periodo mensual); X_pred = última
   fila.
5. **Modelo punto.** Miembros del ensemble: `hgb`, `ridge`, `persistence`
   (y opcional `hgb+latent`). Presupuesto adaptativo por nº de tareas:
   ≤4 tareas: 4 folds / 300 iter / sin cap; ≤10: 3/200/cap 4000; >10:
   2/150/cap 3000 (lo que domina el tiempo son los fits HGB).
6. **Residuos OOF.** walk-forward expansivo con embargo = horizonte; la
   distribución de error queda Student-t (nu por curtosis) o gaussiana.
7. **Joint.** Media mu por tarea + draws correlados por **cópula
   Student-t**: normales correlados (Cholesky sobre correlación de residuos
   OOF, PSD más cercana; identidad si <30 filas) divididos por UN factor
   chi² común por draw -> dependencia de colas (los assets caen juntos, lo
   que la cópula gaussiana subestima). El nu de la cópula sale del exceso de
   curtosis pooled de residuos estandarizados. Las marginales usan la
   **cuantil empírica** del residuo OOF de cada tarea (asimetría real, sin
   fits extra; fallback a t/normal paramétrica si <30 residuos). Escala de
   régimen multiplica la vol en units diarias.
8. **Ensemble.** Score de cada miembro = media por tarea de CRPS empírico de
   sus residuos OOF contra los residuos realizados, **normalizado por la media
   cross-member de cada tarea** (así JPY ~140 no ahoga a UST ~4). Pesos =
   inverso del score. Pooling de los tensores de draws preservando la
   dependencia conjunta dentro de cada draw.
9. **Escritura.** Tensor [n_draws, n_assets, n_horizons] -> parquet largo
   (int32/string/int32/float64), meta JSON, rationale markdown. n_draws:
   max(--n-draws, n_draws_min, 500; 1000 para familia F4).

## 4. Decisiones con evidencia

- **Ensemble por defecto**: en boj-ust-channel, CRPS OOF -> ridge 2.42,
  hgb+latent 2.72, hgb 2.83, persistence 4.30 (pesos 30/27/26/17). En
  dollar-squeeze gana persistence (régimen de crisis -> los modelos
  sobre-ajustan calma). El ensemble se adapta por unit sin coste relevante
  (ridge/persistence son ~gratis frente a HGB).
- **Timeout**: el peor caso medido (dollar-squeeze, 10 assets × 2 horizontes)
  pasó de 2058 s a 225 s con row_cap=3000 y máquina libre — holgado vs 1800 s.
- **Sweep completo**: 104/104 units OK, 0 fallbacks completos, 0 fallbacks de
  tarea, 0 errores de texto (verificado sobre forecast.parquet escrito).

## 5. Bugs encontrados y corregidos

1. `_gaussian_walk`: para target `log_return` usaba diffs de nivel; ahora
   camina sobre log(1+r) con ancla 0; dtypes de salida int32; F4 >= 1000 draws.
2. Alineado mensual: features diarias vs targets mensuales generaban y todo
   NaN -> colapso por periodo en dataset/targets.
3. `ridge` moría con columnas 100 % NaN -> imputación mediana + fillna(0).
4. `LatentRegressor.predict`: `.loc` duplicaba filas cuando la fecha predicha
   ya estaba en la cola -> indexado posicional.
5. Fallbacks mensuales: usaban la key de horizonte como nº de pasos -> ahora
   `_horizon_steps()` traduce por observation_periods (también en embargo).
6. Cache LLM en workdir no escribible -> `$LLM_CACHE_DIR` + escritura
   best-effort; `enable_thinking: false` ahorra tokens House.
7. `main()` captura excepciones de pipeline y emite igualmente el gaussian
   walk (una unit nunca queda sin output).

## 6. Jerarquía de robustez

modelo completo -> por tarea: fallback persistence+sigma_histórico ->
unit entera: random-walk gaussiano conjunto (siempre válido para schema).

## 7. Cumplimiento

- Cutoff: todo se computa de filas <= asof; PCA refit por fold; residuos OOF;
  textos filtrados por timestamp <= asof y agregados causalmente.
- Determinismo: seed = QFBENCH_SEED / --seed gobierna HGB, PCA, draws,
  pooling y la llamada House (temperature 0, seed 0).
- Descriptor: `submission.json` con los 12 campos C5 1.1.0; `models[]`
  declara el House model `nvidia/nemotron-3-super-120b-a12b` (rl-030326-fp8)
  porque llm.py lo usa cuando el harness inyecta el endpoint.
- Sin dependencias de qfbench2-common en la imagen del modelo.

## 8. Qué NO hace (a propósito)

- No empaqueta pesos ni datos (todo se ajusta en runtime <= asof).
- No reimplementa el scoring (CRPS/variogram viven en el toolkit común).
- No usa la rama de texto si el endpoint House no está inyectado.
- El arbitraje del ensemble es por CRPS OOF normalizado; no busca el "mejor
  modelo" global sino el mejor mix por unit.
- La cópula Student-t depende de que el kurtosis pooled refleje co-movimiento
  de colas; con residuos casi gaussianos degenera a cópula gaussiana.

## 9. Comandos esenciales

```
python -m custom_model.forecast --panels <unit>/panels --text <unit>/text \
    --asof YYYY-MM-DD --out <out>/forecast.parquet
python run_all_units.py --mode structural            # chequeo barato 104/104
python run_all_units.py --mode forecast --workers 6  # sweep completo
docker build -f submission/Dockerfile -t <img> .     # desde la raíz del repo
```