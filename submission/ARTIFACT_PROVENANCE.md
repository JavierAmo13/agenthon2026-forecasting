# Artifact provenance — Track 2 custom forecasting model

Per `docs/ARTIFACT-POLICY.md`: this record lists what the submission image
contains, where each artifact came from, and how the information cutoff is
respected.

## What the image contains

| Artifact | Source | Version / revision | Role |
|---|---|---|---|
| `custom_model/` Python code | This repository (participant-authored) | pinned by image digest | loader, features, targets, dataset, model, probabilistic, joint, ensemble, regime, latent, llm, text_features, forecast CLI |
| numpy | PyPI | 2.3.3 | numeric core |
| pandas | PyPI | 2.3.2 | panel handling |
| pyarrow | PyPI | 22.0.0 | parquet IO |
| scipy | PyPI | 1.16.3 | residual kurtosis → Student-t ν |
| scikit-learn | PyPI | 1.7.2 | HistGradientBoosting, Ridge, PCA |

No dataset, fitted parameters, embeddings, checkpoints, or unit answers are
packaged. All model parameters are fit **inside the container at run time**, on
the unit's own `/input` panels truncated at `--asof`, so the task cutoff holds
by construction for every unit.

## Learned models and the `models[]` disclosure

- The gradient-boosting / ridge fits produced at run time are per-unit learned
  models under the artifact policy's "fitted non-neural" allowance; nothing
  fitted is shipped in the image.
- The House model (`nvidia/nemotron-3-super-120b-a12b`, snapshot
  `rl-030326-fp8`) is called only when the harness injects `MODEL_ENDPOINT`,
  `MODEL_NAME`, `MODEL_TOKEN`, and only through
  `POST $MODEL_ENDPOINT/v1/chat/completions`, temperature 0, seed 0,
  ≤25 requests per unit, ≤400 output tokens per call. Its five-field
  `models[]` row is in `submission.json`.

## Cutoff discipline

- Panels and text are read from `/input` only; every feature is computed from
  observations `<= asof` (see `features.build_features`, which truncates each
  panel at the as-of before computing rolling windows).
- LLM text features use only documents whose `corpus_index.json` timestamp is
  `<= asof`; the aggregation is causal (a document influences rows after its
  timestamp only, via forward-fill + exponential decay).
- Walk-forward residual estimation is strictly expanding-window with an
  embargo equal to the horizon in observation steps.
- PCA (when `--model ...+latent`) is refit inside each walk-forward fold and
  on the training rows only — never on held-out or post-asof data.
- The random seed comes from `QFBENCH_SEED` (or `--seed`) and controls all
  stochastic components: HGB `random_state`, Monte-Carlo draws, and the
  House-model `seed` parameter.