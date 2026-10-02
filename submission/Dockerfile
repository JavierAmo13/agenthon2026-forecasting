# Participant submission image — Track 2 custom forecasting model.
#
# Implements the `forecast` verb on PATH:
#   forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD \
#            --out /output/forecast.parquet
#
# Build from the repository ROOT (context must see custom_model/):
#   docker build -f submission/Dockerfile -t <registry>/<repo>:<tag> .
#
# Smoke run (no network, mirrors the harness mount contract):
#   docker run --rm --network=none \
#     -v <unit-dir>:/input:ro -v <out-dir>:/output \
#     <image> forecast --panels /input/panels --text /input/text \
#             --asof 2023-07-21 --out /output/forecast.parquet

FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0"
LABEL qfbench2.track="forecasting"
LABEL qfbench2.verb="forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_CACHE_DIR=/tmp/llm-cache

# Pinned to the versions the model was built and swept against. HGB is the
# non-linear learner; Ridge/persistence are the cheap ensemble members.
RUN pip install --no-cache-dir \
        "numpy==2.3.3" \
        "pandas==2.3.2" \
        "pyarrow==22.0.0" \
        "scipy==1.16.3" \
        "scikit-learn==1.7.2"

WORKDIR /work
COPY custom_model /opt/custom_model
ENV PYTHONPATH=/opt

# The verb, as an executable on PATH.
RUN printf '#!/bin/sh\nexec python3 -m custom_model.forecast "$@"\n' \
      > /usr/local/bin/forecast \
 && chmod +x /usr/local/bin/forecast

RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["forecast", "--help"]