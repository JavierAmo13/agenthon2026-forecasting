"""Custom generic forecasting pipeline for Agenthon 2026 Track 2.

# One pipeline handles all units: loader -> features -> targets -> dataset ->
# model -> probabilistic -> joint -> forecast.parquet
# """