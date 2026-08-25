# Data and reproducibility

## Data source

The reference input is a Yahoo Finance chart response for SPY daily data (`range=10y`, `interval=1d`, with dividend/split events requested). The pipeline uses the `adjclose` series, not raw close, because dividend and split adjustments affect return and realised-variance calculations.

Yahoo is a convenient vendor endpoint, not a contractual research data API. Retrieval timing and vendor revisions can change the bytes or the available history. The canonical private audit run therefore retains the raw response locally and records its SHA-256 in `run_manifest.json` and `REFERENCE_RUN_PROVENANCE.md`; raw vendor bytes are not redistributed by default.

## Reproduce with your own compatible data

Provide either a Yahoo chart JSON containing `chart.result[0].indicators.adjclose[0].adjclose` or a CSV with `date` and `adjusted_close` columns:

```bash
spy-validate run --config configs/default.json --input-json /path/to/your/chart.json --output-dir runs/your_run
spy-validate validate --run-dir runs/your_run --source-path /path/to/your/chart.json
```

For CSV input:

```bash
spy-validate run --config configs/default.json --input-csv /path/to/spy_adjusted_close.csv --output-dir runs/your_run
```

## Reproducibility tiers

1. **Tier 1 — public synthetic/CI:** the synthetic end-to-end smoke test runs without vendor data and exercises features, purged folds, all four models, reporting, manifest hashing and validation.
2. **Tier 2 — live vendor rerun:** a fresh Yahoo request can be run locally, but vendor revisions, endpoint availability and retrieval timing may change the data.
3. **Tier 3 — canonical audit rerun:** the exact private raw snapshot is identified by SHA-256 and rerun in the locked Python 3.12 environment. It is private because raw vendor bytes and point-level evidence are not redistributed by default.

The reference report names the exact environment file and Git commit. No random split or test-set hyperparameter tuning is used.

For provenance, `environment.json` separates the `code_commit` that generated
predictions and metrics from the `artifact_repository_commit` that introduced
the derived reference bundle. The public synthetic demo is a deterministic
price-only functionality check and must not be combined with the SPY reference
metrics.
