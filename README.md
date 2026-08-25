# SPY Public-Market Model Validation

Evidence-first validation of simple volatility-proxy models on SPY daily adjusted-close data.

## What this repository demonstrates

- adjusted-close data parsing with a hashed source manifest;
- a five-day forward realised-variance target;
- 27 purged expanding walk-forward folds in the reference configuration;
- historical mean, Ridge and Random Forest benchmarks;
- point-level out-of-fold predictions and fold metrics;
- calibration bins, prior-volatility regime slices and bootstrap fold comparisons;
- Ridge coefficient, random-forest feature and seed-stability diagnostics;
- an illustrative volatility-targeting turnover and transaction-cost sensitivity layer;
- explicit failure analysis and reproducibility hashes.

This is a **model-validation research project**, not a trading strategy. It makes no alpha, return, P&L, Sharpe-ratio, portfolio-performance or investment-advice claim.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .[dev]
pytest
```

## Run with a local Yahoo chart snapshot

```bash
spy-validate run \
  --config configs/default.json \
  --input-json /path/to/spy_yahoo_chart.json \
  --output-dir runs/reference_run
spy-validate validate --run-dir runs/reference_run
```

## Run with a CSV

The CSV must contain `date` and `adjusted_close` columns.

```bash
spy-validate run --input-csv /path/to/spy_adjusted_close.csv
```

## Fetch at runtime

Omit `--input-json` and `--input-csv` to fetch a new Yahoo chart response. The endpoint is convenient rather than contractual, so the exact raw response is hashed and local snapshots are preferred for deterministic audit reruns.

## Reference run

The 2026-08-25 reference run used 27 purged expanding walk-forward folds and 1,620 out-of-fold observations per model. Random Forest had the best aggregate QLIKE (0.555) and RMSE (0.002231); its fold-bootstrap RMSE difference versus the historical-mean baseline was negative in the 95% interval. Ridge improved RMSE on average but showed severe QLIKE instability in stress periods, which is retained in the failure report rather than hidden.

The committed reference summary contains derived metrics, figures, documentation and a manifest. Raw vendor data and private point-level evidence are excluded from the public repository by default. Re-run the code to regenerate the complete evidence bundle.

See:

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- [`docs/RESUME_EVIDENCE.md`](docs/RESUME_EVIDENCE.md)
- [`docs/GITHUB_PUBLISH_CHECKLIST.md`](docs/GITHUB_PUBLISH_CHECKLIST.md)
