# Resume evidence card

## Safe concise bullet after a successful reference run

> Built a public-market model-validation pipeline on SPY daily adjusted-close data with 27 purged expanding walk-forward folds; benchmarked a historical mean, Ridge and Random Forest across regime robustness, calibration, stability, decision-cost sensitivity and temporal-leakage controls.

> Saved point-level out-of-fold predictions, fold metrics, calibration/regime/stability summaries, decision-cost outputs, failure analysis and file hashes in a reproducible run manifest.

## Evidence required before using the bullet

- `run_manifest.json` says 27 folds and all temporal checks pass.
- `predictions_oof.csv` exists and contains fold IDs, dates, actuals and predictions.
- `fold_metrics.csv`, `regime_metrics.csv`, `calibration_bins.csv`, stability tables and `decision_cost_sensitivity.csv` exist.
- `failure_analysis.md` retains negative and unstable results.
- The source file and each output have SHA-256 values.
- No claim of trading profitability or investment performance is added.

## Three-minute interview defence

Be able to explain adjusted close, the five-day realised-variance target, the purge, why random splitting would leak time, why QLIKE is used, how calibration is measured, what the regime split means, why transaction costs are only an illustrative turnover diagnostic, which model underperformed, and the single-instrument limitation.
