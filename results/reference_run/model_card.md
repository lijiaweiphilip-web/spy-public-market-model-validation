# Model card

## Intended research question
Can simple historical and machine-learning models produce stable, calibrated five-day realised-variance forecasts under purged expanding walk-forward validation on SPY daily adjusted-close data?

## Models
- Historical mean baseline
- Ridge regression on a log target with training-only calibration
- Random forest on a log target with training-only calibration

## Validation
- 27 expanding walk-forward folds
- 60 test rows per fold
- 5-row purge/embargo to prevent overlapping training labels from entering each test period
- Fixed hyperparameters; no test-set tuning

## Intended use
Research-method demonstration, RA interview discussion, and reproducibility evidence.

## Not intended for
Live trading, investment recommendations, alpha claims, P&L claims, portfolio construction or production risk management.

## Known limitations
See `failure_analysis.md`.
