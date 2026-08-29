# Model card

## Intended research question
Can simple historical, EWMA and machine-learning models produce stable forecasts of a five-day realised-variance proxy under purged expanding walk-forward validation on SPY daily adjusted-close data?

## Models
- Historical mean baseline
- RiskMetrics-style EWMA baseline with fixed lambda=0.94; a daily conditional-variance estimate is multiplied by the five-day horizon
- Ridge regression on a log target with `inner_temporal_block` calibration inside each outer training set
- Random forest on a log target with `inner_temporal_block` calibration inside each outer training set

## Validation
- 27 expanding walk-forward folds
- 60 test rows per fold
- 5-row purge/embargo to prevent overlapping training labels from entering each test period
- Fixed hyperparameters; no random split and no test-set tuning

## Intended use
Research-method demonstration, RA interview discussion, and reproducibility evidence.

## Not intended for
Live trading, investment recommendations, alpha claims, P&L claims, portfolio construction or production risk management.

## Known limitations
See `failure_analysis.md` and `docs/CALIBRATION_AUDIT.md`.
