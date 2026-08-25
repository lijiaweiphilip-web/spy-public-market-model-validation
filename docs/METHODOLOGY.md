# Methodology

## Research question

The project asks whether simple historical and machine-learning models produce stable and calibrated forecasts of a five-trading-day realised-variance proxy on SPY daily adjusted-close data under a strict temporal evaluation protocol.

## Target

For feature date `t`, the target is the sum of squared log returns from `t+1` through `t+5`. This is a noisy realised-variance proxy, not latent volatility.

## Features

All features use information available on or before `t`: return lags, momentum, rolling realised variance, absolute-return summaries, downside variance, drawdown and trend features.

## Split protocol

The default configuration produces 27 non-overlapping 60-row test folds after an 800-row minimum training history. Training expands over time. A five-row purge removes training examples whose forward targets overlap the next test period.

## Models

- Historical mean baseline
- Ridge regression on a log target
- Random forest on a log target

Machine-learning predictions use training-only calibration and training-derived clipping bounds. Hyperparameters are fixed before test evaluation.

## Evaluation

The evidence bundle includes point-level out-of-fold predictions, MAE, RMSE, scale-invariant QLIKE, calibration ratio, rank correlation, calibration bins, prior-volatility regime slices, bootstrap fold comparisons, Ridge coefficient stability, random-forest feature importance, random-forest seed stability and a prediction-to-decision turnover cost diagnostic.

## Transaction-cost sensitivity

Predictions are converted into an illustrative volatility-targeting exposure capped at one unit. Turnover is multiplied by 1, 5 and 10 basis points. This measures the cost burden of a hypothetical decision layer. It does not report returns, alpha, Sharpe ratio, P&L or strategy profitability.
