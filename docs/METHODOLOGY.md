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
- RiskMetrics-style EWMA baseline with a fixed lambda=0.94 (see the [RiskMetrics Technical Document](https://www.msci.com/research-and-insights/paper/1996-riskmetrics-technical-document)). Its daily conditional-variance estimate is multiplied by the five-day horizon under a constant-variance approximation; lambda is not selected on the test set.
- Ridge regression on a log target
- Random forest on a log target

Machine-learning predictions use a terminal inner temporal calibration block inside each outer training set. The inner estimator is fit strictly before that block and is separated from the outer test block by the target horizon. Clipping bounds are derived from training data only. Hyperparameters are fixed before test evaluation. See `CALIBRATION_AUDIT.md` for the remaining calibration limitations.

## Evaluation

The evidence bundle includes point-level out-of-fold predictions, MAE, RMSE, scale-invariant QLIKE, calibration ratio, rank correlation, calibration bins, prior-volatility regime slices, bootstrap fold comparisons, Ridge coefficient stability, random-forest feature importance, random-forest seed stability and a prediction-to-decision turnover cost diagnostic. QLIKE is included as a volatility-forecast loss with source context from [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034). The private canonical manifest hashes every written artifact after REPORT, environment and diagnostics are complete; the public derived bundle is listed in [`PUBLIC_REFERENCE_MANIFEST.json`](../results/reference_run/PUBLIC_REFERENCE_MANIFEST.json).

## Transaction-cost sensitivity

Predictions are converted into an illustrative volatility-targeting exposure capped at one unit. Turnover is multiplied by 1, 5 and 10 basis points. This measures the cost burden of a hypothetical decision layer. It does not report returns, alpha, Sharpe ratio, P&L or strategy profitability.
