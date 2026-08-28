# Mathematical contract

This document states the definitions and information-availability assumptions
implemented by the public SPY model-validation code. They are executable
protocol definitions and tested invariants, not new causal-inference theorems.

## Data and target

The input is a positive, timestamped adjusted-close series (P_t). Log returns
are

\[
r_t = \log(P_t) - \log(P_{t-1}).
\]

For a horizon (H=5), the forward realised-variance proxy is

\[
y_t^{(5)} = \sum_{h=1}^{5} r_{t+h}^{2}.
\]

The target window is strictly after the feature timestamp. The implementation
rejects non-positive prices and checks the timestamp ordering.

## Non-anticipating EWMA baseline

The fixed-λ baseline uses only returns available at or before (t):

\[
\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)r_t^2,
\qquad \lambda=0.94,
\]

and maps the daily estimate to the five-day horizon with

\[
\hat y_t = 5\sigma_t^2.
\]

“Non-anticipating” describes temporal information availability. It does not
claim causal identification or causal forecasting.

## Validation and calibration

- Folds are expanding walk-forward folds with an embargo/purge at least as
  large as the target horizon, so training labels end before each test window.
- Hyperparameters are fixed before test evaluation; no random split or test-set
  tuning is used.
- Ridge and Random Forest calibration uses an inner temporal block inside each
  outer training set. The outer test block is not used for calibration.

## Metrics and uncertainty

QLIKE is evaluated with a positive floor:

\[
QLIKE(y,\hat y) = \frac{y}{\hat y} - \log\left(\frac{y}{\hat y}\right) - 1.
\]

RMSE is the square root of the mean squared forecast error. Bootstrap model
comparisons resample complete folds, not individual out-of-fold rows. Every
model must have exactly one row per fold; missing or duplicate fold/model pairs
are rejected. The resulting intervals are descriptive fold-level uncertainty
intervals, not a claim that 1,620 rows are independent observations.

## Public evidence boundary

The public reference bundle contains derived summaries and a deterministic
synthetic path. The canonical vendor snapshot and point-level predictions are
private evidence. No trading, alpha, P&L, profitability, latent-volatility or
cross-asset generalisation claim follows from this contract.
