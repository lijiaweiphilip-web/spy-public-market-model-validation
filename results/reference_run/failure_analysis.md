# Failure and limitation analysis

This file is generated from the reference run. It is intentionally candid: model validation is stronger when unstable or negative results remain visible.

- **random_forest** outperformed the historical-mean baseline on aggregate RMSE by 0.000173237; QLIKE difference was -0.741916.
- **ridge** outperformed the historical-mean baseline on aggregate RMSE by 2.4196e-05; QLIKE difference was 19.2152.
- **random_forest** showed material aggregate calibration drift: predicted/actual mean ratio 0.750.
- **ridge** showed material aggregate calibration drift: predicted/actual mean ratio 0.744.
- **random_forest** beat the mean baseline on RMSE in 81.5% of folds.
- **ridge** beat the mean baseline on RMSE in 74.1% of folds.
- The highest illustrative annualised decision-turnover cost was 126.9 bps for random_forest under a 10-bps assumption. This is a cost diagnostic, not a P&L result.
- **ridge** required training-bounded prediction clipping on 13 test predictions.
- **mean_baseline** mean calibration-bin absolute log ratio: 0.555.
- **random_forest** mean calibration-bin absolute log ratio: 0.268.
- **ridge** mean calibration-bin absolute log ratio: 0.545.

## Structural limitations

- One preselected ETF only; no cross-asset or cross-market generalisation claim.
- Five-day realised variance is a noisy proxy, not latent volatility and not a causal outcome.
- Adjusted-close history comes from a vendor endpoint that can revise data or become unavailable.
- Hyperparameters are fixed before test evaluation; no claim is made that they are optimal.
- The illustrative exposure and transaction-cost layer measures decision turnover only. It does not report returns, alpha, Sharpe ratio, P&L or deployable strategy performance.
