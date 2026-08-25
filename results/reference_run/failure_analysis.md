# Failure and limitation analysis

This file is generated from the reference run. It is intentionally candid: model validation is stronger when unstable or negative results remain visible.
Calibration factors for Ridge and Random Forest are estimated from an inner temporal block inside each outer training set; the outer test block is never used. This is stricter than in-sample calibration but remains a small-sample diagnostic, not a guarantee of calibrated probabilities or variance forecasts.

- **ewma_baseline** outperformed the historical-mean baseline on aggregate RMSE by 0.000281333; QLIKE difference was -0.806744.
- **random_forest** outperformed the historical-mean baseline on aggregate RMSE by 0.000110368; QLIKE difference was -0.719675.
- **ridge** underperformed the historical-mean baseline on aggregate RMSE by 3.7684e-05; QLIKE difference was 18.6723.
- **random_forest** showed material aggregate calibration drift: predicted/actual mean ratio 0.795.
- **ewma_baseline** beat the mean baseline on RMSE in 81.5% of folds.
- **random_forest** beat the mean baseline on RMSE in 85.2% of folds.
- **ridge** beat the mean baseline on RMSE in 77.8% of folds.
- The highest illustrative annualised decision-turnover cost was 121.5 bps for ridge under a 10-bps assumption. This is a cost diagnostic, not a P&L result.
- **ridge** required training-bounded prediction clipping on 15 test predictions.
- **ewma_baseline** mean calibration-bin absolute log ratio: 0.163.
- **mean_baseline** mean calibration-bin absolute log ratio: 0.555.
- **random_forest** mean calibration-bin absolute log ratio: 0.257.
- **ridge** mean calibration-bin absolute log ratio: 0.485.

## Structural limitations

- One preselected ETF only; no cross-asset or cross-market generalisation claim.
- Five-day realised variance is a noisy proxy, not latent volatility and not a causal outcome.
- The EWMA baseline uses a fixed RiskMetrics-style lambda=0.94 and maps a daily conditional-variance estimate to five days by multiplying by the horizon; lambda is not selected on the test set.
- Adjusted-close history comes from a vendor endpoint that can revise data or become unavailable.
- Hyperparameters are fixed before test evaluation; no claim is made that they are optimal.
- The illustrative exposure and transaction-cost layer measures decision turnover only. It does not report returns, alpha, Sharpe ratio, P&L or deployable strategy performance.
