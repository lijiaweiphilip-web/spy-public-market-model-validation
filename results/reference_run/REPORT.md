# Reference-run report

- Data: SPY daily adjusted close, 2016-11-18 to 2026-08-17
- Target: next 5-trading-day realised variance proxy
- Validation: 27 purged expanding walk-forward folds
- Best aggregate QLIKE in this run: **random_forest** (0.555169)
- No alpha, return, Sharpe ratio, profitability or production claim is made.

## Aggregate metrics

| model         |    n |         mae |       rmse |     qlike |   calibration_ratio |    spearman |
|:--------------|-----:|------------:|-----------:|----------:|--------------------:|------------:|
| mean_baseline | 1620 | 0.000776807 | 0.00240431 |  1.29709  |            0.865878 | -0.00850132 |
| random_forest | 1620 | 0.000590616 | 0.00223107 |  0.555169 |            0.749626 |  0.627289   |
| ridge         | 1620 | 0.000640927 | 0.00238011 | 20.5123   |            0.744124 |  0.601151   |

## Bootstrap comparison against the historical mean

| metric   | model         | baseline      |   mean_difference |   ci_lower_95 |   ci_upper_95 |   share_folds_better |
|:---------|:--------------|:--------------|------------------:|--------------:|--------------:|---------------------:|
| rmse     | random_forest | mean_baseline |      -0.000169115 |  -0.000251284 |  -9.47817e-05 |             0.814815 |
| rmse     | ridge         | mean_baseline |      -0.00011272  |  -0.00020058  |  -2.06152e-05 |             0.740741 |
| qlike    | random_forest | mean_baseline |      -0.741916    |  -1.92141     |  -0.0934913   |             0.777778 |
| qlike    | ridge         | mean_baseline |      19.2152      |  -0.2277      |  58.0129      |             0.740741 |

See `failure_analysis.md`, `model_card.md`, the point-level predictions, regime tables, calibration bins, stability tables and hashed run manifest for the complete evidence trail.
