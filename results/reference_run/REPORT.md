# Reference-run report

- Data: SPY daily adjusted close, 2016-11-18 to 2026-08-17
- Target: next 5-trading-day realised variance proxy
- Validation: 27 purged expanding walk-forward folds
- Best aggregate QLIKE in this run: **ewma_baseline** (0.490342)
- Exact environment: [`environment.json`](environment.json), git commit `2711eb2d7b496ac44f7ed820d02b35433b7c9d2b`
- No alpha, return, Sharpe ratio, profitability or production claim is made.

## Aggregate metrics

| model         |    n |         mae |       rmse |     qlike |   calibration_ratio |    spearman |
|:--------------|-----:|------------:|-----------:|----------:|--------------------:|------------:|
| ewma_baseline | 1620 | 0.000666834 | 0.00212298 |  0.490342 |            1.00023  |  0.53377    |
| mean_baseline | 1620 | 0.000776807 | 0.00240431 |  1.29709  |            0.865878 | -0.00850132 |
| random_forest | 1620 | 0.000640056 | 0.00229394 |  0.57741  |            0.794853 |  0.630997   |
| ridge         | 1620 | 0.000717334 | 0.00244199 | 19.9694   |            0.894873 |  0.542046   |

## Bootstrap comparison against the historical mean

| metric   | model         | baseline      |   mean_difference |   ci_lower_95 |   ci_upper_95 |   share_folds_better |
|:---------|:--------------|:--------------|------------------:|--------------:|--------------:|---------------------:|
| rmse     | ewma_baseline | mean_baseline |      -0.0001545   |  -0.000308722 |  -2.69647e-05 |             0.814815 |
| rmse     | random_forest | mean_baseline |      -0.000106947 |  -0.000240533 |   5.88144e-05 |             0.851852 |
| rmse     | ridge         | mean_baseline |      -2.01171e-05 |  -0.000174383 |   0.00017076  |             0.777778 |
| qlike    | ewma_baseline | mean_baseline |      -0.806744    |  -2.14654     |  -0.107434    |             0.814815 |
| qlike    | random_forest | mean_baseline |      -0.719675    |  -1.79803     |  -0.133161    |             0.777778 |
| qlike    | ridge         | mean_baseline |      18.6723      |  -0.224325    |  56.3872      |             0.740741 |

The public reference bundle contains derived summaries only; raw vendor bytes, point-level predictions and exposure paths remain private evidence.
The final `run_manifest.json` hashes every artifact written above.
