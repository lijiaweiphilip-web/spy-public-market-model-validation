# Limitations and non-claims

- One ETF only. No cross-asset or cross-market generalisation claim.
- A realised-variance proxy is noisy and does not identify causal volatility drivers.
- Vendor data can be revised or become unavailable. Every run records the input hash and retrieval metadata.
- The code intentionally uses adjusted close. Raw close would create dividend-related return distortions.
- Models are deliberately simple and hyperparameters are fixed rather than exhaustively tuned.
- The EWMA lambda is fixed at 0.94; the five-day forecast assumes a constant daily conditional variance and is not a fitted GARCH model.
- Inner-block calibration avoids outer-test leakage but remains a finite-sample diagnostic and is not a guarantee of full calibration.
- The transaction-cost layer is a turnover sensitivity diagnostic, not a backtested investment strategy.
- No alpha, return, P&L, Sharpe ratio, portfolio-performance or investment-advice claim is made.
