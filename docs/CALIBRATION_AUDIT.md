# Calibration audit

The historical mean and EWMA baselines do not estimate a fitted calibration factor. Ridge and Random Forest are fit on the outer training set, while their multiplicative calibration factor is estimated from a terminal **inner temporal block** inside that outer training set. The inner estimator is fit only on observations before that block, with a five-row horizon purge. The outer test block is never used to select or fit the factor.

This is stricter than V4.4's in-sample training calibration, but it is still a small-sample calibration diagnostic. It does not make the variance forecasts fully calibrated, does not tune any test result, and does not support a probabilistic-calibration claim beyond the reported bins and calibration ratio.

Each fold records `calibration_method`, `calibration_rows`, `calibration_factor`, and the train-derived clipping bounds in `fold_metrics.csv`. The validator checks the presence and hash of that table; tests cover the inner-block boundary and positive/clipped predictions.
