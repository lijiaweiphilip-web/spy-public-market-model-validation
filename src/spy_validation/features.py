from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_0",
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "return_lag_10",
    "momentum_5",
    "momentum_20",
    "realised_var_5",
    "realised_var_20",
    "realised_var_60",
    "mean_abs_return_5",
    "mean_abs_return_20",
    "downside_var_20",
    "drawdown_20",
    "trend_20",
    "trend_60",
]
EWMA_FEATURE_COLUMN = "ewma_variance_0_94"


def ewma_variance(values: pd.Series | np.ndarray, decay: float = 0.94) -> np.ndarray:
    """Return a non-anticipating EWMA of squared log returns.

    The value at t uses only returns observed on or before t. This is a
    temporal information-availability property, not a causal-inference claim.
    The first valid return seeds the recursion; missing values remain missing
    until a valid return is observed.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be between 0 and 1")
    arr = np.asarray(values, dtype=float)
    result = np.full(arr.shape, np.nan, dtype=float)
    previous = np.nan
    for index, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        squared = float(value) ** 2
        previous = squared if not np.isfinite(previous) else decay * previous + (1.0 - decay) * squared
        result[index] = previous
    return result


def build_feature_frame(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if (frame["adjusted_close"] <= 0).any():
        raise ValueError("adjusted_close must be positive")

    frame["log_return"] = np.log(frame["adjusted_close"]).diff()
    frame["return_0"] = frame["log_return"]
    for lag in (1, 2, 5, 10):
        frame[f"return_lag_{lag}"] = frame["log_return"].shift(lag)

    frame["momentum_5"] = np.log(frame["adjusted_close"] / frame["adjusted_close"].shift(5))
    frame["momentum_20"] = np.log(frame["adjusted_close"] / frame["adjusted_close"].shift(20))
    squared = frame["log_return"].pow(2)
    frame["realised_var_5"] = squared.rolling(5).sum()
    frame["realised_var_20"] = squared.rolling(20).sum()
    frame["realised_var_60"] = squared.rolling(60).sum()
    frame["mean_abs_return_5"] = frame["log_return"].abs().rolling(5).mean()
    frame["mean_abs_return_20"] = frame["log_return"].abs().rolling(20).mean()
    downside = frame["log_return"].clip(upper=0).pow(2)
    frame["downside_var_20"] = downside.rolling(20).sum()
    frame["drawdown_20"] = frame["adjusted_close"] / frame["adjusted_close"].rolling(20).max() - 1
    frame["trend_20"] = frame["adjusted_close"] / frame["adjusted_close"].rolling(20).mean() - 1
    frame["trend_60"] = frame["adjusted_close"] / frame["adjusted_close"].rolling(60).mean() - 1
    frame[EWMA_FEATURE_COLUMN] = ewma_variance(frame["log_return"], decay=0.94)

    future_terms = [squared.shift(-step) for step in range(1, horizon + 1)]
    frame["target_realised_variance"] = sum(future_terms)
    frame["feature_timestamp"] = frame["date"]
    frame["target_start_timestamp"] = frame["date"].shift(-1)
    frame["target_end_timestamp"] = frame["date"].shift(-horizon)

    keep = [
        "date",
        "adjusted_close",
        "feature_timestamp",
        "target_start_timestamp",
        "target_end_timestamp",
        "target_realised_variance",
        EWMA_FEATURE_COLUMN,
        *FEATURE_COLUMNS,
    ]
    result = frame[keep].dropna().reset_index(drop=True)
    if not (
        result["feature_timestamp"] < result["target_start_timestamp"]
    ).all():
        raise AssertionError("Feature timestamp must precede target start")
    if not (
        result["target_start_timestamp"] <= result["target_end_timestamp"]
    ).all():
        raise AssertionError("Target interval is invalid")
    return result
