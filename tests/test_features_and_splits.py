from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spy_validation.data import load_adjusted_close_csv, load_yahoo_chart
from spy_validation.features import (
    EWMA_FEATURE_COLUMN,
    FEATURE_COLUMNS,
    build_feature_frame,
    ewma_variance,
)
from spy_validation.splits import expanding_purged_folds


def test_feature_timestamps_precede_targets(synthetic_prices):
    frame = build_feature_frame(synthetic_prices, horizon=5)
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    assert (frame["feature_timestamp"] < frame["target_start_timestamp"]).all()
    assert (frame["target_start_timestamp"] <= frame["target_end_timestamp"]).all()


def test_purged_fold_has_no_training_label_overlap(synthetic_prices):
    frame = build_feature_frame(synthetic_prices, horizon=5)
    folds = expanding_purged_folds(frame, 400, 40, 40, 5)
    assert folds
    for fold in folds:
        assert fold.train_end_timestamp < fold.test_start_timestamp
        assert len(set(fold.train_indices).intersection(fold.test_indices)) == 0


def test_five_day_target_matches_exact_forward_squared_return_sum():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC"),
            "adjusted_close": np.exp(np.arange(100, dtype=float) * 0.01),
        }
    )
    frame = build_feature_frame(prices, horizon=5)
    row = frame.iloc[0]
    returns = np.diff(np.log(prices["adjusted_close"].to_numpy()))
    source_index = int(prices.index[prices["date"] == row["feature_timestamp"]][0])
    expected = float(np.sum(returns[source_index : source_index + 5] ** 2))
    assert row["target_realised_variance"] == pytest.approx(expected)
    assert row["target_start_timestamp"] == prices.iloc[source_index + 1]["date"]
    assert row["target_end_timestamp"] == prices.iloc[source_index + 5]["date"]


def test_ewma_is_non_anticipating_and_uses_fixed_lambda():
    returns = np.array([np.nan, 0.1, 0.2, 0.3])
    result = ewma_variance(returns, decay=0.94)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(0.1**2)
    assert result[2] == pytest.approx(0.94 * 0.1**2 + 0.06 * 0.2**2)


def test_ewma_feature_is_present_and_positive(synthetic_prices):
    frame = build_feature_frame(synthetic_prices, horizon=5)
    assert EWMA_FEATURE_COLUMN in frame.columns
    assert (frame[EWMA_FEATURE_COLUMN] > 0).all()
    assert not frame[FEATURE_COLUMNS].isna().any().any()


def test_adjusted_close_csv_is_sorted_and_deduplicated(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-01", "2020-01-01"],
            "adjusted_close": [3.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)
    frame = load_adjusted_close_csv(path)
    assert frame["date"].is_monotonic_increasing
    assert len(frame) == 2


def test_raw_close_only_yahoo_payload_is_rejected(tmp_path):
    path = tmp_path / "raw_only.json"
    payload = {"chart": {"result": [{"timestamp": [1], "indicators": {"quote": [{"close": [1.0]}]}}]}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Adjusted-close"):
        load_yahoo_chart(path)


def test_purged_folds_are_deterministic_and_have_expected_shape(synthetic_prices):
    frame = build_feature_frame(synthetic_prices, horizon=5)
    first = expanding_purged_folds(frame, 800, 60, 60, 5)
    second = expanding_purged_folds(frame, 800, 60, 60, 5)
    assert len(first) == len(second) == 5
    assert all(len(fold.test_indices) == 60 for fold in first)
    assert [fold.fold_id for fold in first] == [fold.fold_id for fold in second]
