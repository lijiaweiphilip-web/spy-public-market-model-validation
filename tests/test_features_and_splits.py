from __future__ import annotations

from spy_validation.features import FEATURE_COLUMNS, build_feature_frame
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
