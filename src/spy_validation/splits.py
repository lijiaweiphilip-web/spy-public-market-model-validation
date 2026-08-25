from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_end_timestamp: pd.Timestamp
    test_start_timestamp: pd.Timestamp
    test_end_timestamp: pd.Timestamp


def expanding_purged_folds(
    frame: pd.DataFrame,
    minimum_training_rows: int,
    test_rows: int,
    step_rows: int,
    embargo_rows: int,
) -> list[PurgedFold]:
    first_test_start = minimum_training_rows + embargo_rows
    folds: list[PurgedFold] = []
    fold_id = 0
    test_start = first_test_start
    while test_start + test_rows <= len(frame):
        test_indices = np.arange(test_start, test_start + test_rows)
        test_start_timestamp = pd.Timestamp(frame.iloc[test_start]["feature_timestamp"])
        candidate_indices = np.arange(0, test_start)
        target_ends = pd.to_datetime(frame.iloc[candidate_indices]["target_end_timestamp"], utc=True)
        train_indices = candidate_indices[target_ends < test_start_timestamp]
        if len(train_indices) < minimum_training_rows:
            test_start += step_rows
            continue
        train_end_timestamp = pd.Timestamp(
            frame.iloc[train_indices[-1]]["target_end_timestamp"]
        )
        if not train_end_timestamp < test_start_timestamp:
            raise AssertionError("Purging failed: training target overlaps the test period")
        folds.append(
            PurgedFold(
                fold_id=fold_id,
                train_indices=train_indices,
                test_indices=test_indices,
                train_end_timestamp=train_end_timestamp,
                test_start_timestamp=test_start_timestamp,
                test_end_timestamp=pd.Timestamp(
                    frame.iloc[test_indices[-1]]["feature_timestamp"]
                ),
            )
        )
        fold_id += 1
        test_start += step_rows
    if not folds:
        raise ValueError("No folds were created; revise training and test sizes")
    return folds
