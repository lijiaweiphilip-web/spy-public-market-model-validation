from __future__ import annotations

import os
from pathlib import Path

import pytest

from spy_validation.config import RunConfig
from spy_validation.data import load_yahoo_chart
from spy_validation.features import build_feature_frame
from spy_validation.splits import expanding_purged_folds


def test_private_reference_snapshot_creates_27_folds():
    raw_path = os.environ.get("SPY_REFERENCE_JSON")
    if not raw_path:
        pytest.skip("SPY_REFERENCE_JSON is not set; public CI does not require vendor data")
    candidate = Path(raw_path)
    cfg = RunConfig.load(Path(__file__).resolve().parents[1] / "configs" / "default.json")
    prices = load_yahoo_chart(candidate)
    frame = build_feature_frame(prices, cfg.target_horizon_days)
    folds = expanding_purged_folds(
        frame,
        cfg.minimum_training_rows,
        cfg.test_rows_per_fold,
        cfg.step_rows,
        cfg.embargo_rows,
    )
    assert len(folds) == 27
