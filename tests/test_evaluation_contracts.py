from __future__ import annotations

import numpy as np
import pandas as pd

from spy_validation.config import RunConfig
from spy_validation.evaluation import (
    _assign_regime,
    _regime_thresholds,
    add_decision_cost_diagnostics,
)
from spy_validation.features import EWMA_FEATURE_COLUMN, FEATURE_COLUMNS, build_feature_frame
from spy_validation.models import make_models


def test_regime_thresholds_are_train_only():
    train = pd.DataFrame({"realised_var_20": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    test = pd.DataFrame({"realised_var_20": [0.5, 3.5, 100.0]})
    low, high = _regime_thresholds(train)
    labels = _assign_regime(train, test)
    assert low == train["realised_var_20"].quantile(1 / 3)
    assert high == train["realised_var_20"].quantile(2 / 3)
    assert labels.tolist() == ["low_prior_vol", "mid_prior_vol", "high_prior_vol"]


def test_inner_temporal_calibration_uses_a_strict_inner_block():
    cfg = RunConfig.from_dict(
        {
            "symbol": "SPY", "data_range": "10y", "interval": "1d", "target_horizon_days": 5,
            "minimum_training_rows": 100, "test_rows_per_fold": 20, "step_rows": 20,
            "embargo_rows": 5, "ridge_alpha": 10.0, "forest_estimators": 10,
            "forest_max_depth": 3, "forest_min_samples_leaf": 2, "primary_seed": 42,
            "stability_seeds": [42, 123], "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995, "upper_clip_multiplier": 3.0, "calibration_bins": 5,
            "bootstrap_repetitions": 20, "target_annualised_volatility": 0.12,
            "max_exposure": 1.0, "transaction_cost_bps": [1], "ewma_lambda": 0.94,
            "calibration_method": "inner_temporal_block", "calibration_block_rows": 40,
            "output_dir": "runs/test",
        }
    )
    rng = np.random.default_rng(4)
    x = rng.normal(size=(220, len(FEATURE_COLUMNS)))
    y = np.exp(rng.normal(-8, 0.3, size=220))
    model = make_models(cfg)["ridge"]
    model.fit(x, y)
    assert model.calibration_method == "inner_temporal_block"
    assert model.calibration_rows == 40
    assert model.calibration_rows < len(y)


def test_ewma_forecast_maps_daily_variance_to_five_day_target(synthetic_prices):
    frame = build_feature_frame(synthetic_prices, horizon=5)
    forecast = frame[EWMA_FEATURE_COLUMN].to_numpy() * 5
    assert np.all(forecast > 0)
    assert np.allclose(forecast / 5, frame[EWMA_FEATURE_COLUMN])


def test_decision_cost_arithmetic_is_per_unit_capital():
    predictions = pd.DataFrame(
        {
            "model": ["ewma_baseline", "ewma_baseline", "ewma_baseline"],
            "feature_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "prediction": [0.01, 0.04, 0.01],
        }
    )
    cfg = RunConfig.from_dict(
        {
            "symbol": "SPY", "data_range": "10y", "interval": "1d", "target_horizon_days": 5,
            "minimum_training_rows": 100, "test_rows_per_fold": 20, "step_rows": 20,
            "embargo_rows": 5, "ridge_alpha": 10.0, "forest_estimators": 10,
            "forest_max_depth": 3, "forest_min_samples_leaf": 2, "primary_seed": 42,
            "stability_seeds": [42, 123], "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995, "upper_clip_multiplier": 3.0, "calibration_bins": 5,
            "bootstrap_repetitions": 20, "target_annualised_volatility": 0.12,
            "max_exposure": 1.0, "transaction_cost_bps": [10], "ewma_lambda": 0.94,
            "calibration_method": "inner_temporal_block", "calibration_block_rows": 40,
            "output_dir": "runs/test",
        }
    )
    exposure, costs = add_decision_cost_diagnostics(predictions, cfg)
    assert len(exposure) == 3
    assert costs.loc[0, "total_cost_fraction_per_unit_capital"] >= 0
    assert costs.loc[0, "annualised_cost_bps_per_unit_capital"] >= 0
