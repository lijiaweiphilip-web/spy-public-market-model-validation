from __future__ import annotations

import numpy as np

from spy_validation.config import RunConfig
from spy_validation.metrics import regression_metrics
from spy_validation.models import make_models


def config() -> RunConfig:
    return RunConfig.from_dict(
        {
            "symbol": "SPY",
            "data_range": "10y",
            "interval": "1d",
            "target_horizon_days": 5,
            "minimum_training_rows": 100,
            "test_rows_per_fold": 20,
            "step_rows": 20,
            "embargo_rows": 5,
            "ridge_alpha": 10.0,
            "forest_estimators": 30,
            "forest_max_depth": 4,
            "forest_min_samples_leaf": 5,
            "primary_seed": 42,
            "stability_seeds": [42, 123],
            "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995,
            "upper_clip_multiplier": 3.0,
            "calibration_bins": 5,
            "bootstrap_repetitions": 100,
            "target_annualised_volatility": 0.12,
            "max_exposure": 1.0,
            "transaction_cost_bps": [1, 5],
            "output_dir": "runs/test",
        }
    )


def test_models_produce_positive_deterministic_predictions():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 6))
    y = np.exp(rng.normal(-8, 0.8, size=300))
    x_test = rng.normal(size=(50, 6))
    first = make_models(config())
    second = make_models(config())
    for name in first:
        first[name].fit(x, y)
        second[name].fit(x, y)
        p1 = first[name].predict(x_test)
        p2 = second[name].predict(x_test)
        assert np.all(p1 > 0)
        assert np.allclose(p1, p2)


def test_qlike_is_nonnegative_and_zero_at_perfect_prediction():
    actual = np.array([0.1, 0.2, 0.3])
    perfect = regression_metrics(actual, actual)
    assert perfect["qlike"] == 0.0
    imperfect = regression_metrics(actual, np.array([0.2, 0.1, 0.4]))
    assert imperfect["qlike"] > 0
