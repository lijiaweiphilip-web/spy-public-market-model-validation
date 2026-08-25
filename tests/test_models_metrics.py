from __future__ import annotations

import numpy as np
import pytest

from spy_validation.config import RunConfig
from spy_validation.metrics import qlike_loss, regression_metrics
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


def test_qlike_known_value_matches_definition():
    actual = np.array([2.0])
    prediction = np.array([1.0])
    assert qlike_loss(actual, prediction)[0] == pytest.approx(2.0 - np.log(2.0) - 1.0)


def test_random_forest_seed_is_deterministic_with_stability_seeds():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(120, 6))
    y = np.exp(rng.normal(-8, 0.4, size=120))
    cfg = config()
    first = make_models(cfg, seed=123)["random_forest"]
    second = make_models(cfg, seed=123)["random_forest"]
    first.fit(x, y)
    second.fit(x, y)
    assert np.allclose(first.predict(x[:20]), second.predict(x[:20]))


def test_train_only_clipping_bounds_are_positive_and_recorded():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(240, 6))
    y = np.exp(rng.normal(-8, 0.8, size=240))
    model = make_models(config())["ridge"]
    model.fit(x, y)
    prediction = model.predict(np.full((10, 6), 100.0))
    assert np.all(prediction >= model.lower_bound)
    assert np.all(prediction <= model.upper_bound)
    assert model.calibration_rows == 120
