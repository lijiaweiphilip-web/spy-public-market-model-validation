from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-01", periods=1200, freq="B", tz="UTC")
    returns = rng.normal(0.0002, 0.01, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": dates, "adjusted_close": prices})


@pytest.fixture(scope="session")
def cached_small_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Build one complete small run and let contract tests copy it cheaply."""
    from spy_validation.cli import generate_synthetic_adjusted_close
    from spy_validation.config import RunConfig
    from spy_validation.evaluation import add_decision_cost_diagnostics, run_walk_forward
    from spy_validation.features import build_feature_frame
    from spy_validation.reporting import write_outputs

    root = tmp_path_factory.mktemp("cached_small_run")
    source = root / "synthetic_adjusted_close.csv"
    generate_synthetic_adjusted_close(rows=700, seed=8).to_csv(source, index=False)
    cfg = RunConfig.from_dict(
        {
            "symbol": "SPY",
            "data_range": "synthetic",
            "interval": "1d",
            "target_horizon_days": 5,
            "minimum_training_rows": 550,
            "test_rows_per_fold": 10,
            "step_rows": 100,
            "embargo_rows": 5,
            "ridge_alpha": 10.0,
            "forest_estimators": 1,
            "forest_max_depth": 3,
            "forest_min_samples_leaf": 2,
            "primary_seed": 42,
            "stability_seeds": [42, 123],
            "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995,
            "upper_clip_multiplier": 3.0,
            "calibration_bins": 5,
            "bootstrap_repetitions": 10,
            "target_annualised_volatility": 0.12,
            "max_exposure": 1.0,
            "transaction_cost_bps": [1, 5],
            "ewma_lambda": 0.94,
            "calibration_method": "inner_temporal_block",
            "calibration_block_rows": 20,
            "output_dir": str(root / "run"),
        }
    )
    config_path = root / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    prices = pd.read_csv(source)
    frame = build_feature_frame(prices, cfg.target_horizon_days)
    frame.attrs["raw_price_rows"] = len(prices)
    outputs = run_walk_forward(frame, cfg)
    exposure, costs = add_decision_cost_diagnostics(outputs["predictions"], cfg)
    write_outputs(
        Path(cfg.output_dir), config_path, cfg, source, frame, outputs, exposure, costs, "cached-small-run"
    )
    return Path(cfg.output_dir), source, config_path
