from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from spy_validation.cli import main, validate_run_dir
from spy_validation.config import RunConfig
from spy_validation.data import load_adjusted_close_csv, load_yahoo_chart
from spy_validation.evaluation import add_decision_cost_diagnostics, run_walk_forward
from spy_validation.features import build_feature_frame
from spy_validation.reporting import write_outputs


def _small_config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "symbol": "SPY",
            "data_range": "synthetic",
            "interval": "1d",
            "target_horizon_days": 5,
            "minimum_training_rows": 100,
            "test_rows_per_fold": 20,
            "step_rows": 20,
            "embargo_rows": 5,
            "ridge_alpha": 10.0,
            "forest_estimators": 8,
            "forest_max_depth": 3,
            "forest_min_samples_leaf": 2,
            "primary_seed": 42,
            "stability_seeds": [42, 123],
            "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995,
            "upper_clip_multiplier": 3.0,
            "calibration_bins": 5,
            "bootstrap_repetitions": 20,
            "target_annualised_volatility": 0.12,
            "max_exposure": 1.0,
            "transaction_cost_bps": [1, 5],
            "ewma_lambda": 0.94,
            "calibration_method": "inner_temporal_block",
            "calibration_block_rows": 40,
            "output_dir": str(tmp_path / "run"),
        }
    )


def _small_source(tmp_path: Path) -> Path:
    rng = np.random.default_rng(8)
    dates = pd.date_range("2010-01-01", periods=710, freq="B", tz="UTC")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, len(dates))))
    path = tmp_path / "synthetic_adjusted_close.csv"
    pd.DataFrame({"date": dates, "adjusted_close": prices}).to_csv(path, index=False)
    return path


def _build_small_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = _small_source(tmp_path)
    cfg = _small_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    prices = load_adjusted_close_csv(source)
    frame = build_feature_frame(prices, cfg.target_horizon_days)
    frame.attrs["raw_price_rows"] = len(prices)
    outputs = run_walk_forward(frame, cfg)
    exposure, costs = add_decision_cost_diagnostics(outputs["predictions"], cfg)
    write_outputs(
        Path(cfg.output_dir), config_path, cfg, source, frame, outputs, exposure, costs, "hardening-test"
    )
    return Path(cfg.output_dir), source, config_path


def _yahoo_payload(*, timestamps: list[int] | None = None, adjusted: list[float | None] | None = None) -> dict:
    timestamps = timestamps if timestamps is not None else [1_700_000_000 + i * 86_400 for i in range(1000)]
    adjusted = adjusted if adjusted is not None else [100.0 + i * 0.01 for i in range(len(timestamps))]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "adjclose": [{"adjclose": adjusted}],
                        "quote": [{"volume": [1_000] * len(timestamps)}],
                    },
                }
            ],
            "error": None,
        }
    }


def test_load_yahoo_chart_accepts_valid_adjusted_close_payload(tmp_path: Path):
    path = tmp_path / "chart.json"
    path.write_text(json.dumps(_yahoo_payload()), encoding="utf-8")

    frame = load_yahoo_chart(path)

    assert len(frame) == 1000
    assert list(frame.columns) == ["date", "adjusted_close", "volume"]
    assert frame["date"].is_monotonic_increasing
    assert frame["adjusted_close"].iloc[0] == 100.0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"chart": {"result": [], "error": {"description": "bad symbol"}}}, "no result"),
        (
            {"chart": {"result": [{"timestamp": list(range(1000)), "indicators": {}}]}},
            "Adjusted-close",
        ),
        (
            {
                "chart": {
                    "result": [
                        {
                            "timestamp": [],
                            "indicators": {"adjclose": [{"adjclose": [100.0] * 1000}]},
                        }
                    ]
                }
            },
            "misaligned",
        ),
    ],
)
def test_load_yahoo_chart_rejects_missing_required_components(tmp_path: Path, payload: dict, message: str):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_yahoo_chart(path)


def test_load_yahoo_chart_rejects_misaligned_timestamp_and_adjusted_close_arrays(tmp_path: Path):
    payload = _yahoo_payload(timestamps=list(range(1000)), adjusted=[100.0] * 999)
    path = tmp_path / "misaligned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="misaligned"):
        load_yahoo_chart(path)


def test_load_yahoo_chart_drops_null_adjusted_close_but_enforces_minimum(tmp_path: Path):
    timestamps = [1_700_000_000 + i * 86_400 for i in range(1001)]
    adjusted = [None] + [100.0 + i * 0.01 for i in range(1000)]
    path = tmp_path / "null_adjusted_close.json"
    path.write_text(json.dumps(_yahoo_payload(timestamps=timestamps, adjusted=adjusted)), encoding="utf-8")

    frame = load_yahoo_chart(path)

    assert len(frame) == 1000
    assert frame["adjusted_close"].notna().all()
    assert frame["date"].iloc[0] == pd.to_datetime(timestamps[1], unit="s", utc=True).normalize()


def test_load_adjusted_close_csv_rejects_invalid_schema(tmp_path: Path):
    path = tmp_path / "raw_close_only.csv"
    pd.DataFrame({"date": ["2025-01-01"], "close": [100.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="CSV must contain"):
        load_adjusted_close_csv(path)


def test_cli_run_and_validate_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = _small_source(tmp_path)
    cfg = _small_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    output_dir = tmp_path / "cli_run"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spy-validate",
            "run",
            "--config",
            str(config_path),
            "--input-csv",
            str(source),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spy-validate",
            "validate",
            "--run-dir",
            str(output_dir),
            "--source-path",
            str(source),
            "--config-path",
            str(config_path),
        ],
    )
    assert main() == 0


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_artifact", "artifacts are missing"),
        ("wrong_source", "Supplied source hash"),
        ("wrong_config", "Supplied config hash"),
        ("duplicate_oof", "Duplicate fold/model/feature-date"),
    ],
)
def test_validate_run_dir_rejects_manifest_contract_failures(
    tmp_path: Path,
    mode: str,
    message: str,
    cached_small_run: tuple[Path, Path, Path],
):
    cached_run, cached_source, cached_config = cached_small_run
    run_dir = tmp_path / "run"
    shutil.copytree(cached_run, run_dir)
    source = tmp_path / "synthetic_adjusted_close.csv"
    config_path = tmp_path / "config.json"
    shutil.copy2(cached_source, source)
    shutil.copy2(cached_config, config_path)

    if mode == "missing_artifact":
        (run_dir / "aggregate_metrics.csv").unlink()
        args = (run_dir, source, config_path)
    elif mode == "wrong_source":
        wrong_source = tmp_path / "wrong_source.csv"
        original = source.read_text(encoding="utf-8")
        wrong_source.write_text(original.replace("100.", "101.", 1), encoding="utf-8")
        args = (run_dir, wrong_source, config_path)
    elif mode == "wrong_config":
        wrong_config = tmp_path / "wrong_config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["ridge_alpha"] = 11.0
        wrong_config.write_text(json.dumps(payload), encoding="utf-8")
        args = (run_dir, source, wrong_config)
    else:
        predictions_path = run_dir / "predictions_oof.csv"
        predictions = pd.read_csv(predictions_path)
        predictions.iloc[1] = predictions.iloc[0]
        predictions.to_csv(predictions_path, index=False)
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import hashlib

        manifest["artifact_sha256"]["predictions_oof.csv"] = hashlib.sha256(
            predictions_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        args = (run_dir, source, config_path)

    with pytest.raises(RuntimeError, match=message):
        validate_run_dir(*args)
