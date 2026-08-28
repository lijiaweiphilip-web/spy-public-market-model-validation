from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spy_validation.cli import validate_run_dir
from spy_validation.config import RunConfig
from spy_validation.evaluation import add_decision_cost_diagnostics, run_walk_forward
from spy_validation.features import build_feature_frame
from spy_validation.reporting import write_outputs


def small_config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "symbol": "SPY", "data_range": "synthetic", "interval": "1d", "target_horizon_days": 5,
            "minimum_training_rows": 100, "test_rows_per_fold": 20, "step_rows": 20,
            "embargo_rows": 5, "ridge_alpha": 10.0, "forest_estimators": 8,
            "forest_max_depth": 3, "forest_min_samples_leaf": 2, "primary_seed": 42,
            "stability_seeds": [42, 123], "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995, "upper_clip_multiplier": 3.0, "calibration_bins": 5,
            "bootstrap_repetitions": 20, "target_annualised_volatility": 0.12,
            "max_exposure": 1.0, "transaction_cost_bps": [1, 5], "ewma_lambda": 0.94,
            "calibration_method": "inner_temporal_block", "calibration_block_rows": 40,
            "output_dir": str(tmp_path / "run"),
        }
    )


def synthetic_source(tmp_path: Path) -> Path:
    rng = np.random.default_rng(8)
    dates = pd.date_range("2010-01-01", periods=710, freq="B", tz="UTC")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, len(dates))))
    source = tmp_path / "synthetic_adjusted_close.csv"
    pd.DataFrame({"date": dates, "adjusted_close": prices}).to_csv(source, index=False)
    return source


def test_synthetic_end_to_end_pipeline_and_manifest(tmp_path):
    source = synthetic_source(tmp_path)
    cfg = small_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    prices = pd.read_csv(source)
    frame = build_feature_frame(prices, cfg.target_horizon_days)
    frame.attrs["raw_price_rows"] = len(prices)
    outputs = run_walk_forward(frame, cfg)
    exposure, costs = add_decision_cost_diagnostics(outputs["predictions"], cfg)
    manifest = write_outputs(
        output_dir=Path(cfg.output_dir), config_path=config_path, cfg=cfg,
        source_path=source, feature_frame=frame, outputs=outputs,
        exposure_path=exposure, cost_summary=costs, command="synthetic-smoke",
    )
    assert manifest["folds"] == 27
    assert set(manifest["models"]) == {"mean_baseline", "ewma_baseline", "ridge", "random_forest"}
    result = validate_run_dir(Path(cfg.output_dir), source, config_path)
    assert result["status"] == "PASS"
    assert result["oof_rows_per_model"]["ewma_baseline"] == 540


def test_manifest_hash_tampering_fails_validation(tmp_path):
    source = synthetic_source(tmp_path)
    cfg = small_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    prices = pd.read_csv(source)
    frame = build_feature_frame(prices, cfg.target_horizon_days)
    frame.attrs["raw_price_rows"] = len(prices)
    outputs = run_walk_forward(frame, cfg)
    exposure, costs = add_decision_cost_diagnostics(outputs["predictions"], cfg)
    write_outputs(Path(cfg.output_dir), config_path, cfg, source, frame, outputs, exposure, costs, "tamper-test")
    target = Path(cfg.output_dir) / "aggregate_metrics.csv"
    original = target.read_bytes()
    try:
        target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(RuntimeError, match="hash mismatch"):
            validate_run_dir(Path(cfg.output_dir), source, config_path)
    finally:
        target.write_bytes(original)


def test_public_allowlist_excludes_private_artifacts():
    allowlist = Path(__file__).resolve().parents[1] / "PUBLISH_ALLOWLIST.txt"
    text = allowlist.read_text(encoding="utf-8")
    assert "requirements/" in text
    assert "src/spy_validation/" in text
    assert "src/\n" not in text
    assert "private_local_evidence" not in text
    assert "predictions_oof.csv" not in text
    assert "illustrative_exposure_path.csv" not in text


def test_public_reference_manifest_covers_math_claims_with_explicit_hash_mode():
    root = Path(__file__).resolve().parents[1]
    reference_dir = root / "results" / "reference_run"
    manifest = json.loads((reference_dir / "PUBLIC_REFERENCE_MANIFEST.json").read_text(encoding="utf-8"))
    claims_path = reference_dir / "MATH_CLAIMS.json"
    assert manifest["manifest_version"] >= 3
    assert "MATH_CLAIMS.json" in manifest["public_artifact_sha256"]
    assert manifest["hash_contract"]["artifact_sha256_mode"] == "raw_bytes"
    details = manifest["public_artifact_hash_details"]["MATH_CLAIMS.json"]
    assert details["hash_algorithm"] == "sha256"
    assert details["text_normalization"] == "git-lf-v1"
    assert details["raw_sha256"] == hashlib.sha256(claims_path.read_bytes()).hexdigest()
    normalized = claims_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert details["normalized_sha256"] == hashlib.sha256(normalized).hexdigest()
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    assert claims["reference_manifest_version"] == manifest["manifest_version"]
    assert claims["canonical_experiment_code_commit"] == manifest["canonical_code_commit"]
