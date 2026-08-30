from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spy_validation.cli import validate_run_dir
from spy_validation.config import RunConfig
from spy_validation.evaluation import add_decision_cost_diagnostics, run_walk_forward
from spy_validation.features import build_feature_frame
from spy_validation.reporting import write_outputs
from spy_validation.validation import validate_reference_contract


def small_config(tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "symbol": "SPY", "data_range": "synthetic", "interval": "1d", "target_horizon_days": 5,
            "minimum_training_rows": 550, "test_rows_per_fold": 10, "step_rows": 100,
            "embargo_rows": 5, "ridge_alpha": 10.0, "forest_estimators": 1,
            "forest_max_depth": 3, "forest_min_samples_leaf": 2, "primary_seed": 42,
            "stability_seeds": [42, 123], "prediction_floor": 1e-10,
            "upper_clip_quantile": 0.995, "upper_clip_multiplier": 3.0, "calibration_bins": 5,
            "bootstrap_repetitions": 10, "target_annualised_volatility": 0.12,
            "max_exposure": 1.0, "transaction_cost_bps": [1, 5], "ewma_lambda": 0.94,
            "calibration_method": "inner_temporal_block", "calibration_block_rows": 20,
            "output_dir": str(tmp_path / "run"),
        }
    )


def synthetic_source(tmp_path: Path) -> Path:
    rng = np.random.default_rng(8)
    dates = pd.date_range("2010-01-01", periods=700, freq="B", tz="UTC")
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
    assert manifest["folds"] == 1
    assert set(manifest["models"]) == {"mean_baseline", "ewma_baseline", "ridge", "random_forest"}
    result = validate_run_dir(Path(cfg.output_dir), source, config_path)
    assert result["status"] == "PASS"
    assert result["oof_rows_per_model"]["ewma_baseline"] == 10


def test_manifest_hash_tampering_fails_validation(tmp_path, cached_small_run):
    cached_run, cached_source, cached_config = cached_small_run
    run_dir = tmp_path / "run"
    shutil.copytree(cached_run, run_dir)
    source = tmp_path / "synthetic_adjusted_close.csv"
    config_path = tmp_path / "config.json"
    shutil.copy2(cached_source, source)
    shutil.copy2(cached_config, config_path)
    target = run_dir / "aggregate_metrics.csv"
    original = target.read_bytes()
    try:
        target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(RuntimeError, match="hash mismatch"):
            validate_run_dir(run_dir, source, config_path)
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
    assert manifest["hash_contract"]["artifact_sha256_mode"] == "canonical_by_hash_mode"
    details = manifest["public_artifact_hash_details"]["MATH_CLAIMS.json"]
    assert details["content_type"] == "text"
    assert details["hash_algorithm"] == "sha256"
    assert details["hash_mode"] == "git-lf-v1"
    normalized = claims_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert details["canonical_sha256"] == manifest["public_artifact_sha256"]["MATH_CLAIMS.json"]
    assert details["canonical_sha256"] == hashlib.sha256(normalized).hexdigest()
    assert len(details["source_raw_sha256"]) == 64
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    assert claims["reference_manifest_version"] == manifest["manifest_version"]
    assert claims["canonical_experiment_code_commit"] == manifest["canonical_code_commit"]
    assert manifest["manifest_schema_version"] == manifest["manifest_version"] == 3
    assert manifest["canonical_experiment_code_commit"] == manifest["canonical_code_commit"]
    assert manifest["reference_artifact_commit"] == manifest["artifact_repository_commit"]
    assert manifest["canonical_experiment_package_version"] == "0.1.0"
    assert manifest["validator_release_version"] == "0.2.0"
    assert manifest["release_validator_commit"] is None
    assert claims["math_claims_schema_version"] == manifest["math_claims_schema_version"] == "1.0"
    assert claims["validator_release_version"] == manifest["validator_release_version"]


def test_public_reference_contract_validator_checks_schema_and_provenance():
    reference_dir = Path(__file__).resolve().parents[1] / "results" / "reference_run"
    result = validate_reference_contract(reference_dir)
    assert result["status"] == "PASS"
    assert result["manifest_version"] == 3
    assert result["folds"] == 27


def test_generic_validator_does_not_require_reference_fold_count(tmp_path):
    """A one-fold generated run proves validation is not SPY-reference hard-coded."""
    source = synthetic_source(tmp_path)
    cfg = replace(small_config(tmp_path), minimum_training_rows=610, forest_estimators=2)
    run_dir = tmp_path / "toy_run"
    cfg = replace(cfg, output_dir=str(run_dir))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    prices = pd.read_csv(source)
    frame = build_feature_frame(prices, cfg.target_horizon_days)
    frame.attrs["raw_price_rows"] = len(prices)
    outputs = run_walk_forward(frame, cfg)
    exposure, costs = add_decision_cost_diagnostics(outputs["predictions"], cfg)
    write_outputs(
        output_dir=run_dir,
        config_path=config,
        cfg=cfg,
        source_path=source,
        feature_frame=frame,
        outputs=outputs,
        exposure_path=exposure,
        cost_summary=costs,
        command="generic-one-fold",
    )
    result = validate_run_dir(run_dir, source, config)
    assert result["status"] == "PASS"
    assert result["folds"] == 1
    assert result["models"] == ["ewma_baseline", "mean_baseline", "random_forest", "ridge"]
