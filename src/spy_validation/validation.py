"""Generic run validation and the stricter public-reference contract.

The generic validator checks the self-describing contract emitted by any run of
the pipeline.  The reference validator deliberately adds SPY-specific claims
only when validating the checked-in public reference bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_RUN_ARTIFACTS = {
    "predictions_oof.csv",
    "fold_metrics.csv",
    "aggregate_metrics.csv",
    "regime_metrics.csv",
    "calibration_bins.csv",
    "decision_cost_sensitivity.csv",
    "failure_analysis.md",
    "model_card.md",
    "REPORT.md",
    "environment.json",
    "config_used.json",
}
PREDICTION_COLUMNS = {
    "fold",
    "model",
    "feature_date",
    "target_start_date",
    "target_end_date",
    "actual",
    "prediction",
    "regime",
}
REFERENCE_MODELS = {"mean_baseline", "ewma_baseline", "ridge", "random_forest"}
BINARY_REFERENCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".parquet"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object expected: {path}")
    return payload


def _verify_run_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Manifest has no artifact_sha256 map")
    missing = [name for name in hashes if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Manifest artifacts are missing: {missing}")
    actual_files = {
        path.name
        for path in run_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    unexpected = sorted(actual_files - set(hashes))
    if unexpected:
        raise RuntimeError(f"Unhashed run artifacts present: {unexpected}")
    mismatched = {
        name: (expected, sha256(run_dir / name))
        for name, expected in hashes.items()
        if expected != sha256(run_dir / name)
    }
    if mismatched:
        raise RuntimeError(f"Manifest hash mismatch: {mismatched}")
    return manifest, {str(name): str(value) for name, value in hashes.items()}


def _validate_prediction_tables(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[set[str], int, dict[str, int]]:
    models = {str(value) for value in manifest.get("models", [])}
    if not models:
        raise RuntimeError("Manifest must declare at least one model")
    try:
        folds = int(manifest["folds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Manifest must declare a positive integer fold count") from exc
    if folds <= 0:
        raise RuntimeError("Manifest fold count must be positive")

    predictions = pd.read_csv(run_dir / "predictions_oof.csv")
    missing_columns = sorted(PREDICTION_COLUMNS - set(predictions.columns))
    if missing_columns:
        raise RuntimeError(f"Prediction columns are incomplete: {missing_columns}")
    if set(predictions["model"].unique()) != models:
        raise RuntimeError("Prediction model set differs from the manifest")
    if predictions.duplicated(["fold", "model", "feature_date"]).any():
        raise RuntimeError("Duplicate fold/model/feature-date prediction rows")
    numeric_predictions = predictions[["actual", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_predictions).all():
        raise RuntimeError("Predictions contain non-finite values")
    if (numeric_predictions <= 0).any():
        raise RuntimeError("Actual or prediction is non-positive")
    feature_date = pd.to_datetime(predictions["feature_date"], utc=True)
    target_start = pd.to_datetime(predictions["target_start_date"], utc=True)
    target_end = pd.to_datetime(predictions["target_end_date"], utc=True)
    if not (feature_date < target_start).all() or not (target_start <= target_end).all():
        raise RuntimeError("Prediction timestamp ordering is invalid")

    expected_rows = manifest.get("oof_rows_per_model", {})
    if not isinstance(expected_rows, dict) or set(map(str, expected_rows)) != models:
        raise RuntimeError("Manifest oof_rows_per_model keys differ from models")
    actual_rows = predictions.groupby("model").size().astype(int).to_dict()
    expected_rows_normalized = {str(key): int(value) for key, value in expected_rows.items()}
    if actual_rows != expected_rows_normalized:
        raise RuntimeError(f"OOF row counts differ: expected {expected_rows}, actual {actual_rows}")

    test_rows_per_fold = manifest.get("test_rows_per_fold")
    if test_rows_per_fold is not None:
        try:
            expected_per_fold = int(test_rows_per_fold)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("test_rows_per_fold must be an integer") from exc
        if expected_per_fold <= 0:
            raise RuntimeError("test_rows_per_fold must be positive")
        counts = predictions.groupby(["model", "fold"]).size()
        if len(counts) != len(models) * folds or (counts != expected_per_fold).any():
            raise RuntimeError("Each model/fold does not have the configured test row count")

    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    if set(fold_metrics["model"].unique()) != models:
        raise RuntimeError("Fold metric model set differs from the manifest")
    if len(fold_metrics) != len(models) * folds:
        raise RuntimeError("Fold metric row count is inconsistent with models and folds")
    if fold_metrics.duplicated(["fold", "model"]).any():
        raise RuntimeError("Duplicate fold/model metric rows")
    train_end = pd.to_datetime(fold_metrics["train_target_end"], utc=True)
    test_start = pd.to_datetime(fold_metrics["test_start"], utc=True)
    if not (train_end < test_start).all():
        raise RuntimeError("Training labels overlap the test period")
    for name in ("mae", "rmse", "qlike", "calibration_ratio", "spearman"):
        if name in fold_metrics and not np.isfinite(pd.to_numeric(fold_metrics[name]).to_numpy()).all():
            raise RuntimeError(f"Fold metrics contain non-finite values: {name}")
    return models, folds, actual_rows


def validate_run_dir(
    run_dir: Path,
    source_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a self-describing run without reference-specific assumptions."""

    run_dir = Path(run_dir).resolve()
    manifest, hashes = _verify_run_manifest(run_dir)
    config_file = run_dir / str(manifest.get("config_file", "config_used.json"))
    if not config_file.is_file() or sha256(config_file) != manifest.get("config_sha256"):
        raise RuntimeError("Config hash does not match the manifest")
    if source_path is not None and sha256(Path(source_path)) != manifest.get("source_sha256"):
        raise RuntimeError("Supplied source hash does not match the manifest")
    if config_path is not None and sha256(Path(config_path)) != manifest.get("config_sha256"):
        raise RuntimeError("Supplied config hash does not match the manifest")
    if not REQUIRED_RUN_ARTIFACTS.issubset(hashes):
        raise RuntimeError(
            f"Manifest required artifacts are incomplete: {sorted(REQUIRED_RUN_ARTIFACTS - set(hashes))}"
        )
    checks = manifest.get("checks", {})
    if checks.get("artifact_hashes_complete") != "PASS":
        raise RuntimeError("Manifest artifact completeness check is absent")
    if any(value != "PASS" for value in checks.values()):
        raise RuntimeError(f"A manifest check is not PASS: {checks}")
    models, folds, actual_rows = _validate_prediction_tables(run_dir, manifest)
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "manifest_version": manifest.get("manifest_version"),
        "folds": folds,
        "models": sorted(models),
        "artifact_hashes_verified": len(hashes),
        "oof_rows_per_model": actual_rows,
    }


def _validate_json_schema(document: dict[str, Any], schema_path: Path, label: str) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised in dependency setup
        raise RuntimeError("jsonschema>=4.21,<5 is required for schema validation") from exc
    schema = _read_json(schema_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.path))
    if errors:
        message = "; ".join(error.message for error in errors[:3])
        raise RuntimeError(f"{label} schema validation failed: {message}")


def validate_reference_contract(reference_dir: Path) -> dict[str, Any]:
    """Validate the checked-in public SPY reference bundle contract."""

    reference_dir = Path(reference_dir).resolve()
    manifest_path = reference_dir / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("manifest_version") != 3:
        raise RuntimeError("Public reference manifest must use manifest_version 3")
    repo_root = Path(__file__).resolve().parents[2]
    _validate_json_schema(manifest, repo_root / "schemas" / "public_reference_manifest.schema.json", "reference manifest")
    hashes = manifest.get("public_artifact_sha256")
    details = manifest.get("public_artifact_hash_details")
    if not isinstance(hashes, dict) or not isinstance(details, dict):
        raise TypeError("Public reference manifest hash maps are incomplete")
    actual_files = {
        path.name for path in reference_dir.iterdir() if path.is_file() and path.name != manifest_path.name
    }
    if actual_files != set(hashes):
        raise RuntimeError(f"Public reference file set differs from manifest: {sorted(actual_files ^ set(hashes))}")
    for name, expected in hashes.items():
        path = reference_dir / name
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Public reference hash mismatch: {name}")
        item = details.get(name)
        if item is None and Path(name).suffix.lower() in BINARY_REFERENCE_SUFFIXES:
            # Binary figures are covered by the raw-byte map; LF normalization
            # is defined only for public text artifacts.
            continue
        if not isinstance(item, dict) or item.get("hash_algorithm") != "sha256":
            raise RuntimeError(f"Hash details are incomplete: {name}")
        if item.get("raw_sha256") != actual:
            raise RuntimeError(f"Raw hash detail mismatch: {name}")
        if item.get("text_normalization") == "git-lf-v1" and item.get("normalized_sha256") != _normalized_text_sha256(path):
            raise RuntimeError(f"Normalized hash detail mismatch: {name}")
    contract = manifest.get("hash_contract", {})
    if contract.get("artifact_sha256_mode") != "raw_bytes" or contract.get("text_normalization_mode") != "git-lf-v1":
        raise RuntimeError("Public hash contract mode is incomplete")
    required = {"aggregate_metrics.csv", "fold_metrics.csv", "config_used.json", "environment.json", "MATH_CLAIMS.json"}
    if not required.issubset(hashes):
        raise RuntimeError(f"Reference artifacts are incomplete: {sorted(required - set(hashes))}")

    if manifest.get("folds") != 27 or set(manifest.get("models", [])) != REFERENCE_MODELS:
        raise RuntimeError("Reference SPY fold/model contract is not canonical")
    rows = {str(key): int(value) for key, value in manifest.get("oof_rows_per_model", {}).items()}
    if rows != {model: 1620 for model in REFERENCE_MODELS}:
        raise RuntimeError(f"Reference OOF row contract is not canonical: {rows}")
    if any(value != "PASS" for value in manifest.get("checks", {}).values()):
        raise RuntimeError("Reference manifest contains a non-PASS check")

    claims_path = reference_dir / "MATH_CLAIMS.json"
    claims = _read_json(claims_path)
    _validate_json_schema(claims, repo_root / "schemas" / "math_claims.schema.json", "math claims")
    if claims.get("reference_manifest_version") != 3:
        raise RuntimeError("MATH_CLAIMS reference manifest version mismatch")
    if claims.get("canonical_run_id") != manifest.get("canonical_run_id"):
        raise RuntimeError("MATH_CLAIMS canonical run mismatch")
    if claims.get("canonical_experiment_code_commit") != manifest.get("canonical_code_commit"):
        raise RuntimeError("MATH_CLAIMS canonical code commit mismatch")
    return {
        "status": "PASS",
        "reference_dir": str(reference_dir),
        "manifest_version": 3,
        "folds": 27,
        "models": sorted(REFERENCE_MODELS),
        "artifact_hashes_verified": len(hashes),
        "math_claims": "MATH_CLAIMS.json",
    }
