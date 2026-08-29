"""Cross-artifact run validation and the stricter public-reference contract.

``validate_run_dir`` validates the flat run schema emitted by this repository.
It is intentionally generic *within that schema*: it recomputes metrics from
the point-level predictions and checks that every derived table agrees. It
does not claim to validate arbitrary machine-learning pipelines.

``validate_reference_contract`` adds the SPY-specific invariants that apply to
the checked-in public reference bundle, where point-level predictions are
deliberately not redistributed. Its validation scope is summary/provenance
validation, not full point-level recomputation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .evaluation import add_decision_cost_diagnostics
from .metrics import calibration_table, regression_metrics
from .provenance import (
    BINARY_SUFFIXES,
    RAW_HASH_MODE,
    TEXT_HASH_MODE,
    canonical_bytes,
    is_safe_flat_path,
    sha256_bytes,
)

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
METRIC_COLUMNS = ("n", "mae", "rmse", "qlike", "calibration_ratio", "spearman")
REFERENCE_MODELS = {"mean_baseline", "ewma_baseline", "ridge", "random_forest"}
BINARY_REFERENCE_SUFFIXES = BINARY_SUFFIXES
HASH_PATTERN = set("0123456789abcdef")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    return sha256_bytes(canonical_bytes(path, TEXT_HASH_MODE))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object expected: {path}")
    return payload


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return result


def _strict_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{name} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{name} must contain non-empty strings")
        result.append(item)
    if len(set(result)) != len(result):
        raise RuntimeError(f"{name} contains duplicates")
    return result


def _strict_integral_series(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise RuntimeError(f"{name} must contain finite integer values")
    return numeric.astype("int64")


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for name in columns:
        if name not in frame.columns:
            raise RuntimeError(f"{label} is missing column: {name}")
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label} contains non-finite values: {name}")


def _compare_metric_table(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    required = keys + list(METRIC_COLUMNS)
    missing = sorted(set(required) - set(actual.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")
    if actual.duplicated(keys).any():
        raise RuntimeError(f"Duplicate {label.lower()} rows for keys {keys}")
    _finite_columns(actual, METRIC_COLUMNS, label)
    actual_keys = {tuple(row) for row in actual[keys].itertuples(index=False, name=None)}
    expected_keys = {tuple(row) for row in expected[keys].itertuples(index=False, name=None)}
    if actual_keys != expected_keys:
        raise RuntimeError(f"{label} keys differ from predictions")
    merged = actual[keys + list(METRIC_COLUMNS)].merge(
        expected[keys + list(METRIC_COLUMNS)],
        on=keys,
        suffixes=("", "_expected"),
        how="left",
    )
    for column in METRIC_COLUMNS:
        left = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[f"{column}_expected"], errors="coerce").to_numpy(dtype=float)
        matches = np.equal(left, right) if column == "n" else np.isclose(left, right, rtol=1e-6, atol=1e-10)
        if not bool(matches.all()):
            raise RuntimeError(f"{label} does not match predictions for {column}")


def _verify_run_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Manifest has no artifact_sha256 map")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"Manifest artifact name is not a flat relative path: {name!r}")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or set(expected.lower()) - HASH_PATTERN
        ):
            raise RuntimeError(f"Manifest hash is not a SHA-256 digest: {name}")
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
    run_dir: Path, manifest: dict[str, Any], config_file: Path
) -> tuple[set[str], int, dict[str, int]]:
    models = set(_strict_string_list(manifest.get("models"), "Manifest models"))
    folds = _strict_int(manifest.get("folds"), "Manifest fold count", minimum=1)

    predictions = pd.read_csv(run_dir / "predictions_oof.csv")
    missing_columns = sorted(PREDICTION_COLUMNS - set(predictions.columns))
    if missing_columns:
        raise RuntimeError(f"Prediction columns are incomplete: {missing_columns}")
    predictions["fold"] = _strict_integral_series(predictions["fold"], "Prediction folds")
    if predictions["model"].map(lambda value: not isinstance(value, str) or not value.strip()).any():
        raise RuntimeError("Prediction models must be non-empty strings")
    predictions["model"] = predictions["model"].astype(str)
    if set(predictions["model"]) != models:
        raise RuntimeError("Prediction model set differs from the manifest")
    if predictions["regime"].map(lambda value: not isinstance(value, str) or not value.strip()).any():
        raise RuntimeError("Prediction regimes must be non-empty strings")
    if predictions.duplicated(["fold", "model", "feature_date"]).any():
        raise RuntimeError("Duplicate fold/model/feature-date prediction rows")
    numeric_predictions = predictions[["actual", "prediction"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric_predictions).all():
        raise RuntimeError("Predictions contain non-finite values")
    if (numeric_predictions <= 0).any():
        raise RuntimeError("Actual or prediction is non-positive")
    for column in ("feature_date", "target_start_date", "target_end_date"):
        parsed = pd.to_datetime(predictions[column], format="mixed", utc=True, errors="coerce")
        if parsed.isna().any():
            raise RuntimeError(f"Prediction timestamps are invalid: {column}")
        predictions[column] = parsed
    if not (predictions["feature_date"] < predictions["target_start_date"]).all() or not (
        predictions["target_start_date"] <= predictions["target_end_date"]
    ).all():
        raise RuntimeError("Prediction timestamp ordering is invalid")

    fold_ids = {int(value) for value in predictions["fold"].unique()}
    if len(fold_ids) != folds:
        raise RuntimeError("Manifest fold count differs from predictions")
    expected_pairs = {(fold, model) for fold in fold_ids for model in models}
    actual_pairs = set(zip(predictions["fold"], predictions["model"], strict=False))
    if actual_pairs != expected_pairs:
        raise RuntimeError("Prediction model/fold pairs are incomplete")
    expected_rows = manifest.get("oof_rows_per_model")
    if not isinstance(expected_rows, dict) or set(map(str, expected_rows)) != models:
        raise RuntimeError("Manifest oof_rows_per_model keys differ from models")
    expected_rows_normalized = {
        str(key): _strict_int(value, f"OOF rows for {key}", minimum=1)
        for key, value in expected_rows.items()
    }
    actual_rows = predictions.groupby("model").size().astype(int).to_dict()
    if actual_rows != expected_rows_normalized:
        raise RuntimeError(f"OOF row counts differ: expected {expected_rows}, actual {actual_rows}")

    test_rows_per_fold = manifest.get("test_rows_per_fold")
    if test_rows_per_fold is not None:
        expected_per_fold = _strict_int(test_rows_per_fold, "test_rows_per_fold", minimum=1)
        counts = predictions.groupby(["model", "fold"]).size()
        if len(counts) != len(expected_pairs) or (counts != expected_per_fold).any():
            raise RuntimeError("Each model/fold does not have the configured test row count")

    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    fold_metrics["fold"] = _strict_integral_series(fold_metrics["fold"], "Fold metric folds")
    if fold_metrics["model"].map(lambda value: not isinstance(value, str) or not value.strip()).any():
        raise RuntimeError("Fold metric models must be non-empty strings")
    fold_metrics["model"] = fold_metrics["model"].astype(str)
    _finite_columns(fold_metrics, METRIC_COLUMNS, "Fold metrics")
    if set(fold_metrics["model"]) != models:
        raise RuntimeError("Fold metric model set differs from the manifest")
    fold_pairs = set(zip(fold_metrics["fold"], fold_metrics["model"], strict=False))
    if fold_pairs != expected_pairs:
        raise RuntimeError("Fold metric model/fold pairs differ from predictions")
    if fold_metrics.duplicated(["fold", "model"]).any():
        raise RuntimeError("Duplicate fold/model metric rows")
    for column in ("train_target_end", "test_start"):
        parsed = pd.to_datetime(fold_metrics[column], format="mixed", utc=True, errors="coerce")
        if parsed.isna().any():
            raise RuntimeError(f"Fold metric timestamps are invalid: {column}")
        fold_metrics[column] = parsed
    if not (fold_metrics["train_target_end"] < fold_metrics["test_start"]).all():
        raise RuntimeError("Training labels overlap the test period")

    fold_expected_rows: list[dict[str, Any]] = []
    for (fold, model), group in predictions.groupby(["fold", "model"], sort=True):
        fold_expected_rows.append(
            {"fold": int(fold), "model": model, **regression_metrics(group["actual"], group["prediction"])}
        )
    _compare_metric_table(
        fold_metrics,
        pd.DataFrame(fold_expected_rows),
        ["fold", "model"],
        "Fold metrics",
    )

    aggregate = pd.read_csv(run_dir / "aggregate_metrics.csv")
    aggregate_expected = pd.DataFrame(
        [
            {"model": model, **regression_metrics(group["actual"], group["prediction"])}
            for model, group in predictions.groupby("model", sort=True)
        ]
    )
    _compare_metric_table(aggregate, aggregate_expected, ["model"], "Aggregate metrics")

    regime_metrics = pd.read_csv(run_dir / "regime_metrics.csv")
    regime_expected = pd.DataFrame(
        [
            {"model": model, "regime": regime, **regression_metrics(group["actual"], group["prediction"])}
            for (model, regime), group in predictions.groupby(["model", "regime"], sort=True)
        ]
    )
    _compare_metric_table(regime_metrics, regime_expected, ["model", "regime"], "Regime metrics")

    calibration_bins = pd.read_csv(run_dir / "calibration_bins.csv")
    cfg = RunConfig.load(config_file)
    calibration_expected = calibration_table(predictions, cfg.calibration_bins)
    calibration_keys = ["model", "calibration_bin"]
    for column in calibration_keys:
        if column not in calibration_bins:
            raise RuntimeError(f"Calibration bins is missing column: {column}")
    if calibration_bins.duplicated(calibration_keys).any():
        raise RuntimeError("Duplicate calibration-bin rows")
    calibration_columns = ["n", "mean_prediction", "mean_actual", "absolute_log_ratio"]
    _finite_columns(calibration_bins, tuple(calibration_columns), "Calibration bins")
    calibration_actual_keys = {
        tuple(row) for row in calibration_bins[calibration_keys].itertuples(index=False, name=None)
    }
    calibration_expected_keys = {
        tuple(row) for row in calibration_expected[calibration_keys].itertuples(index=False, name=None)
    }
    if calibration_actual_keys != calibration_expected_keys:
        raise RuntimeError("Calibration-bin keys differ from predictions")
    merged_calibration = calibration_bins[calibration_keys + calibration_columns].merge(
        calibration_expected[calibration_keys + calibration_columns],
        on=calibration_keys,
        suffixes=("", "_expected"),
        how="left",
    )
    for column in calibration_columns:
        left = pd.to_numeric(merged_calibration[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged_calibration[f"{column}_expected"], errors="coerce").to_numpy(dtype=float)
        matches = np.equal(left, right) if column == "n" else np.isclose(left, right, rtol=1e-6, atol=1e-10)
        if not bool(matches.all()):
            raise RuntimeError(f"Calibration bins do not match predictions for {column}")

    _exposure, expected_costs = add_decision_cost_diagnostics(predictions, cfg)
    costs = pd.read_csv(run_dir / "decision_cost_sensitivity.csv")
    _compare_cost_table(
        costs,
        expected_costs,
        ["model", "cost_bps"],
        [
            "observations",
            "total_turnover",
            "mean_daily_turnover",
            "mean_exposure",
            "total_cost_fraction_per_unit_capital",
            "annualised_cost_bps_per_unit_capital",
        ],
    )
    return models, folds, actual_rows


def _compare_cost_table(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    columns: list[str],
) -> None:
    required = keys + columns
    missing = sorted(set(required) - set(actual.columns))
    if missing:
        raise RuntimeError(f"Decision-cost table is missing columns: {missing}")
    if actual.duplicated(keys).any():
        raise RuntimeError("Duplicate decision-cost rows")
    _finite_columns(actual, tuple(columns), "Decision-cost table")
    actual_keys = {tuple(row) for row in actual[keys].itertuples(index=False, name=None)}
    expected_keys = {tuple(row) for row in expected[keys].itertuples(index=False, name=None)}
    if actual_keys != expected_keys:
        raise RuntimeError("Decision-cost keys differ from predictions/config")
    merged = actual[required].merge(expected[required], on=keys, suffixes=("", "_expected"), how="left")
    for column in columns:
        left = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[f"{column}_expected"], errors="coerce").to_numpy(dtype=float)
        matches = np.equal(left, right) if column == "observations" else np.isclose(left, right, rtol=1e-6, atol=1e-10)
        if not bool(matches.all()):
            raise RuntimeError(f"Decision-cost table does not match predictions for {column}")


def validate_run_dir(
    run_dir: Path,
    source_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a self-describing run and recompute all derived tables."""

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
    if not isinstance(checks, dict) or checks.get("artifact_hashes_complete") != "PASS":
        raise RuntimeError("Manifest artifact completeness check is absent")
    if any(value != "PASS" for value in checks.values()):
        raise RuntimeError(f"A manifest check is not PASS: {checks}")
    models, folds, actual_rows = _validate_prediction_tables(run_dir, manifest, config_file)
    return {
        "status": "PASS",
        "validation_scope": "generic_full_recomputation",
        "run_dir": str(run_dir),
        "manifest_version": manifest.get("manifest_version"),
        "folds": folds,
        "models": sorted(models),
        "artifact_hashes_verified": len(hashes),
        "oof_rows_per_model": actual_rows,
        "derived_tables_recomputed": True,
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


def _schema_path(name: str) -> Path:
    """Resolve a schema from an installed wheel or a source checkout."""

    packaged = Path(__file__).resolve().parent / "schemas" / name
    if packaged.is_file():
        return packaged
    source_tree = Path(__file__).resolve().parents[2] / "schemas" / name
    return source_tree


def _validate_reference_summary_tables(reference_dir: Path, manifest: dict[str, Any]) -> None:
    models = set(_strict_string_list(manifest.get("models"), "Reference models"))
    folds = _strict_int(manifest.get("folds"), "Reference fold count", minimum=1)
    fold_metrics = pd.read_csv(reference_dir / "fold_metrics.csv")
    fold_metrics["fold"] = _strict_integral_series(fold_metrics["fold"], "Reference fold metrics folds")
    fold_metrics["model"] = fold_metrics["model"].astype(str)
    if set(fold_metrics["model"]) != models or len(fold_metrics) != len(models) * folds:
        raise RuntimeError("Reference fold metrics are inconsistent with manifest")
    if fold_metrics.duplicated(["fold", "model"]).any():
        raise RuntimeError("Reference fold metrics contain duplicate fold/model rows")
    _finite_columns(fold_metrics, METRIC_COLUMNS, "Reference fold metrics")
    for column in ("train_target_end", "test_start"):
        parsed = pd.to_datetime(fold_metrics[column], format="mixed", utc=True, errors="coerce")
        if parsed.isna().any():
            raise RuntimeError(f"Reference fold metric timestamps are invalid: {column}")
        fold_metrics[column] = parsed
    if not (fold_metrics["train_target_end"] < fold_metrics["test_start"]).all():
        raise RuntimeError("Reference fold metrics contain temporal overlap")
    aggregate = pd.read_csv(reference_dir / "aggregate_metrics.csv")
    aggregate["model"] = aggregate["model"].astype(str)
    if set(aggregate["model"]) != models:
        raise RuntimeError("Reference aggregate models differ from manifest")
    if aggregate.duplicated(["model"]).any():
        raise RuntimeError("Reference aggregate metrics contain duplicate models")
    _finite_columns(aggregate, METRIC_COLUMNS, "Reference aggregate metrics")
    rows = {
        str(key): _strict_int(value, f"Reference rows for {key}", minimum=1)
        for key, value in manifest["oof_rows_per_model"].items()
    }
    for _, row in aggregate.iterrows():
        if int(row["n"]) != rows[str(row["model"])]:
            raise RuntimeError("Reference aggregate OOF counts differ from manifest")
    regime = pd.read_csv(reference_dir / "regime_metrics.csv")
    regime_models = set(regime["model"].astype(str))
    if regime.empty or not regime_models.issubset(models):
        raise RuntimeError("Reference regime metrics are empty or contain unknown models")
    _finite_columns(regime, METRIC_COLUMNS, "Reference regime metrics")
    calibration = pd.read_csv(reference_dir / "calibration_bins.csv")
    if calibration.empty:
        raise RuntimeError("Reference calibration bins are empty")
    _finite_columns(
        calibration,
        ("n", "mean_prediction", "mean_actual", "absolute_log_ratio"),
        "Reference calibration bins",
    )
    costs = pd.read_csv(reference_dir / "decision_cost_sensitivity.csv")
    if set(costs["model"].astype(str)) != models:
        raise RuntimeError("Reference decision-cost models differ from manifest")
    _finite_columns(
        costs,
        (
            "observations",
            "total_turnover",
            "mean_daily_turnover",
            "mean_exposure",
            "total_cost_fraction_per_unit_capital",
            "annualised_cost_bps_per_unit_capital",
        ),
        "Reference decision-cost table",
    )


def validate_reference_contract(reference_dir: Path) -> dict[str, Any]:
    """Validate the checked-in public SPY reference bundle contract."""

    reference_dir = Path(reference_dir).resolve()
    manifest_path = reference_dir / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("manifest_version") != 3:
        raise RuntimeError("Public reference manifest must use manifest_version 3")
    _validate_json_schema(
        manifest,
        _schema_path("public_reference_manifest.schema.json"),
        "reference manifest",
    )
    hashes = manifest.get("public_artifact_sha256")
    details = manifest.get("public_artifact_hash_details")
    if not isinstance(hashes, dict) or not isinstance(details, dict):
        raise TypeError("Public reference manifest hash maps are incomplete")
    actual_files = {
        path.name
        for path in reference_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    if actual_files != set(hashes):
        raise RuntimeError(f"Public reference file set differs from manifest: {sorted(actual_files ^ set(hashes))}")
    for name, expected in hashes.items():
        if not is_safe_flat_path(name) or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"Invalid public artifact hash entry: {name}")
        path = reference_dir / name
        item = details.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"Hash details are incomplete: {name}")  # noqa: TRY004
        content_type = item.get("content_type")
        hash_mode = item.get("hash_mode")
        if content_type == "text" and hash_mode != TEXT_HASH_MODE:
            raise RuntimeError(f"Text artifact must use {TEXT_HASH_MODE}: {name}")
        if content_type == "binary" and hash_mode != RAW_HASH_MODE:
            raise RuntimeError(f"Binary artifact must use {RAW_HASH_MODE}: {name}")
        if content_type not in {"text", "binary"} or item.get("hash_algorithm") != "sha256":
            raise RuntimeError(f"Hash details are incomplete: {name}")
        actual = sha256_bytes(canonical_bytes(path, str(hash_mode)))
        canonical = item.get("canonical_sha256")
        if not isinstance(canonical, str) or len(canonical) != 64 or set(canonical.lower()) - HASH_PATTERN:
            raise RuntimeError(f"Invalid canonical hash detail: {name}")
        if actual != canonical or actual != expected:
            raise RuntimeError(f"Public reference hash mismatch: {name}")
        if content_type == "text":
            source_raw = item.get("source_raw_sha256")
            if not isinstance(source_raw, str) or len(source_raw) != 64 or set(source_raw.lower()) - HASH_PATTERN:
                raise RuntimeError(f"Text source raw hash detail is invalid: {name}")
    contract = manifest.get("hash_contract", {})
    if (
        contract.get("artifact_sha256_mode") != "canonical_by_hash_mode"
        or contract.get("text_normalization_mode") != TEXT_HASH_MODE
        or contract.get("binary_hash_mode") != RAW_HASH_MODE
        or contract.get("canonical_hash_field") != "canonical_sha256"
    ):
        raise RuntimeError("Public hash contract mode is incomplete")
    required = {"aggregate_metrics.csv", "fold_metrics.csv", "config_used.json", "environment.json", "MATH_CLAIMS.json"}
    if not required.issubset(hashes):
        raise RuntimeError(f"Reference artifacts are incomplete: {sorted(required - set(hashes))}")

    if manifest.get("folds") != 27 or set(manifest.get("models", [])) != REFERENCE_MODELS:
        raise RuntimeError("Reference SPY fold/model contract is not canonical")
    rows = {
        str(key): _strict_int(value, f"Reference rows for {key}", minimum=1)
        for key, value in manifest.get("oof_rows_per_model", {}).items()
    }
    if rows != {model: 1620 for model in REFERENCE_MODELS}:
        raise RuntimeError(f"Reference OOF row contract is not canonical: {rows}")
    if any(value != "PASS" for value in manifest.get("checks", {}).values()):
        raise RuntimeError("Reference manifest contains a non-PASS check")
    _validate_reference_summary_tables(reference_dir, manifest)

    claims_path = reference_dir / "MATH_CLAIMS.json"
    claims = _read_json(claims_path)
    _validate_json_schema(claims, _schema_path("math_claims.schema.json"), "math claims")
    if claims.get("reference_manifest_version") != 3:
        raise RuntimeError("MATH_CLAIMS reference manifest version mismatch")
    if claims.get("canonical_run_id") != manifest.get("canonical_run_id"):
        raise RuntimeError("MATH_CLAIMS canonical run mismatch")
    if claims.get("canonical_experiment_code_commit") != manifest.get("canonical_code_commit"):
        raise RuntimeError("MATH_CLAIMS canonical code commit mismatch")
    return {
        "status": "PASS",
        "validation_scope": "public_reference_summary_and_provenance_validation",
        "reference_dir": str(reference_dir),
        "manifest_version": 3,
        "folds": 27,
        "models": sorted(REFERENCE_MODELS),
        "artifact_hashes_verified": len(hashes),
        "math_claims": "MATH_CLAIMS.json",
        "summary_tables_checked": True,
    }
