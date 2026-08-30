from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from spy_validation.validation import validate_run_dir


def _copy_run(tmp_path: Path, cached_small_run: tuple[Path, Path, Path]) -> tuple[Path, Path, Path]:
    cached_run, cached_source, cached_config = cached_small_run
    run_dir = tmp_path / "run"
    shutil.copytree(cached_run, run_dir)
    source = tmp_path / "source.csv"
    config = tmp_path / "config.json"
    shutil.copy2(cached_source, source)
    shutil.copy2(cached_config, config)
    return run_dir, source, config


def _rewrite_manifest_hash(run_dir: Path, artifact: str) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][artifact] = hashlib.sha256((run_dir / artifact).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_aggregate_metrics_are_recomputed_from_predictions(tmp_path: Path, cached_small_run) -> None:
    run_dir, source, config = _copy_run(tmp_path, cached_small_run)
    aggregate = pd.read_csv(run_dir / "aggregate_metrics.csv")
    aggregate.loc[aggregate["model"] == "ridge", "rmse"] += 0.1
    aggregate.to_csv(run_dir / "aggregate_metrics.csv", index=False)
    _rewrite_manifest_hash(run_dir, "aggregate_metrics.csv")

    with pytest.raises(RuntimeError, match="Aggregate metrics does not match predictions"):
        validate_run_dir(run_dir, source, config)


def test_non_integral_manifest_fold_count_is_rejected(tmp_path: Path, cached_small_run) -> None:
    run_dir, source, config = _copy_run(tmp_path, cached_small_run)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises((RuntimeError, TypeError), match="fold count must be an integer"):
        validate_run_dir(run_dir, source, config)


def test_invalid_prediction_timestamp_is_rejected(tmp_path: Path, cached_small_run) -> None:
    run_dir, source, config = _copy_run(tmp_path, cached_small_run)
    predictions_path = run_dir / "predictions_oof.csv"
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, "target_end_date"] = "not-a-date"
    predictions.to_csv(predictions_path, index=False)
    _rewrite_manifest_hash(run_dir, "predictions_oof.csv")

    with pytest.raises(RuntimeError, match="timestamps are invalid"):
        validate_run_dir(run_dir, source, config)


def test_decision_cost_table_is_recomputed_from_predictions(tmp_path: Path, cached_small_run) -> None:
    run_dir, source, config = _copy_run(tmp_path, cached_small_run)
    costs = pd.read_csv(run_dir / "decision_cost_sensitivity.csv")
    costs.loc[0, "total_turnover"] += 1.0
    costs.to_csv(run_dir / "decision_cost_sensitivity.csv", index=False)
    _rewrite_manifest_hash(run_dir, "decision_cost_sensitivity.csv")

    with pytest.raises(RuntimeError, match="Decision-cost table does not match predictions"):
        validate_run_dir(run_dir, source, config)
