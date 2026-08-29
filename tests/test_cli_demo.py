from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from spy_validation.cli import generate_synthetic_adjusted_close, main, validate_run_dir


def test_synthetic_demo_series_is_deterministic_and_price_only():
    first = generate_synthetic_adjusted_close(rows=1180, seed=20260825)
    second = generate_synthetic_adjusted_close(rows=1180, seed=20260825)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["date", "adjusted_close"]
    assert len(first) == 1180
    assert (first["adjusted_close"] > 0).all()


def test_cli_demo_runs_complete_pipeline_and_validates(tmp_path: Path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "demo"
    config_path = repo / "configs" / "default.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spy-validate",
            "demo",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--rows",
            "500",
            "--seed",
            "20260825",
        ],
    )

    assert main() == 0
    source = output_dir / "synthetic_adjusted_close.csv"
    result = validate_run_dir(output_dir, source_path=source)
    assert result["status"] == "PASS"
    assert result["folds"] == 4
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_type"] == "synthetic_demo"
    assert manifest["oof_rows_per_model"] == {
        "ewma_baseline": 120,
        "mean_baseline": 120,
        "random_forest": 120,
        "ridge": 120,
    }
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "synthetic adjusted-close series" in report
    assert "SPY canonical" not in report


def test_cli_validate_reference_command(tmp_path: Path, monkeypatch):
    reference_dir = Path(__file__).resolve().parents[1] / "results" / "reference_run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["spy-validate", "validate-reference", "--reference-dir", str(reference_dir)],
    )
    assert main() == 0
