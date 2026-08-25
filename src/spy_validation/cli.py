from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .data import fetch_yahoo_chart, load_adjusted_close_csv, load_yahoo_chart
from .evaluation import add_decision_cost_diagnostics, run_walk_forward
from .features import build_feature_frame
from .reporting import write_outputs


def _execute_pipeline(
    config_path: Path,
    cfg: RunConfig,
    output_dir: Path,
    source_path: Path,
    prices: pd.DataFrame,
    *,
    run_label: str = "Reference-run",
    data_description: str | None = None,
    run_type: str = "local_or_canonical",
    repository_path: Path | None = None,
) -> dict:
    feature_frame = build_feature_frame(prices, cfg.target_horizon_days)
    feature_frame.attrs["raw_price_rows"] = len(prices)
    outputs = run_walk_forward(feature_frame, cfg)
    exposure, costs = add_decision_cost_diagnostics(outputs["predictions"], cfg)
    command = " ".join(shlex.quote(token) for token in sys.argv)
    manifest = write_outputs(
        output_dir=output_dir,
        config_path=config_path,
        cfg=cfg,
        source_path=source_path,
        feature_frame=feature_frame,
        outputs=outputs,
        exposure_path=exposure,
        cost_summary=costs,
        command=command,
        run_label=run_label,
        data_description=data_description,
        run_type=run_type,
        repository_path=repository_path,
    )
    return manifest


def run_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    cfg = RunConfig.load(config_path)
    output_dir = Path(args.output_dir or cfg.output_dir).resolve()
    if args.input_json:
        source_path = Path(args.input_json).resolve()
        prices = load_yahoo_chart(source_path)
    elif args.input_csv:
        source_path = Path(args.input_csv).resolve()
        prices = load_adjusted_close_csv(source_path)
    else:
        source_path = output_dir / "private_raw_yahoo_chart.json"
        fetch_yahoo_chart(cfg.symbol, cfg.data_range, cfg.interval, source_path)
        prices = load_yahoo_chart(source_path)

    manifest = _execute_pipeline(config_path, cfg, output_dir, source_path, prices)
    print(json.dumps({"output_dir": str(output_dir), "manifest": manifest}, indent=2))
    return 0


def generate_synthetic_adjusted_close(rows: int = 1180, seed: int = 20260825) -> pd.DataFrame:
    """Generate deterministic price-only data for the public demo.

    The demo deliberately creates only adjusted-close prices. Targets and all
    validation artefacts are still derived by the normal feature/evaluation
    pipeline, so synthetic metrics cannot be mistaken for SPY evidence.
    """
    if rows < 200:
        raise ValueError("demo requires at least 200 price rows")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=rows, freq="B", tz="UTC")
    index = np.arange(rows)
    volatility = np.where((index // 180) % 2 == 0, 0.008, 0.018)
    returns = 0.00015 + rng.normal(0.0, volatility, rows)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": dates, "adjusted_close": prices})


def demo_command(args: argparse.Namespace) -> int:
    base_config_path = Path(args.config).resolve()
    base_cfg = RunConfig.load(base_config_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_values = base_cfg.to_dict()
    demo_values.update(
        {
            "symbol": "SYNTHETIC",
            "data_range": "synthetic",
            "interval": "1d",
            "minimum_training_rows": 300,
            "test_rows_per_fold": 30,
            "step_rows": 30,
            "forest_estimators": min(base_cfg.forest_estimators, 12),
            "calibration_block_rows": min(base_cfg.calibration_block_rows, 60),
            "bootstrap_repetitions": min(base_cfg.bootstrap_repetitions, 100),
            "output_dir": str(output_dir),
        }
    )
    demo_config_path = output_dir / "demo_config.json"
    demo_config_path.write_text(json.dumps(demo_values, indent=2), encoding="utf-8")
    cfg = RunConfig.load(demo_config_path)
    source_path = output_dir / "synthetic_adjusted_close.csv"
    generate_synthetic_adjusted_close(rows=args.rows, seed=args.seed).to_csv(
        source_path, index=False
    )
    prices = load_adjusted_close_csv(source_path)
    manifest = _execute_pipeline(
        demo_config_path,
        cfg,
        output_dir,
        source_path,
        prices,
        run_label="Synthetic demo",
        data_description="Deterministic synthetic adjusted-close series (reproducibility demonstration; not SPY reference data)",
        run_type="synthetic_demo",
        repository_path=Path(__file__).resolve().parents[2],
    )
    print(json.dumps({"output_dir": str(output_dir), "manifest": manifest}, indent=2))
    return 0


def validate_run_dir(
    run_dir: Path,
    source_path: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Manifest has no artifact_sha256 map")
    missing = [name for name in hashes if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Manifest artifacts are missing: {missing}")
    actual_files = {
        path.name for path in run_dir.iterdir() if path.is_file() and path.name != "run_manifest.json"
    }
    unexpected = sorted(actual_files - set(hashes))
    if unexpected:
        raise RuntimeError(f"Unhashed run artifacts present: {unexpected}")
    mismatched = {
        name: (expected, _sha256(run_dir / name))
        for name, expected in hashes.items()
        if expected != _sha256(run_dir / name)
    }
    if mismatched:
        raise RuntimeError(f"Manifest hash mismatch: {mismatched}")

    config_file = run_dir / str(manifest.get("config_file", "config_used.json"))
    if not config_file.is_file() or _sha256(config_file) != manifest.get("config_sha256"):
        raise RuntimeError("Config hash does not match the manifest")
    if config_path is not None and _sha256(config_path) != manifest.get("config_sha256"):
        raise RuntimeError("Supplied config hash does not match the manifest")
    if source_path is not None and _sha256(source_path) != manifest.get("source_sha256"):
        raise RuntimeError("Supplied source hash does not match the manifest")

    required = {"predictions_oof.csv", "fold_metrics.csv", "aggregate_metrics.csv", "regime_metrics.csv", "calibration_bins.csv", "decision_cost_sensitivity.csv", "failure_analysis.md", "model_card.md", "REPORT.md", "environment.json", "config_used.json"}
    if not required.issubset(hashes):
        raise RuntimeError(f"Manifest required artifacts are incomplete: {sorted(required - set(hashes))}")
    if manifest.get("folds") != 27:
        raise RuntimeError(f"Expected 27 folds in the reference configuration; got {manifest.get('folds')}")
    models = set(manifest.get("models", []))
    if models != {"mean_baseline", "ewma_baseline", "ridge", "random_forest"}:
        raise RuntimeError(f"Unexpected models in manifest: {sorted(models)}")
    if manifest.get("checks", {}).get("artifact_hashes_complete") != "PASS":
        raise RuntimeError("Manifest artifact completeness check is absent")
    if any(value != "PASS" for value in manifest.get("checks", {}).values()):
        raise RuntimeError(f"A manifest check is not PASS: {manifest.get('checks')}")

    predictions = pd.read_csv(run_dir / "predictions_oof.csv")
    expected_columns = {"fold", "model", "feature_date", "target_start_date", "target_end_date", "actual", "prediction", "regime"}
    if not expected_columns.issubset(predictions.columns):
        raise RuntimeError(f"Prediction columns are incomplete: {sorted(expected_columns - set(predictions.columns))}")
    if set(predictions["model"].unique()) != models:
        raise RuntimeError("Prediction model set differs from the manifest")
    if predictions.duplicated(["fold", "model", "feature_date"]).any():
        raise RuntimeError("Duplicate fold/model/feature-date prediction rows")
    if not np.isfinite(predictions[["actual", "prediction"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Predictions contain non-finite values")
    if (predictions[["actual", "prediction"]] <= 0).any().any():
        raise RuntimeError("Actual or prediction is non-positive")
    feature_date = pd.to_datetime(predictions["feature_date"], utc=True)
    target_start = pd.to_datetime(predictions["target_start_date"], utc=True)
    target_end = pd.to_datetime(predictions["target_end_date"], utc=True)
    if not (feature_date < target_start).all() or not (target_start <= target_end).all():
        raise RuntimeError("Prediction timestamp ordering is invalid")
    expected_rows = manifest.get("oof_rows_per_model", {})
    actual_rows = predictions.groupby("model").size().astype(int).to_dict()
    if actual_rows != {str(k): int(v) for k, v in expected_rows.items()}:
        raise RuntimeError(f"OOF row counts differ: expected {expected_rows}, actual {actual_rows}")
    expected_rows_per_fold = int(manifest.get("test_rows_per_fold", 0))
    if predictions.groupby(["model", "fold"]).size().nunique() != 1 or (predictions.groupby(["model", "fold"]).size() != expected_rows_per_fold).any():
        raise RuntimeError("Each model/fold does not have the configured test row count")

    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    if len(fold_metrics) != len(models) * int(manifest["folds"]):
        raise RuntimeError("Fold metric row count is inconsistent with models and folds")
    train_end = pd.to_datetime(fold_metrics["train_target_end"], utc=True)
    test_start = pd.to_datetime(fold_metrics["test_start"], utc=True)
    if not (train_end < test_start).all():
        raise RuntimeError("Training labels overlap the test period")
    if fold_metrics.duplicated(["fold", "model"]).any():
        raise RuntimeError("Duplicate fold/model metric rows")
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "folds": int(manifest["folds"]),
        "models": sorted(models),
        "artifact_hashes_verified": len(hashes),
        "oof_rows_per_model": actual_rows,
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_command(args: argparse.Namespace) -> int:
    result = validate_run_dir(
        Path(args.run_dir),
        Path(args.source_path).resolve() if args.source_path else None,
        Path(args.config_path).resolve() if args.config_path else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPY public-market model validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the full validation pipeline")
    run.add_argument("--config", default="configs/default.json")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--input-json")
    group.add_argument("--input-csv")
    run.add_argument("--output-dir")
    run.set_defaults(func=run_command)
    demo = subparsers.add_parser("demo", help="run the public synthetic end-to-end demo")
    demo.add_argument("--config", default="configs/default.json")
    demo.add_argument("--output-dir", default="runs/demo")
    demo.add_argument("--rows", type=int, default=1180)
    demo.add_argument("--seed", type=int, default=20260825)
    demo.set_defaults(func=demo_command)
    validate = subparsers.add_parser("validate", help="validate a completed run directory")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--source-path")
    validate.add_argument("--config-path")
    validate.set_defaults(func=validate_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
