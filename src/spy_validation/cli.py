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
from .validation import validate_reference_contract, validate_run_dir


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


def validate_command(args: argparse.Namespace) -> int:
    result = validate_run_dir(
        Path(args.run_dir),
        Path(args.source_path).resolve() if args.source_path else None,
        Path(args.config_path).resolve() if args.config_path else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def validate_reference_command(args: argparse.Namespace) -> int:
    result = validate_reference_contract(Path(args.reference_dir))
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
    reference = subparsers.add_parser(
        "validate-reference", help="validate the checked-in canonical public reference bundle"
    )
    reference.add_argument("--reference-dir", required=True)
    reference.set_defaults(func=validate_reference_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
