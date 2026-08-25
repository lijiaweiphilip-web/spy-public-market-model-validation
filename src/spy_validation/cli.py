from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .config import RunConfig
from .data import fetch_yahoo_chart, load_adjusted_close_csv, load_yahoo_chart
from .evaluation import add_decision_cost_diagnostics, run_walk_forward
from .features import build_feature_frame
from .reporting import write_outputs


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
    )
    print(json.dumps({"output_dir": str(output_dir), "manifest": manifest}, indent=2))
    return 0


def validate_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = [
        "predictions_oof.csv",
        "fold_metrics.csv",
        "aggregate_metrics.csv",
        "regime_metrics.csv",
        "calibration_bins.csv",
        "decision_cost_sensitivity.csv",
        "failure_analysis.md",
        "model_card.md",
        "REPORT.md",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}")
    if manifest.get("folds") != 27:
        raise RuntimeError(f"Expected 27 folds in the reference configuration; got {manifest.get('folds')}")
    if manifest.get("checks", {}).get("point_predictions_saved") != "PASS":
        raise RuntimeError("Point prediction check is absent")
    print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "folds": manifest["folds"]}, indent=2))
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
    validate = subparsers.add_parser("validate", help="validate a completed run directory")
    validate.add_argument("--run-dir", required=True)
    validate.set_defaults(func=validate_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
