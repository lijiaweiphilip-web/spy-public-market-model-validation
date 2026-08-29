"""Build the public machine-readable mathematical claim contract.

The output is derived from checked-in public reference artifacts. It excludes
point-level predictions and the working-tree commit so provenance is explicit
and non-circular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from spy_validation.provenance import artifact_hash_detail, raw_sha256
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from spy_validation.provenance import artifact_hash_detail, raw_sha256


def build_math_claims(repo_root: Path, manifest: dict | None = None) -> dict:
    repo_root = repo_root.resolve()
    reference_dir = repo_root / "results" / "reference_run"
    manifest_path = reference_dir / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest = manifest or json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((reference_dir / "config_used.json").read_text(encoding="utf-8"))
    environment = json.loads((reference_dir / "environment.json").read_text(encoding="utf-8"))
    aggregate = pd.read_csv(reference_dir / "aggregate_metrics.csv")
    fold_metrics = pd.read_csv(reference_dir / "fold_metrics.csv")
    bootstrap = pd.read_csv(reference_dir / "bootstrap_model_comparison.csv")
    best = aggregate.sort_values(["qlike", "rmse"], kind="stable").iloc[0]
    ridge = aggregate.loc[aggregate["model"] == "ridge"].iloc[0]
    ewma = aggregate.loc[aggregate["model"] == "ewma_baseline"].iloc[0]
    contract_path = repo_root / "docs" / "MATHEMATICAL_CONTRACT.md"
    generator_path = Path(__file__).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    generated_from = [
        "results/reference_run/config_used.json",
        "results/reference_run/environment.json",
        "results/reference_run/aggregate_metrics.csv",
        "results/reference_run/fold_metrics.csv",
        "results/reference_run/bootstrap_model_comparison.csv",
        "results/reference_run/failure_analysis.md",
        "docs/MATHEMATICAL_CONTRACT.md",
    ]
    claims = {
        "schema_version": "1.0",
        "task": "SPY five-day realised-variance proxy validation",
        "asset": str(config.get("symbol", "SPY")),
        "data_field": "adjusted_close",
        "target_definition": "sum of squared log returns over the next five trading days",
        "target_horizon_days": int(config["target_horizon_days"]),
        "fold_protocol": "purged_expanding_walk_forward",
        "folds": int(fold_metrics["fold"].nunique()),
        "models": [str(value) for value in manifest["models"]],
        "metrics": [name for name in ("rmse", "qlike") if name in aggregate.columns],
        "bootstrap_metrics": sorted(set(bootstrap["metric"].astype(str))),
        "ewma_lambda": float(config["ewma_lambda"]),
        "metric_definitions": {
            "qlike": "actual/prediction - log(actual/prediction) - 1 with a positive floor",
            "rmse": "square root of mean squared forecast error",
        },
        "oof_rows_per_model": {
            str(key): int(value) for key, value in manifest["oof_rows_per_model"].items()
        },
        "primary_observation": (
            f"{best['model']} has the lowest aggregate QLIKE ({best['qlike']:.6f}) "
            "in this reference run; this is descriptive model validation."
        ),
        "negative_result": (
            f"Ridge QLIKE ({ridge['qlike']:.6f}) is materially less stable than the "
            f"non-anticipating EWMA baseline ({ewma['qlike']:.6f}) in the retained reference evidence."
        ),
        "canonical_run_id": str(manifest["canonical_run_id"]),
        "canonical_experiment_code_commit": str(
            manifest.get("canonical_code_commit", environment.get("code_commit", "UNKNOWN"))
        ),
        "claims_generator_sha256": raw_sha256(generator_path),
        "mathematical_contract_sha256": artifact_hash_detail(contract_path)["source_raw_sha256"],
        "mathematical_contract_normalized_sha256": artifact_hash_detail(contract_path)["canonical_sha256"],
        "mathematical_contract_hash_mode": "git-lf-v1",
        "reference_manifest_version": int(manifest["manifest_version"]),
        "generated_from_artifacts": generated_from,
        "provenance_note": (
            "canonical_experiment_code_commit identifies the locked reference run; "
            "claims_generator_sha256 and mathematical_contract hashes identify the "
            "derivation inputs. The working-tree commit is intentionally not embedded."
        ),
        "non_claims": [
            "causal inference",
            "trading strategy",
            "alpha",
            "P&L",
            "profitability",
            "production benchmark",
            "cross-asset generalisation",
            "independent OOF rows",
        ],
    }
    output_path = reference_dir / "MATH_CLAIMS.json"
    output_path.write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8", newline="\n")
    return claims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    claims = build_math_claims(args.repo_root)
    print(json.dumps({"status": "PASS", "fields": len(claims)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
