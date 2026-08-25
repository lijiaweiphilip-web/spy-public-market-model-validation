"""Extract the public-safe reference bundle from one canonical private run.

The script is intentionally allowlist-first. It never copies point-level
predictions, exposure paths, raw vendor bytes, or the private run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PUBLIC_FILES = (
    "aggregate_metrics.csv",
    "bootstrap_model_comparison.csv",
    "calibration_bins.csv",
    "calibration_bins.png",
    "config_used.json",
    "decision_cost_sensitivity.csv",
    "decision_cost_sensitivity.png",
    "environment.json",
    "failure_analysis.md",
    "fold_metrics.csv",
    "fold_rmse.png",
    "model_card.md",
    "prediction_clipping_audit.csv",
    "random_forest_feature_importance.csv",
    "random_forest_seed_stability.csv",
    "regime_metrics.csv",
    "regime_qlike.png",
    "REPORT.md",
    "ridge_coefficients.csv",
)
PRIVATE_OR_UNSAFE = {
    "predictions_oof.csv",
    "illustrative_exposure_path.csv",
    "run_manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_overview(output_dir: Path) -> None:
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    fold = pd.read_csv(output_dir / "fold_metrics.csv")
    regime = pd.read_csv(output_dir / "regime_metrics.csv")
    labels = {
        "mean_baseline": "Mean",
        "ewma_baseline": "EWMA",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
    }
    order = [model for model in labels if model in set(aggregate["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    left = aggregate.set_index("model").loc[order].reset_index()
    axes[0].bar([labels[m] for m in left["model"]], left["qlike"], color="#315a7d")
    axes[0].set_title("Aggregate QLIKE (lower is better)")
    axes[0].set_ylabel("QLIKE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    for model in order:
        series = fold[fold["model"] == model].sort_values("fold")
        axes[1].plot(series["fold"], series["qlike"], marker=".", linewidth=1, label=labels[model])
    axes[1].set_title("Fold QLIKE and volatility-regime context")
    axes[1].set_xlabel("Purged expanding fold")
    axes[1].set_ylabel("QLIKE")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    # Add a compact regime note without hiding any folds or models.
    regime_counts = regime.groupby("regime")["model"].nunique().to_dict()
    axes[1].text(
        0.01,
        0.01,
        "Regime rows: " + ", ".join(f"{k}={v} models" for k, v in sorted(regime_counts.items())),
        transform=axes[1].transAxes,
        fontsize=7,
        color="#555555",
    )
    assets_dir = output_dir.parent.parent / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets_dir / "overview.png", dpi=180)
    plt.close(fig)


def _refresh_readme(repo_root: Path, aggregate: pd.DataFrame) -> None:
    readme_path = repo_root / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    rows = {
        "mean_baseline": "historical mean",
        "ewma_baseline": "EWMA (lambda=0.94)",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
    }
    lines = [
        "<!-- BEGIN CANONICAL_RESULTS -->",
        "| Model | RMSE | QLIKE | Calibration ratio |",
        "|---|---:|---:|---:|",
    ]
    for model in ("mean_baseline", "ewma_baseline", "ridge", "random_forest"):
        row = aggregate.loc[aggregate["model"] == model].iloc[0]
        lines.append(
            f"| {rows[model]} | {row['rmse']:.6f} | {row['qlike']:.6f} | {row['calibration_ratio']:.6f} |"
        )
    lines.append("<!-- END CANONICAL_RESULTS -->")
    replacement = "\n".join(lines)
    pattern = re.compile(
        r"<!-- BEGIN CANONICAL_RESULTS -->.*?<!-- END CANONICAL_RESULTS -->|\| Model \| RMSE \| QLIKE \| Calibration ratio \|.*?(?=\n\nThe canonical table)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError("README canonical-results block was not found")
    text = pattern.sub(replacement, text, count=1)
    readme_path.write_text(text, encoding="utf-8")


def refresh(canonical_run: Path, repo_root: Path) -> dict:
    canonical_run = canonical_run.resolve()
    repo_root = repo_root.resolve()
    if not (canonical_run / "run_manifest.json").is_file():
        raise FileNotFoundError(canonical_run / "run_manifest.json")
    manifest = json.loads((canonical_run / "run_manifest.json").read_text(encoding="utf-8"))
    output_dir = repo_root / "results" / "reference_run"
    _reset_directory(output_dir)
    for name in PUBLIC_FILES:
        source = canonical_run / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)
    _write_overview(output_dir)
    aggregate = pd.read_csv(output_dir / "aggregate_metrics.csv")
    _refresh_readme(repo_root, aggregate)

    # Byte-level comparison for copied derived evidence is the publication gate.
    for name in PUBLIC_FILES:
        if (canonical_run / name).read_bytes() != (output_dir / name).read_bytes():
            raise AssertionError(f"Public artifact differs from canonical run: {name}")
    public_names = {p.name for p in output_dir.iterdir() if p.is_file()}
    if "predictions_oof.csv" in public_names or "illustrative_exposure_path.csv" in public_names:
        raise AssertionError("Private point-level or exposure artifact entered public bundle")
    if PRIVATE_OR_UNSAFE & public_names:
        raise AssertionError(f"Unsafe artifacts entered public bundle: {sorted(PRIVATE_OR_UNSAFE & public_names)}")

    environment = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    public_manifest = {
        "manifest_version": 2,
        "scope": "SPY five-day realised-variance model validation; no trading-performance claim",
        "canonical_run_id": manifest["run_id"],
        "canonical_git_commit": manifest.get("git_commit", "UNKNOWN"),
        "source_sha256": manifest["source_sha256"],
        "config_sha256": manifest["config_sha256"],
        "environment": environment,
        "folds": manifest["folds"],
        "models": manifest["models"],
        "oof_rows_per_model": manifest["oof_rows_per_model"],
        "checks": manifest["checks"],
        "public_note": "Derived summaries only; raw vendor bytes, point predictions, exposure paths and private run manifest are excluded.",
        "public_artifact_sha256": {},
    }
    manifest_path = output_dir / "PUBLIC_REFERENCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    public_manifest["public_artifact_sha256"] = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != manifest_path.name
    }
    manifest_path.write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    return public_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-run", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = refresh(args.canonical_run, args.repo_root)
    print(json.dumps({"status": "PASS", "canonical_run_id": result["canonical_run_id"], "public_files": len(result["public_artifact_sha256"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
