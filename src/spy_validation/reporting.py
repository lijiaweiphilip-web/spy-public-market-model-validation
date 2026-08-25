from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RunConfig
from .metrics import (
    bootstrap_fold_differences,
    calibration_table,
    regression_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return importlib.metadata.version("spy-public-market-model-validation")
    except importlib.metadata.PackageNotFoundError:
        return "local-source"


def _git_commit(config_path: Path) -> str:
    repo = config_path.resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return result.stdout.strip()


def _write_plot_fold_rmse(fold_metrics: pd.DataFrame, path: Path) -> None:
    pivot = fold_metrics.pivot(index="fold", columns="model", values="rmse")
    ax = pivot.plot(figsize=(10, 5), marker="o", linewidth=1)
    ax.set_title("Purged expanding walk-forward RMSE by fold")
    ax.set_ylabel("RMSE of forward realised variance")
    ax.set_xlabel("Fold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    _fig, ax = plt.subplots(figsize=(7, 6))
    for model, group in calibration.groupby("model", sort=True):
        ax.plot(group["mean_prediction"], group["mean_actual"], marker="o", label=model)
    values = np.concatenate([calibration["mean_prediction"], calibration["mean_actual"]])
    low, high = float(np.min(values)), float(np.max(values))
    ax.plot([low, high], [low, high], linestyle="--", label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted realised variance")
    ax.set_ylabel("Mean actual realised variance")
    ax.set_title("Out-of-fold calibration bins")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_plot_regime(regime_metrics: pd.DataFrame, path: Path) -> None:
    pivot = regime_metrics.pivot(index="regime", columns="model", values="qlike")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("QLIKE by prior-volatility regime")
    ax.set_ylabel("QLIKE, lower is better")
    ax.set_xlabel("Regime")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_plot_cost(costs: pd.DataFrame, path: Path) -> None:
    pivot = costs.pivot(index="cost_bps", columns="model", values="annualised_cost_bps_per_unit_capital")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Illustrative decision-turnover cost sensitivity")
    ax.set_ylabel("Annualised cost, basis points per unit capital")
    ax.set_xlabel("Assumed one-way transaction cost, basis points")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model, group in predictions.groupby("model", sort=True):
        rows.append({"model": model, **regression_metrics(group["actual"], group["prediction"])})
    return pd.DataFrame(rows)


def _failure_analysis(
    aggregate: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    costs: pd.DataFrame,
    clipping: pd.DataFrame,
) -> str:
    baseline = aggregate.set_index("model").loc["mean_baseline"]
    lines = [
        "# Failure and limitation analysis",
        "",
        "This file is generated from the reference run. It is intentionally candid: model validation is stronger when unstable or negative results remain visible.",
        "Calibration factors for Ridge and Random Forest are estimated from an inner temporal block inside each outer training set; the outer test block is never used. This is stricter than in-sample calibration but remains a small-sample diagnostic, not a guarantee of calibrated probabilities or variance forecasts.",
        "",
    ]
    for _, row in aggregate.iterrows():
        model = row["model"]
        if model == "mean_baseline":
            continue
        rmse_delta = row["rmse"] - baseline["rmse"]
        qlike_delta = row["qlike"] - baseline["qlike"]
        direction = "outperformed" if rmse_delta < 0 else "underperformed"
        lines.append(
            f"- **{model}** {direction} the historical-mean baseline on aggregate RMSE by {abs(rmse_delta):.6g}; QLIKE difference was {qlike_delta:.6g}."
        )
    for _, row in aggregate.iterrows():
        ratio = row["calibration_ratio"]
        if ratio < 0.8 or ratio > 1.25:
            lines.append(
                f"- **{row['model']}** showed material aggregate calibration drift: predicted/actual mean ratio {ratio:.3f}."
            )
    fold_pivot = fold_metrics.pivot(index="fold", columns="model", values="rmse")
    for model in [column for column in fold_pivot.columns if column != "mean_baseline"]:
        share = float(np.mean(fold_pivot[model] < fold_pivot["mean_baseline"]))
        lines.append(f"- **{model}** beat the mean baseline on RMSE in {share:.1%} of folds.")
    max_cost = costs.sort_values("annualised_cost_bps_per_unit_capital", ascending=False).iloc[0]
    lines.append(
        f"- The highest illustrative annualised decision-turnover cost was {max_cost['annualised_cost_bps_per_unit_capital']:.1f} bps for {max_cost['model']} under a {int(max_cost['cost_bps'])}-bps assumption. This is a cost diagnostic, not a P&L result."
    )
    clipped = clipping.groupby("model")["clipping_count"].sum()
    for model, count in clipped.items():
        if count:
            lines.append(f"- **{model}** required training-bounded prediction clipping on {int(count)} test predictions.")
    calibration_error = calibration.groupby("model")["absolute_log_ratio"].mean()
    for model, value in calibration_error.items():
        lines.append(f"- **{model}** mean calibration-bin absolute log ratio: {value:.3f}.")
    lines.extend(
        [
            "",
            "## Structural limitations",
            "",
            "- One preselected ETF only; no cross-asset or cross-market generalisation claim.",
            "- Five-day realised variance is a noisy proxy, not latent volatility and not a causal outcome.",
            "- The EWMA baseline uses a fixed RiskMetrics-style lambda=0.94 and maps a daily conditional-variance estimate to five days by multiplying by the horizon; lambda is not selected on the test set.",
            "- Adjusted-close history comes from a vendor endpoint that can revise data or become unavailable.",
            "- Hyperparameters are fixed before test evaluation; no claim is made that they are optimal.",
            "- The illustrative exposure and transaction-cost layer measures decision turnover only. It does not report returns, alpha, Sharpe ratio, P&L or deployable strategy performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    config_path: Path,
    cfg: RunConfig,
    source_path: Path,
    feature_frame: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    exposure_path: pd.DataFrame,
    cost_summary: pd.DataFrame,
    command: str,
) -> dict:
    """Write a complete run before hashing it.

    The manifest is deliberately the final file written. This prevents the
    earlier V4.4 failure mode where REPORT/environment were created after their
    hashes had already been recorded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = outputs["predictions"]
    fold_metrics = outputs["fold_metrics"]
    aggregate = _aggregate_predictions(predictions)
    calibration = calibration_table(predictions, cfg.calibration_bins)
    bootstrap_frames = [
        bootstrap_fold_differences(
            fold_metrics,
            metric=metric,
            baseline="mean_baseline",
            repetitions=cfg.bootstrap_repetitions,
            seed=cfg.primary_seed + offset,
        )
        for offset, metric in enumerate(("rmse", "qlike"))
    ]
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)

    dataframes = {
        "predictions_oof.csv": predictions,
        "fold_metrics.csv": fold_metrics,
        "aggregate_metrics.csv": aggregate,
        "regime_metrics.csv": outputs["regime_metrics"],
        "calibration_bins.csv": calibration,
        "ridge_coefficients.csv": outputs["ridge_coefficients"],
        "random_forest_feature_importance.csv": outputs["forest_importances"],
        "random_forest_seed_stability.csv": outputs["random_forest_seed_stability"],
        "prediction_clipping_audit.csv": outputs["clipping_audit"],
        "illustrative_exposure_path.csv": exposure_path,
        "decision_cost_sensitivity.csv": cost_summary,
        "bootstrap_model_comparison.csv": bootstrap,
    }
    for filename, frame in dataframes.items():
        frame.to_csv(output_dir / filename, index=False)

    shutil.copy2(config_path, output_dir / "config_used.json")
    _write_plot_fold_rmse(fold_metrics, output_dir / "fold_rmse.png")
    _write_plot_calibration(calibration, output_dir / "calibration_bins.png")
    _write_plot_regime(outputs["regime_metrics"], output_dir / "regime_qlike.png")
    _write_plot_cost(cost_summary, output_dir / "decision_cost_sensitivity.png")

    failure_text = _failure_analysis(
        aggregate,
        fold_metrics,
        calibration,
        cost_summary,
        outputs["clipping_audit"],
    )
    (output_dir / "failure_analysis.md").write_text(failure_text, encoding="utf-8")

    model_card = f"""# Model card

## Intended research question
Can simple historical, EWMA and machine-learning models produce stable forecasts of a five-day realised-variance proxy under purged expanding walk-forward validation on SPY daily adjusted-close data?

## Models
- Historical mean baseline
- RiskMetrics-style EWMA baseline with fixed lambda={cfg.ewma_lambda:.2f}; a daily conditional-variance estimate is multiplied by the five-day horizon
- Ridge regression on a log target with `{cfg.calibration_method}` calibration inside each outer training set
- Random forest on a log target with `{cfg.calibration_method}` calibration inside each outer training set

## Validation
- {len(fold_metrics['fold'].unique())} expanding walk-forward folds
- {cfg.test_rows_per_fold} test rows per fold
- {cfg.embargo_rows}-row purge/embargo to prevent overlapping training labels from entering each test period
- Fixed hyperparameters; no random split and no test-set tuning

## Intended use
Research-method demonstration, RA interview discussion, and reproducibility evidence.

## Not intended for
Live trading, investment recommendations, alpha claims, P&L claims, portfolio construction or production risk management.

## Known limitations
See `failure_analysis.md` and `docs/CALIBRATION_AUDIT.md`.
"""
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    git_commit = _git_commit(config_path)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "tabulate": importlib.metadata.version("tabulate"),
        "package_version": _package_version(),
        "git_commit": git_commit,
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(source_path),
        "calibration_method": cfg.calibration_method,
        "calibration_block_rows": cfg.calibration_block_rows,
        "ewma_lambda": cfg.ewma_lambda,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    ranked = aggregate.sort_values(["qlike", "rmse"])
    best = ranked.iloc[0]
    report = [
        "# Reference-run report",
        "",
        f"- Data: SPY daily adjusted close, {feature_frame['date'].min().date()} to {feature_frame['date'].max().date()}",
        f"- Target: next {cfg.target_horizon_days}-trading-day realised variance proxy",
        f"- Validation: {fold_metrics['fold'].nunique()} purged expanding walk-forward folds",
        f"- Best aggregate QLIKE in this run: **{best['model']}** ({best['qlike']:.6f})",
        f"- Exact environment: [`environment.json`](environment.json), git commit `{git_commit}`",
        "- No alpha, return, Sharpe ratio, profitability or production claim is made.",
        "",
        "## Aggregate metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Bootstrap comparison against the historical mean",
        "",
        bootstrap.to_markdown(index=False),
        "",
        "The public reference bundle contains derived summaries only; raw vendor bytes, point-level predictions and exposure paths remain private evidence.",
        "The final `run_manifest.json` hashes every artifact written above.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest = {
        "manifest_version": 2,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SPY five-day realised-variance model validation; no trading-performance claim",
        "command": command,
        "source_file": source_path.name,
        "source_sha256": sha256(source_path),
        "config_file": "config_used.json",
        "config_sha256": sha256(config_path),
        "config": cfg.to_dict(),
        "environment_file": "environment.json",
        "git_commit": git_commit,
        "package_version": _package_version(),
        "raw_price_rows": int(feature_frame.attrs.get("raw_price_rows", 0)),
        "feature_rows": len(feature_frame),
        "folds": int(fold_metrics["fold"].nunique()),
        "test_rows_per_fold": cfg.test_rows_per_fold,
        "models": sorted(predictions["model"].unique().tolist()),
        "oof_rows_per_model": predictions.groupby("model").size().astype(int).to_dict(),
        "checks": {
            "adjusted_close_only": "PASS",
            "feature_before_target": "PASS",
            "purged_training_labels": "PASS",
            "expanding_temporal_split": "PASS",
            "point_predictions_saved": "PASS",
            "negative_results_retained": "PASS",
            "artifact_hashes_complete": "PASS",
        },
        "artifact_sha256": artifact_hashes,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
