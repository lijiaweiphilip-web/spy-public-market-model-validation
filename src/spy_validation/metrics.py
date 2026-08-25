from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.maximum(np.asarray(actual, dtype=float), 1e-12)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 1e-12)
    error = actual - prediction
    ratio = actual / prediction
    qlike = ratio - np.log(ratio) - 1.0
    rank_actual = pd.Series(actual).rank(method="average").to_numpy(dtype=float)
    rank_prediction = pd.Series(prediction).rank(method="average").to_numpy(dtype=float)
    if np.ptp(rank_actual) <= 1e-15 or np.ptp(rank_prediction) <= 1e-15:
        spearman = 0.0
    else:
        centered_actual = rank_actual - np.mean(rank_actual)
        centered_prediction = rank_prediction - np.mean(rank_prediction)
        denominator = np.sqrt(np.sum(centered_actual**2) * np.sum(centered_prediction**2))
        spearman = float(np.sum(centered_actual * centered_prediction) / denominator) if denominator > 0 else 0.0
    return {
        "n": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "qlike": float(np.mean(qlike)),
        "calibration_ratio": float(np.mean(prediction) / np.mean(actual)),
        "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
    }


def calibration_table(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for model, group in frame.groupby("model", sort=True):
        ranked = group["prediction"].rank(method="first")
        labels = pd.qcut(ranked, q=min(bins, len(group)), labels=False, duplicates="drop")
        table = (
            group.assign(calibration_bin=labels)
            .groupby("calibration_bin", as_index=False)
            .agg(
                n=("actual", "size"),
                mean_prediction=("prediction", "mean"),
                mean_actual=("actual", "mean"),
            )
        )
        table.insert(0, "model", model)
        table["absolute_log_ratio"] = np.abs(
            np.log((table["mean_prediction"] + 1e-12) / (table["mean_actual"] + 1e-12))
        )
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def bootstrap_fold_differences(
    fold_metrics: pd.DataFrame,
    metric: str,
    baseline: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    pivot = fold_metrics.pivot(index="fold", columns="model", values=metric).dropna()
    if baseline not in pivot.columns:
        raise ValueError(f"Baseline {baseline!r} is absent")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for model in pivot.columns:
        if model == baseline:
            continue
        differences = (pivot[model] - pivot[baseline]).to_numpy(dtype=float)
        samples = np.empty(repetitions, dtype=float)
        for idx in range(repetitions):
            draw = rng.choice(differences, size=len(differences), replace=True)
            samples[idx] = float(np.mean(draw))
        rows.append(
            {
                "metric": metric,
                "model": model,
                "baseline": baseline,
                "mean_difference": float(np.mean(differences)),
                "ci_lower_95": float(np.quantile(samples, 0.025)),
                "ci_upper_95": float(np.quantile(samples, 0.975)),
                "share_folds_better": float(np.mean(differences < 0)),
            }
        )
    return pd.DataFrame(rows)
