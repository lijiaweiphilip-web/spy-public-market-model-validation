from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RunConfig
from .features import FEATURE_COLUMNS
from .metrics import regression_metrics
from .models import make_models
from .splits import expanding_purged_folds


def _assign_regime(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    low, high = train["realised_var_20"].quantile([1 / 3, 2 / 3]).tolist()
    return np.select(
        [test["realised_var_20"] <= low, test["realised_var_20"] >= high],
        ["low_prior_vol", "high_prior_vol"],
        default="mid_prior_vol",
    )


def run_walk_forward(frame: pd.DataFrame, cfg: RunConfig) -> dict[str, pd.DataFrame]:
    folds = expanding_purged_folds(
        frame=frame,
        minimum_training_rows=cfg.minimum_training_rows,
        test_rows=cfg.test_rows_per_fold,
        step_rows=cfg.step_rows,
        embargo_rows=cfg.embargo_rows,
    )
    prediction_rows: list[dict] = []
    fold_rows: list[dict] = []
    ridge_coefficient_rows: list[dict] = []
    forest_importance_rows: list[dict] = []
    seed_prediction_store: dict[tuple[int, int], np.ndarray] = {}
    clipping_rows: list[dict] = []

    for fold in folds:
        train = frame.iloc[fold.train_indices]
        test = frame.iloc[fold.test_indices]
        x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train["target_realised_variance"].to_numpy(dtype=float)
        x_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test["target_realised_variance"].to_numpy(dtype=float)
        regimes = _assign_regime(train, test)
        models = make_models(cfg)

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            metrics = regression_metrics(y_test, prediction)
            fold_rows.append(
                {
                    "fold": fold.fold_id,
                    "model": model_name,
                    "train_rows": len(train),
                    "train_target_end": fold.train_end_timestamp.date().isoformat(),
                    "test_start": fold.test_start_timestamp.date().isoformat(),
                    "test_end": fold.test_end_timestamp.date().isoformat(),
                    "calibration_factor": model.calibration_factor,
                    "prediction_clipping_count": model.clipping_count,
                    **metrics,
                }
            )
            clipping_rows.append(
                {
                    "fold": fold.fold_id,
                    "model": model_name,
                    "clipping_count": model.clipping_count,
                    "test_rows": len(test),
                }
            )
            for row_idx, (_, source_row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "fold": fold.fold_id,
                        "model": model_name,
                        "feature_date": source_row["feature_timestamp"].date().isoformat(),
                        "target_start_date": source_row["target_start_timestamp"].date().isoformat(),
                        "target_end_date": source_row["target_end_timestamp"].date().isoformat(),
                        "actual": float(y_test[row_idx]),
                        "prediction": float(prediction[row_idx]),
                        "regime": str(regimes[row_idx]),
                    }
                )

            if model_name == "ridge" and model.estimator is not None:
                coefficients = model.estimator.named_steps["ridge"].coef_
                for feature, coefficient in zip(FEATURE_COLUMNS, coefficients):
                    ridge_coefficient_rows.append(
                        {
                            "fold": fold.fold_id,
                            "feature": feature,
                            "standardised_coefficient": float(coefficient),
                        }
                    )
            if model_name == "random_forest" and model.estimator is not None:
                importances = model.estimator.feature_importances_
                for feature, importance in zip(FEATURE_COLUMNS, importances):
                    forest_importance_rows.append(
                        {
                            "fold": fold.fold_id,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )

        for seed in cfg.stability_seeds:
            stable_model = make_models(cfg, seed=seed)["random_forest"]
            stable_model.fit(x_train, y_train)
            seed_prediction_store[(fold.fold_id, int(seed))] = stable_model.predict(x_test)

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    ridge_coefficients = pd.DataFrame(ridge_coefficient_rows)
    forest_importances = pd.DataFrame(forest_importance_rows)
    clipping = pd.DataFrame(clipping_rows)

    regime_rows: list[dict] = []
    for (model, regime), group in predictions.groupby(["model", "regime"], sort=True):
        regime_rows.append(
            {"model": model, "regime": regime, **regression_metrics(group["actual"], group["prediction"])}
        )
    regime_metrics = pd.DataFrame(regime_rows)

    stability_rows: list[dict] = []
    for fold in folds:
        seed_values = [seed_prediction_store[(fold.fold_id, seed)] for seed in cfg.stability_seeds]
        correlations: list[float] = []
        for left in range(len(seed_values)):
            for right in range(left + 1, len(seed_values)):
                left_values = seed_values[left]
                right_values = seed_values[right]
                left_constant = np.ptp(left_values) <= 1e-15
                right_constant = np.ptp(right_values) <= 1e-15
                if left_constant or right_constant:
                    correlation = 1.0 if np.allclose(left_values, right_values) else 0.0
                else:
                    left_centered = left_values - np.mean(left_values)
                    right_centered = right_values - np.mean(right_values)
                    denominator = np.sqrt(
                        np.sum(left_centered**2) * np.sum(right_centered**2)
                    )
                    correlation = (
                        float(np.sum(left_centered * right_centered) / denominator)
                        if denominator > 1e-30
                        else 0.0
                    )
                correlations.append(correlation)
        stacked = np.vstack(seed_values)
        stability_rows.append(
            {
                "fold": fold.fold_id,
                "seed_count": len(seed_values),
                "mean_pairwise_prediction_correlation": float(np.nanmean(correlations)),
                "mean_pointwise_prediction_std": float(np.mean(np.std(stacked, axis=0))),
                "max_pointwise_prediction_std": float(np.max(np.std(stacked, axis=0))),
            }
        )
    stability = pd.DataFrame(stability_rows)

    return {
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "regime_metrics": regime_metrics,
        "ridge_coefficients": ridge_coefficients,
        "forest_importances": forest_importances,
        "random_forest_seed_stability": stability,
        "clipping_audit": clipping,
    }


def add_decision_cost_diagnostics(
    predictions: pd.DataFrame, cfg: RunConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    horizon = cfg.target_horizon_days
    for model, group in predictions.groupby("model", sort=True):
        ordered = group.sort_values("feature_date").copy()
        daily_variance = np.maximum(ordered["prediction"].to_numpy(dtype=float) / horizon, 1e-12)
        annualised_volatility = np.sqrt(daily_variance * 252.0)
        exposure = np.clip(
            cfg.target_annualised_volatility / annualised_volatility,
            0.0,
            cfg.max_exposure,
        )
        turnover = np.abs(np.diff(exposure, prepend=exposure[0]))
        ordered["predicted_annualised_volatility"] = annualised_volatility
        ordered["illustrative_exposure"] = exposure
        ordered["turnover"] = turnover
        exposure_rows.append(ordered)
        years = max(len(ordered) / 252.0, 1e-12)
        for basis_points in cfg.transaction_cost_bps:
            cost_fraction = turnover * basis_points / 10_000.0
            summary_rows.append(
                {
                    "model": model,
                    "cost_bps": basis_points,
                    "observations": len(ordered),
                    "total_turnover": float(np.sum(turnover)),
                    "mean_daily_turnover": float(np.mean(turnover)),
                    "mean_exposure": float(np.mean(exposure)),
                    "total_cost_fraction_per_unit_capital": float(np.sum(cost_fraction)),
                    "annualised_cost_bps_per_unit_capital": float(np.sum(cost_fraction) / years * 10_000.0),
                }
            )
    return pd.concat(exposure_rows, ignore_index=True), pd.DataFrame(summary_rows)
