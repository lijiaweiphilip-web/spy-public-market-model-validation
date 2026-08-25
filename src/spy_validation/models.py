from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RunConfig


@dataclass
class PositivePredictionModel:
    name: str
    estimator: RegressorMixin | None
    prediction_floor: float
    upper_clip_quantile: float
    upper_clip_multiplier: float
    calibration_factor: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    clipping_count: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> PositivePredictionModel:
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("Target must be non-negative")
        self.lower_bound = max(
            self.prediction_floor,
            float(np.quantile(y, 0.005)) * 0.25,
        )
        self.upper_bound = max(
            self.prediction_floor,
            float(np.quantile(y, self.upper_clip_quantile)) * self.upper_clip_multiplier,
        )
        if self.estimator is None:
            self.calibration_factor = 1.0
            self._mean = float(np.mean(y))
            return self
        log_y = np.log(y + self.prediction_floor)
        self.estimator.fit(x, log_y)
        train_raw = np.exp(self.estimator.predict(x)) - self.prediction_floor
        train_raw = np.clip(train_raw, self.prediction_floor, self.upper_bound)
        denominator = max(float(np.mean(train_raw)), self.prediction_floor)
        self.calibration_factor = float(np.clip(np.mean(y) / denominator, 0.25, 4.0))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            pred = np.full(len(x), self._mean, dtype=float)
        else:
            pred = (np.exp(self.estimator.predict(x)) - self.prediction_floor) * self.calibration_factor
        before = pred.copy()
        pred = np.clip(pred, self.lower_bound, self.upper_bound)
        self.clipping_count += int(np.sum(before != pred))
        return pred


def make_models(cfg: RunConfig, seed: int | None = None) -> dict[str, PositivePredictionModel]:
    random_seed = cfg.primary_seed if seed is None else int(seed)
    return {
        "mean_baseline": PositivePredictionModel(
            name="mean_baseline",
            estimator=None,
            prediction_floor=cfg.prediction_floor,
            upper_clip_quantile=cfg.upper_clip_quantile,
            upper_clip_multiplier=cfg.upper_clip_multiplier,
        ),
        "ridge": PositivePredictionModel(
            name="ridge",
            estimator=Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=cfg.ridge_alpha)),
                ]
            ),
            prediction_floor=cfg.prediction_floor,
            upper_clip_quantile=cfg.upper_clip_quantile,
            upper_clip_multiplier=cfg.upper_clip_multiplier,
        ),
        "random_forest": PositivePredictionModel(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=cfg.forest_estimators,
                max_depth=cfg.forest_max_depth,
                min_samples_leaf=cfg.forest_min_samples_leaf,
                random_state=random_seed,
                n_jobs=-1,
            ),
            prediction_floor=cfg.prediction_floor,
            upper_clip_quantile=cfg.upper_clip_quantile,
            upper_clip_multiplier=cfg.upper_clip_multiplier,
        ),
    }
