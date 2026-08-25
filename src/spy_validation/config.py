from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    symbol: str
    data_range: str
    interval: str
    target_horizon_days: int
    minimum_training_rows: int
    test_rows_per_fold: int
    step_rows: int
    embargo_rows: int
    ridge_alpha: float
    forest_estimators: int
    forest_max_depth: int
    forest_min_samples_leaf: int
    primary_seed: int
    stability_seeds: tuple[int, ...]
    prediction_floor: float
    upper_clip_quantile: float
    upper_clip_multiplier: float
    calibration_bins: int
    bootstrap_repetitions: int
    target_annualised_volatility: float
    max_exposure: float
    transaction_cost_bps: tuple[int, ...]
    ewma_lambda: float
    calibration_method: str
    calibration_block_rows: int
    output_dir: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        return cls(
            symbol=str(payload["symbol"]),
            data_range=str(payload["data_range"]),
            interval=str(payload["interval"]),
            target_horizon_days=int(payload["target_horizon_days"]),
            minimum_training_rows=int(payload["minimum_training_rows"]),
            test_rows_per_fold=int(payload["test_rows_per_fold"]),
            step_rows=int(payload["step_rows"]),
            embargo_rows=int(payload["embargo_rows"]),
            ridge_alpha=float(payload["ridge_alpha"]),
            forest_estimators=int(payload["forest_estimators"]),
            forest_max_depth=int(payload["forest_max_depth"]),
            forest_min_samples_leaf=int(payload["forest_min_samples_leaf"]),
            primary_seed=int(payload["primary_seed"]),
            stability_seeds=tuple(int(v) for v in payload["stability_seeds"]),
            prediction_floor=float(payload["prediction_floor"]),
            upper_clip_quantile=float(payload["upper_clip_quantile"]),
            upper_clip_multiplier=float(payload["upper_clip_multiplier"]),
            calibration_bins=int(payload["calibration_bins"]),
            bootstrap_repetitions=int(payload["bootstrap_repetitions"]),
            target_annualised_volatility=float(payload["target_annualised_volatility"]),
            max_exposure=float(payload["max_exposure"]),
            transaction_cost_bps=tuple(int(v) for v in payload["transaction_cost_bps"]),
            ewma_lambda=float(payload.get("ewma_lambda", 0.94)),
            calibration_method=str(payload.get("calibration_method", "inner_temporal_block")),
            calibration_block_rows=int(payload.get("calibration_block_rows", 120)),
            output_dir=str(payload["output_dir"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.from_dict(payload)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.target_horizon_days < 1:
            raise ValueError("target_horizon_days must be positive")
        if self.embargo_rows < self.target_horizon_days:
            raise ValueError("embargo_rows must be at least the target horizon")
        if self.minimum_training_rows < 100:
            raise ValueError("minimum_training_rows is unexpectedly small")
        if self.test_rows_per_fold < 1 or self.step_rows < 1:
            raise ValueError("test and step rows must be positive")
        if not 0.9 < self.upper_clip_quantile < 1.0:
            raise ValueError("upper_clip_quantile must be between 0.9 and 1")
        if self.max_exposure <= 0:
            raise ValueError("max_exposure must be positive")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be between 0 and 1")
        if self.calibration_method not in {"inner_temporal_block", "in_sample"}:
            raise ValueError("calibration_method must be inner_temporal_block or in_sample")
        if self.calibration_block_rows < self.target_horizon_days + 1:
            raise ValueError("calibration_block_rows must exceed the target horizon")

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["stability_seeds"] = list(self.stability_seeds)
        data["transaction_cost_bps"] = list(self.transaction_cost_bps)
        return data
