"""Pydantic response models for model performance data."""

from pydantic import BaseModel


class ModelCVResult(BaseModel):
    model_name: str
    log_loss_mean: float
    log_loss_std: float
    brier_mean: float
    brier_std: float
    accuracy_mean: float
    accuracy_std: float


class ModelPerformance(BaseModel):
    computed_at: str
    test_period: str
    cross_validation_folds: int
    cross_validation: list[ModelCVResult]
    calibration_ece: dict[str, float]
    final_model: str