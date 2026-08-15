"""GET /model/performance — reads the persisted metrics file (see
modeling/record_performance.py) rather than recomputing anything live."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..schemas.performance import ModelCVResult, ModelPerformance

router = APIRouter()

METRICS_PATH = Path("modeling/model_registry/performance_metrics.json")


@router.get("/performance", response_model=ModelPerformance)
def get_model_performance():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No performance metrics recorded yet. Run "
                   "'python -m modeling.record_performance' first.",
        )

    data = json.loads(METRICS_PATH.read_text())

    cv_results = [
        ModelCVResult(model_name=name, **metrics)
        for name, metrics in data["cross_validation"].items()
    ]
    cv_results.sort(key=lambda r: r.log_loss_mean)

    return ModelPerformance(
        computed_at=data["computed_at"],
        test_period=data["test_period"],
        cross_validation_folds=data["cross_validation_folds"],
        cross_validation=cv_results,
        calibration_ece=data["calibration_ece"],
        final_model=data["final_model"],
    )