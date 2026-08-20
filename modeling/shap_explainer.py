"""
Computes SHAP (SHapley Additive exPlanations) values for a single
match's prediction — the mathematical attribution of how much each
feature pushed the predicted probability up or down, relative to the
model's average output.

This is the grounding mechanism for Phase 6: every sentence the
narrative generator produces traces back to a real number here, not an
invented reason. TreeExplainer is used because it's exact and fast for
tree-based models specifically (CatBoost included) — no sampling or
approximation needed, unlike SHAP's model-agnostic explainers.
"""

import numpy as np
import shap
from catboost import CatBoostClassifier

from modeling.dataset import FEATURE_COLUMNS


def compute_shap_contributions(
    model: CatBoostClassifier,
    feature_values: list[float],
    class_index: int,
) -> dict[str, float]:
    """
    Returns {feature_name: shap_value} for ONE match, for ONE specific
    class (e.g. the predicted outcome). A positive value means that
    feature pushed the probability for this class UP; negative means it
    pushed it DOWN. Values are in the model's raw output space, not
    probability percentage points directly — useful for ranking which
    features mattered most, not for reading as "this added exactly X%."
    """
    explainer = shap.TreeExplainer(model)
    shap_output = explainer.shap_values(np.array([feature_values]))

    # CatBoost multiclass SHAP output shape varies by shap library
    # version — handle both known shapes rather than assuming one and
    # failing confusingly on the other.
    if isinstance(shap_output, list):
        # One array per class: shap_output[class_index][0] = per-feature values
        class_shap_values = shap_output[class_index][0]
    elif shap_output.ndim == 3:
        # Single array shaped (n_samples, n_features, n_classes)
        class_shap_values = shap_output[0, :, class_index]
    else:
        raise RuntimeError(
            f"Unexpected SHAP output shape: {np.shape(shap_output)}. "
            f"This likely means the installed shap library version handles "
            f"CatBoost multiclass differently than expected — check "
            f"shap.__version__ and adjust the shape handling above."
        )

    return dict(zip(FEATURE_COLUMNS, class_shap_values))


def top_contributing_features(
    contributions: dict[str, float], n: int = 4
) -> list[tuple[str, float]]:
    """Returns the top N features by ABSOLUTE contribution (biggest
    impact in either direction), sorted strongest-first."""
    return sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]