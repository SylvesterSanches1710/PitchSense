"""
Shared evaluation logic — every model in this project gets scored the
same way, by the same code, so comparisons between them are actually
apples-to-apples rather than subtly different per-script implementations.
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

CLASSES = ["A", "D", "H"]


def multiclass_brier_score(y_true_labels: np.ndarray, y_proba: np.ndarray) -> float:
    """sklearn's brier_score_loss is binary-only. This is the standard
    multiclass generalization: mean squared error between the predicted
    probability vector and a one-hot encoding of the true class."""
    one_hot = np.zeros((len(y_true_labels), len(CLASSES)))
    for i, label in enumerate(y_true_labels):
        one_hot[i, CLASSES.index(label)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def compute_metrics(y_test, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba, labels=CLASSES),
        "brier_score": multiclass_brier_score(y_test.to_numpy(), y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr", labels=CLASSES),
    }


def assert_class_order(model_classes, model_name: str) -> None:
    """Every model here has its own internal class ordering. If it ever
    doesn't match CLASSES, every metric above silently computes against
    the wrong columns — this catches that immediately instead of letting
    it fail silently."""
    assert list(model_classes) == CLASSES, (
        f"{model_name}: label order mismatch — model has {list(model_classes)}, "
        f"expected {CLASSES}"
    )