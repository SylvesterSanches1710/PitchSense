"""
Computes and PERSISTS model performance metrics — cross-validation
comparison across all 5 models, and calibration (ECE) for the final
CatBoost model. Run this once after retraining/reevaluating, not on
every dashboard page load; the API just reads the saved JSON.

Reuses the actual evaluation logic from modeling.cross_validate and
modeling.calibration_check rather than reimplementing it — same
principle as betting_math.py, one source of truth for how a number
is computed.

Usage:
    python -m modeling.record_performance
"""

import datetime
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from database.session import SessionLocal
from modeling.calibration_check import (
    CLASS_NAMES, N_BINS, expected_calibration_error, train_final_catboost,
)
from modeling.cross_validate import N_SPLITS, build_models, evaluate_fold
from modeling.dataset import FEATURE_COLUMNS, load_training_dataframe
from modeling.metrics import CLASSES

OUTPUT_PATH = Path("modeling/model_registry/performance_metrics.json")


def compute_cross_validation_summary(df) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df["result"]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    models = build_models()
    results_by_model = {name: [] for name in models}

    for train_idx, test_idx in tscv.split(X):
        X_train_df, X_test_df = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        for name, (kind, model) in models.items():
            metrics = evaluate_fold(kind, model, X_train_df, y_train, X_test_df, y_test)
            results_by_model[name].append(metrics)

    summary = {}
    for name, fold_results in results_by_model.items():
        log_losses = [r["log_loss"] for r in fold_results]
        briers = [r["brier_score"] for r in fold_results]
        accs = [r["accuracy"] for r in fold_results]
        summary[name] = {
            "log_loss_mean": float(np.mean(log_losses)), "log_loss_std": float(np.std(log_losses)),
            "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        }
    return summary


def compute_calibration_summary(df) -> dict:
    model, X_test, y_test = train_final_catboost(df)
    y_proba = model.predict_proba(X_test)
    class_order = list(model.classes_)

    summary = {}
    for class_label in CLASSES:
        class_idx = class_order.index(class_label)
        y_true_binary = (y_test == class_label).astype(int).to_numpy()
        y_prob_this_class = y_proba[:, class_idx]
        ece = expected_calibration_error(y_true_binary, y_prob_this_class, N_BINS)
        summary[CLASS_NAMES[class_label]] = round(ece, 4)
    return summary


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)

        print("Running cross-validation across all 5 models (takes a minute or two)...")
        cv_summary = compute_cross_validation_summary(df)

        print("Computing calibration for final CatBoost model...")
        calibration_summary = compute_calibration_summary(df)

        output = {
            "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "test_period": "2025-2026 season (held out)",
            "cross_validation_folds": N_SPLITS,
            "cross_validation": cv_summary,
            "calibration_ece": calibration_summary,
            "final_model": "CatBoost",
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(output, indent=2))
        print(f"\nSaved performance metrics to {OUTPUT_PATH}")

    finally:
        session.close()


if __name__ == "__main__":
    main()