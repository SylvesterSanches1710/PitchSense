"""
Trains Random Forest, XGBoost, LightGBM, and CatBoost, evaluates all four
(plus a re-run Logistic Regression baseline) on the same held-out 2025-26
season, and prints a comparison table.

XGBoost/LightGBM/CatBoost use a two-phase process to guard against
overfitting on a small dataset (~760 training rows is genuinely small
for gradient boosting):
  Phase 1: train on 2023-24 only, validate on 2024-25, with early
           stopping to find how many boosting rounds is actually useful
           before the model starts memorizing training noise.
  Phase 2: retrain from scratch on 2023-24+2024-25 COMBINED, using the
           round count discovered in phase 1 (not early stopping again
           — we already know the answer from phase 1's validation data).
The 2025-26 test season is never touched until final evaluation, in
either phase — using it for early-stopping decisions would be the same
category of leakage as a random train/test split, just one step removed.

Usage:
    python -m modeling.train_compare
"""

import time

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from database.session import SessionLocal
from modeling.dataset import (
    FEATURE_COLUMNS,
    load_training_dataframe,
    time_based_split,
    time_based_split_three_way,
)
from modeling.metrics import CLASSES, assert_class_order, compute_metrics

EARLY_STOPPING_ROUNDS = 30
MAX_ESTIMATORS = 500  # ceiling — early stopping should halt well before this


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test) -> dict:
    start = time.monotonic()
    model.fit(X_train, y_train)
    train_seconds = time.monotonic() - start

    assert_class_order(model.classes_, name)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["model"] = name
    metrics["train_seconds"] = train_seconds
    return metrics


def tune_then_refit_xgboost(train_only_df, validation_df, full_train_df, X_test, y_test) -> dict:
    X_tune = train_only_df[FEATURE_COLUMNS]
    y_tune = train_only_df["result"].map({label: i for i, label in enumerate(CLASSES)})
    X_val = validation_df[FEATURE_COLUMNS]
    y_val = validation_df["result"].map({label: i for i, label in enumerate(CLASSES)})

    tuner = XGBClassifier(
        n_estimators=MAX_ESTIMATORS, max_depth=3, learning_rate=0.05,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=42, n_jobs=-1,
    )
    tuner.fit(X_tune, y_tune, eval_set=[(X_val, y_val)], verbose=False)
    best_rounds = tuner.best_iteration + 1
    print(f"  XGBoost: early stopping found {best_rounds} rounds (of {MAX_ESTIMATORS} max)")

    X_full_train = full_train_df[FEATURE_COLUMNS]
    y_full_train = full_train_df["result"].map({label: i for i, label in enumerate(CLASSES)})

    final_model = XGBClassifier(
        n_estimators=best_rounds, max_depth=3, learning_rate=0.05,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    start = time.monotonic()
    final_model.fit(X_full_train, y_full_train)
    train_seconds = time.monotonic() - start

    proba = final_model.predict_proba(X_test)
    pred_encoded = final_model.predict(X_test)
    pred = np.array(CLASSES)[pred_encoded]
    metrics = compute_metrics(y_test, pred, proba)
    metrics["model"] = "XGBoost"
    metrics["train_seconds"] = train_seconds
    return metrics


def tune_then_refit_lightgbm(train_only_df, validation_df, full_train_df, X_test, y_test) -> dict:
    import lightgbm as lgb

    X_tune = train_only_df[FEATURE_COLUMNS]
    y_tune = train_only_df["result"]
    X_val = validation_df[FEATURE_COLUMNS]
    y_val = validation_df["result"]

    tuner = LGBMClassifier(
        n_estimators=MAX_ESTIMATORS, max_depth=3, learning_rate=0.05,
        objective="multiclass", reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    tuner.fit(
        X_tune, y_tune, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_rounds = tuner.best_iteration_
    print(f"  LightGBM: early stopping found {best_rounds} rounds (of {MAX_ESTIMATORS} max)")

    X_full_train = full_train_df[FEATURE_COLUMNS]
    y_full_train = full_train_df["result"]

    final_model = LGBMClassifier(
        n_estimators=best_rounds, max_depth=3, learning_rate=0.05,
        objective="multiclass", reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    return train_and_evaluate("LightGBM", final_model, X_full_train, y_full_train, X_test, y_test)


def tune_then_refit_catboost(train_only_df, validation_df, full_train_df, X_test, y_test) -> dict:
    X_tune = train_only_df[FEATURE_COLUMNS]
    y_tune = train_only_df["result"]
    X_val = validation_df[FEATURE_COLUMNS]
    y_val = validation_df["result"]

    tuner = CatBoostClassifier(
        iterations=MAX_ESTIMATORS, depth=3, learning_rate=0.05,
        loss_function="MultiClass", l2_leaf_reg=6.0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=42, verbose=False,
    )
    tuner.fit(X_tune, y_tune, eval_set=(X_val, y_val))
    best_rounds = tuner.get_best_iteration() + 1
    print(f"  CatBoost: early stopping found {best_rounds} rounds (of {MAX_ESTIMATORS} max)")

    X_full_train = full_train_df[FEATURE_COLUMNS]
    y_full_train = full_train_df["result"]

    final_model = CatBoostClassifier(
        iterations=best_rounds, depth=3, learning_rate=0.05,
        loss_function="MultiClass", l2_leaf_reg=6.0, random_state=42, verbose=False,
    )
    return train_and_evaluate("CatBoost", final_model, X_full_train, y_full_train, X_test, y_test)


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)
        train_df, test_df = time_based_split(df)
        train_only_df, validation_df, _ = time_based_split_three_way(df)

        X_train_raw = train_df[FEATURE_COLUMNS]
        y_train = train_df["result"]
        X_test_raw = test_df[FEATURE_COLUMNS]
        y_test = test_df["result"]

        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_test_imputed = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        results = []

        results.append(
            train_and_evaluate(
                "Logistic Regression",
                LogisticRegression(max_iter=1000),
                X_train_scaled, y_train, X_test_scaled, y_test,
            )
        )

        results.append(
            train_and_evaluate(
                "Random Forest",
                RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1),
                X_train_imputed, y_train, X_test_imputed, y_test,
            )
        )

        print("Tuning boosted models with early stopping (train=2023-24, validate=2024-25)...")
        results.append(tune_then_refit_xgboost(train_only_df, validation_df, train_df, X_test_raw, y_test))
        results.append(tune_then_refit_lightgbm(train_only_df, validation_df, train_df, X_test_raw, y_test))
        results.append(tune_then_refit_catboost(train_only_df, validation_df, train_df, X_test_raw, y_test))

        print(f"\n{'Model':<22}{'Accuracy':>10}{'Log Loss':>12}{'Brier':>10}{'ROC-AUC':>10}{'Time (s)':>10}")
        print("-" * 74)
        for r in sorted(results, key=lambda r: r["log_loss"]):
            print(
                f"{r['model']:<22}{r['accuracy']:>10.3f}{r['log_loss']:>12.3f}"
                f"{r['brier_score']:>10.3f}{r['roc_auc']:>10.3f}{r['train_seconds']:>10.2f}"
            )

        print(f"\n(Sorted by log loss, ascending — best-calibrated model first.)")
        print(f"For reference: random guessing scores log loss ≈ {np.log(3):.3f}, "
              f"Brier ≈ 0.667, ROC-AUC = 0.5")

    finally:
        session.close()


if __name__ == "__main__":
    main()