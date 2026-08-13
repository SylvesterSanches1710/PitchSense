"""
Trains Random Forest, XGBoost, LightGBM, and CatBoost, evaluates all four
(plus a re-run Logistic Regression baseline) on the same held-out 2025-26
season, and prints a comparison table.

All models train in well under a minute on this dataset size (~760 rows)
even on a CPU-only laptop — this is nowhere near "big data," so don't
expect (or need) long training times to feel like real progress.

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
from modeling.dataset import FEATURE_COLUMNS, load_training_dataframe, time_based_split
from modeling.metrics import CLASSES, assert_class_order, compute_metrics


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


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)
        train_df, test_df = time_based_split(df)

        X_train_raw = train_df[FEATURE_COLUMNS]
        y_train = train_df["result"]
        X_test_raw = test_df[FEATURE_COLUMNS]
        y_test = test_df["result"]

        # Imputed + scaled version — needed for Logistic Regression and
        # Random Forest, neither of which handles NaN natively.
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
                RandomForestClassifier(
                    n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
                ),
                # Random Forest doesn't need scaling (tree splits are
                # scale-invariant) but does need the imputed version.
                X_train_imputed, y_train, X_test_imputed, y_test,
            )
        )

        # From here on: raw features WITH NaN, no imputation — these
        # three handle missing values internally.
        y_train_encoded = y_train.map({label: i for i, label in enumerate(CLASSES)})
        y_test_for_xgb = y_test  # XGBoost's predict_proba still needs decoding below

        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )
        start = time.monotonic()
        xgb_model.fit(X_train_raw, y_train_encoded)
        xgb_train_seconds = time.monotonic() - start
        # XGBoost's sklearn wrapper doesn't expose string classes_ the
        # same way — we encoded manually above in CLASSES order, so the
        # predict_proba column order is already guaranteed correct.
        xgb_proba = xgb_model.predict_proba(X_test_raw)
        xgb_pred_encoded = xgb_model.predict(X_test_raw)
        xgb_pred = np.array(CLASSES)[xgb_pred_encoded]
        xgb_metrics = compute_metrics(y_test_for_xgb, xgb_pred, xgb_proba)
        xgb_metrics["model"] = "XGBoost"
        xgb_metrics["train_seconds"] = xgb_train_seconds
        results.append(xgb_metrics)

        results.append(
            train_and_evaluate(
                "LightGBM",
                LGBMClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    objective="multiclass", num_class=3,
                    random_state=42, n_jobs=-1, verbose=-1,
                ),
                X_train_raw, y_train, X_test_raw, y_test,
            )
        )

        results.append(
            train_and_evaluate(
                "CatBoost",
                CatBoostClassifier(
                    iterations=300, depth=4, learning_rate=0.05,
                    loss_function="MultiClass", random_state=42, verbose=False,
                ),
                X_train_raw, y_train, X_test_raw, y_test,
            )
        )

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