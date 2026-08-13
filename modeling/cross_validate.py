"""
Time-series cross-validation across the whole chronologically-sorted
dataset (not season-aligned) — more folds than the 2 season-boundary
splits 3 seasons alone would give, for a more statistically meaningful
read on whether one model's edge over the others is real or just an
artifact of this particular train/test split.

Simplification stated plainly: this uses FIXED, moderate hyperparameters
for the boosted models (informed by the early-stopping rounds discovered
in train_compare.py — roughly the same ballpark, not re-tuned per fold)
rather than running early stopping inside every fold. Nested tuning
within CV is the more rigorous approach in principle, but the added
complexity isn't worth it at this project's scale — this run exists to
answer "is the ranking stable across time," not to extract another
decimal point of accuracy.

Usage:
    python -m modeling.cross_validate
"""

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from database.session import SessionLocal
from modeling.dataset import FEATURE_COLUMNS, load_training_dataframe
from modeling.metrics import CLASSES, assert_class_order, compute_metrics

N_SPLITS = 5
FIXED_ESTIMATORS = 80  # informed by earlier early-stopping results (26-108 range)


def build_models():
    return {
        "Logistic Regression": ("scaled", LogisticRegression(max_iter=1000)),
        "Random Forest": (
            "imputed",
            RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1),
        ),
        "XGBoost": (
            "raw_encoded",
            XGBClassifier(
                n_estimators=FIXED_ESTIMATORS, max_depth=3, learning_rate=0.05,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1,
            ),
        ),
        "LightGBM": (
            "raw",
            LGBMClassifier(
                n_estimators=FIXED_ESTIMATORS, max_depth=3, learning_rate=0.05,
                objective="multiclass", reg_lambda=2.0, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
            ),
        ),
        "CatBoost": (
            "raw",
            CatBoostClassifier(
                iterations=FIXED_ESTIMATORS, depth=3, learning_rate=0.05,
                loss_function="MultiClass", l2_leaf_reg=6.0,
                random_state=42, verbose=False,
            ),
        ),
    }


def evaluate_fold(kind, model, X_train_df, y_train, X_test_df, y_test) -> dict:
    if kind in ("scaled", "imputed"):
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)
        if kind == "scaled":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        assert_class_order(model.classes_, "model")
        return compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test))

    if kind == "raw":
        model.fit(X_train_df, y_train)
        assert_class_order(model.classes_, "model")
        return compute_metrics(y_test, model.predict(X_test_df), model.predict_proba(X_test_df))

    if kind == "raw_encoded":
        y_train_enc = y_train.map({label: i for i, label in enumerate(CLASSES)})
        model.fit(X_train_df, y_train_enc)
        proba = model.predict_proba(X_test_df)
        pred = np.array(CLASSES)[model.predict(X_test_df)]
        return compute_metrics(y_test, pred, proba)

    raise ValueError(kind)


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)  # already chronologically sorted
        X = df[FEATURE_COLUMNS]
        y = df["result"]

        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        models = build_models()
        results_by_model = {name: [] for name in models}

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train_df, X_test_df = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"\nFold {fold_idx}: train={len(train_idx)} matches, test={len(test_idx)} matches")

            for name, (kind, model) in models.items():
                metrics = evaluate_fold(kind, model, X_train_df, y_train, X_test_df, y_test)
                results_by_model[name].append(metrics)
                print(
                    f"  {name:<22} log_loss={metrics['log_loss']:.3f}  "
                    f"brier={metrics['brier_score']:.3f}  acc={metrics['accuracy']:.3f}"
                )

        print(f"\n{'=' * 74}")
        print(f"Summary across {N_SPLITS} folds (mean ± std)")
        print(f"{'=' * 74}")
        print(f"{'Model':<22}{'Log Loss':>20}{'Brier':>18}{'Accuracy':>18}")
        for name, fold_results in results_by_model.items():
            log_losses = [r["log_loss"] for r in fold_results]
            briers = [r["brier_score"] for r in fold_results]
            accs = [r["accuracy"] for r in fold_results]
            print(
                f"{name:<22}"
                f"{np.mean(log_losses):>9.3f} ± {np.std(log_losses):<8.3f}"
                f"{np.mean(briers):>8.3f} ± {np.std(briers):<7.3f}"
                f"{np.mean(accs):>8.3f} ± {np.std(accs):<7.3f}"
            )

    finally:
        session.close()


if __name__ == "__main__":
    main()