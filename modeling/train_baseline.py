"""
Baseline model: Logistic Regression. The floor every other model needs
to beat, and the model whose calibration is easiest to reason about
before moving to less transparent ones.

Usage:
    python -m modeling.train_baseline
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from database.session import SessionLocal
from modeling.dataset import FEATURE_COLUMNS, load_training_dataframe, time_based_split

# Fixed label order everywhere — sklearn's predict_proba column order
# must match this exactly, or every downstream metric silently scrambles.
CLASSES = ["A", "D", "H"]


def multiclass_brier_score(y_true_labels: np.ndarray, y_proba: np.ndarray) -> float:
    """
    sklearn's brier_score_loss is binary-only. The multiclass generalization
    (Brier, 1950) is the mean squared error between the predicted
    probability vector and a one-hot encoding of the true class — this is
    a hand-rolled implementation of that, not a library shortcut, since
    understanding what this number actually measures matters here.
    """
    one_hot = np.zeros((len(y_true_labels), len(CLASSES)))
    for i, label in enumerate(y_true_labels):
        one_hot[i, CLASSES.index(label)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)
        train_df, test_df = time_based_split(df)

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["result"]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["result"]

        # Fit imputer/scaler on TRAIN ONLY, then apply (never re-fit) to
        # test — fitting on the combined or test data would leak
        # information about the test set's distribution back into
        # training, the same category of mistake as a random train/test
        # split, just subtler.
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # model.classes_ is sklearn's own alphabetical ordering (A, D, H
        # for our labels — happens to already match CLASSES here, but
        # asserting it rather than assuming protects against a silent
        # mismatch if labels ever change).
        assert list(model.classes_) == CLASSES, (
            f"Label order mismatch: model has {list(model.classes_)}, expected {CLASSES}"
        )

        y_proba = model.predict_proba(X_test_scaled)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba, labels=CLASSES)
        brier = multiclass_brier_score(y_test.to_numpy(), y_proba)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", labels=CLASSES)

        print("=== Logistic Regression baseline (test = 2025-26 season) ===")
        print(f"Accuracy:        {accuracy:.3f}")
        print(f"Log loss:        {loss:.3f}  (lower is better; random guessing ≈ {np.log(3):.3f})")
        print(f"Brier score:     {brier:.3f}  (lower is better; 0 = perfect, ~0.67 = random 3-class)")
        print(f"ROC-AUC (OvR):   {auc:.3f}  (0.5 = random, 1.0 = perfect)")

        # A cheap but telling sanity check: a model that just always
        # predicts the most common class (Home) sets a floor accuracy
        # needs to clear to be worth anything at all.
        majority_class_accuracy = (y_test == y_test.mode()[0]).mean()
        print(f"\n(For reference, always predicting '{y_test.mode()[0]}' gets "
              f"{majority_class_accuracy:.3f} accuracy — the model should beat this.)")

    finally:
        session.close()


if __name__ == "__main__":
    main()