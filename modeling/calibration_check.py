"""
Calibration check for the final CatBoost model: for each outcome class
(Home/Draw/Away), does "predicted 60%" actually happen about 60% of the
time across matches where the model said that?

Uses the same tune-on-2023-24 -> validate-on-2024-25 -> refit-on-both
process as train_compare.py, then checks calibration on the held-out
2025-26 test season — never touched during training or tuning.

Limitation stated plainly: 380 test matches split 3 ways is a small
sample for calibration analysis. Using 5 equal-count bins (not the more
common 10) to keep each bin's estimate from being too noisy to trust,
and printing Expected Calibration Error as a numeric summary alongside
the visual, since eyeballing a noisy curve from this few points can be
misleading either direction.

Usage:
    python -m modeling.calibration_check
"""

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve

from database.session import SessionLocal
from modeling.dataset import (
    FEATURE_COLUMNS,
    load_training_dataframe,
    time_based_split,
    time_based_split_three_way,
)
from modeling.metrics import CLASSES

N_BINS = 5
CLASS_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def train_final_catboost(df):
    train_only_df, validation_df, _ = time_based_split_three_way(df)
    full_train_df, test_df = time_based_split(df)

    X_tune = train_only_df[FEATURE_COLUMNS]
    y_tune = train_only_df["result"]
    X_val = validation_df[FEATURE_COLUMNS]
    y_val = validation_df["result"]

    tuner = CatBoostClassifier(
        iterations=500, depth=3, learning_rate=0.05,
        loss_function="MultiClass", l2_leaf_reg=6.0,
        early_stopping_rounds=30, random_state=42, verbose=False,
    )
    tuner.fit(X_tune, y_tune, eval_set=(X_val, y_val))
    best_rounds = tuner.get_best_iteration() + 1
    print(f"Early stopping found {best_rounds} rounds.")

    X_full_train = full_train_df[FEATURE_COLUMNS]
    y_full_train = full_train_df["result"]
    final_model = CatBoostClassifier(
        iterations=best_rounds, depth=3, learning_rate=0.05,
        loss_function="MultiClass", l2_leaf_reg=6.0,
        random_state=42, verbose=False,
    )
    final_model.fit(X_full_train, y_full_train)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["result"]
    return final_model, X_test, y_test


def expected_calibration_error(y_true_binary, y_prob, n_bins) -> float:
    """Weighted average of |predicted - observed| across bins — a single
    number summarizing how far off calibration is, weighted by how many
    predictions fell in each bin."""
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9  # include the max value in the last bin
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        predicted_mean = y_prob[in_bin].mean()
        observed_mean = y_true_binary[in_bin].mean()
        weight = in_bin.sum() / len(y_prob)
        ece += weight * abs(predicted_mean - observed_mean)
    return ece


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)
        model, X_test, y_test = train_final_catboost(df)

        y_proba = model.predict_proba(X_test)
        class_order = list(model.classes_)  # CatBoost's own ordering

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        print(f"\n{'Class':<10}{'ECE':>10}   (lower is better; 0 = perfectly calibrated)")
        print("-" * 35)

        for ax, class_label in zip(axes, CLASSES):
            class_idx = class_order.index(class_label)
            y_true_binary = (y_test == class_label).astype(int).to_numpy()
            y_prob_this_class = y_proba[:, class_idx]

            observed_freq, predicted_mean = calibration_curve(
                y_true_binary, y_prob_this_class, n_bins=N_BINS, strategy="quantile"
            )
            ece = expected_calibration_error(y_true_binary, y_prob_this_class, N_BINS)
            print(f"{CLASS_NAMES[class_label]:<10}{ece:>10.3f}")

            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
            ax.plot(predicted_mean, observed_freq, "o-", label="CatBoost")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"{CLASS_NAMES[class_label]} (ECE={ece:.3f})")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        output_path = "modeling/calibration_plot.png"
        plt.savefig(output_path, dpi=120)
        print(f"\nSaved calibration plot to {output_path} — open it to look at the curves.")

        print(
            "\nHow to read the plot: points ABOVE the diagonal mean the model is "
            "UNDER-confident for that probability range (things happen more often "
            "than predicted). Points BELOW the diagonal mean OVER-confident "
            "(things happen less often than predicted) — over-confidence is the "
            "more dangerous direction for betting decisions, since it means the "
            "model's edge over the bookmaker's price may be smaller than it looks."
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()