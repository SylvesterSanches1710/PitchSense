"""
Trains the final CatBoost model (same tune-then-refit process as
calibration_check.py and train_compare.py) and saves it to disk, so
Phase 4 can load a trained model instead of retraining from scratch
every time.

Usage:
    python -m modeling.save_final_model
"""

import json
from pathlib import Path

from catboost import CatBoostClassifier

from database.session import SessionLocal
from modeling.dataset import (
    FEATURE_COLUMNS,
    load_training_dataframe,
    time_based_split,
    time_based_split_three_way,
)

MODEL_DIR = Path("modeling/model_registry")
MODEL_PATH = MODEL_DIR / "catboost_v1.cbm"
METADATA_PATH = MODEL_DIR / "catboost_v1_metadata.json"


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)
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

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(MODEL_PATH))

        # Metadata alongside the model file — without this, six months
        # from now there's no record of which features this model
        # expects, in what order, or what data it was trained on. The
        # model file alone doesn't answer any of that.
        metadata = {
            "model_type": "CatBoostClassifier",
            "feature_columns": FEATURE_COLUMNS,
            "class_order": list(final_model.classes_),
            "trained_on_seasons": sorted(full_train_df["season"].unique().tolist()),
            "iterations": best_rounds,
            "hyperparameters": {
                "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 6.0,
            },
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))

        print(f"Saved model to {MODEL_PATH}")
        print(f"Saved metadata to {METADATA_PATH}")

    finally:
        session.close()


if __name__ == "__main__":
    main()