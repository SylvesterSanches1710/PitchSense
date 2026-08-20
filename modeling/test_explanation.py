"""
Standalone test: generates a real explanation for one upcoming match.
Run this BEFORE wiring SHAP into the API — if the shap output shape
handling in shap_explainer.py doesn't match your installed shap
version, this is where you'll find out, with a clear error message
pointing at exactly what to check.

Usage:
    python -m modeling.test_explanation <match_id>
"""

import json
import sys
from pathlib import Path

from catboost import CatBoostClassifier

from database.models import Match, MatchFeature, Team
from database.session import SessionLocal
from modeling.dataset import FEATURE_COLUMNS
from modeling.narrative_generator import generate_explanation
from modeling.shap_explainer import compute_shap_contributions, top_contributing_features

MODEL_PATH = Path("modeling/model_registry/catboost_v1.cbm")
METADATA_PATH = Path("modeling/model_registry/catboost_v1_metadata.json")

CLASS_TO_LABEL = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m modeling.test_explanation <match_id>")
        return

    match_id = int(sys.argv[1])

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    class_order = json.loads(METADATA_PATH.read_text())["class_order"]

    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        feature_row = session.query(MatchFeature).filter_by(match_id=match_id).first()
        home_team = session.get(Team, match.home_team_id).name
        away_team = session.get(Team, match.away_team_id).name

        feature_values = [getattr(feature_row, col) for col in FEATURE_COLUMNS]
        proba = model.predict_proba([feature_values])[0]
        model_probs = {label: proba[i] for i, label in enumerate(class_order)}

        predicted_class = max(model_probs, key=model_probs.get)
        predicted_prob = model_probs[predicted_class]
        class_index = class_order.index(predicted_class)

        print(f"{home_team} vs {away_team}")
        print(f"Predicted: {CLASS_TO_LABEL[predicted_class]} ({predicted_prob:.1%})\n")

        print("Computing SHAP contributions...")
        contributions = compute_shap_contributions(model, feature_values, class_index)

        print("\nTop 8 contributing features (raw SHAP values) — this matches the "
              "candidate pool generate_explanation() actually considers, so every "
              "feature cited below should be traceable back to this list:")
        for name, value in top_contributing_features(contributions, n=8):
            print(f"  {name:<30} {value:+.4f}")

        explanation = generate_explanation(
            home_team=home_team, away_team=away_team,
            predicted_outcome_label=CLASS_TO_LABEL[predicted_class],
            predicted_probability=predicted_prob,
            contributions=contributions, feature_row=feature_row,
        )

        print(f"\nGenerated explanation:\n{explanation}")

    finally:
        session.close()


if __name__ == "__main__":
    main()