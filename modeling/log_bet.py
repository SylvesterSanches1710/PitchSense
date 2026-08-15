"""
Logs a bet YOU actually placed on Stake. Pulls the model's probability
and EV automatically from a fresh prediction (so you don't retype
numbers by hand and risk a transcription error) — you only need to
supply what only you know: the outcome you bet, your actual stake, and
the actual odds Stake gave you (which may have moved slightly since the
last time odds were fetched into the Odds table).

Usage:
    python -m modeling.log_bet <match_id> <outcome:H|D|A> <stake> <odds_taken> [--notes "..."]

Example:
    python -m modeling.log_bet 1146 A 10 16.00 --notes "Coventry upset — logging despite low-data warning, small stake only"
"""

import argparse
import datetime
import json
from pathlib import Path

from catboost import CatBoostClassifier

from database.models import Bet, Match
from database.session import SessionLocal
from modeling.dataset import FEATURE_COLUMNS
from database.models import MatchFeature

MODEL_PATH = Path("modeling/model_registry/catboost_v1.cbm")
METADATA_PATH = Path("modeling/model_registry/catboost_v1_metadata.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("match_id", type=int)
    parser.add_argument("outcome", choices=["H", "D", "A"])
    parser.add_argument("stake", type=float)
    parser.add_argument("odds_taken", type=float)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()

    if args.odds_taken <= 1.0:
        print(f"odds_taken={args.odds_taken} looks wrong — decimal odds should be > 1.0. Aborting.")
        return

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    class_order = json.loads(METADATA_PATH.read_text())["class_order"]

    session = SessionLocal()
    try:
        match = session.get(Match, args.match_id)
        if match is None:
            print(f"No match with id {args.match_id}.")
            return

        feature_row = session.query(MatchFeature).filter_by(match_id=match.id).first()
        if feature_row is None:
            print(f"No features computed for match {args.match_id} — run build_live_features first.")
            return

        feature_values = [[getattr(feature_row, col) for col in FEATURE_COLUMNS]]
        proba = model.predict_proba(feature_values)[0]
        model_prob = {label: proba[i] for i, label in enumerate(class_order)}[args.outcome]

        ev = model_prob * args.odds_taken - 1

        bet = Bet(
            match_id=match.id,
            outcome=args.outcome,
            stake=args.stake,
            odds_taken=args.odds_taken,
            bookmaker="Stake",
            model_prob_at_bet=model_prob,
            ev_at_bet=ev,
            placed_at=datetime.datetime.now(datetime.timezone.utc),
            notes=args.notes,
        )
        session.add(bet)
        session.commit()

        print(f"Logged bet #{bet.id}: {args.outcome} @ {args.odds_taken}, stake={args.stake}")
        print(f"  Model probability at bet time: {model_prob:.1%}")
        print(f"  EV at bet time: {ev:+.1%}")
        if ev <= 0:
            print("  NOTE: this bet was logged with non-positive EV per the model. "
                  "That's not necessarily wrong (you may have information the model "
                  "doesn't), but worth knowing it's not the model's own recommendation.")
    finally:
        session.close()


if __name__ == "__main__":
    main()