"""
The actual decision-support output: loads the saved CatBoost model, runs
it against live pre-match features, compares against your Stake odds,
and reports implied probability, EV, and Kelly stake sizing per outcome.

Usage:
    python -m modeling.betting_analysis
"""

import json
from pathlib import Path

from catboost import CatBoostClassifier

from database.models import Match, MatchFeature, Odds, Team
from database.session import SessionLocal
from modeling.dataset import FEATURE_COLUMNS
from modeling.betting_math import (
    remove_vig,
    calculate_ev,
    kelly_stake_fraction,
    check_low_data_warning,
    CLASS_TO_LABEL,
    KELLY_FRACTION,
)

MODEL_PATH = Path("modeling/model_registry/catboost_v1.cbm")
METADATA_PATH = Path("modeling/model_registry/catboost_v1_metadata.json")

# Fractional Kelly, not full Kelly — full Kelly assumes your probability
# estimate is exactly correct, which is a dangerous assumption for a
# model with the calibration profile confirmed earlier (specifically:
# overconfident on high-probability Away Win predictions). Quarter-Kelly
# gives most of Kelly's growth benefit with much less exposure to being
# wrong about your own edge.
# KELLY_FRACTION = 0.25

# CLASS_TO_LABEL = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

# # A team whose Elo still sits within this many points of the 1500
# # starting default has effectively played very few real matches — Elo
# # moves away from 1500 quickly once a team has genuine history. Combined
# # with form_pre being None (zero prior matches on record at all), this
# # catches newly-promoted or otherwise data-sparse teams automatically,
# # instead of relying on manually remembering to check.
# LOW_DATA_ELO_THRESHOLD = 15.0


# def check_low_data_warning(
#     team_name: str, elo_pre: float | None, form_pre: float | None
# ) -> str | None:
#     if form_pre is None:
#         return f"{team_name}: zero prior matches on record — likely newly promoted or new to this dataset."
#     if elo_pre is not None and abs(elo_pre - 1500.0) < LOW_DATA_ELO_THRESHOLD:
#         return f"{team_name}: Elo still near the 1500 default ({elo_pre:.1f}) — very little real history to draw on."
#     return None


# def remove_vig(home_odds: float, draw_odds: float, away_odds: float) -> dict:
#     """Converts raw bookmaker odds to fair (vig-removed) implied
#     probability, via the standard proportional method: raw implied
#     probabilities are computed per outcome, then normalized to sum to 1.
#     This is a stated simplification — more sophisticated de-vig methods
#     (e.g. Shin's method) exist and can differ slightly, but proportional
#     is the standard, simplest, and widely-used approach."""
#     raw = {"H": 1 / home_odds, "D": 1 / draw_odds, "A": 1 / away_odds}
#     overround = sum(raw.values())
#     return {k: v / overround for k, v in raw.items()}, overround


# def calculate_ev(model_prob: float, decimal_odds: float) -> float:
#     """Expected value per unit staked, using the ACTUAL odds offered
#     (not the vig-removed 'fair' odds) — that's the price you'd really
#     get paid at, so it's what determines real expected return."""
#     return model_prob * decimal_odds - 1


# def kelly_stake_fraction(model_prob: float, decimal_odds: float) -> float:
#     """Full Kelly fraction: what fraction of bankroll to stake for
#     theoretically optimal long-run growth, GIVEN that model_prob is
#     exactly correct. Negative when there's no edge (EV <= 0) — always
#     clamp to 0 in that case, never bet with a negative Kelly fraction."""
#     b = decimal_odds - 1  # net odds
#     numerator = model_prob * decimal_odds - 1
#     return max(0.0, numerator / b)


def main():
    if not MODEL_PATH.exists():
        print(
            f"No saved model found at {MODEL_PATH}. Run modeling.save_final_model first."
        )
        return

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    metadata = json.loads(METADATA_PATH.read_text())
    class_order = metadata["class_order"]

    session = SessionLocal()
    try:
        matches_with_odds = (
            session.query(Match, Odds)
            .join(Odds, Odds.match_id == Match.id)
            .filter(Odds.bookmaker == "Stake")
            .order_by(Match.kickoff_utc.asc())
            .all()
        )

        if not matches_with_odds:
            print("No matches with Stake odds found. Run load_manual_odds first.")
            return

        print(f"Analyzing {len(matches_with_odds)} matches with Stake odds.\n")
        print("=" * 100)

        total_outcomes_shown = 0
        total_positive_ev = 0

        for match, odds in matches_with_odds:
            feature_row = (
                session.query(MatchFeature).filter_by(match_id=match.id).first()
            )
            if feature_row is None:
                print(
                    f"Match {match.id}: no features computed yet — run build_live_features first. Skipping.\n"
                )
                continue

            home_team = session.get(Team, match.home_team_id)
            away_team = session.get(Team, match.away_team_id)

            feature_values = [[getattr(feature_row, col) for col in FEATURE_COLUMNS]]
            proba = model.predict_proba(feature_values)[0]
            model_probs = {label: proba[i] for i, label in enumerate(class_order)}

            fair_probs, overround = remove_vig(
                odds.home_odds, odds.draw_odds, odds.away_odds
            )
            bookmaker_odds = {
                "H": odds.home_odds,
                "D": odds.draw_odds,
                "A": odds.away_odds,
            }

            # Confidence: margin between the model's top and second pick —
            # a simple, interpretable measure of how decisive the
            # prediction is, NOT a rigorous statistical confidence
            # interval (that would need e.g. bootstrapped predictions,
            # which this doesn't attempt).
            sorted_probs = sorted(model_probs.values(), reverse=True)
            confidence_margin = sorted_probs[0] - sorted_probs[1]
            confidence_label = (
                "High"
                if confidence_margin > 0.25
                else "Medium" if confidence_margin > 0.10 else "Low"
            )

            print(
                f"\n{home_team.name} vs {away_team.name}  ({match.kickoff_utc.strftime('%Y-%m-%d %H:%M UTC')})"
            )

            home_warning = check_low_data_warning(
                home_team.name, feature_row.elo_home_pre, feature_row.form_home_pre
            )
            away_warning = check_low_data_warning(
                away_team.name, feature_row.elo_away_pre, feature_row.form_away_pre
            )
            if home_warning or away_warning:
                print(
                    "*** LOW DATA WARNING — treat this match's predictions with significant "
                    "extra caution, the model may be extrapolating outside its training "
                    "experience: ***"
                )
                if home_warning:
                    print(f"    {home_warning}")
                if away_warning:
                    print(f"    {away_warning}")

            print(f"Stake's market margin (vig): {(overround - 1) * 100:.1f}%")
            print(
                f"Model confidence: {confidence_label} (top pick beats runner-up by {confidence_margin:.1%})"
            )
            print(
                f"{'Outcome':<10}{'Model %':>10}{'Stake Fair %':>15}{'Stake Odds':>12}{'EV':>10}{'Kelly (1/4)':>13}"
            )
            print("-" * 70)

            for outcome in ["H", "D", "A"]:
                mp = model_probs[outcome]
                fp = fair_probs[outcome]
                bo = bookmaker_odds[outcome]
                ev = calculate_ev(mp, bo)
                kelly = kelly_stake_fraction(mp, bo) * KELLY_FRACTION

                flag = "  <-- +EV" if ev > 0 else ""
                total_outcomes_shown += 1
                if ev > 0:
                    total_positive_ev += 1
                print(
                    f"{CLASS_TO_LABEL[outcome]:<10}{mp:>10.1%}{fp:>15.1%}{bo:>12.2f}"
                    f"{ev:>+10.1%}{kelly:>13.1%}{flag}"
                )

        print("\n" + "=" * 100)

        ev_rate = (
            total_positive_ev / total_outcomes_shown if total_outcomes_shown else 0
        )
        print(
            f"\n+EV flagged on {total_positive_ev} of {total_outcomes_shown} outcomes shown ({ev_rate:.0%})."
        )
        if ev_rate > 0.30:
            print(
                "*** This rate is unusually high. In an efficiently-priced market, genuine "
                "value should be rare — a model that disagrees with the market on more than "
                "roughly a third of outcomes is more likely showing calibration noise than "
                "a real, systematic edge. Treat +EV flags as 'worth a closer look', not "
                "'worth betting', especially in a batch with a high overall rate like this one. ***"
            )

        print(
            "\nReminders:\n"
            "- EV and Kelly are only as good as the model's probability estimate. A "
            "positive EV here means 'the model disagrees with the market,' not a "
            "guarantee — the model could simply be wrong.\n"
            "- Away Win predictions are known to run overconfident at high probabilities "
            "(from the calibration check) — treat high-confidence Away +EV flags with "
            "extra skepticism specifically.\n"
            "- Kelly figures shown are already quarter-Kelly, not full Kelly, and are for "
            "education/reference — not a instruction to stake that amount."
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()
