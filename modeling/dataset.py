"""
Assembles the final training dataset: one row per finished match, all
Phase 2 features as columns, target = result from the home team's
perspective (H/D/A). This is the single source both training and later
live prediction will pull from — one definition of "what a training row
looks like," not duplicated logic scattered across scripts.

Usage:
    python -m modeling.dataset
"""

import pandas as pd

from database.models import Match, MatchFeature, MatchStatus
from database.session import SessionLocal

FEATURE_COLUMNS = [
    "elo_home_pre", "elo_away_pre",
    "form_home_pre", "form_away_pre",
    "home_venue_form_pre", "away_venue_form_pre",
    "home_goals_scored_avg_pre", "home_goals_conceded_avg_pre",
    "away_goals_scored_avg_pre", "away_goals_conceded_avg_pre",
    "h2h_home_ppg_pre", "h2h_away_ppg_pre", "h2h_meetings_pre",
    "home_rest_days_pre", "away_rest_days_pre",
    "home_shots_avg_pre", "away_shots_avg_pre",
    "home_possession_avg_pre", "away_possession_avg_pre",
    "home_corners_avg_pre", "away_corners_avg_pre",
    "home_position_pre", "away_position_pre",
    "home_injuries_count_pre", "away_injuries_count_pre",
    "home_suspensions_count_pre", "away_suspensions_count_pre",
    "home_yellow_cards_avg_pre", "away_yellow_cards_avg_pre",
    "home_red_cards_avg_pre", "away_red_cards_avg_pre",
    "away_travel_km_pre",
]


def _result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def load_training_dataframe(session) -> pd.DataFrame:
    rows = (
        session.query(Match, MatchFeature)
        .join(MatchFeature, MatchFeature.match_id == Match.id)
        .filter(Match.status == MatchStatus.FINISHED)
        .filter(Match.home_score.isnot(None))
        .filter(Match.away_score.isnot(None))
        .order_by(Match.kickoff_utc.asc())
        .all()
    )

    records = []
    for match, feature in rows:
        record = {
            "match_id": match.id,
            "season": match.season,
            "kickoff_utc": match.kickoff_utc,
            "result": _result_label(match.home_score, match.away_score),
        }
        for col in FEATURE_COLUMNS:
            record[col] = getattr(feature, col)
        records.append(record)

    return pd.DataFrame.from_records(records)


def time_based_split(
    df: pd.DataFrame, test_season: str = "2025-2026"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split, never random. A random split would let the model
    train on a May 2025 match while being tested on a January 2025 match
    — future information leaking into training, which silently inflates
    every metric and produces a model that looks great in evaluation but
    is quietly cheating.

    Testing on an entire held-out SEASON (rather than just a date cutoff
    partway through one) also means the test set has zero overlap with
    the rolling-window histories used anywhere in training — every
    feature for a 2025-26 match was built from 2023/24 results only, up
    to that point, exactly like a real live prediction would be.
    """
    train_df = df[df["season"] != test_season].reset_index(drop=True)
    test_df = df[df["season"] == test_season].reset_index(drop=True)
    return train_df, test_df


def time_based_split_three_way(
    df: pd.DataFrame,
    train_season: str = "2023-2024",
    validation_season: str = "2024-2025",
    test_season: str = "2025-2026",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Same no-leakage principle as time_based_split, split three ways
    instead of two — needed for early stopping. The VALIDATION season is
    what boosting models watch during training to know when to stop
    adding trees (to avoid overfitting); the TEST season stays completely
    untouched until final evaluation, exactly like time_based_split's
    test set. Never use the test season for any tuning decision,
    including early stopping — that would be the same category of
    leakage as a random split, just one step removed.
    """
    train_df = df[df["season"] == train_season].reset_index(drop=True)
    validation_df = df[df["season"] == validation_season].reset_index(drop=True)
    test_df = df[df["season"] == test_season].reset_index(drop=True)
    return train_df, validation_df, test_df


def main():
    session = SessionLocal()
    try:
        df = load_training_dataframe(session)
        print(f"Loaded {len(df)} matches with features.\n")

        print("Result distribution (H/D/A):")
        print(df["result"].value_counts())
        print(f"\nAs percentages:\n{(df['result'].value_counts(normalize=True) * 100).round(1)}")

        print("\nMissing values per feature (NaN count):")
        missing = df[FEATURE_COLUMNS].isnull().sum()
        print(missing[missing > 0].sort_values(ascending=False))

        train_df, test_df = time_based_split(df)
        print(f"\nTrain: {len(train_df)} matches, seasons {sorted(train_df['season'].unique())}")
        print(f"Test:  {len(test_df)} matches, seasons {sorted(test_df['season'].unique())}")
    finally:
        session.close()


if __name__ == "__main__":
    main()