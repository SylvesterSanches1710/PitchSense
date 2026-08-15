"""
Reads the filled-in odds_template.csv and loads it into the Odds table
as bookmaker="Stake". Validates each row before committing anything —
a typo in one row shouldn't silently corrupt the whole batch.

Usage:
    python -m data_collection.load_manual_odds
"""

import csv
import datetime

from database.models import Match, Odds
from database.session import SessionLocal

INPUT_PATH = "odds_template.csv"


def validate_row(row: dict, row_num: int) -> tuple[float, float, float] | None:
    """Returns (home_odds, draw_odds, away_odds) if valid, or None (with
    a printed reason) if this row should be skipped rather than crash
    the whole load."""
    if not row["home_odds"] or not row["draw_odds"] or not row["away_odds"]:
        print(f"  Row {row_num} (match {row['match_id']}): skipping — odds not filled in yet.")
        return None

    try:
        home_odds = float(row["home_odds"])
        draw_odds = float(row["draw_odds"])
        away_odds = float(row["away_odds"])
    except ValueError:
        print(f"  Row {row_num} (match {row['match_id']}): skipping — odds aren't valid numbers.")
        return None

    for label, value in [("home", home_odds), ("draw", draw_odds), ("away", away_odds)]:
        if value <= 1.0:
            print(
                f"  Row {row_num} (match {row['match_id']}): skipping — {label}_odds={value} "
                f"looks wrong (decimal odds should always be > 1.0; if you typed American "
                f"or fractional odds by mistake, convert to decimal first)."
            )
            return None

    return home_odds, draw_odds, away_odds


def main():
    session = SessionLocal()
    try:
        with open(INPUT_PATH, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        loaded_count = 0
        for row_num, row in enumerate(rows, start=2):  # row 1 is the header
            validated = validate_row(row, row_num)
            if validated is None:
                continue

            home_odds, draw_odds, away_odds = validated
            match_id = int(row["match_id"])

            match = session.get(Match, match_id)
            if match is None:
                print(f"  Row {row_num}: match_id {match_id} not found — skipping.")
                continue

            # Overwrite any existing Stake entry for this match rather than
            # accumulating duplicates if you re-run this after updating odds.
            existing = (
                session.query(Odds)
                .filter_by(match_id=match_id, bookmaker="Stake", market="h2h")
                .first()
            )
            if existing:
                existing.home_odds = home_odds
                existing.draw_odds = draw_odds
                existing.away_odds = away_odds
                existing.fetched_at = datetime.datetime.now(datetime.timezone.utc)
            else:
                session.add(
                    Odds(
                        match_id=match_id, bookmaker="Stake", market="h2h",
                        home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds,
                        fetched_at=datetime.datetime.now(datetime.timezone.utc),
                    )
                )
            loaded_count += 1

        session.commit()
        print(f"\nLoaded {loaded_count} matches' odds from {INPUT_PATH}.")
    finally:
        session.close()


if __name__ == "__main__":
    main()