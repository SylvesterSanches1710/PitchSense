"""
Generates a CSV of upcoming matches with blank odds columns, ready to
fill in by hand from Stake's site.

Usage:
    python -m data_collection.generate_odds_template [--days 14]
"""

import argparse
import csv
import datetime

from database.models import Match, MatchStatus, Team
from database.session import SessionLocal

DEFAULT_DAYS_AHEAD = 14
OUTPUT_PATH = "odds_template.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_AHEAD)
    args = parser.parse_args()

    session = SessionLocal()
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = now + datetime.timedelta(days=args.days)

        matches = (
            session.query(Match)
            .filter(Match.kickoff_utc >= now)
            .filter(Match.kickoff_utc <= cutoff)
            .filter(Match.status == MatchStatus.SCHEDULED)
            .order_by(Match.kickoff_utc.asc())
            .all()
        )

        if not matches:
            print(
                f"No scheduled matches found in the next {args.days} days. "
                f"Run data_collection.scheduler --once first if this seems wrong."
            )
            return

        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "match_id", "kickoff_utc", "home_team", "away_team",
                "home_odds", "draw_odds", "away_odds",
            ])
            for match in matches:
                home_team = session.get(Team, match.home_team_id)
                away_team = session.get(Team, match.away_team_id)
                writer.writerow([
                    match.id, match.kickoff_utc.isoformat(),
                    home_team.name, away_team.name,
                    "", "", "",  # blank — fill these in from Stake
                ])

        print(f"Wrote {len(matches)} matches to {OUTPUT_PATH}")
        print("Open it, fill in home_odds/draw_odds/away_odds from Stake "
              "(decimal odds, e.g. 2.35), save, then run "
              "data_collection.load_manual_odds")
    finally:
        session.close()


if __name__ == "__main__":
    main()