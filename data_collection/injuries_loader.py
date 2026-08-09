"""
Backfills injury/suspension reports, one fixture at a time. Reuses the
api_football_fixture_id mapping already built by
match_stats_loader.py --map — no separate mapping step needed here.

Usage:
    python -m data_collection.injuries_loader --fetch [--limit 90]
"""

import argparse
import datetime

from data_collection.api_clients.api_football import ApiFootballClient
from database.models import Injury, Match
from database.session import SessionLocal

DEFAULT_FETCH_LIMIT = 90


def fetch_injuries(limit: int) -> None:
    client = ApiFootballClient()
    session = SessionLocal()
    try:
        base_query = session.query(Match).filter(Match.api_football_fixture_id.isnot(None))
        remaining_before = base_query.filter(Match.injuries_fetched_at.is_(None)).count()
        pending = (
            base_query.filter(Match.injuries_fetched_at.is_(None))
            .order_by(Match.kickoff_utc.asc())
            .limit(limit)
            .all()
        )

        print(
            f"{remaining_before} matches still need injury data. "
            f"Fetching up to {len(pending)} this run..."
        )

        fetched_count = 0
        for match in pending:
            response = client.get_fixture_injuries(match.api_football_fixture_id)

            for record in response:
                team_api_id = str(record["team"]["id"])
                if team_api_id == match.home_team.api_football_id:
                    team = match.home_team
                elif team_api_id == match.away_team.api_football_id:
                    team = match.away_team
                else:
                    # Shouldn't normally happen, but one odd record
                    # shouldn't crash the whole run — skip and move on.
                    continue

                session.add(
                    Injury(
                        team_id=team.id,
                        match_id=match.id,
                        player_name=record["player"]["name"],
                        status=record.get("reason") or record.get("type") or "Unknown",
                    )
                )

            match.injuries_fetched_at = datetime.datetime.now(datetime.timezone.utc)
            session.commit()  # per-match commit — safe to interrupt mid-run
            fetched_count += 1

            if not response:
                print(f"  Match {match.id}: no injuries reported.")

        remaining_after = remaining_before - fetched_count
        print(f"\nFetched {fetched_count} matches this run. {remaining_after} remaining.")
        if remaining_after > 0:
            days_left = (remaining_after + limit - 1) // limit
            print(f"At {limit}/day, that's about {days_left} more day(s). Run this again tomorrow.")
        else:
            print("Injuries backfill complete!")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--limit", type=int, default=DEFAULT_FETCH_LIMIT)
    args = parser.parse_args()
    if not args.fetch:
        parser.error("Pass --fetch")
    fetch_injuries(args.limit)


if __name__ == "__main__":
    main()