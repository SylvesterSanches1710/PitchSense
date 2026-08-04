"""
One-off backfill: loads Premier League teams and 3 seasons of match
history into the database. Uses the shared create-or-update logic in
data_collection.sync, so re-running this is always safe.

Usage:
    python -m data_collection.historical_loader
"""

from data_collection.api_clients.football_data_org import FootballDataClient
from data_collection.sync import (
    COMPETITION_CODE,
    get_or_create_league,
    get_or_create_team,
    sync_season,
)
from database.models import League, Team
from database.session import SessionLocal

SEASONS = ["2023", "2024", "2025"]  # 2023-24, 2024-25, 2025-26


def load_current_teams(session, client: FootballDataClient, league: League) -> dict[str, Team]:
    teams_data = client.get_teams(COMPETITION_CODE)
    teams_by_external_id = {}
    for team_data in teams_data:
        team = get_or_create_team(session, team_data, league)
        teams_by_external_id[team.external_id] = team
    session.commit()
    print(f"Loaded {len(teams_by_external_id)} current teams.")
    return teams_by_external_id


def main():
    client = FootballDataClient()
    session = SessionLocal()

    try:
        league = get_or_create_league(session, client)
        teams_by_external_id = load_current_teams(session, client, league)

        total_created, total_updated = 0, 0
        for season in SEASONS:
            created, updated = sync_season(
                session, client, league, season, teams_by_external_id
            )
            print(f"Season {season}: {created} created, {updated} updated.")
            total_created += created
            total_updated += updated

        print(
            f"\nDone. {total_created} matches created, {total_updated} updated "
            f"across {len(SEASONS)} seasons."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()