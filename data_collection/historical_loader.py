"""
One-off backfill: loads Premier League teams and 3 seasons of match history
into the database. Safe to re-run — every insert is "get or create" keyed
on external_id, so running this twice won't create duplicates.

Usage:
    python -m data_collection.historical_loader
"""

import datetime

from data_collection.api_clients.football_data_org import FootballDataClient
from database.models import League, Match, MatchStatus, Team
from database.session import SessionLocal

COMPETITION_CODE = "PL"
SEASONS = ["2023", "2024", "2025"]  # 2023-24, 2024-25, 2025-26

# football-data.org statuses -> our simplified MatchStatus enum.
# IN_PLAY/PAUSED/SUSPENDED are treated as SCHEDULED here since this loader
# is for historical backfill, not live tracking — the daily update job
# (built next) will refine in-progress matches as they finish.
_STATUS_MAP = {
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,
    "IN_PLAY": MatchStatus.SCHEDULED,
    "PAUSED": MatchStatus.SCHEDULED,
    "SUSPENDED": MatchStatus.SCHEDULED,
    "FINISHED": MatchStatus.FINISHED,
    "POSTPONED": MatchStatus.POSTPONED,
    "CANCELLED": MatchStatus.CANCELLED,
}


def get_or_create_league(session, client: FootballDataClient) -> League:
    league = session.query(League).filter_by(external_id=COMPETITION_CODE).first()
    if league:
        return league

    competition = client.get_competition(COMPETITION_CODE)
    league = League(
        external_id=COMPETITION_CODE,
        name=competition["name"],
        country=competition["area"]["name"],
    )
    session.add(league)
    session.flush()  # populate league.id without committing yet
    print(f"Created league: {league.name}")
    return league


def get_or_create_team(session, team_data: dict, league: League) -> Team:
    external_id = str(team_data["id"])
    team = session.query(Team).filter_by(external_id=external_id).first()
    if team:
        return team

    team = Team(
        external_id=external_id,
        name=team_data["name"],
        short_name=team_data.get("shortName"),
        league_id=league.id,
    )
    session.add(team)
    session.flush()
    print(f"  Created team: {team.name}")
    return team


def load_teams(session, client: FootballDataClient, league: League) -> dict[str, Team]:
    """Seed the current squad list up front — not the only source of teams
    (see load_matches_for_season, which discovers teams on the fly too),
    but this gives current teams clean data from the dedicated teams
    endpoint rather than the leaner info embedded in match records."""
    teams_data = client.get_teams(COMPETITION_CODE)
    teams_by_external_id = {}
    for team_data in teams_data:
        team = get_or_create_team(session, team_data, league)
        teams_by_external_id[team.external_id] = team
    session.commit()
    print(f"Loaded {len(teams_by_external_id)} current teams.")
    return teams_by_external_id


def load_matches_for_season(
    session,
    client: FootballDataClient,
    league: League,
    season: str,
    teams_by_external_id: dict[str, Team],
) -> int:
    """
    teams_by_external_id is mutated in place as new teams are discovered —
    e.g. a team relegated before the current season, which wasn't in the
    current-squad list from load_teams. get_or_create_team checks the DB
    first, so a team is never duplicated across seasons or across runs.
    """
    matches_data = client.get_matches(COMPETITION_CODE, season)
    loaded_count = 0

    for match_data in matches_data:
        external_id = str(match_data["id"])
        existing = session.query(Match).filter_by(external_id=external_id).first()
        if existing:
            continue  # already loaded, skip (safe to re-run this script)

        home_external_id = str(match_data["homeTeam"]["id"])
        away_external_id = str(match_data["awayTeam"]["id"])

        for side_external_id, side_data in (
            (home_external_id, match_data["homeTeam"]),
            (away_external_id, match_data["awayTeam"]),
        ):
            if side_external_id not in teams_by_external_id:
                team = get_or_create_team(session, side_data, league)
                teams_by_external_id[side_external_id] = team
                print(f"  Discovered team from match history: {team.name}")

        score = match_data.get("score", {}).get("fullTime", {})
        status = _STATUS_MAP.get(match_data["status"], MatchStatus.SCHEDULED)

        match = Match(
            external_id=external_id,
            league_id=league.id,
            season=f"{season}-{int(season) + 1}",
            home_team_id=teams_by_external_id[home_external_id].id,
            away_team_id=teams_by_external_id[away_external_id].id,
            kickoff_utc=datetime.datetime.fromisoformat(
                match_data["utcDate"].replace("Z", "+00:00")
            ),
            status=status,
            home_score=score.get("home"),
            away_score=score.get("away"),
            venue=match_data.get("venue"),
        )
        session.add(match)
        loaded_count += 1

    session.commit()
    print(f"Season {season}: loaded {loaded_count} new matches.")
    return loaded_count


def main():
    client = FootballDataClient()
    session = SessionLocal()

    try:
        league = get_or_create_league(session, client)
        teams_by_external_id = load_teams(session, client, league)

        total_matches = 0
        for season in SEASONS:
            total_matches += load_matches_for_season(
                session, client, league, season, teams_by_external_id
            )

        print(f"\nDone. {total_matches} new matches loaded across {len(SEASONS)} seasons.")
    finally:
        session.close()


if __name__ == "__main__":
    main()