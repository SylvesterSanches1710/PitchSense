"""
Shared load/sync logic for leagues, teams, and matches.

Both historical_loader.py (one-off backfill) and scheduler.py (recurring
updates) import from here. The difference between them is only *which*
seasons they ask for and *how often* they run — the actual "how do I turn
an API match dict into a database row" logic lives in exactly one place.
"""

import datetime

from data_collection.api_clients.football_data_org import FootballDataClient
from database.models import League, Match, MatchStatus, Team

COMPETITION_CODE = "PL"

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
    session.flush()
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


def sync_season(
    session,
    client: FootballDataClient,
    league: League,
    season: str,
    teams_by_external_id: dict[str, Team],
) -> tuple[int, int]:
    """
    Create-or-update every match in the given season.

    Returns (created_count, updated_count). Unlike the original backfill
    behavior (skip if already present), this UPDATES existing matches —
    which is exactly what the scheduler needs: a match that was
    SCHEDULED yesterday should become FINISHED with a real score today,
    not get silently skipped because a row already exists.
    """
    matches_data = client.get_matches(COMPETITION_CODE, season)
    created_count = 0
    updated_count = 0

    for match_data in matches_data:
        external_id = str(match_data["id"])

        home_external_id = str(match_data["homeTeam"]["id"])
        away_external_id = str(match_data["awayTeam"]["id"])
        for side_external_id, side_data in (
            (home_external_id, match_data["homeTeam"]),
            (away_external_id, match_data["awayTeam"]),
        ):
            if side_external_id not in teams_by_external_id:
                team = get_or_create_team(session, side_data, league)
                teams_by_external_id[side_external_id] = team

        score = match_data.get("score", {}).get("fullTime", {})
        status = _STATUS_MAP.get(match_data["status"], MatchStatus.SCHEDULED)
        kickoff_utc = datetime.datetime.fromisoformat(
            match_data["utcDate"].replace("Z", "+00:00")
        )

        match = session.query(Match).filter_by(external_id=external_id).first()

        if match is None:
            match = Match(
                external_id=external_id,
                league_id=league.id,
                season=f"{season}-{int(season) + 1}",
                home_team_id=teams_by_external_id[home_external_id].id,
                away_team_id=teams_by_external_id[away_external_id].id,
                kickoff_utc=kickoff_utc,
                status=status,
                home_score=score.get("home"),
                away_score=score.get("away"),
                venue=match_data.get("venue"),
            )
            session.add(match)
            created_count += 1
        else:
            changed = (
                match.status != status
                or match.home_score != score.get("home")
                or match.away_score != score.get("away")
                or match.kickoff_utc != kickoff_utc
            )
            if changed:
                match.status = status
                match.home_score = score.get("home")
                match.away_score = score.get("away")
                match.kickoff_utc = kickoff_utc
                updated_count += 1

    session.commit()
    return created_count, updated_count


def current_season_start_year() -> str:
    """
    Premier League seasons start in August. Before August, we're still in
    the season that started the previous calendar year.
    """
    today = datetime.date.today()
    year = today.year if today.month >= 8 else today.year - 1
    return str(year)