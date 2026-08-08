"""
Two-stage process for pulling match stats from API-Football:

  1. map_fixtures()   — cheap (3 requests total): matches our existing
                         Match rows to API-Football's own fixture IDs, by
                         team name + date. Run once (safe to re-run).

  2. fetch_statistics(limit) — expensive (1 request per fixture): pulls
                         shots/possession/corners/cards for matches that
                         have a mapped fixture ID but no stats yet.
                         Resumable — run once a day until done.

Usage:
    python -m data_collection.match_stats_loader --map
    python -m data_collection.match_stats_loader --fetch --limit 90
"""

import argparse
import datetime
import re

from data_collection.api_clients.api_football import (
    PREMIER_LEAGUE_ID,
    ApiFootballClient,
)
from database.models import Match, MatchStats, Team
from database.session import SessionLocal

SEASONS = ["2023", "2024", "2025"]

# A conservative default — leaves headroom under the 100/day cap for
# other calls (or a retry) on the same day, rather than cutting it exactly
# to 100 and risking a failed run if anything else touches the API today.
DEFAULT_FETCH_LIMIT = 90

# Known name mismatches that substring matching (below) can't catch on
# its own — e.g. "Wolves" isn't a substring of "Wolverhampton Wanderers",
# and "Sheffield Utd" doesn't share "United" spelled out. Add to this if
# map_fixtures() prints an "unmatched" warning for a team you recognize.
MANUAL_NAME_OVERRIDES = {
    "wolves": "wolverhampton wanderers",
    "sheffield utd": "sheffield united",
}


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = MANUAL_NAME_OVERRIDES.get(name, name)
    for suffix in (" fc", " afc", " f.c.", " a.f.c."):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for prefix in ("afc ",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = name.replace("&", "and").replace(".", "").replace("'", "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def resolve_team(api_name: str, teams_by_normalized_name: dict[str, "Team"]):
    """
    Exact match first, then a substring fallback — this is what handles
    API-Football using colloquial names ("Tottenham", "West Ham") against
    our official names ("Tottenham Hotspur FC", "West Ham United FC")
    without needing an override hand-written for every single team.

    Only returns a substring match if it's UNIQUE — an ambiguous partial
    match (which could theoretically happen with short names) is treated
    as no match at all rather than guessing wrong silently.
    """
    normalized = normalize_name(api_name)

    exact = teams_by_normalized_name.get(normalized)
    if exact is not None:
        return exact

    candidates = [
        team
        for norm_name, team in teams_by_normalized_name.items()
        if normalized in norm_name or norm_name in normalized
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def map_fixtures() -> None:
    client = ApiFootballClient()
    session = SessionLocal()
    try:
        teams_by_normalized_name = {
            normalize_name(team.name): team for team in session.query(Team).all()
        }

        matched_count = 0
        unmatched_teams: set[str] = set()
        skipped_seasons: list[str] = []

        for season in SEASONS:
            try:
                fixtures = client.get_season_fixtures(PREMIER_LEAGUE_ID, season)
            except RuntimeError as e:
                # Free-tier plans on API-Football restrict which seasons
                # you can query, and that window rolls over time (it may
                # not include the most recent season). Skip this season
                # rather than aborting the whole mapping run — 2 out of 3
                # seasons of stats is still useful, and matches in the
                # skipped season simply keep null stats-derived features,
                # same as any other legitimately-missing data in this
                # project (see the nullable-columns note in models.py).
                print(f"Season {season}: unavailable on your plan ({e}). Skipping.")
                skipped_seasons.append(season)
                continue

            print(f"Season {season}: {len(fixtures)} fixtures from API-Football.")

            for fixture in fixtures:
                fixture_id = str(fixture["fixture"]["id"])
                fixture_date = datetime.datetime.fromisoformat(
                    fixture["fixture"]["date"]
                ).date()
                home_name = fixture["teams"]["home"]["name"]
                away_name = fixture["teams"]["away"]["name"]

                home_team = resolve_team(home_name, teams_by_normalized_name)
                away_team = resolve_team(away_name, teams_by_normalized_name)

                if home_team is None:
                    unmatched_teams.add(home_name)
                if away_team is None:
                    unmatched_teams.add(away_name)
                if home_team is None or away_team is None:
                    continue

                # Record the API-Football team IDs while we're here — cheap
                # and may be useful for other endpoints later (e.g. injuries).
                home_team.api_football_id = str(fixture["teams"]["home"]["id"])
                away_team.api_football_id = str(fixture["teams"]["away"]["id"])

                match = (
                    session.query(Match)
                    .filter(
                        Match.home_team_id == home_team.id,
                        Match.away_team_id == away_team.id,
                        Match.kickoff_utc >= datetime.datetime.combine(
                            fixture_date, datetime.time.min, tzinfo=datetime.timezone.utc
                        ),
                        Match.kickoff_utc < datetime.datetime.combine(
                            fixture_date + datetime.timedelta(days=1),
                            datetime.time.min,
                            tzinfo=datetime.timezone.utc,
                        ),
                    )
                    .first()
                )

                if match is None:
                    print(
                        f"  No matching Match row for {home_name} vs {away_name} "
                        f"on {fixture_date} — skipping."
                    )
                    continue

                match.api_football_fixture_id = fixture_id
                matched_count += 1

            session.commit()

        print(f"\nMapped {matched_count} matches to API-Football fixture IDs.")
        if skipped_seasons:
            print(
                f"\nSkipped season(s) not covered by your plan: {', '.join(skipped_seasons)}. "
                f"Matches from these seasons will have no stats-derived features "
                f"(shots/possession/corners/cards) until you either upgrade or your "
                f"plan's available-seasons window rolls forward to include them."
            )
        if unmatched_teams:
            print(
                f"\nUnmatched team names (add overrides to MANUAL_NAME_OVERRIDES "
                f"if these are real teams with a naming mismatch):"
            )
            for name in sorted(unmatched_teams):
                print(f"  - {name}")
    finally:
        session.close()


def _extract_stat(statistics: list[dict], stat_type: str) -> float | None:
    for stat in statistics:
        if stat["type"] == stat_type:
            value = stat["value"]
            if value is None:
                return None
            if isinstance(value, str) and value.endswith("%"):
                return float(value.rstrip("%"))
            return float(value)
    return None


def fetch_statistics(limit: int) -> None:
    client = ApiFootballClient()
    session = SessionLocal()
    try:
        already_fetched_ids = {
            row.match_id for row in session.query(MatchStats.match_id).all()
        }
        pending = (
            session.query(Match)
            .filter(Match.api_football_fixture_id.isnot(None))
            .filter(~Match.id.in_(already_fetched_ids) if already_fetched_ids else True)
            .order_by(Match.kickoff_utc.asc())
            .limit(limit)
            .all()
        )

        total_mapped = (
            session.query(Match).filter(Match.api_football_fixture_id.isnot(None)).count()
        )
        remaining_before = total_mapped - len(already_fetched_ids)

        print(
            f"{remaining_before} matches still need stats. "
            f"Fetching up to {len(pending)} this run..."
        )

        fetched_count = 0
        for match in pending:
            response = client.get_fixture_statistics(match.api_football_fixture_id)

            home_stats, away_stats = None, None
            for team_block in response:
                if str(team_block["team"]["id"]) == match.home_team.api_football_id:
                    home_stats = team_block["statistics"]
                elif str(team_block["team"]["id"]) == match.away_team.api_football_id:
                    away_stats = team_block["statistics"]

            match_stats = MatchStats(
                match_id=match.id,
                fetched_at=datetime.datetime.now(datetime.timezone.utc),
                home_shots_total=_extract_stat(home_stats, "Total Shots") if home_stats else None,
                away_shots_total=_extract_stat(away_stats, "Total Shots") if away_stats else None,
                home_possession_pct=_extract_stat(home_stats, "Ball Possession") if home_stats else None,
                away_possession_pct=_extract_stat(away_stats, "Ball Possession") if away_stats else None,
                home_corners=_extract_stat(home_stats, "Corner Kicks") if home_stats else None,
                away_corners=_extract_stat(away_stats, "Corner Kicks") if away_stats else None,
                home_yellow_cards=_extract_stat(home_stats, "Yellow Cards") if home_stats else None,
                away_yellow_cards=_extract_stat(away_stats, "Yellow Cards") if away_stats else None,
                home_red_cards=_extract_stat(home_stats, "Red Cards") if home_stats else None,
                away_red_cards=_extract_stat(away_stats, "Red Cards") if away_stats else None,
            )
            session.add(match_stats)
            session.commit()  # commit per match — safe to interrupt mid-run
            fetched_count += 1

            if not response:
                print(f"  Match {match.id}: no stats coverage available (empty response).")

        remaining_after = remaining_before - fetched_count
        print(f"\nFetched {fetched_count} matches this run. {remaining_after} remaining.")
        if remaining_after > 0:
            days_left = (remaining_after + limit - 1) // limit
            print(f"At {limit}/day, that's about {days_left} more day(s). Run this again tomorrow.")
        else:
            print("Stats backfill complete!")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="store_true", help="Map matches to API-Football fixture IDs")
    parser.add_argument("--fetch", action="store_true", help="Fetch statistics for mapped matches")
    parser.add_argument("--limit", type=int, default=DEFAULT_FETCH_LIMIT)
    args = parser.parse_args()

    if args.map:
        map_fixtures()
    elif args.fetch:
        fetch_statistics(args.limit)
    else:
        parser.error("Pass --map or --fetch")


if __name__ == "__main__":
    main()