"""
Fetches match-winner (1X2) odds for UPCOMING Premier League fixtures.

Historical odds backfill isn't possible: API-Football's /odds endpoint
only retains the last 7 days of data by design (confirmed from their own
docs), regardless of plan tier — this isn't something a paid plan
unlocks. This loader targets what the endpoint is actually built for:
odds on fixtures that haven't been played yet.

Two-stage process, same shape as match_stats_loader.py:
  --map    maps upcoming fixtures to API-Football fixture IDs (using
           get_next_fixtures, NOT get_season_fixtures — a season-archive
           pull isn't the right tool for "what's coming up next")
  --fetch  pulls odds for those mapped fixtures

Usage:
    python -m data_collection.odds_loader --map
    python -m data_collection.odds_loader --fetch [--limit 90]
"""

import argparse
import datetime

from data_collection.api_clients.api_football import (
    PREMIER_LEAGUE_ID,
    ApiFootballClient,
)
from database.models import Match, Odds, Team
from database.session import SessionLocal

DEFAULT_FETCH_LIMIT = 90
DEFAULT_DAYS_AHEAD = 14
PREFERRED_BOOKMAKERS = ["Bet365", "Pinnacle", "1xBet"]
MATCH_WINNER_MARKET_NAMES = {"match winner", "1x2", "fulltime result"}
OUTCOME_LABEL_MAP = {
    "home": "home", "1": "home",
    "draw": "draw", "x": "draw",
    "away": "away", "2": "away",
}


def map_upcoming_fixtures(days_ahead: int) -> None:
    """
    Same team-matching approach as match_stats_loader.py --map, but the
    fixture DISCOVERY works differently: instead of asking API-Football
    "what's coming up" (the paid-only `next` parameter), we already know
    what's coming up from our own database (synced independently via
    football-data.org's scheduler). We just ask API-Football, one date
    at a time, to confirm the fixture ID for dates we already care about
    — a query shape that isn't gated behind season-archive or paid-param
    restrictions.
    """
    from data_collection.match_stats_loader import normalize_name, resolve_team

    client = ApiFootballClient()
    session = SessionLocal()
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = now + datetime.timedelta(days=days_ahead)

        upcoming_matches = (
            session.query(Match)
            .filter(Match.kickoff_utc >= now)
            .filter(Match.kickoff_utc <= cutoff)
            .filter(Match.api_football_fixture_id.is_(None))
            .order_by(Match.kickoff_utc.asc())
            .all()
        )

        if not upcoming_matches:
            print(
                f"No unmapped upcoming matches found in the next {days_ahead} days. "
                f"Run data_collection.scheduler to sync more fixtures first if this "
                f"seems wrong."
            )
            return

        unique_dates = sorted({m.kickoff_utc.date() for m in upcoming_matches})
        print(f"{len(upcoming_matches)} unmapped upcoming matches across "
              f"{len(unique_dates)} distinct dates.")

        teams_by_normalized_name = {
            normalize_name(team.name): team for team in session.query(Team).all()
        }

        matched_count = 0
        for date in unique_dates:
            date_str = date.isoformat()
            fixtures = client.get_fixtures_by_date(date_str)
            pl_fixtures = [f for f in fixtures if f["league"]["id"] == PREMIER_LEAGUE_ID]
            print(f"{date_str}: {len(pl_fixtures)} Premier League fixtures found.")

            for fixture in pl_fixtures:
                fixture_id = str(fixture["fixture"]["id"])
                home_name = fixture["teams"]["home"]["name"]
                away_name = fixture["teams"]["away"]["name"]

                home_team = resolve_team(home_name, teams_by_normalized_name)
                away_team = resolve_team(away_name, teams_by_normalized_name)
                if home_team is None or away_team is None:
                    print(f"  Couldn't match teams for {home_name} vs {away_name} — skipping.")
                    continue

                match = next(
                    (
                        m for m in upcoming_matches
                        if m.home_team_id == home_team.id
                        and m.away_team_id == away_team.id
                        and m.kickoff_utc.date() == date
                    ),
                    None,
                )
                if match is None:
                    continue  # this date's PL fixture isn't one we're tracking as upcoming

                match.api_football_fixture_id = fixture_id
                matched_count += 1

        session.commit()
        print(f"\nMapped {matched_count} upcoming fixtures.")
    finally:
        session.close()


def _pick_bookmaker(bookmakers: list[dict]) -> dict | None:
    by_name = {b["name"]: b for b in bookmakers}
    for preferred in PREFERRED_BOOKMAKERS:
        if preferred in by_name:
            return by_name[preferred]
    return bookmakers[0] if bookmakers else None


def _extract_match_winner_odds(bookmaker: dict) -> dict | None:
    for bet in bookmaker.get("bets", []):
        if bet.get("name", "").strip().lower() in MATCH_WINNER_MARKET_NAMES:
            odds = {}
            for value in bet.get("values", []):
                label = OUTCOME_LABEL_MAP.get(str(value.get("value", "")).strip().lower())
                if label:
                    odds[label] = float(value["odd"])
            if {"home", "draw", "away"}.issubset(odds.keys()):
                return odds
    return None


def fetch_odds(limit: int) -> None:
    client = ApiFootballClient()
    session = SessionLocal()
    try:
        base_query = session.query(Match).filter(Match.api_football_fixture_id.isnot(None))
        remaining_before = base_query.filter(Match.odds_fetched_at.is_(None)).count()
        pending = (
            base_query.filter(Match.odds_fetched_at.is_(None))
            .order_by(Match.kickoff_utc.asc())
            .limit(limit)
            .all()
        )

        print(f"{remaining_before} matches still need odds. Fetching up to {len(pending)} this run...")

        fetched_count, parsed_count, error_count = 0, 0, 0
        for match in pending:
            try:
                response = client.get_fixture_odds(match.api_football_fixture_id)
            except Exception as e:
                # A persistent rate-limit or transient error shouldn't
                # kill the whole run — skip this match WITHOUT marking
                # odds_fetched_at, so it's retried on the next run rather
                # than being permanently (and wrongly) treated as "checked,
                # no data."
                print(f"  Match {match.id}: request failed ({e}). Will retry next run.")
                error_count += 1
                continue

            if response:
                bookmakers = response[0].get("bookmakers", [])
                bookmaker = _pick_bookmaker(bookmakers)

                if bookmaker is None:
                    print(f"  Match {match.id}: no bookmakers in response.")
                else:
                    parsed = _extract_match_winner_odds(bookmaker)
                    if parsed is None:
                        print(
                            f"  Match {match.id}: couldn't parse odds from "
                            f"'{bookmaker.get('name')}'. Raw bet names: "
                            f"{[b.get('name') for b in bookmaker.get('bets', [])]}"
                        )
                    else:
                        session.add(
                            Odds(
                                match_id=match.id,
                                bookmaker=bookmaker["name"],
                                market="h2h",
                                home_odds=parsed["home"],
                                draw_odds=parsed["draw"],
                                away_odds=parsed["away"],
                                fetched_at=datetime.datetime.now(datetime.timezone.utc),
                            )
                        )
                        parsed_count += 1
            else:
                print(f"  Match {match.id}: no odds posted yet (try again closer to kickoff).")

            match.odds_fetched_at = datetime.datetime.now(datetime.timezone.utc)
            session.commit()
            fetched_count += 1

        remaining_after = remaining_before - fetched_count
        print(
            f"\nFetched {fetched_count} matches this run ({parsed_count} had usable odds, "
            f"{error_count} failed and will retry). {remaining_after} remaining."
        )
        if remaining_after > 0:
            days_left = (remaining_after + limit - 1) // limit
            print(f"At {limit}/day, that's about {days_left} more day(s).")
        else:
            print("Odds fetch complete for currently-mapped fixtures!")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="store_true", help="Map upcoming fixtures")
    parser.add_argument("--fetch", action="store_true", help="Fetch odds for mapped fixtures")
    parser.add_argument("--limit", type=int, default=DEFAULT_FETCH_LIMIT)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_AHEAD,
                         help="How many days ahead to look for upcoming fixtures (with --map)")
    args = parser.parse_args()

    if args.map:
        map_upcoming_fixtures(args.days)
    elif args.fetch:
        fetch_odds(args.limit)
    else:
        parser.error("Pass --map or --fetch")


if __name__ == "__main__":
    main()