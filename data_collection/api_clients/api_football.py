"""
Thin wrapper around API-Football (api-sports.io) v3.

Same design as football_data_org.py — dumb client, knows nothing about
our database, just turns API calls into plain dicts.
"""

import os
import time

import requests

BASE_URL = "https://v3.football.api-sports.io"

PREMIER_LEAGUE_ID = 39

# The free plan enforces 10 requests/minute (not just the 100/day cap).
# 6.5s between requests keeps us under that with margin.
_MIN_SECONDS_BETWEEN_REQUESTS = 6.5

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 65  # a bit over a minute — the per-minute window resets


class ApiFootballClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["API_FOOTBALL_KEY"]
        self._session = requests.Session()
        self._session.headers.update({"x-apisports-key": self.api_key})
        self._last_request_time: float = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(1, _MAX_RETRIES + 1):
            self._respect_rate_limit()
            response = self._session.get(f"{BASE_URL}{path}", params=params)
            self._last_request_time = time.monotonic()

            if response.status_code == 429:
                if attempt == _MAX_RETRIES:
                    response.raise_for_status()
                print(
                    f"  Rate limited (attempt {attempt}/{_MAX_RETRIES}). "
                    f"Waiting {_RETRY_BACKOFF_SECONDS}s for the per-minute "
                    f"window to reset..."
                )
                time.sleep(_RETRY_BACKOFF_SECONDS)
                continue

            response.raise_for_status()
            data = response.json()

            # API-Football returns HTTP 200 even for quota-exceeded and
            # invalid-parameter errors — the actual error lives in the body.
            if data.get("errors"):
                raise RuntimeError(f"API-Football error on {path}: {data['errors']}")

            return data

        raise RuntimeError(f"Failed {path} after {_MAX_RETRIES} attempts (rate limited).")

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

    def get_season_fixtures(self, league_id: int, season: str) -> list[dict]:
        """One call returns every fixture (with API-Football's own fixture
        ID and team IDs) for the whole season — this is the cheap bulk call."""
        data = self._get("/fixtures", params={"league": league_id, "season": season})
        return data["response"]

    def get_fixtures_by_date(self, date_str: str) -> list[dict]:
        """Fixtures across ALL competitions on a specific calendar date
        (format: YYYY-MM-DD) — not a season archive pull and not the
        paid-only 'next' parameter, so neither restriction applies here.
        Returns fixtures from every league on that date; filter by
        league ID yourself for Premier League specifically."""
        data = self._get("/fixtures", params={"date": date_str})
        return data["response"]

    def get_fixture_statistics(self, fixture_id: str) -> list[dict]:
        """One call per fixture — this is the expensive part that the
        free tier's limits actually constrain. Returns a list of
        (usually) 2 blocks, one per team, each with a list of named stats
        (Shots on Goal, Ball Possession, Corner Kicks, Yellow Cards, etc).
        Can legitimately return an empty list if this fixture has no
        stats coverage — that's a valid response, not an error."""
        data = self._get("/fixtures/statistics", params={"fixture": fixture_id})
        return data["response"]

    def get_fixture_injuries(self, fixture_id: str) -> list[dict]:
        """One call per fixture. Returns one record per player ruled out
        for THIS specific match, each tagged with a reason (e.g.
        'Hamstring Injury', 'Suspended', 'Illness'). An empty list can
        mean either a genuinely clean team news report or no injury
        coverage for this fixture — the response alone can't distinguish
        the two, same ambiguity as fixture statistics."""
        data = self._get("/injuries", params={"fixture": fixture_id})
        return data["response"]

    def get_fixture_odds(self, fixture_id: str) -> list[dict]:
        """One call per fixture. Returns odds grouped by bookmaker, each
        with a list of markets ('bets'), each with a list of outcomes
        ('values'). An empty list means no odds coverage for this fixture
        — for UPCOMING fixtures this usually means odds just haven't been
        posted yet (try again closer to kickoff); for anything more than
        7 days old, API-Football simply doesn't retain odds data at all."""
        data = self._get("/odds", params={"fixture": fixture_id})
        return data["response"]