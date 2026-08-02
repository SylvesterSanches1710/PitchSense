"""
Thin wrapper around the football-data.org v4 API.

Design note: this client knows nothing about our database or SQLAlchemy
models — it only knows how to talk to the API and hand back plain dicts.
Keeping API clients "dumb" like this means swapping providers later (or
adding a second one) never touches the loading/parsing logic downstream.
"""

import os
import time

import requests

BASE_URL = "https://api.football-data.org/v4"

# Free tier: 10 requests/minute. We pace ourselves under that rather than
# hitting the limit and getting a 429 — sleeping 6.5s keeps us under 10/min
# with margin for clock drift.
_MIN_SECONDS_BETWEEN_REQUESTS = 6.5


class FootballDataClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["FOOTBALL_DATA_API_KEY"]
        self._session = requests.Session()
        self._session.headers.update({"X-Auth-Token": self.api_key})
        self._last_request_time: float = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._respect_rate_limit()
        response = self._session.get(f"{BASE_URL}{path}", params=params)
        self._last_request_time = time.monotonic()
        response.raise_for_status()
        return response.json()

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

    def get_teams(self, competition_code: str) -> list[dict]:
        """e.g. competition_code='PL' for Premier League."""
        data = self._get(f"/competitions/{competition_code}/teams")
        return data["teams"]

    def get_matches(self, competition_code: str, season: str) -> list[dict]:
        """
        `season` is the year the season started, as a string, e.g. "2023"
        for the 2023-2024 season.
        """
        data = self._get(
            f"/competitions/{competition_code}/matches",
            params={"season": season},
        )
        return data["matches"]

    def get_competition(self, competition_code: str) -> dict:
        return self._get(f"/competitions/{competition_code}")