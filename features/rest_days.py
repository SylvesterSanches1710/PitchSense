"""
Rest days feature: how many days since each team's previous match, as of
this match's kickoff. Captures fixture congestion / fatigue — a team on
3 days' rest after a midweek European game is a genuinely different
proposition than the same team on a full week off.

Depends on MatchResult carrying `kickoff_utc` (see setup guide — this is
the one shared-type change the other feature modules didn't need).
"""

import datetime
from dataclasses import dataclass

from features.elo import MatchResult


@dataclass
class RestDaysSnapshot:
    match_id: int
    home_rest_days_pre: int | None  # None = team's first match in our dataset
    away_rest_days_pre: int | None


def compute_rest_days_features(matches: list[MatchResult]) -> list[RestDaysSnapshot]:
    """`matches` MUST be sorted chronologically, same as every other
    feature module here."""
    last_match_date: dict[int, datetime.datetime] = {}
    snapshots: list[RestDaysSnapshot] = []

    for match in matches:
        home_last = last_match_date.get(match.home_team_id)
        away_last = last_match_date.get(match.away_team_id)

        home_rest_days_pre = (
            (match.kickoff_utc - home_last).days if home_last is not None else None
        )
        away_rest_days_pre = (
            (match.kickoff_utc - away_last).days if away_last is not None else None
        )

        snapshots.append(
            RestDaysSnapshot(
                match_id=match.match_id,
                home_rest_days_pre=home_rest_days_pre,
                away_rest_days_pre=away_rest_days_pre,
            )
        )

        last_match_date[match.home_team_id] = match.kickoff_utc
        last_match_date[match.away_team_id] = match.kickoff_utc

    return snapshots