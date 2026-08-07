"""
Head-to-head feature: how each team has historically performed
specifically against THIS opponent (not opponents in general).

Different shape from form.py/venue_form.py/goals.py: those track one
history per team. This tracks one history per (team, opponent) PAIR,
since "how City do against United" and "how City do overall" are
different questions.

Uses average points-per-game rather than a raw sum, because — unlike the
10-match form window, which is almost always full — two teams meet at
most 2x/season, so the number of prior meetings varies from 0 to ~6
across a 3-season dataset. A sum would make "3 points from 1 meeting"
look identical to "3 points from 3 meetings" (1 win vs. 3 draws) when
those are very different signals; an average tells them apart, and
`meetings_pre` alongside it tells you how much to trust that average.
"""

from collections import deque
from dataclasses import dataclass

from features.elo import MatchResult

# Small cap (not a "rolling window" in the same sense as form — with only
# 3 seasons of data, two teams rarely meet more than 5-6 times total, so
# this mostly just guards against unbounded growth if the dataset grows).
H2H_MAX_MEETINGS = 5


@dataclass
class H2HSnapshot:
    match_id: int
    h2h_home_ppg_pre: float | None  # home team's avg points/game vs THIS opponent
    h2h_away_ppg_pre: float | None  # away team's avg points/game vs THIS opponent
    h2h_meetings_pre: int  # how many prior meetings this is based on (0 if none)


def _points_earned(team_score: int, opponent_score: int) -> int:
    if team_score > opponent_score:
        return 3
    if team_score == opponent_score:
        return 1
    return 0


def compute_head_to_head_features(
    matches: list[MatchResult], max_meetings: int = H2H_MAX_MEETINGS
) -> list[H2HSnapshot]:
    """`matches` MUST be sorted chronologically, same as every other
    feature module here."""
    # Keyed by (team_id, opponent_id) — asymmetric on purpose, since we
    # want "City's points earned vs United" tracked separately from
    # "United's points earned vs City", even though they're drawn from
    # the same matches.
    history: dict[tuple[int, int], deque[int]] = {}
    snapshots: list[H2HSnapshot] = []

    for match in matches:
        home_key = (match.home_team_id, match.away_team_id)
        away_key = (match.away_team_id, match.home_team_id)

        home_hist = history.get(home_key)
        away_hist = history.get(away_key)

        h2h_home_ppg_pre = (sum(home_hist) / len(home_hist)) if home_hist else None
        h2h_away_ppg_pre = (sum(away_hist) / len(away_hist)) if away_hist else None
        meetings_pre = len(home_hist) if home_hist else 0

        snapshots.append(
            H2HSnapshot(
                match_id=match.match_id,
                h2h_home_ppg_pre=h2h_home_ppg_pre,
                h2h_away_ppg_pre=h2h_away_ppg_pre,
                h2h_meetings_pre=meetings_pre,
            )
        )

        home_points = _points_earned(match.home_score, match.away_score)
        away_points = _points_earned(match.away_score, match.home_score)

        history.setdefault(home_key, deque(maxlen=max_meetings)).append(home_points)
        history.setdefault(away_key, deque(maxlen=max_meetings)).append(away_points)

    return snapshots