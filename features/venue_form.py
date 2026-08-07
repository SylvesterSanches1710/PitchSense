"""
Venue-specific form: points from a team's last N matches AT HOME (for the
home team) and last N matches AWAY (for the away team) — as opposed to
form.py, which mixes both venues together.

Same pure-function pattern as elo.py and form.py.
"""

from collections import deque
from dataclasses import dataclass

from features.elo import MatchResult

VENUE_FORM_WINDOW = 5


@dataclass
class VenueFormSnapshot:
    match_id: int
    home_venue_form_pre: float | None  # home team's form in their last N HOME matches
    away_venue_form_pre: float | None  # away team's form in their last N AWAY matches


def _points_earned(team_score: int, opponent_score: int) -> int:
    if team_score > opponent_score:
        return 3
    if team_score == opponent_score:
        return 1
    return 0


def compute_venue_form_features(
    matches: list[MatchResult], window: int = VENUE_FORM_WINDOW
) -> list[VenueFormSnapshot]:
    """
    `matches` MUST be sorted chronologically, same requirement as the
    other feature computations.

    Note the asymmetry versus form.py: here, a team's home-venue history
    is ONLY updated when they play at home (their away matches don't
    touch it, and vice versa). That's the entire point of this feature —
    isolating "how do they do specifically at home" from overall form.
    """
    home_history: dict[int, deque[int]] = {}
    away_history: dict[int, deque[int]] = {}
    snapshots: list[VenueFormSnapshot] = []

    for match in matches:
        home_hist = home_history.setdefault(match.home_team_id, deque(maxlen=window))
        away_hist = away_history.setdefault(match.away_team_id, deque(maxlen=window))

        home_venue_form_pre = sum(home_hist) if home_hist else None
        away_venue_form_pre = sum(away_hist) if away_hist else None

        snapshots.append(
            VenueFormSnapshot(
                match_id=match.match_id,
                home_venue_form_pre=home_venue_form_pre,
                away_venue_form_pre=away_venue_form_pre,
            )
        )

        home_points = _points_earned(match.home_score, match.away_score)
        away_points = _points_earned(match.away_score, match.home_score)
        home_hist.append(home_points)
        away_hist.append(away_points)

    return snapshots