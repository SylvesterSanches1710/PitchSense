"""
Rolling average goals scored and conceded over each team's last N matches
(home + away combined) — same window size as form.py for consistency.

Same pure-function pattern as the other feature modules.
"""

from collections import deque
from dataclasses import dataclass

from features.elo import MatchResult

GOALS_WINDOW = 10


@dataclass
class GoalsSnapshot:
    match_id: int
    home_goals_scored_avg_pre: float | None
    home_goals_conceded_avg_pre: float | None
    away_goals_scored_avg_pre: float | None
    away_goals_conceded_avg_pre: float | None


def compute_goals_features(
    matches: list[MatchResult], window: int = GOALS_WINDOW
) -> list[GoalsSnapshot]:
    """
    `matches` MUST be sorted chronologically — same requirement as every
    other feature computation in this package.

    Tracks scored and conceded as two separate rolling windows per team
    (not one window of goal difference) — a model can always derive goal
    difference from the two, but can't recover the split once it's been
    collapsed into a single number, and Over/Under modeling in Phase 4
    cares about scored/conceded separately, not just the net.
    """
    scored_history: dict[int, deque[int]] = {}
    conceded_history: dict[int, deque[int]] = {}
    snapshots: list[GoalsSnapshot] = []

    def avg_or_none(history: deque[int]) -> float | None:
        return (sum(history) / len(history)) if history else None

    for match in matches:
        home_scored_hist = scored_history.setdefault(match.home_team_id, deque(maxlen=window))
        home_conceded_hist = conceded_history.setdefault(match.home_team_id, deque(maxlen=window))
        away_scored_hist = scored_history.setdefault(match.away_team_id, deque(maxlen=window))
        away_conceded_hist = conceded_history.setdefault(match.away_team_id, deque(maxlen=window))

        snapshots.append(
            GoalsSnapshot(
                match_id=match.match_id,
                home_goals_scored_avg_pre=avg_or_none(home_scored_hist),
                home_goals_conceded_avg_pre=avg_or_none(home_conceded_hist),
                away_goals_scored_avg_pre=avg_or_none(away_scored_hist),
                away_goals_conceded_avg_pre=avg_or_none(away_conceded_hist),
            )
        )

        home_scored_hist.append(match.home_score)
        home_conceded_hist.append(match.away_score)
        away_scored_hist.append(match.away_score)
        away_conceded_hist.append(match.home_score)

    return snapshots