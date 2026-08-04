"""
Rolling "form" feature: points earned from each team's last N matches
(home and away combined), as of just before the match being featured.

Same design as elo.py — pure function, no database dependency.
"""

from collections import deque
from dataclasses import dataclass

from features.elo import MatchResult

FORM_WINDOW = 10


@dataclass
class FormSnapshot:
    match_id: int
    form_home_pre: float | None  # None = team has zero prior matches on record
    form_away_pre: float | None


def _points_earned(team_score: int, opponent_score: int) -> int:
    if team_score > opponent_score:
        return 3
    if team_score == opponent_score:
        return 1
    return 0


def compute_form_features(
    matches: list[MatchResult], window: int = FORM_WINDOW
) -> list[FormSnapshot]:
    """
    `matches` MUST already be sorted chronologically — same requirement
    as compute_elo_ratings, same reason: this is a sequential, stateful
    computation.

    A team with fewer than `window` prior matches gets form computed over
    whatever history it does have (e.g. 1st match of the dataset for a
    team → based on 0 games → None; 4th match → based on 3 games). This
    is a deliberate tradeoff: using a partial window still carries real
    signal, versus discarding it entirely and losing that team's early
    matches from every model trained on this feature.
    """
    history: dict[int, deque[int]] = {}
    snapshots: list[FormSnapshot] = []

    for match in matches:
        home_history = history.setdefault(match.home_team_id, deque(maxlen=window))
        away_history = history.setdefault(match.away_team_id, deque(maxlen=window))

        form_home_pre = sum(home_history) if home_history else None
        form_away_pre = sum(away_history) if away_history else None

        snapshots.append(
            FormSnapshot(
                match_id=match.match_id,
                form_home_pre=form_home_pre,
                form_away_pre=form_away_pre,
            )
        )

        home_points = _points_earned(match.home_score, match.away_score)
        away_points = _points_earned(match.away_score, match.home_score)
        home_history.append(home_points)
        away_history.append(away_points)

    return snapshots