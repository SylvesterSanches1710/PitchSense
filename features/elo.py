"""
Elo rating computation for football.

Deliberately has zero database or SQLAlchemy imports — it takes plain
data in and returns plain data out. This makes it trivially unit-testable
and reusable (e.g. you could run it against a CSV in a notebook while
debugging, with no DB required).
"""

from dataclasses import dataclass

STARTING_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 65.0  # applied only when computing expected score


@dataclass
class MatchResult:
    match_id: int
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int


@dataclass
class EloSnapshot:
    match_id: int
    elo_home_pre: float
    elo_away_pre: float
    elo_home_post: float
    elo_away_post: float


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that side A beats side B, per the standard Elo formula."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _actual_score(team_score: int, opponent_score: int) -> float:
    if team_score > opponent_score:
        return 1.0
    if team_score == opponent_score:
        return 0.5
    return 0.0


def compute_elo_ratings(
    matches: list[MatchResult],
) -> tuple[list[EloSnapshot], dict[int, float]]:
    """
    `matches` MUST already be sorted chronologically (oldest first) — Elo
    is inherently sequential; feeding it matches out of order silently
    produces wrong ratings with no error to warn you.

    Returns (snapshots, final_ratings). `final_ratings` (team_id -> Elo)
    is handy for showing "current strength" on a dashboard without
    re-running the whole computation.
    """
    ratings: dict[int, float] = {}
    snapshots: list[EloSnapshot] = []

    for match in matches:
        elo_home = ratings.get(match.home_team_id, STARTING_ELO)
        elo_away = ratings.get(match.away_team_id, STARTING_ELO)

        expected_home = _expected_score(elo_home + HOME_ADVANTAGE, elo_away)
        actual_home = _actual_score(match.home_score, match.away_score)

        elo_home_post = elo_home + K_FACTOR * (actual_home - expected_home)
        # Away team's actual/expected are the mirror image of home's.
        elo_away_post = elo_away + K_FACTOR * ((1 - actual_home) - (1 - expected_home))

        snapshots.append(
            EloSnapshot(
                match_id=match.match_id,
                elo_home_pre=elo_home,
                elo_away_pre=elo_away,
                elo_home_post=elo_home_post,
                elo_away_post=elo_away_post,
            )
        )

        ratings[match.home_team_id] = elo_home_post
        ratings[match.away_team_id] = elo_away_post

    return snapshots, ratings