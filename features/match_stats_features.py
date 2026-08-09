"""
Rolling averages for shots, possession, and corners — same window-based
pattern as goals.py, but with one important difference: the raw data
comes from match_stats, which is still being backfilled daily (see
match_stats_loader.py). Matches without stats yet (or matches from the
2025-26 season, which the free API plan doesn't cover at all) must still
get a valid pre-match feature snapshot — just computed from whatever
history exists so far, without that particular match contributing to
anyone's rolling history.

This is why raw_stats_by_match_id.get(match.match_id) can legitimately
be None here, and why the update step is skipped (not defaulted to zero)
when it is.
"""

from collections import deque
from dataclasses import dataclass

from features.elo import MatchResult

STATS_WINDOW = 10


@dataclass
class RawMatchStats:
    home_shots_total: float | None
    away_shots_total: float | None
    home_possession_pct: float | None
    away_possession_pct: float | None
    home_corners: float | None
    away_corners: float | None


@dataclass
class MatchStatsFeatureSnapshot:
    match_id: int
    home_shots_avg_pre: float | None
    away_shots_avg_pre: float | None
    home_possession_avg_pre: float | None
    away_possession_avg_pre: float | None
    home_corners_avg_pre: float | None
    away_corners_avg_pre: float | None


def compute_match_stats_features(
    matches: list[MatchResult],
    raw_stats_by_match_id: dict[int, RawMatchStats],
    window: int = STATS_WINDOW,
) -> list[MatchStatsFeatureSnapshot]:
    """`matches` MUST be sorted chronologically, same as every other
    feature module here. `raw_stats_by_match_id` need not have an entry
    for every match — missing entries (not yet fetched, or an
    uncovered season) simply don't update the rolling history."""
    shots_hist: dict[int, deque[float]] = {}
    possession_hist: dict[int, deque[float]] = {}
    corners_hist: dict[int, deque[float]] = {}
    snapshots: list[MatchStatsFeatureSnapshot] = []

    def avg_or_none(history: deque[float]) -> float | None:
        return (sum(history) / len(history)) if history else None

    for match in matches:
        home_shots_h = shots_hist.setdefault(match.home_team_id, deque(maxlen=window))
        away_shots_h = shots_hist.setdefault(match.away_team_id, deque(maxlen=window))
        home_poss_h = possession_hist.setdefault(match.home_team_id, deque(maxlen=window))
        away_poss_h = possession_hist.setdefault(match.away_team_id, deque(maxlen=window))
        home_corn_h = corners_hist.setdefault(match.home_team_id, deque(maxlen=window))
        away_corn_h = corners_hist.setdefault(match.away_team_id, deque(maxlen=window))

        snapshots.append(
            MatchStatsFeatureSnapshot(
                match_id=match.match_id,
                home_shots_avg_pre=avg_or_none(home_shots_h),
                away_shots_avg_pre=avg_or_none(away_shots_h),
                home_possession_avg_pre=avg_or_none(home_poss_h),
                away_possession_avg_pre=avg_or_none(away_poss_h),
                home_corners_avg_pre=avg_or_none(home_corn_h),
                away_corners_avg_pre=avg_or_none(away_corn_h),
            )
        )

        raw = raw_stats_by_match_id.get(match.match_id)
        if raw is not None:
            if raw.home_shots_total is not None:
                home_shots_h.append(raw.home_shots_total)
            if raw.away_shots_total is not None:
                away_shots_h.append(raw.away_shots_total)
            if raw.home_possession_pct is not None:
                home_poss_h.append(raw.home_possession_pct)
            if raw.away_possession_pct is not None:
                away_poss_h.append(raw.away_possession_pct)
            if raw.home_corners is not None:
                home_corn_h.append(raw.home_corners)
            if raw.away_corners is not None:
                away_corn_h.append(raw.away_corners)

    return snapshots