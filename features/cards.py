"""
Rolling average yellow/red cards per match, over each team's last N
matches — same pattern as match_stats_features.py (shots/possession/
corners), same raw source (match_stats), same graceful handling of
matches with no stats yet.
"""

from collections import deque
from dataclasses import dataclass

from features.elo import MatchResult

CARDS_WINDOW = 10


@dataclass
class RawCardStats:
    home_yellow_cards: float | None
    away_yellow_cards: float | None
    home_red_cards: float | None
    away_red_cards: float | None


@dataclass
class CardsFeatureSnapshot:
    match_id: int
    home_yellow_cards_avg_pre: float | None
    away_yellow_cards_avg_pre: float | None
    home_red_cards_avg_pre: float | None
    away_red_cards_avg_pre: float | None


def compute_cards_features(
    matches: list[MatchResult],
    raw_cards_by_match_id: dict[int, RawCardStats],
    window: int = CARDS_WINDOW,
) -> list[CardsFeatureSnapshot]:
    """`matches` MUST be sorted chronologically. `raw_cards_by_match_id`
    need not cover every match — same missing-data handling as
    match_stats_features.py: a match without a raw entry simply doesn't
    update the rolling history, it still gets a snapshot from whatever
    history exists so far."""
    yellow_hist: dict[int, deque[float]] = {}
    red_hist: dict[int, deque[float]] = {}
    snapshots: list[CardsFeatureSnapshot] = []

    def avg_or_none(history: deque[float]) -> float | None:
        return (sum(history) / len(history)) if history else None

    for match in matches:
        home_yellow_h = yellow_hist.setdefault(match.home_team_id, deque(maxlen=window))
        away_yellow_h = yellow_hist.setdefault(match.away_team_id, deque(maxlen=window))
        home_red_h = red_hist.setdefault(match.home_team_id, deque(maxlen=window))
        away_red_h = red_hist.setdefault(match.away_team_id, deque(maxlen=window))

        snapshots.append(
            CardsFeatureSnapshot(
                match_id=match.match_id,
                home_yellow_cards_avg_pre=avg_or_none(home_yellow_h),
                away_yellow_cards_avg_pre=avg_or_none(away_yellow_h),
                home_red_cards_avg_pre=avg_or_none(home_red_h),
                away_red_cards_avg_pre=avg_or_none(away_red_h),
            )
        )

        raw = raw_cards_by_match_id.get(match.match_id)
        if raw is not None:
            if raw.home_yellow_cards is not None:
                home_yellow_h.append(raw.home_yellow_cards)
            if raw.away_yellow_cards is not None:
                away_yellow_h.append(raw.away_yellow_cards)
            if raw.home_red_cards is not None:
                home_red_h.append(raw.home_red_cards)
            if raw.away_red_cards is not None:
                away_red_h.append(raw.away_red_cards)

    return snapshots