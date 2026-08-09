"""
League position feature: each team's standings rank as of just before
this match, computed from that SEASON's results only (unlike every other
feature so far, this one resets at season boundaries rather than
flowing continuously — a team's Elo carries over from last season, but
their league position obviously starts fresh).

Depends on MatchResult carrying `season` (see setup guide).

Tiebreak simplification, stated plainly: official Premier League
tiebreakers use head-to-head results when teams are level on points and
goal difference. This uses the more common simplified sort (points →
goal difference → goals scored) that ignores head-to-head — it can place
two level teams in a different order than the real table by a spot or
two. Not fixed here deliberately: the complexity of replicating official
tiebreak rules isn't worth it for a feature whose job is "is this team
top-half or bottom-half," not "reproduce the exact table."
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby

from features.elo import MatchResult


@dataclass
class LeaguePositionSnapshot:
    match_id: int
    home_position_pre: int | None  # None = team's first match of the season
    away_position_pre: int | None


def _points_earned(team_score: int, opponent_score: int) -> int:
    if team_score > opponent_score:
        return 3
    if team_score == opponent_score:
        return 1
    return 0


def compute_league_position_features(
    matches: list[MatchResult],
) -> list[LeaguePositionSnapshot]:
    """`matches` MUST be sorted chronologically overall — grouping by
    season below preserves that relative order within each group, it
    doesn't re-sort."""
    matches_by_season: dict[str, list[MatchResult]] = defaultdict(list)
    for match in matches:
        matches_by_season[match.season].append(match)

    snapshots_by_match_id: dict[int, LeaguePositionSnapshot] = {}

    for season_matches in matches_by_season.values():
        # Fresh table every season — this dict is intentionally scoped
        # inside the loop, not shared across seasons.
        table: dict[int, dict[str, int]] = {}

        # Group matches by exact kickoff time, not one-at-a-time. This
        # matters because a full matchday's worth of fixtures often kicks
        # off simultaneously — most obviously the final matchday, where
        # every match starts at the same time by design (no team should
        # have an information advantage). Processing them one-at-a-time
        # in an arbitrary tiebreak order (since SQL doesn't guarantee a
        # stable order among equal timestamps) would let one same-round
        # match's result leak into another same-round match's "pre-match"
        # snapshot, even though in reality neither had happened yet
        # relative to the other. groupby requires equal keys to be
        # contiguous, which sorting by kickoff_utc alone already
        # guarantees — it doesn't need a second sort pass here.
        for _kickoff_time, group_iter in groupby(
            season_matches, key=lambda m: m.kickoff_utc
        ):
            group = list(group_iter)

            ranked_team_ids = sorted(
                (team_id for team_id, s in table.items() if s["played"] > 0),
                key=lambda team_id: (
                    -table[team_id]["points"],
                    -(table[team_id]["gf"] - table[team_id]["ga"]),
                    -table[team_id]["gf"],
                ),
            )
            position_by_team_id = {
                team_id: rank + 1 for rank, team_id in enumerate(ranked_team_ids)
            }

            # Every match in this group gets a snapshot from the SAME
            # pre-round table state — this is the actual fix.
            for match in group:
                snapshots_by_match_id[match.match_id] = LeaguePositionSnapshot(
                    match_id=match.match_id,
                    home_position_pre=position_by_team_id.get(match.home_team_id),
                    away_position_pre=position_by_team_id.get(match.away_team_id),
                )

            # Only now, after every match in the group has its snapshot,
            # apply all of their results to the table.
            for match in group:
                home_stats = table.setdefault(
                    match.home_team_id, {"points": 0, "gf": 0, "ga": 0, "played": 0}
                )
                away_stats = table.setdefault(
                    match.away_team_id, {"points": 0, "gf": 0, "ga": 0, "played": 0}
                )

                home_stats["points"] += _points_earned(match.home_score, match.away_score)
                home_stats["gf"] += match.home_score
                home_stats["ga"] += match.away_score
                home_stats["played"] += 1

                away_stats["points"] += _points_earned(match.away_score, match.home_score)
                away_stats["gf"] += match.away_score
                away_stats["ga"] += match.home_score
                away_stats["played"] += 1

    # Reassemble in the original chronological order the caller passed in.
    return [snapshots_by_match_id[match.match_id] for match in matches]