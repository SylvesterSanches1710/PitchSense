"""
Injury/suspension counts. Structurally different from every other module
in this package: no chronological walk, no rolling window, no state
carried between matches. API-Football's injuries endpoint already scopes
each record to a specific fixture — "who's ruled out for THIS match" —
so there's nothing to compute "as of before this match"; the data is
already there.

Splits into two counts using the free-text `status` field on each Injury
row: anything containing "suspen" counts as a suspension, everything
else as an injury. This is a simple heuristic, not a guaranteed-accurate
classification against every possible phrasing API-Football might use —
worth spot-checking a handful of records if a count looks surprising.
"""

from dataclasses import dataclass


@dataclass
class MatchInjuryInput:
    match_id: int
    home_team_id: int
    away_team_id: int
    injuries_fetched: bool


@dataclass
class InjurySnapshot:
    match_id: int
    home_injuries_count_pre: int | None
    away_injuries_count_pre: int | None
    home_suspensions_count_pre: int | None
    away_suspensions_count_pre: int | None


def compute_injury_features(
    matches: list[MatchInjuryInput],
    counts_by_match_team: dict[tuple[int, int], dict[str, int]],
) -> list[InjurySnapshot]:
    """
    `counts_by_match_team` keys are (match_id, team_id); values are
    {"injury": n, "suspension": n} from a GROUP BY over the Injury table.

    The null-vs-zero distinction matters here, same as everywhere else in
    this package: a match with injuries_fetched=False gets None (we
    haven't checked yet — NOT "assumed clean"). A fetched match with no
    entry in counts_by_match_team gets a real 0 (we checked, genuinely
    nothing to report).
    """
    snapshots = []
    for match in matches:
        if not match.injuries_fetched:
            snapshots.append(
                InjurySnapshot(
                    match_id=match.match_id,
                    home_injuries_count_pre=None,
                    away_injuries_count_pre=None,
                    home_suspensions_count_pre=None,
                    away_suspensions_count_pre=None,
                )
            )
            continue

        home_counts = counts_by_match_team.get((match.match_id, match.home_team_id), {})
        away_counts = counts_by_match_team.get((match.match_id, match.away_team_id), {})

        snapshots.append(
            InjurySnapshot(
                match_id=match.match_id,
                home_injuries_count_pre=home_counts.get("injury", 0),
                away_injuries_count_pre=away_counts.get("injury", 0),
                home_suspensions_count_pre=home_counts.get("suspension", 0),
                away_suspensions_count_pre=away_counts.get("suspension", 0),
            )
        )
    return snapshots