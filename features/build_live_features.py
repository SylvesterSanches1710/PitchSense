"""
Computes pre-match features for UPCOMING matches, reusing every existing
feature-computation function completely unchanged.

The trick: every compute_*_features() function in this project takes a
snapshot of a match's pre-match state BEFORE using that match's own
score to update history for whatever comes after it. An upcoming match's
score is a harmless placeholder (0-0) here — it's never read before its
own snapshot is taken, because that's how every one of these functions
is written. Append the target match to the end of the full chronological
match list, run the same pipeline, and its snapshot comes out correct.

Usage:
    python -m features.build_live_features <match_id> [match_id ...]
"""

import sys
from collections import defaultdict

from database.models import Injury, Match, MatchFeature, MatchStats, MatchStatus, Team
from database.session import SessionLocal
from features.cards import RawCardStats, compute_cards_features
from features.elo import MatchResult, compute_elo_ratings
from features.form import compute_form_features
from features.goals import compute_goals_features
from features.head_to_head import compute_head_to_head_features
from features.injuries import MatchInjuryInput, compute_injury_features
from features.league_position import compute_league_position_features
from features.match_stats_features import RawMatchStats, compute_match_stats_features
from features.rest_days import compute_rest_days_features
from features.travel_distance import compute_travel_features
from features.venue_form import compute_venue_form_features


def build_combined_match_list(session, target_match_ids: list[int]) -> list[MatchResult]:
    finished = (
        session.query(Match)
        .filter(Match.status == MatchStatus.FINISHED)
        .filter(Match.home_score.isnot(None))
        .filter(Match.away_score.isnot(None))
        .order_by(Match.kickoff_utc.asc())
        .all()
    )
    targets = session.query(Match).filter(Match.id.in_(target_match_ids)).all()
    found_ids = {m.id for m in targets}
    missing = set(target_match_ids) - found_ids
    if missing:
        raise ValueError(f"Match ids not found in database: {missing}")

    # CRITICAL SAFETY CHECK: if the same team appears in more than one
    # target match, an earlier target's placeholder score would get
    # treated as a real result and contaminate a later target's history
    # (form/Elo/goals/position all wrongly updated from a match that
    # hasn't actually been played). This exact bug produced a nonsense
    # league position for a team's genuine season opener. Rather than
    # risk this silently recurring, refuse to proceed rather than
    # produce quietly-wrong features — run one match_id at a time
    # instead if you need features for multiple matches involving
    # overlapping teams.
    team_appearance_count: dict[int, int] = {}
    for m in targets:
        team_appearance_count[m.home_team_id] = team_appearance_count.get(m.home_team_id, 0) + 1
        team_appearance_count[m.away_team_id] = team_appearance_count.get(m.away_team_id, 0) + 1
    overlapping_teams = [team_id for team_id, count in team_appearance_count.items() if count > 1]
    if overlapping_teams:
        raise ValueError(
            f"Team id(s) {overlapping_teams} appear in more than one of the target "
            f"matches in this batch. Processing them together would let an earlier "
            f"target match's placeholder score contaminate a later one's computed "
            f"history. Run build_live_features with ONE match_id at a time instead — "
            f"e.g. call this script separately for each match rather than passing "
            f"multiple match IDs that share a team."
        )

    combined = sorted(list(finished) + list(targets), key=lambda m: m.kickoff_utc)

    return [
        MatchResult(
            match_id=m.id,
            home_team_id=m.home_team_id,
            away_team_id=m.away_team_id,
            home_score=m.home_score if m.home_score is not None else 0,
            away_score=m.away_score if m.away_score is not None else 0,
            kickoff_utc=m.kickoff_utc,
            season=m.season,
        )
        for m in combined
    ]


def load_raw_match_stats(session) -> dict[int, RawMatchStats]:
    rows = session.query(MatchStats).all()
    return {
        row.match_id: RawMatchStats(
            home_shots_total=row.home_shots_total, away_shots_total=row.away_shots_total,
            home_possession_pct=row.home_possession_pct, away_possession_pct=row.away_possession_pct,
            home_corners=row.home_corners, away_corners=row.away_corners,
        )
        for row in rows
    }


def load_raw_cards(session) -> dict[int, RawCardStats]:
    rows = session.query(MatchStats).all()
    return {
        row.match_id: RawCardStats(
            home_yellow_cards=row.home_yellow_cards, away_yellow_cards=row.away_yellow_cards,
            home_red_cards=row.home_red_cards, away_red_cards=row.away_red_cards,
        )
        for row in rows
    }


def load_injury_counts(session) -> dict[tuple[int, int], dict[str, int]]:
    counts = defaultdict(lambda: {"injury": 0, "suspension": 0})
    rows = session.query(Injury.match_id, Injury.team_id, Injury.status).filter(Injury.match_id.isnot(None))
    for match_id, team_id, status in rows:
        bucket = "suspension" if "suspen" in (status or "").lower() else "injury"
        counts[(match_id, team_id)][bucket] += 1
    return dict(counts)


def build_live_features(match_ids: list[int]) -> None:
    session = SessionLocal()
    try:
        matches = build_combined_match_list(session, match_ids)
        target_ids = set(match_ids)

        raw_stats = load_raw_match_stats(session)
        raw_cards = load_raw_cards(session)
        injury_counts = load_injury_counts(session)
        injury_fetch_status = dict(session.query(Match.id, Match.injuries_fetched_at))
        team_name_by_id = {t.id: t.name for t in session.query(Team).all()}

        match_injury_inputs = [
            MatchInjuryInput(
                match_id=m.match_id, home_team_id=m.home_team_id, away_team_id=m.away_team_id,
                injuries_fetched=injury_fetch_status.get(m.match_id) is not None,
            )
            for m in matches
        ]
        match_team_names = [
            (m.match_id, team_name_by_id[m.home_team_id], team_name_by_id[m.away_team_id])
            for m in matches
        ]

        print(f"Running all 9 feature pipelines across {len(matches)} matches "
              f"({len(matches) - len(target_ids)} historical + {len(target_ids)} upcoming)...")

        elo_snapshots, _ = compute_elo_ratings(matches)
        form_snapshots = compute_form_features(matches)
        venue_form_snapshots = compute_venue_form_features(matches)
        goals_snapshots = compute_goals_features(matches)
        h2h_snapshots = compute_head_to_head_features(matches)
        rest_days_snapshots = compute_rest_days_features(matches)
        match_stats_snapshots = compute_match_stats_features(matches, raw_stats)
        league_position_snapshots = compute_league_position_features(matches)
        cards_snapshots = compute_cards_features(matches, raw_cards)
        injury_snapshots = compute_injury_features(match_injury_inputs, injury_counts)
        travel_snapshots = compute_travel_features(match_team_names)

        def by_id(snaps):
            return {s.match_id: s for s in snaps}

        elo_d, form_d, venue_d = by_id(elo_snapshots), by_id(form_snapshots), by_id(venue_form_snapshots)
        goals_d, h2h_d, rest_d = by_id(goals_snapshots), by_id(h2h_snapshots), by_id(rest_days_snapshots)
        stats_d, pos_d, cards_d = by_id(match_stats_snapshots), by_id(league_position_snapshots), by_id(cards_snapshots)
        inj_d, travel_d = by_id(injury_snapshots), by_id(travel_snapshots)

        written = 0
        for mid in target_ids:
            feature_row = session.query(MatchFeature).filter_by(match_id=mid).first()
            if feature_row is None:
                feature_row = MatchFeature(match_id=mid)
                session.add(feature_row)

            e, f, v, g, h = elo_d[mid], form_d[mid], venue_d[mid], goals_d[mid], h2h_d[mid]
            r, s, p, c, i, t = rest_d[mid], stats_d[mid], pos_d[mid], cards_d[mid], inj_d[mid], travel_d[mid]

            feature_row.elo_home_pre, feature_row.elo_away_pre = e.elo_home_pre, e.elo_away_pre
            feature_row.form_home_pre, feature_row.form_away_pre = f.form_home_pre, f.form_away_pre
            feature_row.home_venue_form_pre = v.home_venue_form_pre
            feature_row.away_venue_form_pre = v.away_venue_form_pre
            feature_row.home_goals_scored_avg_pre = g.home_goals_scored_avg_pre
            feature_row.home_goals_conceded_avg_pre = g.home_goals_conceded_avg_pre
            feature_row.away_goals_scored_avg_pre = g.away_goals_scored_avg_pre
            feature_row.away_goals_conceded_avg_pre = g.away_goals_conceded_avg_pre
            feature_row.h2h_home_ppg_pre = h.h2h_home_ppg_pre
            feature_row.h2h_away_ppg_pre = h.h2h_away_ppg_pre
            feature_row.h2h_meetings_pre = h.h2h_meetings_pre
            feature_row.home_rest_days_pre, feature_row.away_rest_days_pre = r.home_rest_days_pre, r.away_rest_days_pre
            feature_row.home_shots_avg_pre = s.home_shots_avg_pre
            feature_row.away_shots_avg_pre = s.away_shots_avg_pre
            feature_row.home_possession_avg_pre = s.home_possession_avg_pre
            feature_row.away_possession_avg_pre = s.away_possession_avg_pre
            feature_row.home_corners_avg_pre = s.home_corners_avg_pre
            feature_row.away_corners_avg_pre = s.away_corners_avg_pre
            feature_row.home_position_pre, feature_row.away_position_pre = p.home_position_pre, p.away_position_pre
            feature_row.home_yellow_cards_avg_pre = c.home_yellow_cards_avg_pre
            feature_row.away_yellow_cards_avg_pre = c.away_yellow_cards_avg_pre
            feature_row.home_red_cards_avg_pre = c.home_red_cards_avg_pre
            feature_row.away_red_cards_avg_pre = c.away_red_cards_avg_pre
            feature_row.home_injuries_count_pre = i.home_injuries_count_pre
            feature_row.away_injuries_count_pre = i.away_injuries_count_pre
            feature_row.home_suspensions_count_pre = i.home_suspensions_count_pre
            feature_row.away_suspensions_count_pre = i.away_suspensions_count_pre
            feature_row.away_travel_km_pre = t.away_travel_km_pre

            written += 1

        session.commit()
        print(f"Wrote live pre-match features for {written} upcoming matches.")
    finally:
        session.close()


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]]
    if not ids:
        print("Usage: python -m features.build_live_features <match_id> [match_id ...]")
    else:
        build_live_features(ids)