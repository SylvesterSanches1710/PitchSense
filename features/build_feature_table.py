"""
Computes engineered features across all finished matches (in chronological
order) and writes the pre-match values into match_features.

Usage:
    python -m features.build_feature_table
"""

from database.models import Match, MatchFeature, MatchStatus, MatchStats, Team
from database.session import SessionLocal
from features.elo import MatchResult, compute_elo_ratings
from features.form import compute_form_features
from features.venue_form import compute_venue_form_features
from features.goals import compute_goals_features
from features.head_to_head import compute_head_to_head_features
from features.rest_days import compute_rest_days_features
from features.match_stats_features import RawMatchStats, compute_match_stats_features
from features.league_position import compute_league_position_features
from collections import defaultdict
from database.models import Injury
from features.injuries import MatchInjuryInput, compute_injury_features
from features.cards import RawCardStats, compute_cards_features
from features.travel_distance import compute_travel_features


def load_finished_matches_chronological(session) -> list[MatchResult]:
    matches = (
        session.query(Match)
        .filter(Match.status == MatchStatus.FINISHED)
        .filter(Match.home_score.isnot(None))
        .filter(Match.away_score.isnot(None))
        .order_by(Match.kickoff_utc.asc())
        .all()
    )
    return [
        MatchResult(
            match_id=m.id,
            home_team_id=m.home_team_id,
            away_team_id=m.away_team_id,
            home_score=m.home_score,
            away_score=m.away_score,
            kickoff_utc=m.kickoff_utc,
            season=m.season,
        )
        for m in matches
    ]


def load_raw_match_stats(session) -> dict[int, RawMatchStats]:
    rows = session.query(MatchStats).all()
    return {
        row.match_id: RawMatchStats(
            home_shots_total=row.home_shots_total,
            away_shots_total=row.away_shots_total,
            home_possession_pct=row.home_possession_pct,
            away_possession_pct=row.away_possession_pct,
            home_corners=row.home_corners,
            away_corners=row.away_corners,
        )
        for row in rows
    }


def load_raw_cards(session) -> dict[int, RawCardStats]:
    rows = session.query(MatchStats).all()
    return {
        row.match_id: RawCardStats(
            home_yellow_cards=row.home_yellow_cards,
            away_yellow_cards=row.away_yellow_cards,
            home_red_cards=row.home_red_cards,
            away_red_cards=row.away_red_cards,
        )
        for row in rows
    }


def load_match_team_names(session, matches) -> list[tuple[int, str, str]]:
    team_name_by_id = {team.id: team.name for team in session.query(Team).all()}
    return [
        (m.match_id, team_name_by_id[m.home_team_id], team_name_by_id[m.away_team_id])
        for m in matches
    ]


def load_injury_counts(session) -> dict[tuple[int, int], dict[str, int]]:
    counts: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"injury": 0, "suspension": 0}
    )
    rows = session.query(Injury.match_id, Injury.team_id, Injury.status).filter(
        Injury.match_id.isnot(None)
    )
    for match_id, team_id, status in rows:
        bucket = "suspension" if "suspen" in (status or "").lower() else "injury"
        counts[(match_id, team_id)][bucket] += 1
    return dict(counts)


def upsert_features(
    session,
    elo_snapshots,
    form_snapshots,
    venue_form_snapshots,
    goals_snapshots,
    h2h_snapshots,
    rest_days_snapshots,
    match_stats_snapshots,
    league_position_snapshots,
    injury_snapshots,
    cards_snapshots,
    travel_features,
) -> int:
    form_by_match_id = {s.match_id: s for s in form_snapshots}
    venue_form_by_match_id = {s.match_id: s for s in venue_form_snapshots}
    goals_by_match_id = {s.match_id: s for s in goals_snapshots}
    h2h_by_match_id = {s.match_id: s for s in h2h_snapshots}
    rest_days_by_match_id = {s.match_id: s for s in rest_days_snapshots}
    match_stats_by_match_id = {s.match_id: s for s in match_stats_snapshots}
    league_position_by_match_id = {s.match_id: s for s in league_position_snapshots}
    injury_by_match_id = {s.match_id: s for s in injury_snapshots}
    cards_by_match_id = {s.match_id: s for s in cards_snapshots}
    travel_by_match_id = {s.match_id: s for s in travel_features}

    updated_count = 0

    for elo_snapshot in elo_snapshots:
        feature_row = (
            session.query(MatchFeature)
            .filter_by(match_id=elo_snapshot.match_id)
            .first()
        )

        if feature_row is None:
            feature_row = MatchFeature(match_id=elo_snapshot.match_id)
            session.add(feature_row)

        # Elo
        feature_row.elo_home_pre = elo_snapshot.elo_home_pre
        feature_row.elo_away_pre = elo_snapshot.elo_away_pre

        # Overall form
        form_snapshot = form_by_match_id[elo_snapshot.match_id]
        feature_row.form_home_pre = form_snapshot.form_home_pre
        feature_row.form_away_pre = form_snapshot.form_away_pre

        # Venue-specific form
        venue_snapshot = venue_form_by_match_id[elo_snapshot.match_id]
        feature_row.home_venue_form_pre = venue_snapshot.home_venue_form_pre
        feature_row.away_venue_form_pre = venue_snapshot.away_venue_form_pre

        # Goals
        goals_snapshot = goals_by_match_id[elo_snapshot.match_id]
        feature_row.home_goals_scored_avg_pre = goals_snapshot.home_goals_scored_avg_pre
        feature_row.home_goals_conceded_avg_pre = (
            goals_snapshot.home_goals_conceded_avg_pre
        )
        feature_row.away_goals_scored_avg_pre = goals_snapshot.away_goals_scored_avg_pre
        feature_row.away_goals_conceded_avg_pre = (
            goals_snapshot.away_goals_conceded_avg_pre
        )

        # Head-to-head
        h2h_snapshot = h2h_by_match_id[elo_snapshot.match_id]

        feature_row.h2h_home_ppg_pre = h2h_snapshot.h2h_home_ppg_pre
        feature_row.h2h_away_ppg_pre = h2h_snapshot.h2h_away_ppg_pre
        feature_row.h2h_meetings_pre = h2h_snapshot.h2h_meetings_pre

        # Rest days
        rest_days_snapshot = rest_days_by_match_id[elo_snapshot.match_id]
        feature_row.home_rest_days_pre = rest_days_snapshot.home_rest_days_pre
        feature_row.away_rest_days_pre = rest_days_snapshot.away_rest_days_pre

        # Rolling shots / possession / corners
        match_stats_snapshot = match_stats_by_match_id[elo_snapshot.match_id]

        feature_row.home_shots_avg_pre = match_stats_snapshot.home_shots_avg_pre
        feature_row.away_shots_avg_pre = match_stats_snapshot.away_shots_avg_pre
        feature_row.home_possession_avg_pre = (
            match_stats_snapshot.home_possession_avg_pre
        )
        feature_row.away_possession_avg_pre = (
            match_stats_snapshot.away_possession_avg_pre
        )
        feature_row.home_corners_avg_pre = match_stats_snapshot.home_corners_avg_pre
        feature_row.away_corners_avg_pre = match_stats_snapshot.away_corners_avg_pre

        # League position
        league_position_snapshot = league_position_by_match_id[elo_snapshot.match_id]

        feature_row.home_position_pre = league_position_snapshot.home_position_pre
        feature_row.away_position_pre = league_position_snapshot.away_position_pre

        # Injuries / suspensions
        injury_snapshot = injury_by_match_id[elo_snapshot.match_id]

        feature_row.home_injuries_count_pre = injury_snapshot.home_injuries_count_pre
        feature_row.away_injuries_count_pre = injury_snapshot.away_injuries_count_pre
        feature_row.home_suspensions_count_pre = (
            injury_snapshot.home_suspensions_count_pre
        )
        feature_row.away_suspensions_count_pre = (
            injury_snapshot.away_suspensions_count_pre
        )

        # Cards
        cards_snapshot = cards_by_match_id[elo_snapshot.match_id]

        feature_row.home_yellow_cards_avg_pre = cards_snapshot.home_yellow_cards_avg_pre
        feature_row.away_yellow_cards_avg_pre = cards_snapshot.away_yellow_cards_avg_pre
        feature_row.home_red_cards_avg_pre = cards_snapshot.home_red_cards_avg_pre
        feature_row.away_red_cards_avg_pre = cards_snapshot.away_red_cards_avg_pre

        # Travel distance
        travel_snapshot = travel_by_match_id[elo_snapshot.match_id]
        feature_row.away_travel_km_pre = travel_snapshot.away_travel_km_pre

        updated_count += 1

    session.commit()
    return updated_count


def main():
    session = SessionLocal()
    try:
        matches = load_finished_matches_chronological(session)
        print(f"Computing features across {len(matches)} finished matches...")

        raw_stats_by_match_id = load_raw_match_stats(session)
        raw_cards_by_match_id = load_raw_cards(session)

        injury_fetch_status = {
            match_id: injuries_fetched_at
            for match_id, injuries_fetched_at in session.query(
                Match.id,
                Match.injuries_fetched_at,
            )
        }

        match_injury_inputs = [
            MatchInjuryInput(
                match_id=m.match_id,
                home_team_id=m.home_team_id,
                away_team_id=m.away_team_id,
                injuries_fetched=injury_fetch_status.get(m.match_id) is not None,
            )
            for m in matches
        ]

        injury_counts = load_injury_counts(session)

        injury_snapshots = compute_injury_features(
            match_injury_inputs,
            injury_counts,
        )

        elo_snapshots, final_ratings = compute_elo_ratings(matches)
        form_snapshots = compute_form_features(matches)
        venue_form_snapshots = compute_venue_form_features(matches)
        goals_snapshots = compute_goals_features(matches)
        h2h_snapshots = compute_head_to_head_features(matches)
        rest_days_snapshots = compute_rest_days_features(matches)
        match_stats_snapshots = compute_match_stats_features(
            matches,
            raw_stats_by_match_id,
        )
        cards_snapshots = compute_cards_features(
            matches,
            raw_cards_by_match_id,
        )
        league_position_snapshots = compute_league_position_features(matches)
        match_team_names = load_match_team_names(session, matches)
        travel_features = compute_travel_features(match_team_names)

        updated_count = upsert_features(
            session,
            elo_snapshots,
            form_snapshots,
            venue_form_snapshots,
            goals_snapshots,
            h2h_snapshots,
            rest_days_snapshots,
            match_stats_snapshots,
            league_position_snapshots,
            injury_snapshots,
            cards_snapshots,
            travel_features,
        )

        print(f"Wrote features for {updated_count} matches.")

        top_teams = sorted(
            final_ratings.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]

        print("\nCurrent top 5 Elo ratings:")
        for team_id, rating in top_teams:
            print(f"  team_id={team_id}: {rating:.1f}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
