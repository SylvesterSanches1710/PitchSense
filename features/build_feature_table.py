"""
Computes engineered features across all finished matches (in chronological
order) and writes the pre-match values into match_features.

Usage:
    python -m features.build_feature_table
"""

from database.models import Match, MatchFeature, MatchStatus
from database.session import SessionLocal
from features.elo import MatchResult, compute_elo_ratings
from features.form import compute_form_features
from features.venue_form import compute_venue_form_features
from features.goals import compute_goals_features
from features.head_to_head import compute_head_to_head_features


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
        )
        for m in matches
    ]


def upsert_features(
    session,
    elo_snapshots,
    form_snapshots,
    venue_form_snapshots,
    goals_snapshots,
    h2h_snapshots,
) -> int:
    form_by_match_id = {s.match_id: s for s in form_snapshots}
    venue_form_by_match_id = {s.match_id: s for s in venue_form_snapshots}
    goals_by_match_id = {s.match_id: s for s in goals_snapshots}

    h2h_by_match_id = {s.match_id: s for s in h2h_snapshots}

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

        updated_count += 1

    session.commit()
    return updated_count


def main():
    session = SessionLocal()
    try:
        matches = load_finished_matches_chronological(session)
        print(f"Computing features across {len(matches)} finished matches...")

        elo_snapshots, final_ratings = compute_elo_ratings(matches)
        form_snapshots = compute_form_features(matches)
        venue_form_snapshots = compute_venue_form_features(matches)
        goals_snapshots = compute_goals_features(matches)
        h2h_snapshots = compute_head_to_head_features(matches)

        updated_count = upsert_features(
            session,
            elo_snapshots,
            form_snapshots,
            venue_form_snapshots,
            goals_snapshots,
            h2h_snapshots,
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
