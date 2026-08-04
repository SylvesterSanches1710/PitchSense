"""
Computes Elo ratings across all finished matches (in chronological order)
and writes the pre-match ratings into match_features.

Usage:
    python -m features.build_feature_table
"""

from database.models import Match, MatchFeature, MatchStatus
from database.session import SessionLocal
from features.elo import MatchResult, compute_elo_ratings


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


def upsert_elo_features(session, snapshots) -> int:
    updated_count = 0
    for snapshot in snapshots:
        feature_row = (
            session.query(MatchFeature)
            .filter_by(match_id=snapshot.match_id)
            .first()
        )
        if feature_row is None:
            feature_row = MatchFeature(match_id=snapshot.match_id)
            session.add(feature_row)

        feature_row.elo_home_pre = snapshot.elo_home_pre
        feature_row.elo_away_pre = snapshot.elo_away_pre
        updated_count += 1

    session.commit()
    return updated_count


def main():
    session = SessionLocal()
    try:
        matches = load_finished_matches_chronological(session)
        print(f"Computing Elo across {len(matches)} finished matches...")

        snapshots, final_ratings = compute_elo_ratings(matches)
        updated_count = upsert_elo_features(session, snapshots)

        print(f"Wrote Elo features for {updated_count} matches.")

        # Quick sanity check: show the 5 highest-rated teams right now.
        top_teams = sorted(final_ratings.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("\nCurrent top 5 Elo ratings:")
        for team_id, rating in top_teams:
            print(f"  team_id={team_id}: {rating:.1f}")
    finally:
        session.close()


if __name__ == "__main__":
    main()