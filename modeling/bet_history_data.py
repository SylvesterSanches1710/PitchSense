"""
Shared bet history data logic — used by modeling/bet_history.py (the CLI
report) and api/routers/betting.py (the dashboard backend). Same
one-source-of-truth principle as betting_math.py.
"""

from database.models import Bet, BetStatus, Match, Team


def get_bet_history(session) -> dict:
    bets = session.query(Bet).order_by(Bet.placed_at.asc()).all()

    bet_records = []
    for bet in bets:
        match = session.get(Match, bet.match_id)
        home = session.get(Team, match.home_team_id)
        away = session.get(Team, match.away_team_id)
        bet_records.append({
            "id": bet.id,
            "home_team": home.name,
            "away_team": away.name,
            "kickoff_utc": match.kickoff_utc.isoformat(),
            "outcome": bet.outcome,
            "stake": bet.stake,
            "odds_taken": bet.odds_taken,
            "status": bet.status.value,
            "profit_loss": bet.profit_loss,
            "placed_at": bet.placed_at.isoformat(),
        })

    settled_bets = [b for b in bets if b.status != BetStatus.PENDING]
    won_bets = [b for b in bets if b.status == BetStatus.WON]
    decided_bets = [b for b in settled_bets if b.status != BetStatus.VOID]  # excludes void from win-rate denominator
    total_staked = sum(b.stake for b in decided_bets)
    total_pl = sum(b.profit_loss for b in settled_bets if b.profit_loss is not None)

    win_rate = (len(won_bets) / len(decided_bets)) if decided_bets else None
    roi_pct = (total_pl / total_staked * 100) if total_staked > 0 else None

    # Cumulative P/L over time — only settled bets, in the order they
    # were settled, each point building on the last. This is what the
    # dashboard's P/L chart plots.
    cumulative_pl = []
    running_total = 0.0
    for bet in sorted(settled_bets, key=lambda b: b.settled_at):
        running_total += bet.profit_loss or 0.0
        cumulative_pl.append({
            "bet_id": bet.id,
            "settled_at": bet.settled_at.isoformat(),
            "cumulative_pl": round(running_total, 2),
        })

    return {
        "bets": bet_records,
        "summary": {
            "total_bets": len(bets),
            "pending": sum(1 for b in bets if b.status == BetStatus.PENDING),
            "settled": len(settled_bets),
            "won": len(won_bets),
            "win_rate": win_rate,
            "total_staked": round(total_staked, 2),
            "total_profit_loss": round(total_pl, 2),
            "roi_pct": round(roi_pct, 2) if roi_pct is not None else None,
        },
        "cumulative_pl": cumulative_pl,
    }