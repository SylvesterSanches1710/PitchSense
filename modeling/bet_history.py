"""
Prints betting history and running performance stats — this is the raw
data the Phase 5 dashboard's bankroll tracker and P/L graphs will
eventually visualize; for now, a readable text report.

Usage:
    python -m modeling.bet_history
"""

from database.models import Bet, BetStatus, Match, Team
from database.session import SessionLocal


def main():
    session = SessionLocal()
    try:
        bets = session.query(Bet).order_by(Bet.placed_at.asc()).all()

        if not bets:
            print("No bets logged yet.")
            return

        print(f"{'ID':<5}{'Match':<40}{'Outcome':<8}{'Stake':>8}{'Odds':>8}{'Status':<10}{'P/L':>10}")
        print("-" * 95)

        running_total = 0.0
        for bet in bets:
            match = session.get(Match, bet.match_id)
            home = session.get(Team, match.home_team_id).name
            away = session.get(Team, match.away_team_id).name
            matchup = f"{home} vs {away}"[:38]

            pl_str = f"{bet.profit_loss:+.2f}" if bet.profit_loss is not None else "-"
            if bet.profit_loss is not None:
                running_total += bet.profit_loss

            print(
                f"{bet.id:<5}{matchup:<40}{bet.outcome:<8}{bet.stake:>8.2f}"
                f"{bet.odds_taken:>8.2f}{bet.status.value:<10}{pl_str:>10}"
            )

        settled_bets = [b for b in bets if b.status != BetStatus.PENDING]
        won_bets = [b for b in bets if b.status == BetStatus.WON]
        total_staked = sum(b.stake for b in settled_bets if b.status != BetStatus.VOID)
        total_pl = sum(b.profit_loss for b in settled_bets if b.profit_loss is not None)

        print("-" * 95)
        print(f"\nTotal bets logged: {len(bets)}  "
              f"(Pending: {sum(1 for b in bets if b.status == BetStatus.PENDING)}, "
              f"Settled: {len(settled_bets)})")

        if settled_bets:
            win_rate = len(won_bets) / len([b for b in settled_bets if b.status != BetStatus.VOID]) \
                if any(b.status != BetStatus.VOID for b in settled_bets) else 0
            roi = (total_pl / total_staked * 100) if total_staked > 0 else 0
            print(f"Win rate (settled, excl. void): {win_rate:.1%}")
            print(f"Total staked: {total_staked:.2f}")
            print(f"Total P/L: {total_pl:+.2f}")
            print(f"ROI: {roi:+.1f}%")
            print(
                "\n(Small sample caveat: with only a handful of bets, win rate and ROI "
                "are noisy estimates, not reliable long-run performance figures — the "
                "usual statistical caution about small samples applies here just as much "
                "as it did to the model's own evaluation metrics earlier in this project.)"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()