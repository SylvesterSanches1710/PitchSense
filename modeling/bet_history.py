"""
Prints betting history and running performance stats.

Usage:
    python -m modeling.bet_history
"""

from database.session import SessionLocal
from modeling.bet_history_data import get_bet_history


def main():
    session = SessionLocal()
    try:
        data = get_bet_history(session)

        if not data["bets"]:
            print("No bets logged yet.")
            return

        print(
            f"{'ID':<5}{'Match':<40}{'Outcome':<8}"
            f"{'Stake':>8}{'Odds':>8}{'Status':<10}{'P/L':>10}"
        )
        print("-" * 95)

        for bet in data["bets"]:
            matchup = f"{bet['home_team']} vs {bet['away_team']}"[:38]

            pl_str = (
                f"{bet['profit_loss']:+.2f}"
                if bet["profit_loss"] is not None
                else "-"
            )

            print(
                f"{bet['id']:<5}{matchup:<40}{bet['outcome']:<8}"
                f"{bet['stake']:>8.2f}{bet['odds_taken']:>8.2f}"
                f"{bet['status']:<10}{pl_str:>10}"
            )

        summary = data["summary"]

        print("-" * 95)
        print(
            f"\nTotal bets logged: {summary['total_bets']}  "
            f"(Pending: {summary['pending']}, "
            f"Settled: {summary['settled']})"
        )

        if summary["settled"]:
            win_rate = summary["win_rate"]
            roi = summary["roi_pct"]

            print(
                f"Win rate (settled, excl. void): "
                f"{win_rate:.1%}"
                if win_rate is not None
                else "Win rate (settled, excl. void): -"
            )
            print(f"Total staked: {summary['total_staked']:.2f}")
            print(f"Total P/L: {summary['total_profit_loss']:+.2f}")
            print(
                f"ROI: {roi:+.1f}%"
                if roi is not None
                else "ROI: -"
            )

            print(
                "\n(Small sample caveat: with only a handful of bets, "
                "win rate and ROI are noisy estimates, not reliable "
                "long-run performance figures — the usual statistical "
                "caution about small samples applies here just as much "
                "as it did to the model's own evaluation metrics earlier "
                "in this project.)"
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()