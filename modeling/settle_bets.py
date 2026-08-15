"""
Settles PENDING bets whose match has now finished — compares the bet's
outcome against the real result and computes profit/loss.

Usage:
    python -m modeling.settle_bets
"""

import datetime

from database.models import Bet, BetStatus, Match, MatchStatus
from database.session import SessionLocal


def _actual_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def main():
    session = SessionLocal()
    try:
        pending_bets = session.query(Bet).filter(Bet.status == BetStatus.PENDING).all()

        if not pending_bets:
            print("No pending bets to settle.")
            return

        settled_count = 0
        for bet in pending_bets:
            match = session.get(Match, bet.match_id)

            if match.status == MatchStatus.POSTPONED or match.status == MatchStatus.CANCELLED:
                bet.status = BetStatus.VOID
                bet.profit_loss = 0.0
                bet.settled_at = datetime.datetime.now(datetime.timezone.utc)
                settled_count += 1
                print(f"Bet #{bet.id}: VOID (match {match.status.value})")
                continue

            if match.status != MatchStatus.FINISHED or match.home_score is None:
                continue  # not played yet, leave pending

            actual = _actual_result(match.home_score, match.away_score)
            if actual == bet.outcome:
                bet.status = BetStatus.WON
                bet.profit_loss = bet.stake * (bet.odds_taken - 1)
            else:
                bet.status = BetStatus.LOST
                bet.profit_loss = -bet.stake

            bet.settled_at = datetime.datetime.now(datetime.timezone.utc)
            settled_count += 1
            print(f"Bet #{bet.id}: {bet.status.value.upper()}, P/L = {bet.profit_loss:+.2f}")

        session.commit()
        print(f"\nSettled {settled_count} bet(s). "
              f"{len(pending_bets) - settled_count} still pending (match not finished yet).")
    finally:
        session.close()


if __name__ == "__main__":
    main()