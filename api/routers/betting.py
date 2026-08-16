"""GET /betting/history — bet log, summary stats, and cumulative P/L
for the dashboard's bankroll page."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.session import get_session
from modeling.bet_history_data import get_bet_history

from ..schemas.betting import BetHistory

router = APIRouter()


@router.get("/history", response_model=BetHistory)
def get_betting_history(db: Session = Depends(get_session)):
    return get_bet_history(db)