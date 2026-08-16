"""Pydantic response models for bet history / bankroll data."""

from pydantic import BaseModel


class BetRecord(BaseModel):
    id: int
    home_team: str
    away_team: str
    kickoff_utc: str
    outcome: str
    stake: float
    odds_taken: float
    status: str
    profit_loss: float | None
    placed_at: str


class BetSummary(BaseModel):
    total_bets: int
    pending: int
    settled: int
    won: int
    win_rate: float | None
    total_staked: float
    total_profit_loss: float
    roi_pct: float | None


class CumulativePLPoint(BaseModel):
    bet_id: int
    settled_at: str
    cumulative_pl: float


class BetHistory(BaseModel):
    bets: list[BetRecord]
    summary: BetSummary
    cumulative_pl: list[CumulativePLPoint]