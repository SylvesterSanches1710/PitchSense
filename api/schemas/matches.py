"""
Pydantic response models — these define exactly what shape of JSON the
API returns, and FastAPI uses them to auto-generate the /docs page and
validate responses. This is worth having even though it's more typing
upfront: without it, a typo or missing field in an endpoint would only
surface as a confusing bug in the React frontend later, instead of
failing loudly and immediately here.
"""

import datetime

from pydantic import BaseModel


class OutcomeAnalysis(BaseModel):
    outcome: str  # "H", "D", or "A"
    label: str  # "Home Win", "Draw", "Away Win"
    model_probability: float
    stake_fair_probability: float | None
    stake_odds: float | None
    ev: float | None
    kelly_stake_fraction: float | None
    is_positive_ev: bool


class UpcomingMatchPrediction(BaseModel):
    match_id: int
    kickoff_utc: datetime.datetime
    home_team: str
    away_team: str
    confidence_label: str
    confidence_margin: float
    stake_market_margin_pct: float | None
    low_data_warnings: list[str]
    outcomes: list[OutcomeAnalysis]