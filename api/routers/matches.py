"""
GET /matches/upcoming — the core endpoint: every upcoming match with
features computed, the model's prediction, and (where Stake odds have
been logged) the full EV/Kelly/low-data-warning analysis. This is the
same calculation betting_analysis.py prints to a terminal, returned as
structured JSON instead.
"""

import json
from pathlib import Path

from catboost import CatBoostClassifier
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.models import Match, MatchFeature, MatchStatus, Odds, Team
from database.session import get_session
from modeling.betting_math import (
    calculate_ev,
    check_low_data_warning,
    confidence_label,
    kelly_stake_fraction,
    remove_vig,
    CLASS_TO_LABEL,
    KELLY_FRACTION,
)
from modeling.dataset import FEATURE_COLUMNS

from ..schemas.matches import OutcomeAnalysis, UpcomingMatchPrediction

router = APIRouter()

MODEL_PATH = Path("modeling/model_registry/catboost_v1.cbm")
METADATA_PATH = Path("modeling/model_registry/catboost_v1_metadata.json")

_model: CatBoostClassifier | None = None
_class_order: list[str] | None = None


def _get_model() -> tuple[CatBoostClassifier, list[str]]:
    """Loads the model once and reuses it across requests — reloading a
    model file from disk on every single API call would be needlessly
    slow, since the model itself doesn't change between requests."""
    global _model, _class_order
    if _model is None:
        _model = CatBoostClassifier()
        _model.load_model(str(MODEL_PATH))
        _class_order = json.loads(METADATA_PATH.read_text())["class_order"]
    return _model, _class_order


@router.get("/upcoming", response_model=list[UpcomingMatchPrediction])
def get_upcoming_matches(db: Session = Depends(get_session)):
    model, class_order = _get_model()

    upcoming = (
        db.query(Match)
        .filter(Match.status == MatchStatus.SCHEDULED)
        .join(MatchFeature, MatchFeature.match_id == Match.id)
        .order_by(Match.kickoff_utc.asc())
        .all()
    )

    results = []
    for match in upcoming:
        feature_row = db.query(MatchFeature).filter_by(match_id=match.id).first()
        home_team = db.get(Team, match.home_team_id)
        away_team = db.get(Team, match.away_team_id)

        feature_values = [[getattr(feature_row, col) for col in FEATURE_COLUMNS]]
        proba = model.predict_proba(feature_values)[0]
        model_probs = {label: float(proba[i]) for i, label in enumerate(class_order)}

        label, margin = confidence_label(model_probs)

        warnings = []
        home_warning = check_low_data_warning(home_team.name, feature_row.elo_home_pre, feature_row.form_home_pre)
        away_warning = check_low_data_warning(away_team.name, feature_row.elo_away_pre, feature_row.form_away_pre)
        if home_warning:
            warnings.append(home_warning)
        if away_warning:
            warnings.append(away_warning)

        stake_odds_row = (
            db.query(Odds)
            .filter_by(match_id=match.id, bookmaker="Stake", market="h2h")
            .first()
        )

        outcomes = []
        market_margin_pct = None
        if stake_odds_row:
            fair_probs, overround = remove_vig(
                stake_odds_row.home_odds, stake_odds_row.draw_odds, stake_odds_row.away_odds
            )
            market_margin_pct = (overround - 1) * 100
            bookmaker_odds = {
                "H": stake_odds_row.home_odds,
                "D": stake_odds_row.draw_odds,
                "A": stake_odds_row.away_odds,
            }

        for outcome_code in ["H", "D", "A"]:
            mp = model_probs[outcome_code]
            if stake_odds_row:
                bo = bookmaker_odds[outcome_code]
                fp = fair_probs[outcome_code]
                ev = calculate_ev(mp, bo)
                kelly = kelly_stake_fraction(mp, bo) * KELLY_FRACTION
                outcomes.append(OutcomeAnalysis(
                    outcome=outcome_code, label=CLASS_TO_LABEL[outcome_code],
                    model_probability=mp, stake_fair_probability=fp, stake_odds=bo,
                    ev=ev, kelly_stake_fraction=kelly, is_positive_ev=ev > 0,
                ))
            else:
                outcomes.append(OutcomeAnalysis(
                    outcome=outcome_code, label=CLASS_TO_LABEL[outcome_code],
                    model_probability=mp, stake_fair_probability=None, stake_odds=None,
                    ev=None, kelly_stake_fraction=None, is_positive_ev=False,
                ))

        results.append(UpcomingMatchPrediction(
            match_id=match.id, kickoff_utc=match.kickoff_utc,
            home_team=home_team.name, away_team=away_team.name,
            confidence_label=label, confidence_margin=margin,
            stake_market_margin_pct=market_margin_pct,
            low_data_warnings=warnings, outcomes=outcomes,
        ))

    return results