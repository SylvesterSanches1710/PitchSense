"""
Pure betting math — no database, no I/O, no printing. Shared by
modeling/betting_analysis.py (the CLI report) and api/routers/matches.py
(the dashboard backend), so both surfaces compute EV/Kelly/warnings
identically instead of maintaining two copies that could drift apart.
"""

CLASS_TO_LABEL = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

KELLY_FRACTION = 0.25  # quarter-Kelly — see docstring below for why

# A team whose Elo still sits within this many points of the 1500
# starting default has effectively played very few real matches.
LOW_DATA_ELO_THRESHOLD = 15.0


def remove_vig(home_odds: float, draw_odds: float, away_odds: float) -> tuple[dict, float]:
    """Converts raw bookmaker odds to fair (vig-removed) implied
    probability via the standard proportional method. Returns
    (fair_probs_dict, overround) — overround > 1.0 is the bookmaker's
    margin (e.g. 1.04 = 4% margin)."""
    raw = {"H": 1 / home_odds, "D": 1 / draw_odds, "A": 1 / away_odds}
    overround = sum(raw.values())
    return {k: v / overround for k, v in raw.items()}, overround


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Expected value per unit staked, using the ACTUAL odds offered —
    that's the real price, so it's what determines real expected return."""
    return model_prob * decimal_odds - 1


def kelly_stake_fraction(model_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction for theoretically optimal long-run bankroll
    growth, GIVEN that model_prob is exactly correct. Callers should
    multiply by KELLY_FRACTION (quarter-Kelly) before displaying —
    full Kelly assumes perfect probability estimates, which is a
    dangerous assumption to size real stakes on given known calibration
    imperfections (e.g. Away Win overconfidence at high probabilities).
    Returns 0.0 (never negative) when there's no edge.
    """
    b = decimal_odds - 1
    numerator = model_prob * decimal_odds - 1
    return max(0.0, numerator / b)


def check_low_data_warning(team_name: str, elo_pre: float | None, form_pre: float | None) -> str | None:
    """Returns a human-readable warning string if this team has too
    little history to trust the model's output for it, or None if fine."""
    if form_pre is None:
        return f"{team_name}: zero prior matches on record — likely newly promoted or new to this dataset."
    if elo_pre is not None and abs(elo_pre - 1500.0) < LOW_DATA_ELO_THRESHOLD:
        return f"{team_name}: Elo still near the 1500 default ({elo_pre:.1f}) — very little real history to draw on."
    return None


def confidence_label(model_probs: dict) -> tuple[str, float]:
    """Margin between the model's top and second pick — simple,
    interpretable, explicitly NOT a rigorous statistical confidence
    interval. Returns (label, margin)."""
    sorted_probs = sorted(model_probs.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    label = "High" if margin > 0.25 else "Medium" if margin > 0.10 else "Low"
    return label, margin