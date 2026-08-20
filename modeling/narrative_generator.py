"""
Turns SHAP contributions + raw feature values into a plain-English
explanation — grounded entirely in real numbers from shap_explainer.py,
never invented. Each template below states a fact the model actually
used (e.g. "won 13 of 15 possible points at home"), not a generic or
speculative reason.

Only the features most likely to be genuinely explanatory have a
dedicated template. Anything else falls back to a plain, honest
humanization of the column name and value — still grounded in the real
number, just less narratively polished. This is a deliberate scope
choice: better to have 16 features explained well than 31 explained
awkwardly.
"""

from modeling.shap_explainer import top_contributing_features

# A cited reason must be at least this fraction of the STRONGEST
# contributor's magnitude to be included — prevents a weak, borderline-
# noise contributor from being narratively equated with a dominant one
# just to fill a fixed reason count. 0.3 means: if the top reason has
# magnitude 0.095, anything under 0.0285 gets dropped rather than cited.
# Found via a real case: home_red_cards_avg_pre (+0.025) was being cited
# alongside home_position_pre (+0.095, ~4x larger) as if comparably
# important — technically grounded in a real SHAP value, but misleading
# about how much it actually mattered relative to the other reasons.
MIN_RELATIVE_MAGNITUDE = 0.3

# Features excluded from narrative citation even when SHAP ranks them
# highly. Red cards and suspensions are rare events — a rolling average
# built from ~1,140 matches has very few actual occurrences to learn
# from, so the model can latch onto spurious, backward-looking
# correlations (e.g. "high red card average" appearing to predict wins,
# purely from a handful of coincidental training examples, not a real
# football relationship). Confirmed via a real test case where the
# model cited a 0.5 red-card average as a reason FOR a home win — that
# doesn't make football sense and shouldn't be presented as if it does,
# even though it remains a legitimate input to the model itself.
_EXCLUDED_FROM_NARRATIVE = {
    "home_red_cards_avg_pre", "away_red_cards_avg_pre",
    "home_suspensions_count_pre", "away_suspensions_count_pre",
}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Each entry: feature_column_name -> function(raw_value, team_name) -> phrase
# Phrases are written to read naturally mid-sentence, lowercase, no period.
_TEMPLATES = {
    "elo_home_pre": lambda v, team: f"{team}'s strong overall team rating ({v:.0f})",
    "elo_away_pre": lambda v, team: f"{team}'s overall team rating ({v:.0f})",
    "form_home_pre": lambda v, team: f"{team} picking up {v:.0f} of a possible 30 points in their last 10 matches",
    "form_away_pre": lambda v, team: f"{team} picking up {v:.0f} of a possible 30 points in their last 10 matches",
    "home_venue_form_pre": lambda v, team: f"{team}'s home form — {v:.0f} of a possible 15 points from their last 5 home matches",
    "away_venue_form_pre": lambda v, team: f"{team}'s away form — {v:.0f} of a possible 15 points from their last 5 away matches",
    "h2h_home_ppg_pre": lambda v, team: f"{team} averaging {v:.1f} points per game historically against this opponent",
    "h2h_away_ppg_pre": lambda v, team: f"{team} averaging {v:.1f} points per game historically against this opponent",
    "home_position_pre": lambda v, team: f"{team} currently sitting {_ordinal(int(v))} in the table",
    "away_position_pre": lambda v, team: f"{team} currently sitting {_ordinal(int(v))} in the table",
    "home_injuries_count_pre": lambda v, team: f"{team} missing {int(v)} player{'s' if v != 1 else ''} to injury",
    "away_injuries_count_pre": lambda v, team: f"{team} missing {int(v)} player{'s' if v != 1 else ''} to injury",
    "home_goals_scored_avg_pre": lambda v, team: f"{team} averaging {v:.1f} goals scored per game recently",
    "away_goals_scored_avg_pre": lambda v, team: f"{team} averaging {v:.1f} goals scored per game recently",
    "home_goals_conceded_avg_pre": lambda v, team: f"{team} averaging {v:.1f} goals conceded per game recently",
    "away_goals_conceded_avg_pre": lambda v, team: f"{team} averaging {v:.1f} goals conceded per game recently",
    "home_rest_days_pre": lambda v, team: f"{team} having {int(v)} days of rest before this match",
    "away_rest_days_pre": lambda v, team: f"{team} having {int(v)} days of rest before this match",
}


def _humanize_column_name(col: str) -> str:
    return col.replace("_pre", "").replace("_", " ")


def _describe_feature(feature_name: str, raw_value, home_team: str, away_team: str) -> str | None:
    if raw_value is None:
        return None  # can't reference a fact we don't have
    if feature_name in _EXCLUDED_FROM_NARRATIVE:
        return None  # legitimate model input, not a narratable reason — see module docstring

    team = home_team if "home" in feature_name else away_team
    template = _TEMPLATES.get(feature_name)
    if template:
        return template(raw_value, team)

    # Fallback for features without a dedicated template — still
    # grounded in a real value, just less polished phrasing.
    return f"{team}'s {_humanize_column_name(feature_name)} ({raw_value:.1f})"


def generate_explanation(
    home_team: str,
    away_team: str,
    predicted_outcome_label: str,
    predicted_probability: float,
    contributions: dict[str, float],
    feature_row,  # MatchFeature ORM object — read raw values from it
    n_reasons: int = 4,
) -> str:
    """
    Produces a plain-English explanation string. Only features that
    PUSHED TOWARD the predicted outcome (positive SHAP contribution) are
    used as stated reasons — a feature that pushed against the
    prediction wouldn't make sense as a reason FOR it, even if it was
    among the largest-magnitude contributions.
    """
    top_features = top_contributing_features(contributions, n=n_reasons * 2)
    positive_features = [(name, val) for name, val in top_features if val > 0]

    if positive_features:
        # Relative-strength floor: only cite a feature if it's at least
        # this fraction of the STRONGEST positive contributor's
        # magnitude. Without this, a weak signal (e.g. 25% the strength
        # of the top contributor) gets narrated with the same apparent
        # confidence as a dominant one — misleading regardless of
        # whether the feature sounds intuitive or not. This is an
        # objective magnitude criterion applied uniformly, not editorial
        # judgment about which reasons "sound sensible" — that would be
        # its own form of inventing a cleaner story than the model
        # actually produced, which is exactly what this whole module
        # exists to avoid.
        strongest_magnitude = abs(positive_features[0][1])
        positive_features = [
            (name, val) for name, val in positive_features
            if abs(val) >= MIN_RELATIVE_MAGNITUDE * strongest_magnitude
        ][:n_reasons]

    if not positive_features:
        return (
            f"{home_team} vs {away_team}: the model gives {predicted_outcome_label} the highest "
            f"probability ({predicted_probability:.0%}), but no single feature stands out as a "
            f"clear driver — this looks like a genuinely close call."
        )

    reasons = []
    for feature_name, _shap_value in positive_features:
        raw_value = getattr(feature_row, feature_name, None)
        description = _describe_feature(feature_name, raw_value, home_team, away_team)
        if description:
            reasons.append(description)

    if not reasons:
        return (
            f"{home_team} vs {away_team}: the model favors {predicted_outcome_label} "
            f"({predicted_probability:.0%}), but the underlying reasons involve features "
            f"without a clean plain-English description yet."
        )

    if len(reasons) == 1:
        reason_text = reasons[0]
    elif len(reasons) == 2:
        reason_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reason_text = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    if predicted_outcome_label == "Home Win":
        subject_sentence = f"{home_team} has a {predicted_probability:.0%} chance of winning"
    elif predicted_outcome_label == "Away Win":
        subject_sentence = f"{away_team} has a {predicted_probability:.0%} chance of winning away"
    else:  # Draw
        subject_sentence = f"A draw is given a {predicted_probability:.0%} chance"

    return f"{subject_sentence}, driven by {reason_text}."