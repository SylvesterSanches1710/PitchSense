"""
Travel distance feature: straight-line (great-circle) distance the AWAY
team travels to reach this match, in km. No API needed — Premier League
stadiums essentially don't move, so this is a one-time static lookup,
not a rolling computation or an external fetch.

Home team travel is always 0 (they're at their own stadium) — not worth
its own column of constant zeros, so only away_travel_km_pre exists.

Known limitation, stated plainly: Everton relocated from Goodison Park
to the new Everton Stadium during the window this dataset covers. A
single fixed coordinate per team means away trips to Everton are
measured against one point regardless of which stadium was actually in
use for a given match — a deliberate simplification rather than tracking
mid-season venue changes for one club's brief transition period.
"""

import math
from dataclasses import dataclass

# Approximate stadium coordinates (decimal degrees). This is straight-line
# distance, not actual road/rail travel distance — a reasonable proxy for
# relative travel burden between fixtures, not a routing-accurate figure.
STADIUM_COORDINATES: dict[str, tuple[float, float]] = {
    "arsenal": (51.5549, -0.1084),
    "aston villa": (52.5092, -1.8848),
    "bournemouth": (50.7352, -1.8380),
    "brentford": (51.4907, -0.2886),
    "brighton and hove albion": (50.8617, -0.0838),
    "burnley": (53.7890, -2.2308),
    "chelsea": (51.4817, -0.1910),
    "crystal palace": (51.3983, -0.0855),
    "everton": (53.4388, -2.9663),
    "fulham": (51.4750, -0.2216),
    "ipswich town": (52.0546, 1.1451),
    "leeds united": (53.7778, -1.5722),
    "leicester city": (52.6204, -1.1422),
    "liverpool": (53.4308, -2.9608),
    "luton town": (51.8843, -0.4318),
    "manchester city": (53.4831, -2.2004),
    "manchester united": (53.4631, -2.2913),
    "newcastle united": (54.9756, -1.6217),
    "nottingham forest": (52.9400, -1.1327),
    "sheffield united": (53.3701, -1.4708),
    "southampton": (50.9058, -1.3911),
    "sunderland": (54.9144, -1.3883),
    "tottenham hotspur": (51.6043, -0.0664),
    "west ham united": (51.5386, -0.0166),
    "wolverhampton wanderers": (52.5903, -2.1305),
}


@dataclass
class TravelSnapshot:
    match_id: int
    away_travel_km_pre: (
        float | None
    )  # None = a team name wasn't found in the table above


def _normalize(name: str) -> str:
    name = name.lower().strip()
    for suffix in (" fc", " afc"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for prefix in ("afc ",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("&", "and").strip()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def compute_travel_features(
    match_ids_with_team_names: list[
        tuple[int, str, str]
    ],  # (match_id, home_name, away_name)
) -> list[TravelSnapshot]:
    """No chronological state needed — this is a pure per-match lookup,
    same shape as injuries.py, not a rolling window like most of the
    other features here."""
    snapshots: list[TravelSnapshot] = []
    unmatched_names: set[str] = set()

    for match_id, home_name, away_name in match_ids_with_team_names:
        home_coords = STADIUM_COORDINATES.get(_normalize(home_name))
        away_coords = STADIUM_COORDINATES.get(_normalize(away_name))

        if home_coords is None:
            unmatched_names.add(home_name)
        if away_coords is None:
            unmatched_names.add(away_name)

        if home_coords is None or away_coords is None:
            snapshots.append(TravelSnapshot(match_id=match_id, away_travel_km_pre=None))
            continue

        distance_km = _haversine_km(
            home_coords[0], home_coords[1], away_coords[0], away_coords[1]
        )
        snapshots.append(
            TravelSnapshot(match_id=match_id, away_travel_km_pre=round(distance_km, 1))
        )

    if unmatched_names:
        print("Teams not found in STADIUM_COORDINATES (add them to fix):")
        for name in sorted(unmatched_names):
            print(f"  - {name}")

    return snapshots
