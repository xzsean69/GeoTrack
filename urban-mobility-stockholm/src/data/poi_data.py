"""
Points of Interest (POI) module for Stockholm.

Provides a curated catalogue of representative Stockholm POIs across six
categories that drive transit demand.  For each transit stop the module
computes a *proximity score* – a weighted sum of inverse distances to nearby
POIs – for each category.  These scores become additional features for the
ML demand-prediction model.

POI categories
--------------
office          : major business/office areas (generates morning/evening peaks)
university      : universities & higher education (generates off-peak demand)
hospital        : hospitals & clinics (generates steady all-day demand)
shopping        : retail centres & large stores (generates afternoon peaks)
tourist         : museums, parks, landmarks (generates daytime weekend demand)
transit_hub     : major interchanges (multiplier for transfer demand)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional


# ─── Curated Stockholm POI catalogue ─────────────────────────────────────────
_POI_CATALOGUE: list[dict] = [
    # ── Offices / Business districts ─────────────────────────────────────────
    {"name": "Kista Science City",     "lat": 59.4028, "lon": 17.9502, "category": "office",      "weight": 3.0},
    {"name": "Stockholm City / Norrmalm", "lat": 59.3340, "lon": 18.0620, "category": "office",   "weight": 2.5},
    {"name": "Hammarby Sjöstad",       "lat": 59.3040, "lon": 18.0810, "category": "office",      "weight": 1.5},
    {"name": "Alvik Business Park",    "lat": 59.3321, "lon": 17.9755, "category": "office",      "weight": 1.2},
    {"name": "Solna Business Park",    "lat": 59.3640, "lon": 18.0010, "category": "office",      "weight": 1.8},

    # ── Universities ─────────────────────────────────────────────────────────
    {"name": "KTH Royal Institute of Technology", "lat": 59.3474, "lon": 18.0710, "category": "university", "weight": 3.0},
    {"name": "Stockholm University",              "lat": 59.3650, "lon": 18.0580, "category": "university", "weight": 2.5},
    {"name": "Karolinska Institute (Solna)",      "lat": 59.3495, "lon": 18.0246, "category": "university", "weight": 2.0},
    {"name": "Stockholm School of Economics",     "lat": 59.3410, "lon": 18.0590, "category": "university", "weight": 1.5},
    {"name": "Konstfack (University of Arts)",    "lat": 59.3147, "lon": 18.0183, "category": "university", "weight": 1.0},

    # ── Hospitals ────────────────────────────────────────────────────────────
    {"name": "Karolinska University Hospital",  "lat": 59.3495, "lon": 18.0246, "category": "hospital", "weight": 3.0},
    {"name": "Södersjukhuset",                  "lat": 59.3094, "lon": 18.0592, "category": "hospital", "weight": 2.5},
    {"name": "Danderyd Hospital",               "lat": 59.4069, "lon": 18.0422, "category": "hospital", "weight": 2.0},
    {"name": "St Göran Hospital",               "lat": 59.3335, "lon": 18.0340, "category": "hospital", "weight": 1.5},

    # ── Shopping centres ─────────────────────────────────────────────────────
    {"name": "Gallerian (City)",           "lat": 59.3343, "lon": 18.0631, "category": "shopping", "weight": 2.5},
    {"name": "Nordstan (Central Station)", "lat": 59.3305, "lon": 18.0570, "category": "shopping", "weight": 2.0},
    {"name": "Farsta Centrum",             "lat": 59.2409, "lon": 18.0877, "category": "shopping", "weight": 2.0},
    {"name": "Mall of Scandinavia (Solna)","lat": 59.3700, "lon": 18.0040, "category": "shopping", "weight": 3.0},
    {"name": "Täby Centrum",               "lat": 59.4445, "lon": 18.0761, "category": "shopping", "weight": 1.8},
    {"name": "Kungens Kurva (IKEA)",       "lat": 59.2679, "lon": 17.9278, "category": "shopping", "weight": 1.5},

    # ── Tourist attractions ───────────────────────────────────────────────────
    {"name": "Gamla Stan (Old Town)",      "lat": 59.3250, "lon": 18.0710, "category": "tourist", "weight": 3.0},
    {"name": "Vasa Museum",                "lat": 59.3280, "lon": 18.0915, "category": "tourist", "weight": 2.5},
    {"name": "ABBA Museum",                "lat": 59.3264, "lon": 18.0952, "category": "tourist", "weight": 2.0},
    {"name": "Skansen Open-Air Museum",    "lat": 59.3274, "lon": 18.1064, "category": "tourist", "weight": 2.5},
    {"name": "Royal Palace",               "lat": 59.3268, "lon": 18.0717, "category": "tourist", "weight": 2.0},
    {"name": "Djurgården (Park)",          "lat": 59.3260, "lon": 18.1150, "category": "tourist", "weight": 2.0},

    # ── Transit hubs ─────────────────────────────────────────────────────────
    {"name": "Stockholm Central Station", "lat": 59.3310, "lon": 18.0590, "category": "transit_hub", "weight": 5.0},
    {"name": "T-Centralen (Metro hub)",   "lat": 59.3314, "lon": 18.0597, "category": "transit_hub", "weight": 4.0},
    {"name": "Slussen (Hub)",             "lat": 59.3200, "lon": 18.0720, "category": "transit_hub", "weight": 3.5},
    {"name": "Odenplan",                  "lat": 59.3431, "lon": 18.0503, "category": "transit_hub", "weight": 2.5},
    {"name": "Fridhemsplan",              "lat": 59.3327, "lon": 18.0303, "category": "transit_hub", "weight": 2.0},
    {"name": "Gullmarsplan",              "lat": 59.3011, "lon": 18.0792, "category": "transit_hub", "weight": 2.0},
]

_POI_DF: Optional[pd.DataFrame] = None


def get_poi_dataframe() -> pd.DataFrame:
    """Return the full POI catalogue as a DataFrame (cached singleton)."""
    global _POI_DF
    if _POI_DF is None:
        _POI_DF = pd.DataFrame(_POI_CATALOGUE)
    return _POI_DF.copy()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def compute_poi_scores(
    lat: float,
    lon: float,
    radius_km: float = 2.0,
) -> dict[str, float]:
    """
    Compute per-category POI proximity scores for a single location.

    Score = sum(weight / max(distance_km, 0.1)) for all POIs within
    ``radius_km`` of the given point, grouped by category.

    Parameters
    ----------
    lat, lon    : Coordinates of the query point (e.g. a transit stop).
    radius_km   : Only POIs within this radius are included.

    Returns
    -------
    dict mapping category name → score (float ≥ 0).
    """
    categories = {"office", "university", "hospital", "shopping", "tourist", "transit_hub"}
    scores: dict[str, float] = {c: 0.0 for c in categories}

    for poi in _POI_CATALOGUE:
        dist = _haversine_km(lat, lon, poi["lat"], poi["lon"])
        if dist <= radius_km:
            scores[poi["category"]] += poi["weight"] / max(dist, 0.1)

    return scores


def add_poi_features(
    stops_df: pd.DataFrame,
    radius_km: float = 2.0,
) -> pd.DataFrame:
    """
    Compute POI proximity scores for every stop and append them as new columns.

    Input DataFrame must have columns ``stop_lat`` and ``stop_lon``.
    Returns a copy with additional columns:
        poi_office, poi_university, poi_hospital,
        poi_shopping, poi_tourist, poi_transit_hub
    """
    result = stops_df.copy()
    categories = ["office", "university", "hospital", "shopping", "tourist", "transit_hub"]

    scores_list = [
        compute_poi_scores(row["stop_lat"], row["stop_lon"], radius_km)
        for _, row in stops_df.iterrows()
    ]

    for cat in categories:
        result[f"poi_{cat}"] = [s[cat] for s in scores_list]

    # Normalise each category to [0, 1] across all stops
    for cat in categories:
        col = f"poi_{cat}"
        col_max = result[col].max()
        if col_max > 0:
            result[col] = result[col] / col_max

    return result
