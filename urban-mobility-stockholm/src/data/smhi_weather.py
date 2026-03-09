"""
SMHI Weather data module for Stockholm.

Provides synthetic hourly weather observations that mirror the structure of
data returned by the SMHI Open Data API
(https://opendata-download-metobs.smhi.se/api/).

In a production deployment replace ``generate_synthetic_weather`` with a call
to ``fetch_smhi_weather`` which queries the real SMHI REST endpoint.

Columns produced
----------------
timestamp       : UTC datetime
temperature     : degrees Celsius
precipitation   : mm (hourly)
wind_speed      : m/s
relative_humidity : %
is_rainy        : 1 if precipitation > 0.1 mm else 0
weather_demand_factor : composite multiplier (0.5 – 1.2) used to scale
                        passenger demand (rain + cold → more transit demand)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from typing import Optional


# ─── Stockholm bounding-box centre ───────────────────────────────────────────
STOCKHOLM_LAT = 59.3293
STOCKHOLM_LON = 18.0686

# SMHI parameter IDs used for reference
_SMHI_PARAMS = {
    "temperature":         1,   # air temperature, °C
    "precipitation":       5,   # hourly precipitation, mm
    "wind_speed":         4,   # mean wind speed, m/s
    "relative_humidity":  6,   # relative humidity, %
}

# SMHI station closest to Stockholm city (Observatorielunden, station 98210)
_SMHI_STATION_ID = 98210


def fetch_smhi_weather(
    parameter_id: int,
    station_id: int = _SMHI_STATION_ID,
    period: str = "latest-months",
) -> pd.DataFrame:
    """
    Fetch hourly weather data from the SMHI Open Data REST API.

    Parameters
    ----------
    parameter_id : int
        SMHI meteorological parameter (see ``_SMHI_PARAMS``).
    station_id : int
        SMHI weather station identifier.
    period : str
        One of 'latest-hour', 'latest-day', 'latest-months', 'corrected-archive'.

    Returns
    -------
    pd.DataFrame with columns ['timestamp', 'value'].

    Notes
    -----
    Raises ``requests.RequestException`` on network errors.
    """
    url = (
        f"https://opendata-download-metobs.smhi.se/api/version/latest"
        f"/parameter/{parameter_id}/station/{station_id}"
        f"/period/{period}/data.json"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    records = [
        {
            "timestamp": pd.Timestamp(entry[0], unit="ms", tz="UTC"),
            "value": float(entry[1]),
        }
        for entry in data.get("value", [])
    ]
    return pd.DataFrame(records)


def generate_synthetic_weather(
    n_hours: int = 168,
    start: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic hourly weather data for Stockholm.

    The model captures:
    - Seasonal temperature variation (colder in winter, warmer in summer)
    - Diurnal temperature cycle (cooler at night)
    - Stochastic precipitation events (clustered in time)
    - Wind speed with mild diurnal pattern
    - Relative humidity inversely correlated with temperature

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, temperature, precipitation, wind_speed,
        relative_humidity, is_rainy, weather_demand_factor
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n_hours, freq="h")

    hours_arr   = np.array([ts.hour       for ts in timestamps], dtype=float)
    months_arr  = np.array([ts.month      for ts in timestamps], dtype=float)
    doy_arr     = np.array([ts.dayofyear  for ts in timestamps], dtype=float)

    # ── Temperature ──────────────────────────────────────────────────────────
    # Seasonal: ranges ~-5 °C (Jan) to ~22 °C (Jul)
    seasonal_temp = 8.5 + 13.5 * np.sin(2 * np.pi * (doy_arr - 80) / 365)
    # Diurnal: ±3 °C, coldest at ~04:00
    diurnal_temp  = 3.0 * np.sin(2 * np.pi * (hours_arr - 4) / 24)
    noise_temp    = rng.normal(0, 1.5, n_hours)
    temperature   = seasonal_temp + diurnal_temp + noise_temp

    # ── Precipitation ────────────────────────────────────────────────────────
    # Markov-chain weather state (dry / wet) to create realistic clustering
    state      = np.zeros(n_hours, dtype=int)  # 0=dry, 1=wet
    state[0]   = int(rng.random() > 0.7)
    p_dry_wet  = 0.06   # probability of transitioning dry→wet per hour
    p_wet_dry  = 0.15   # probability of transitioning wet→dry per hour
    for t in range(1, n_hours):
        if state[t - 1] == 0:
            state[t] = 1 if rng.random() < p_dry_wet else 0
        else:
            state[t] = 0 if rng.random() < p_wet_dry else 1

    precipitation = np.where(
        state == 1,
        rng.exponential(scale=1.2, size=n_hours),
        rng.uniform(0, 0.05, size=n_hours),
    )
    precipitation = np.clip(precipitation, 0, 20)

    # ── Wind speed ───────────────────────────────────────────────────────────
    # Weibull-distributed, slightly higher during daytime
    base_wind   = rng.weibull(2, n_hours) * 4.5
    diurnal_wind = 1.0 * np.sin(2 * np.pi * (hours_arr - 14) / 24)
    wind_speed  = np.clip(base_wind + diurnal_wind, 0, 25)

    # ── Relative humidity ────────────────────────────────────────────────────
    relative_humidity = np.clip(
        85 - 0.8 * (temperature - 8) + rng.normal(0, 5, n_hours),
        30, 100,
    )

    # ── Composite demand factor ───────────────────────────────────────────────
    # Cold weather: +10 % demand per 10 °C below 10 °C
    cold_boost     = np.clip((10 - temperature) / 10 * 0.1, 0, 0.2)
    # Rain: +15 % demand
    rain_boost     = np.where(precipitation > 0.1, 0.15, 0.0)
    # Strong wind: +5 % demand
    wind_boost     = np.where(wind_speed > 8, 0.05, 0.0)
    # Hot summer: slight dip in transit (-5 %)
    heat_penalty   = np.where(temperature > 25, -0.05, 0.0)
    weather_demand_factor = np.clip(
        1.0 + cold_boost + rain_boost + wind_boost + heat_penalty, 0.5, 1.4
    )

    return pd.DataFrame({
        "timestamp":             timestamps,
        "temperature":           np.round(temperature, 1),
        "precipitation":         np.round(precipitation, 2),
        "wind_speed":            np.round(wind_speed, 1),
        "relative_humidity":     np.round(relative_humidity, 1),
        "is_rainy":              (precipitation > 0.1).astype(int),
        "weather_demand_factor": np.round(weather_demand_factor, 4),
    })
