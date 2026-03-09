"""
SMHI Weather data module for Stockholm.

Provides real hourly weather observations from the SMHI Open Data API
(https://opendata-download-metobs.smhi.se/api/).

The primary entry point is ``load_real_weather`` which fetches all four
meteorological parameters in one call and returns a ready-to-use DataFrame.
``generate_synthetic_weather`` is retained as a fallback for offline / test
environments.

Columns produced
----------------
timestamp       : UTC datetime
temperature     : degrees Celsius
precipitation   : mm (hourly)
wind_speed      : m/s
relative_humidity : %
is_rainy        : 1 if precipitation > 0.1 mm else 0
weather_demand_factor : composite multiplier (0.5 – 1.4) used to scale
                        passenger demand (rain + cold → more transit demand)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import requests
from typing import Optional

logger = logging.getLogger(__name__)


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


def _compute_demand_factor(
    temperature: "pd.Series",
    precipitation: "pd.Series",
    wind_speed: "pd.Series",
) -> "pd.Series":
    """
    Compute the composite weather demand factor from raw observations.

    Replicates the same formula used in ``generate_synthetic_weather`` so
    that real and synthetic data produce comparable multipliers.
    """
    cold_boost = ((10.0 - temperature) / 10.0 * 0.1).clip(0.0, 0.2)
    rain_boost = (precipitation > 0.1).astype(float) * 0.15
    wind_boost = (wind_speed > 8.0).astype(float) * 0.05
    heat_penalty = (temperature > 25.0).astype(float) * (-0.05)
    return (1.0 + cold_boost + rain_boost + wind_boost + heat_penalty).clip(0.5, 1.4)


def load_real_weather(
    station_id: int = _SMHI_STATION_ID,
    period: str = "latest-months",
    n_hours: Optional[int] = None,
    fallback_to_synthetic: bool = True,
    synthetic_n_hours: int = 168,
) -> pd.DataFrame:
    """
    Fetch real hourly weather data from the SMHI Open Data API and return a
    DataFrame in the same format as ``generate_synthetic_weather``.

    All four meteorological parameters (temperature, precipitation, wind
    speed, relative humidity) are fetched independently and then joined on
    the rounded-to-hour UTC timestamp.

    Parameters
    ----------
    station_id : int
        SMHI weather station (default: Observatorielunden, Stockholm, 98210).
    period : str
        SMHI period identifier – 'latest-months' returns the last ~3 months.
    n_hours : int, optional
        If given, only the most-recent ``n_hours`` rows are returned.
    fallback_to_synthetic : bool
        When *True* (default) a ``requests.RequestException`` or a merge
        that yields no rows silently falls back to ``generate_synthetic_weather``.
        Set to *False* to let network errors propagate to the caller.
    synthetic_n_hours : int
        Number of hours to generate when falling back to synthetic data.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, temperature, precipitation, wind_speed,
        relative_humidity, is_rainy, weather_demand_factor
    """
    try:
        raw_temp   = fetch_smhi_weather(_SMHI_PARAMS["temperature"],        station_id, period)
        raw_precip = fetch_smhi_weather(_SMHI_PARAMS["precipitation"],      station_id, period)
        raw_wind   = fetch_smhi_weather(_SMHI_PARAMS["wind_speed"],         station_id, period)
        raw_humid  = fetch_smhi_weather(_SMHI_PARAMS["relative_humidity"],  station_id, period)

        # Rename the generic 'value' column to the parameter name
        raw_temp   = raw_temp.rename(columns={"value": "temperature"})
        raw_precip = raw_precip.rename(columns={"value": "precipitation"})
        raw_wind   = raw_wind.rename(columns={"value": "wind_speed"})
        raw_humid  = raw_humid.rename(columns={"value": "relative_humidity"})

        # Round timestamps to the hour so the four series align correctly
        for frame in (raw_temp, raw_precip, raw_wind, raw_humid):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.floor("h")

        df = (
            raw_temp
            .merge(raw_precip, on="timestamp", how="inner")
            .merge(raw_wind,   on="timestamp", how="inner")
            .merge(raw_humid,  on="timestamp", how="inner")
        )

        if df.empty:
            raise ValueError("SMHI merge produced an empty DataFrame – no overlapping timestamps.")

        # Deduplicate and sort chronologically
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        # Clip to physically plausible ranges
        df["temperature"]      = df["temperature"].clip(-40.0,  45.0).round(1)
        df["precipitation"]    = df["precipitation"].clip(0.0,  50.0).round(2)
        df["wind_speed"]       = df["wind_speed"].clip(0.0,     50.0).round(1)
        df["relative_humidity"] = df["relative_humidity"].clip(0.0, 100.0).round(1)

        # Derived columns
        df["is_rainy"] = (df["precipitation"] > 0.1).astype(int)
        df["weather_demand_factor"] = _compute_demand_factor(
            df["temperature"], df["precipitation"], df["wind_speed"]
        ).round(4)

        # Optionally limit to the most-recent n_hours
        if n_hours and len(df) > n_hours:
            df = df.tail(n_hours).reset_index(drop=True)

        return df[["timestamp", "temperature", "precipitation", "wind_speed",
                   "relative_humidity", "is_rainy", "weather_demand_factor"]]

    except Exception as exc:
        if fallback_to_synthetic:
            logger.warning(
                "Failed to fetch real SMHI weather data (station %s, period %s): %s. "
                "Falling back to synthetic data.",
                station_id, period, exc,
            )
            return generate_synthetic_weather(n_hours=synthetic_n_hours)
        raise
