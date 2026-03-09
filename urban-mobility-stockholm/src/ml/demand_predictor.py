"""
Machine learning demand prediction module.
Train an XGBoost regression model to predict hourly passenger demand.

Extended feature set
--------------------
- Time features        : hour, day_of_week, month, is_weekend
- Lag features         : demand at t-1, t-2, t-3, t-24
- Weather features     : temperature, precipitation, wind_speed,
                         relative_humidity, is_rainy, weather_demand_factor
                         (sourced from SMHI or generated synthetically)
- POI proximity scores : poi_office, poi_university, poi_hospital,
                         poi_shopping, poi_tourist, poi_transit_hub
                         (computed from the Stockholm POI catalogue)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Weather and POI feature names exposed for callers
WEATHER_FEATURE_COLS: list[str] = [
    "temperature", "precipitation", "wind_speed",
    "relative_humidity", "is_rainy", "weather_demand_factor",
]
POI_FEATURE_COLS: list[str] = [
    "poi_office", "poi_university", "poi_hospital",
    "poi_shopping", "poi_tourist", "poi_transit_hub",
]
BASE_FEATURE_COLS: list[str] = [
    "hour", "day_of_week", "month", "is_weekend",
    "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_24",
]
ALL_FEATURE_COLS: list[str] = BASE_FEATURE_COLS + WEATHER_FEATURE_COLS + POI_FEATURE_COLS


def engineer_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    weather_df: Optional[pd.DataFrame] = None,
    poi_scores: Optional[dict[str, dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Feature engineering: extract time, lag, weather, and POI features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: timestamp, zone_id, demand.
    timestamp_col : str
        Name of the timestamp column.
    weather_df : pd.DataFrame, optional
        Hourly weather observations indexed by timestamp with columns:
        temperature, precipitation, wind_speed, relative_humidity,
        is_rainy, weather_demand_factor.
        If None, default neutral weather values are used.
    poi_scores : dict, optional
        Mapping of zone_id → {category: score} returned by
        ``src.data.poi_data.compute_poi_scores``.
        If None, zero scores are used.

    Returns
    -------
    pd.DataFrame with all engineered features (NaN rows from lagging dropped).
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek
    df["month"] = df[timestamp_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── Lag features per zone ─────────────────────────────────────────────────
    df = df.sort_values([timestamp_col])
    for lag in [1, 2, 3, 24]:
        df[f"demand_lag_{lag}"] = df.groupby("zone_id")["demand"].shift(lag)

    # ── Weather features ──────────────────────────────────────────────────────
    if weather_df is not None:
        # Align weather data to the demand DataFrame by timestamp
        weather_indexed = weather_df.set_index("timestamp") if "timestamp" in weather_df.columns else weather_df
        weather_indexed.index = pd.to_datetime(weather_indexed.index)
        # Round timestamps to the hour for matching
        df_hour = df[timestamp_col].dt.floor("h")
        for col in WEATHER_FEATURE_COLS:
            if col in weather_indexed.columns:
                df[col] = df_hour.map(weather_indexed[col]).values
            else:
                df[col] = _weather_defaults()[col]
    else:
        for col, val in _weather_defaults().items():
            df[col] = val

    # ── POI features ──────────────────────────────────────────────────────────
    for cat in ["office", "university", "hospital", "shopping", "tourist", "transit_hub"]:
        col = f"poi_{cat}"
        if poi_scores:
            df[col] = df["zone_id"].map(
                lambda z, c=cat: poi_scores.get(z, {}).get(c, 0.0)
            )
        else:
            df[col] = 0.0

    return df.dropna()


def _weather_defaults() -> dict[str, float]:
    """Neutral (calm, mild) weather values used when no weather data is provided."""
    return {
        "temperature": 10.0,
        "precipitation": 0.0,
        "wind_speed": 3.0,
        "relative_humidity": 70.0,
        "is_rainy": 0,
        "weather_demand_factor": 1.0,
    }


def train_demand_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "demand",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[XGBRegressor, dict]:
    """
    Train an XGBoost regression model to predict passenger demand.
    Returns (model, metrics).
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    return model, metrics


def save_model(model: XGBRegressor, path: Path) -> None:
    """Save the trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> XGBRegressor:
    """Load a trained model from disk."""
    return joblib.load(path)


def predict_demand(
    model: XGBRegressor,
    features: pd.DataFrame,
) -> np.ndarray:
    """Run inference with the trained model."""
    return model.predict(features)


def generate_synthetic_demand(
    n_zones: int = 20,
    n_hours: int = 168,  # 1 week
    seed: int = 42,
    weather_df: Optional[pd.DataFrame] = None,
    poi_scores: Optional[dict[str, dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Generate synthetic hourly passenger demand data, optionally modulated by
    weather conditions (SMHI) and POI proximity scores.

    Parameters
    ----------
    n_zones : int
        Number of demand zones (transit stops).
    n_hours : int
        Number of hourly time steps.
    seed : int
        Random seed for reproducibility.
    weather_df : pd.DataFrame, optional
        Output of ``src.data.smhi_weather.generate_synthetic_weather``.
        Must have a ``timestamp`` index/column and a ``weather_demand_factor``
        column.  If provided, demand is scaled by the weather factor.
    poi_scores : dict, optional
        Mapping of zone_id → {category: score}.
        The composite POI score boosts base demand by up to +50 %.

    Returns
    -------
    pd.DataFrame with columns: timestamp, zone_id, demand.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    zones = [f"zone_{i}" for i in range(n_zones)]

    # Build a quick weather-factor lookup {timestamp: factor}
    weather_factor_map: dict = {}
    if weather_df is not None:
        wdf = weather_df.copy()
        if "timestamp" in wdf.columns:
            wdf = wdf.set_index("timestamp")
        wdf.index = pd.to_datetime(wdf.index)
        weather_factor_map = wdf["weather_demand_factor"].to_dict()

    records = []
    for ts in timestamps:
        w_factor = weather_factor_map.get(ts, 1.0)
        for zone in zones:
            base = 100 + 50 * np.sin(2 * np.pi * ts.hour / 24)

            # POI boost: composite score across all categories (max +50 %)
            poi_boost = 1.0
            if poi_scores and zone in poi_scores:
                zscores = poi_scores[zone]
                composite = sum(zscores.values()) / max(len(zscores), 1)
                poi_boost = 1.0 + 0.5 * min(composite, 1.0)

            noise = rng.normal(0, 10)
            demand = max(0, base * w_factor * poi_boost + noise)
            records.append({"timestamp": ts, "zone_id": zone, "demand": demand})
    return pd.DataFrame(records)
