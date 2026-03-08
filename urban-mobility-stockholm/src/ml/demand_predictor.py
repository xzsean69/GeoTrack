"""
Machine learning demand prediction module.
Train an XGBoost regression model to predict hourly passenger demand.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def engineer_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Feature engineering: extract time features and lag features from a demand DataFrame.
    Expects columns: timestamp, zone_id, demand.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek
    df["month"] = df[timestamp_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Lag features per zone
    df = df.sort_values([timestamp_col])
    for lag in [1, 2, 3, 24]:
        df[f"demand_lag_{lag}"] = df.groupby("zone_id")["demand"].shift(lag)

    return df.dropna()


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
) -> pd.DataFrame:
    """
    Generate synthetic hourly passenger demand data for testing.
    Returns a DataFrame with columns: timestamp, zone_id, demand.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    zones = [f"zone_{i}" for i in range(n_zones)]
    records = []
    for ts in timestamps:
        for zone in zones:
            base = 100 + 50 * np.sin(2 * np.pi * ts.hour / 24)
            noise = rng.normal(0, 10)
            demand = max(0, base + noise)
            records.append({"timestamp": ts, "zone_id": zone, "demand": demand})
    return pd.DataFrame(records)
