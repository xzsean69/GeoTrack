"""
GTFS data ingestion module.
Load and clean GTFS public transport data into pandas DataFrames.
"""
import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_gtfs(url: str, output_path: Path) -> Path:
    """Download a GTFS zip file from a URL."""
    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / "gtfs.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    zip_path.write_bytes(response.content)
    return zip_path


def parse_gtfs_zip(zip_path: Path) -> dict[str, pd.DataFrame]:
    """Parse a GTFS zip file and return DataFrames for each file."""
    frames = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                key = name.replace(".txt", "")
                with zf.open(name) as f:
                    frames[key] = pd.read_csv(f)
    return frames


def clean_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate stops DataFrame."""
    stops = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"])
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    return stops.reset_index(drop=True)


def clean_stop_times(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Clean stop_times DataFrame."""
    stop_times = stop_times.dropna(subset=["trip_id", "stop_id", "stop_sequence"])
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    return stop_times.reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a parquet file."""
    return pd.read_parquet(path)


def load_gtfs(zip_path: Path) -> dict[str, pd.DataFrame]:
    """Load, clean, and return GTFS data as DataFrames."""
    frames = parse_gtfs_zip(zip_path)
    if "stops" in frames:
        frames["stops"] = clean_stops(frames["stops"])
    if "stop_times" in frames:
        frames["stop_times"] = clean_stop_times(frames["stop_times"])
    return frames


def save_gtfs_parquet(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save all GTFS DataFrames as parquet files."""
    for name, df in frames.items():
        save_parquet(df, output_dir / f"{name}.parquet")
