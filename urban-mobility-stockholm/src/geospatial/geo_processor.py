"""
Geospatial processing module.
Perform spatial join between GPS points and administrative zones using GeoPandas.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from math import radians, cos, sin, asin, sqrt
from typing import Optional


def stops_to_geodataframe(stops: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert stops DataFrame with lat/lon to a GeoDataFrame."""
    geometry = [Point(lon, lat) for lon, lat in zip(stops["stop_lon"], stops["stop_lat"])]
    return gpd.GeoDataFrame(stops, geometry=geometry, crs="EPSG:4326")


def create_grid_zones(
    bounds: tuple[float, float, float, float],
    n_rows: int = 10,
    n_cols: int = 10,
) -> gpd.GeoDataFrame:
    """
    Create a grid of rectangular zones over the given bounding box.
    bounds: (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds
    x_step = (maxx - minx) / n_cols
    y_step = (maxy - miny) / n_rows
    cells = []
    zone_ids = []
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = minx + col * x_step
            y0 = miny + row * y_step
            x1 = x0 + x_step
            y1 = y0 + y_step
            cells.append(box(x0, y0, x1, y1))
            zone_ids.append(f"zone_{row}_{col}")
    return gpd.GeoDataFrame({"zone_id": zone_ids, "geometry": cells}, crs="EPSG:4326")


def spatial_join_stops_to_zones(
    stops_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Join stops to zones using a spatial join."""
    return gpd.sjoin(stops_gdf, zones_gdf, how="left", predicate="within")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance between two points using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def build_demand_heatmap(
    stops_with_zones: gpd.GeoDataFrame,
    passenger_counts: pd.Series,
) -> pd.DataFrame:
    """Build a demand heatmap by summing passenger counts per zone."""
    df = stops_with_zones.copy()
    df["demand"] = passenger_counts.values
    return df.groupby("zone_id")["demand"].sum().reset_index()
