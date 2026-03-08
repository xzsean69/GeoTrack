"""
FastAPI service for transit demand prediction and route optimization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ml.demand_predictor import (
    generate_synthetic_demand,
    engineer_features,
    train_demand_model,
    predict_demand,
)
from src.network.graph_builder import (
    build_transit_graph,
    compute_shortest_path,
    get_network_stats,
)
from src.optimization.route_optimizer import (
    identify_congested_edges,
    suggest_alternative_routes,
)

app = FastAPI(title="Urban Mobility Stockholm API", version="1.0.0")

# ──────────────────────────────────────────────
# In-memory state (populated at startup)
# ──────────────────────────────────────────────
_model = None
_feature_cols: list[str] = []
_graph: nx.DiGraph = nx.DiGraph()


@app.on_event("startup")
def startup_event():
    global _model, _feature_cols, _graph
    # Train a quick model on synthetic data
    df = generate_synthetic_demand()
    df = engineer_features(df)
    _feature_cols = ["hour", "day_of_week", "month", "is_weekend",
                     "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_24"]
    _model, _ = train_demand_model(df, _feature_cols)

    # Build a tiny synthetic graph for demo purposes
    stops = pd.DataFrame({
        "stop_id": [f"S{i}" for i in range(5)],
        "stop_name": [f"Stop {i}" for i in range(5)],
        "stop_lat": [59.33 + i * 0.01 for i in range(5)],
        "stop_lon": [18.07 + i * 0.01 for i in range(5)],
    })
    stop_times = pd.DataFrame({
        "trip_id": ["T1"] * 5,
        "stop_id": [f"S{i}" for i in range(5)],
        "stop_sequence": list(range(5)),
    })
    _graph = build_transit_graph(stops, stop_times)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class DemandRequest(BaseModel):
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    demand_lag_1: float
    demand_lag_2: float
    demand_lag_3: float
    demand_lag_24: float


class RouteRequest(BaseModel):
    source: str
    target: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.post("/predict-demand")
def predict_demand_endpoint(req: DemandRequest):
    """Predict passenger demand for the given features."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = pd.DataFrame([req.model_dump()])
    prediction = predict_demand(_model, features[_feature_cols])
    return {"predicted_demand": float(prediction[0])}


@app.post("/optimal-route")
def optimal_route_endpoint(req: RouteRequest):
    """Find the optimal route between two stops."""
    path, length = compute_shortest_path(_graph, req.source, req.target)
    if not path:
        raise HTTPException(status_code=404, detail="No path found between stops")
    return {"path": path, "total_weight": length}


@app.get("/network-stats")
def network_stats_endpoint():
    """Return statistics about the transit network."""
    return get_network_stats(_graph)
