"""
Interactive Streamlit dashboard with map visualizations for urban mobility.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.demand_predictor import generate_synthetic_demand, engineer_features, train_demand_model
from src.geospatial.geo_processor import create_grid_zones, haversine_distance
from src.network.graph_builder import build_transit_graph, get_network_stats
from src.optimization.route_optimizer import identify_congested_edges

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Mobility Stockholm",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 Urban Mobility Stockholm Dashboard")
st.markdown("Demand forecasting · Network analysis · Route optimization")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    n_zones = st.slider("Number of Zones (grid)", 4, 20, 10)
    n_hours = st.slider("Simulation Hours", 24, 336, 168)
    congestion_threshold = st.slider("Congestion Threshold", 100, 2000, 500)

# ──────────────────────────────────────────────
# Data generation (cached)
# ──────────────────────────────────────────────
@st.cache_data
def get_demand_data(n_zones, n_hours):
    return generate_synthetic_demand(n_zones=n_zones, n_hours=n_hours)


@st.cache_data
def get_trained_model(n_zones, n_hours):
    df = get_demand_data(n_zones, n_hours)
    df_feat = engineer_features(df)
    feat_cols = ["hour", "day_of_week", "month", "is_weekend",
                 "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_24"]
    model, metrics = train_demand_model(df_feat, feat_cols)
    return model, metrics, feat_cols


@st.cache_data
def get_graph(n_zones):
    stops = pd.DataFrame({
        "stop_id": [f"S{i}" for i in range(n_zones)],
        "stop_name": [f"Stop {i}" for i in range(n_zones)],
        "stop_lat": [59.30 + (i % 5) * 0.02 for i in range(n_zones)],
        "stop_lon": [18.05 + (i // 5) * 0.02 for i in range(n_zones)],
    })
    stop_times = pd.DataFrame({
        "trip_id": ["T1"] * n_zones,
        "stop_id": [f"S{i}" for i in range(n_zones)],
        "stop_sequence": list(range(n_zones)),
    })
    return build_transit_graph(stops, stop_times), stops


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Demand Heatmap", "🗺️ Network Map", "🔀 Route Optimizer"])

# ── Tab 1: Demand Heatmap ──────────────────────
with tab1:
    st.subheader("Hourly Passenger Demand by Zone")
    demand_df = get_demand_data(n_zones, n_hours)
    demand_df = demand_df.copy()
    demand_df["hour"] = pd.to_datetime(demand_df["timestamp"]).dt.hour

    # Aggregate by hour
    hourly = demand_df.groupby("hour")["demand"].mean().reset_index()
    fig1 = px.bar(
        hourly, x="hour", y="demand",
        title="Average Demand by Hour of Day",
        labels={"hour": "Hour", "demand": "Avg Passengers"},
        color="demand", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap per zone per hour
    pivot = demand_df.pivot_table(index="zone_id", columns="hour", values="demand", aggfunc="mean")
    fig2 = px.imshow(
        pivot,
        title="Demand Heatmap (Zone × Hour)",
        labels={"x": "Hour", "y": "Zone", "color": "Avg Demand"},
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ML Model metrics
    model, metrics, feat_cols = get_trained_model(n_zones, n_hours)
    st.subheader("🤖 ML Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['mae']:.2f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("R²", f"{metrics['r2']:.4f}")

# ── Tab 2: Network Map ─────────────────────────
with tab2:
    st.subheader("Transit Network Graph")
    G, stops_df = get_graph(n_zones)
    stats = get_network_stats(G)

    col1, col2, col3 = st.columns(3)
    col1.metric("Stops", stats["num_stops"])
    col2.metric("Connections", stats["num_connections"])
    col3.metric("Avg Degree", f"{stats['average_degree']:.2f}")

    # Map of stops
    fig3 = px.scatter_mapbox(
        stops_df,
        lat="stop_lat",
        lon="stop_lon",
        hover_name="stop_name",
        zoom=11,
        title="Stop Locations",
        mapbox_style="open-street-map",
    )
    # Add edges
    edge_lats, edge_lons = [], []
    for u, v in G.edges():
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        edge_lats += [u_data.get("lat", 0), v_data.get("lat", 0), None]
        edge_lons += [u_data.get("lon", 0), v_data.get("lon", 0), None]
    fig3.add_trace(go.Scattermapbox(
        lat=edge_lats, lon=edge_lons,
        mode="lines",
        line=dict(width=1, color="blue"),
        name="Routes",
    ))
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Route Optimizer ─────────────────────
with tab3:
    st.subheader("Route Optimization")
    G, stops_df = get_graph(n_zones)

    stop_ids = stops_df["stop_id"].tolist()
    source = st.selectbox("Source Stop", stop_ids, index=0)
    target = st.selectbox("Target Stop", stop_ids, index=min(4, len(stop_ids) - 1))

    if st.button("Find Optimal Route"):
        from src.network.graph_builder import compute_shortest_path
        path, length = compute_shortest_path(G, source, target)
        if path:
            st.success(f"Route: {' → '.join(path)} | Travel weight: {length:.1f}")
        else:
            st.error("No path found between selected stops.")

    st.markdown("---")
    st.subheader("Congestion Analysis")
    st.caption("ℹ️ Congestion loads shown below are synthetic demo values for illustration purposes.")
    # Assign mock loads
    for u, v in G.edges():
        G[u][v]["load"] = np.random.uniform(0, 1000)
    congested = identify_congested_edges(G, threshold=congestion_threshold)
    st.write(f"**Congested edges:** {len(congested)}")
    if congested:
        cong_df = pd.DataFrame(
            [(u, v, d.get("load", 0)) for u, v, d in congested],
            columns=["From", "To", "Load"]
        )
        st.dataframe(cong_df)
