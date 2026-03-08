"""
Interactive Streamlit dashboard with map visualizations for urban mobility.
Enhanced UI with metro schematic map, demand analytics, and route optimization.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.demand_predictor import generate_synthetic_demand, engineer_features, train_demand_model
from src.geospatial.geo_processor import create_grid_zones, haversine_distance
from src.network.graph_builder import build_transit_graph, get_network_stats, compute_shortest_path
from src.optimization.route_optimizer import identify_congested_edges

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Mobility Stockholm",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS – polished, clean design
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Reduce default top padding */
    .block-container { padding-top: 1rem !important; }

    /* App header banner */
    .app-header {
        background: linear-gradient(135deg, #1a2f4e 0%, #2563a8 100%);
        color: white;
        padding: 1.1rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .app-header .title  { font-size: 1.55rem; font-weight: 700; margin: 0; }
    .app-header .subtitle { font-size: 0.83rem; opacity: 0.82; margin-top: 0.25rem; }

    /* Compact tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f3f6;
        padding: 5px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 16px;
        background: transparent;
        border-radius: 7px;
        font-size: 0.87rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1a2f4e !important;
        color: white !important;
    }

    /* Section headings */
    h3 { color: #1a2f4e !important; margin-top: 0.6rem !important; }

    /* Route step card */
    .route-step {
        padding: 0.38rem 0.75rem;
        border-left: 3px solid #2563a8;
        background: #f6f8fb;
        border-radius: 0 6px 6px 0;
        margin: 0.25rem 0;
        font-size: 0.87rem;
    }

    /* Legend card */
    .legend-card {
        background: #f6f8fb;
        border: 1px solid #dde3ec;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.83rem;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    with st.expander("🗺️ Map Options", expanded=True):
        map_style = st.selectbox(
            "Tile Layer",
            ["CartoDB Positron", "OpenStreetMap", "CartoDB Dark Matter"],
            index=0,
            help="Background style for the geographic map",
        )
        show_heatmap = st.checkbox("Demand Heatmap", value=True)
        show_routes  = st.checkbox("Transit Routes",  value=True)

    with st.expander("📊 Simulation", expanded=True):
        n_zones = st.slider("Number of Stops", 5, 30, 15,
                            help="Total stops in the simulated network")
        n_hours = st.slider("Simulation Hours", 24, 336, 168,
                            help="Historical demand window to simulate")
        congestion_threshold = st.slider("Congestion Threshold", 100, 2000, 500,
                                         help="Passengers/hr above which a segment is congested")

    st.markdown("---")
    st.markdown(
        "<small>🚇 <b>Urban Mobility Stockholm</b><br>"
        "Streamlit · Folium · Plotly · XGBoost</small>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Cached data helpers
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
def get_stops_df(n_zones):
    """Generate realistic Stockholm stop locations."""
    center_lat, center_lon = 59.3293, 18.0686
    np.random.seed(42)

    landmarks = [
        ("T-Centralen",  59.3310, 18.0590),
        ("Gamla Stan",   59.3250, 18.0710),
        ("Södermalm",    59.3150, 18.0720),
        ("Östermalm",    59.3370, 18.0890),
        ("Kungsholmen",  59.3340, 18.0280),
        ("Djurgården",   59.3260, 18.1150),
        ("Norrmalm",     59.3380, 18.0550),
        ("Vasastan",     59.3450, 18.0520),
        ("Hornstull",    59.3160, 18.0340),
        ("Slussen",      59.3200, 18.0720),
    ]

    stops = []
    for i in range(n_zones):
        if i < len(landmarks):
            name, lat, lon = landmarks[i]
            stops.append({
                "stop_id":   f"S{i}",
                "stop_name": name,
                "stop_lat":  lat + np.random.uniform(-0.002, 0.002),
                "stop_lon":  lon + np.random.uniform(-0.002, 0.002),
            })
        else:
            angle  = (i / n_zones) * 2 * np.pi
            radius = np.random.uniform(0.01, 0.04)
            stops.append({
                "stop_id":   f"S{i}",
                "stop_name": f"Station {i}",
                "stop_lat":  center_lat + radius * np.cos(angle) + np.random.uniform(-0.005, 0.005),
                "stop_lon":  center_lon + radius * np.sin(angle) * 1.5 + np.random.uniform(-0.005, 0.005),
            })

    return pd.DataFrame(stops)


@st.cache_data
def get_graph(n_zones):
    """Build transit network graph.

    Args:
        n_zones: Number of stops in the simulated network.

    Returns:
        Tuple of (nx.DiGraph, stops_df, stop_times_df).
    """
    stops = get_stops_df(n_zones)
    stop_times_data = []

    n_lines = max(3, n_zones // 4)
    for line in range(n_lines):
        np.random.seed(line + 100)
        line_stops = sorted(
            np.random.choice(n_zones, size=min(n_zones, np.random.randint(4, 8)), replace=False)
        )
        for seq, stop_idx in enumerate(line_stops):
            stop_times_data.append({
                "trip_id":       f"Line {line + 1}",
                "stop_id":       f"S{stop_idx}",
                "stop_sequence": seq,
            })

    # Circular line connecting all stops
    for seq in range(n_zones):
        stop_times_data.append({
            "trip_id": "Circle", "stop_id": f"S{seq}", "stop_sequence": seq,
        })
    stop_times_data.append({
        "trip_id": "Circle", "stop_id": "S0", "stop_sequence": n_zones,
    })

    stop_times = pd.DataFrame(stop_times_data)
    return build_transit_graph(stops, stop_times), stops, stop_times


def get_map_tiles(style):
    return {
        "OpenStreetMap":      "OpenStreetMap",
        "CartoDB Positron":   "CartoDB positron",
        "CartoDB Dark Matter": "CartoDB dark_matter",
    }.get(style, "CartoDB positron")


# ──────────────────────────────────────────────
# Metro schematic builder
# ──────────────────────────────────────────────
_LINE_COLORS = [
    "#e63946",  # Red
    "#0077b6",  # Blue
    "#2dc653",  # Green
    "#f4a261",  # Orange
    "#9d4edd",  # Purple
    "#00b4d8",  # Teal
    "#e76f51",  # Coral
    "#457b9d",  # Steel-blue
]


def create_metro_schematic(stop_times_df: pd.DataFrame, stops_df: pd.DataFrame) -> go.Figure:
    """
    Build a clean schematic metro-style network map using Plotly.

    • Each transit line is drawn as a coloured polyline.
    • Regular stations are white circles with a coloured border.
    • Transfer stations (on ≥ 2 lines) are larger white circles with a
      dark border to make interchanges immediately visible.
    • The legend lets the user toggle individual lines on/off.
    """
    # Build ordered trip → stop lists
    trips_dict = {
        trip_id: group.sort_values("stop_sequence")["stop_id"].tolist()
        for trip_id, group in stop_times_df.groupby("trip_id")
    }

    # Identify transfer stations
    stop_line_count = Counter(
        s for stop_ids in trips_dict.values() for s in set(stop_ids)
    )
    transfers = {s for s, c in stop_line_count.items() if c >= 2}

    stop_lookup = stops_df.set_index("stop_id").to_dict("index")

    fig = go.Figure()

    for i, (trip_id, stop_ids) in enumerate(trips_dict.items()):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]

        valid   = [(s, stop_lookup[s]) for s in stop_ids if s in stop_lookup]
        if len(valid) < 2:
            continue

        lons  = [v["stop_lon"]  for _, v in valid]
        lats  = [v["stop_lat"]  for _, v in valid]
        names = [v["stop_name"] for _, v in valid]
        sids  = [s              for s, _ in valid]

        # Line trace
        fig.add_trace(go.Scatter(
            x=lons, y=lats,
            mode="lines",
            line=dict(width=6, color=color),
            name=trip_id,
            legendgroup=trip_id,
            hoverinfo="none",
        ))

        # Separate regular stops from transfers
        reg_idx = [j for j, s in enumerate(sids) if s not in transfers]
        tr_idx  = [j for j, s in enumerate(sids) if s in transfers]

        if reg_idx:
            fig.add_trace(go.Scatter(
                x=[lons[j] for j in reg_idx],
                y=[lats[j] for j in reg_idx],
                mode="markers+text",
                marker=dict(size=10, color="white", symbol="circle",
                            line=dict(color=color, width=2.5)),
                text=[names[j] for j in reg_idx],
                textposition="top center",
                textfont=dict(size=8, color="#333333"),
                hovertext=[f"{names[j]}<br><i>{trip_id}</i>" for j in reg_idx],
                hoverinfo="text",
                showlegend=False,
                legendgroup=trip_id,
            ))

        if tr_idx:
            fig.add_trace(go.Scatter(
                x=[lons[j] for j in tr_idx],
                y=[lats[j] for j in tr_idx],
                mode="markers+text",
                marker=dict(size=16, color="white", symbol="circle",
                            line=dict(color="#1a2f4e", width=3)),
                text=[names[j] for j in tr_idx],
                textposition="top center",
                textfont=dict(size=8, color="#1a2f4e"),
                hovertext=[f"⇄ {names[j]}<br><i>Transfer station</i>" for j in tr_idx],
                hoverinfo="text",
                showlegend=False,
                legendgroup=trip_id,
            ))

    fig.update_layout(
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#eef2f7",
        height=480,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(
            title=dict(text="<b>Transit Lines</b>", font=dict(size=11)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#d1d5db",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1.8),
        title=dict(
            text="Metro Network – Schematic View  "
                 "<span style='font-size:11px;color:#888'>"
                 "(large circles = transfer stations)</span>",
            font=dict(size=13, color="#1a2f4e"),
            x=0.5, xanchor="center",
        ),
        hovermode="closest",
    )
    return fig


# ──────────────────────────────────────────────
# App header
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="title">🚇 Urban Mobility Stockholm</div>
  <div class="subtitle">
    Real-time transit analytics &nbsp;·&nbsp;
    Demand forecasting &nbsp;·&nbsp;
    Route optimization
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Top KPI strip  – visible from any tab
# ──────────────────────────────────────────────
kpi_graph, kpi_stops, kpi_trips = get_graph(n_zones)
kpi_stats = get_network_stats(kpi_graph)

kc1, kc2, kc3, kc4 = st.columns(4)
kc1.metric("🚏 Stops",       kpi_stats["num_stops"])
kc2.metric("🚇 Lines",       kpi_trips["trip_id"].nunique())
kc3.metric("🔗 Connections", kpi_stats["num_connections"])
kc4.metric("📶 Network",
           "Connected" if kpi_stats["is_weakly_connected"] else "Fragmented")

st.divider()

# ──────────────────────────────────────────────
# Main tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Map & Metro View",
    "📊 Demand Analytics",
    "🔀 Route Planner",
    "📈 Network Stats",
])

# ══════════════════════════════════════════════
# Tab 1 – Map  (geographic  OR  metro schematic)
# ══════════════════════════════════════════════
with tab1:
    G, stops_df, stop_times = get_graph(n_zones)

    view_mode = st.radio(
        "Map view",
        ["📍 Geographic Map", "🚇 Metro Schematic"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "🚇 Metro Schematic":
        # ── Schematic view ──────────────────────
        st.caption(
            "Schematic view · circles = stations · large circles = transfer points · "
            "click legend items to toggle lines"
        )
        st.plotly_chart(create_metro_schematic(stop_times, stops_df),
                        use_container_width=True)

        st.markdown("##### 📋 Line Summary")
        line_summary = (
            stop_times.groupby("trip_id")["stop_id"]
            .nunique()
            .reset_index()
            .rename(columns={"trip_id": "Line", "stop_id": "Unique Stops"})
        )
        st.dataframe(line_summary, use_container_width=True, hide_index=True)

    else:
        # ── Geographic Folium map ────────────────
        col1, col2 = st.columns([3, 1])

        with col1:
            center_lat = stops_df["stop_lat"].mean()
            center_lon = stops_df["stop_lon"].mean()

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles=get_map_tiles(map_style),
            )

            if show_heatmap:
                heat_data = []
                for _, row in stops_df.iterrows():
                    idx = int(row["stop_id"].replace("S", ""))
                    intensity = np.random.uniform(0.3, 1.0) * (1 + np.sin(idx))
                    heat_data.append([row["stop_lat"], row["stop_lon"], intensity])
                HeatMap(
                    heat_data, radius=25, blur=15, max_zoom=13,
                    gradient={0.4: "blue", 0.65: "lime", 0.8: "yellow", 1: "red"},
                ).add_to(m)

            if show_routes:
                for idx, (u, v, _) in enumerate(G.edges(data=True)):
                    u_d, v_d = G.nodes[u], G.nodes[v]
                    if u_d.get("lat") and v_d.get("lat"):
                        folium.PolyLine(
                            [[u_d["lat"], u_d["lon"]], [v_d["lat"], v_d["lon"]]],
                            weight=3,
                            color=_LINE_COLORS[idx % len(_LINE_COLORS)],
                            opacity=0.75,
                        ).add_to(m)

            marker_cluster = MarkerCluster().add_to(m)
            for _, row in stops_df.iterrows():
                popup_html = (
                    f'<div style="font-family:Arial,sans-serif;width:170px;padding:4px">'
                    f'<b style="color:#1a2f4e">🚏 {row["stop_name"]}</b>'
                    f'<hr style="margin:5px 0;border-color:#e0e0e0">'
                    f'<table style="font-size:11px;width:100%">'
                    f'<tr><td><b>ID</b></td><td>{row["stop_id"]}</td></tr>'
                    f'<tr><td><b>Lat</b></td><td>{row["stop_lat"]:.4f}</td></tr>'
                    f'<tr><td><b>Lon</b></td><td>{row["stop_lon"]:.4f}</td></tr>'
                    f'</table></div>'
                )
                folium.Marker(
                    [row["stop_lat"], row["stop_lon"]],
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=row["stop_name"],
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(marker_cluster)

            map_data = st_folium(m, width=None, height=500,
                                 returned_objects=["last_object_clicked"])

        with col2:
            st.markdown("#### 📍 Quick Info")
            stats = get_network_stats(G)
            st.metric("🚏 Stops",       stats["num_stops"])
            st.metric("🔗 Connections", stats["num_connections"])
            st.metric("📊 Avg Degree",  f"{stats['average_degree']:.1f}")

            st.markdown("---")
            st.markdown("#### 🎯 Selected Stop")
            if map_data and map_data.get("last_object_clicked"):
                clicked = map_data["last_object_clicked"]
                st.info(
                    f"📍 **{clicked.get('lat', 0):.4f}°N**\n\n"
                    f"**{clicked.get('lng', 0):.4f}°E**"
                )
            else:
                st.caption("Click a stop on the map for details")

            st.markdown("---")
            st.markdown(
                '<div class="legend-card">'
                "🔵 Transit stops<br>"
                "🌈 Transit routes<br>"
                "🔥 Demand heatmap<br>"
                "🔘 Clustered markers"
                "</div>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════
# Tab 2 – Demand Analytics
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Passenger Demand Analytics")

    demand_df = get_demand_data(n_zones, n_hours).copy()
    demand_df["hour"]        = pd.to_datetime(demand_df["timestamp"]).dt.hour
    demand_df["day_of_week"] = pd.to_datetime(demand_df["timestamp"]).dt.day_name()

    # Quick summary metrics
    avg_demand = demand_df["demand"].mean()
    peak_hour  = demand_df.groupby("hour")["demand"].mean().idxmax()
    peak_day   = demand_df.groupby("day_of_week")["demand"].mean().idxmax()

    s1, s2, s3 = st.columns(3)
    s1.metric("📈 Avg Demand",  f"{avg_demand:.0f} pax/hr")
    s2.metric("⏰ Peak Hour",   f"{peak_hour:02d}:00")
    s3.metric("📅 Busiest Day", (peak_day[:3] if peak_day and len(peak_day) >= 3 else "N/A"))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        hourly = demand_df.groupby("hour")["demand"].mean().reset_index()
        fig1 = px.area(
            hourly, x="hour", y="demand",
            title="📈 Average Demand by Hour",
            labels={"hour": "Hour of Day", "demand": "Avg Passengers"},
            color_discrete_sequence=["#0077b6"],
        )
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily = (
            demand_df.groupby("day_of_week")["demand"]
            .mean()
            .reindex(dow_order)
            .reset_index()
        )
        fig2 = px.bar(
            daily, x="day_of_week", y="demand",
            title="📅 Demand by Day of Week",
            labels={"day_of_week": "Day", "demand": "Avg Passengers"},
            color="demand",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20), coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔥 Zone × Hour Demand Heatmap")
    pivot = demand_df.pivot_table(
        index="zone_id", columns="hour", values="demand", aggfunc="mean"
    )
    fig3 = px.imshow(
        pivot,
        labels={"x": "Hour", "y": "Zone", "color": "Passengers"},
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig3.update_layout(height=350, margin=dict(t=10, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # ML section
    st.markdown("---")
    st.markdown("### 🤖 ML Demand Prediction")

    with st.spinner("Training XGBoost model…"):
        model, metrics, feat_cols = get_trained_model(n_zones, n_hours)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",       f"{metrics['mae']:.2f}",   help="Mean Absolute Error")
    m2.metric("RMSE",      f"{metrics['rmse']:.2f}",  help="Root Mean Squared Error")
    m3.metric("R² Score",  f"{metrics['r2']:.4f}",    help="Coefficient of Determination")
    m4.metric("Features",  len(feat_cols))

    st.markdown("#### 🔮 Demand Prediction")
    p1, p2, p3, p4 = st.columns(4)
    with p1: pred_hour    = st.number_input("Hour (0–23)",         0, 23,  8)
    with p2: pred_dow     = st.number_input("Day of Week (0=Mon)", 0,  6,  1)
    with p3: pred_month   = st.number_input("Month (1–12)",        1, 12,  3)
    with p4: pred_weekend = st.selectbox("Weekend?", [0, 1], index=0)

    if st.button("🔮 Predict", type="primary"):
        input_df = pd.DataFrame([{
            "hour": pred_hour, "day_of_week": pred_dow,
            "month": pred_month, "is_weekend": pred_weekend,
            "demand_lag_1": 120.0, "demand_lag_2": 115.0,
            "demand_lag_3": 110.0, "demand_lag_24": 125.0,
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Demand: **{prediction:.0f} passengers**")

# ══════════════════════════════════════════════
# Tab 3 – Route Planner
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🔀 Route Planner")

    G, stops_df, _ = get_graph(n_zones)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📍 Select Origin & Destination")
        stop_options = {
            f"{row['stop_name']} ({row['stop_id']})": row["stop_id"]
            for _, row in stops_df.iterrows()
        }
        source_display = st.selectbox("🟢 From", list(stop_options.keys()), index=0)
        target_display = st.selectbox("🔴 To",   list(stop_options.keys()),
                                      index=min(4, len(stop_options) - 1))
        source = stop_options[source_display]
        target = stop_options[target_display]
        find_route = st.button("🔍 Find Optimal Route", type="primary",
                               use_container_width=True)

    with col2:
        if find_route:
            path, length = compute_shortest_path(G, source, target)

            if path:
                st.success(
                    f"✅ Route found · **{len(path) - 1} stops** · "
                    f"{length:.1f} min estimated travel time"
                )

                route_map = folium.Map(
                    location=[stops_df["stop_lat"].mean(), stops_df["stop_lon"].mean()],
                    zoom_start=13, tiles=get_map_tiles(map_style),
                )
                route_coords = [[G.nodes[s]["lat"], G.nodes[s]["lon"]] for s in path]
                folium.PolyLine(route_coords, weight=5,
                                color="#e63946", opacity=0.85).add_to(route_map)

                for i, stop_id in enumerate(path):
                    node = G.nodes[stop_id]
                    if i == 0:
                        icon = folium.Icon(color="green", icon="play")
                    elif i == len(path) - 1:
                        icon = folium.Icon(color="red", icon="stop")
                    else:
                        icon = folium.Icon(color="blue", icon="info-sign")
                    folium.Marker(
                        [node["lat"], node["lon"]],
                        popup=node.get("name", stop_id),
                        tooltip=node.get("name", stop_id),
                        icon=icon,
                    ).add_to(route_map)

                st_folium(route_map, width=None, height=380)

                st.markdown("#### 🗺️ Step-by-step Directions")
                route_names = [G.nodes[s].get("name", s) for s in path]
                for i, name in enumerate(route_names):
                    marker = "🟢" if i == 0 else ("🔴" if i == len(route_names) - 1 else "🔵")
                    st.markdown(
                        f'<div class="route-step">{marker} {name}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.error("❌ No route found between the selected stops.")
        else:
            st.info("👆 Select origin & destination, then click **Find Optimal Route**")
            preview_map = folium.Map(
                location=[stops_df["stop_lat"].mean(), stops_df["stop_lon"].mean()],
                zoom_start=13, tiles=get_map_tiles(map_style),
            )
            for _, row in stops_df.iterrows():
                folium.CircleMarker(
                    [row["stop_lat"], row["stop_lon"]],
                    radius=5, color="#0077b6", fill=True,
                    popup=row["stop_name"],
                ).add_to(preview_map)
            st_folium(preview_map, width=None, height=380)

    # Congestion analysis
    st.markdown("---")
    st.markdown("### 🚦 Congestion Analysis")

    for u, v in G.edges():
        G[u][v]["load"] = np.random.uniform(0, 1000)

    congested = identify_congested_edges(G, threshold=congestion_threshold)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric("⚠️ Congested Segments", len(congested))
        st.caption(f"Threshold: {congestion_threshold} pax/hr")
        if congested:
            cong_df = pd.DataFrame(
                [(G.nodes[u].get("name", u), G.nodes[v].get("name", v),
                  int(d.get("load", 0)))
                 for u, v, d in congested],
                columns=["From", "To", "Load"],
            )
            st.dataframe(cong_df, use_container_width=True, hide_index=True)

    with c2:
        if congested:
            fig_cong = px.bar(
                cong_df.head(10),
                x="Load",
                y=[f"{r['From']} → {r['To']}" for _, r in cong_df.head(10).iterrows()],
                orientation="h",
                title="Top Congested Segments",
                color="Load",
                color_continuous_scale="Reds",
            )
            fig_cong.update_layout(
                yaxis_title="", showlegend=False,
                margin=dict(t=40, b=10), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cong, use_container_width=True)
        else:
            st.success("✅ Network flowing smoothly — no congestion detected.")

# ══════════════════════════════════════════════
# Tab 4 – Network Statistics
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Network Statistics")

    G, stops_df, _ = get_graph(n_zones)
    stats = get_network_stats(G)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🚏 Stops",       stats["num_stops"])
    mc2.metric("🔗 Connections", stats["num_connections"])
    mc3.metric("📊 Avg Degree",  f"{stats['average_degree']:.2f}")
    mc4.metric("🔄 Connected",   "Yes" if stats["is_weakly_connected"] else "No")

    col1, col2 = st.columns(2)

    with col1:
        degrees = [d for _, d in G.degree()]
        fig_deg = px.histogram(
            x=degrees, nbins=20,
            title="Stop Connectivity Distribution",
            labels={"x": "Connections", "y": "Count"},
            color_discrete_sequence=["#0077b6"],
        )
        fig_deg.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    with col2:
        edge_x, edge_y = [], []
        for u, v in G.edges():
            u_d, v_d = G.nodes[u], G.nodes[v]
            edge_x += [u_d.get("lon", 0), v_d.get("lon", 0), None]
            edge_y += [u_d.get("lat", 0), v_d.get("lat", 0), None]

        node_x   = [G.nodes[n].get("lon", 0)  for n in G.nodes()]
        node_y   = [G.nodes[n].get("lat", 0)  for n in G.nodes()]
        node_txt = [G.nodes[n].get("name", n) for n in G.nodes()]
        node_deg = [G.degree(n)               for n in G.nodes()]

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1, color="#c0c0c0"), hoverinfo="none",
        ))
        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=[10 + d * 2 for d in node_deg],
                color=node_deg, colorscale="Blues", showscale=True,
                colorbar=dict(title="Degree", thickness=12),
            ),
            text=node_txt, textposition="top center",
            textfont=dict(size=8), hoverinfo="text",
        ))
        fig_net.update_layout(
            title="Network Graph",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("#### 📋 All Stops")
    disp = stops_df.copy()
    disp["Connections"] = disp["stop_id"].apply(lambda x: G.degree(x))
    disp.columns = ["ID", "Name", "Latitude", "Longitude", "Connections"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#aaa;font-size:0.78rem;padding:0.4rem 0">'
    "🚇 Urban Mobility Stockholm &nbsp;·&nbsp; "
    "Streamlit · Folium · Plotly · XGBoost"
    "</div>",
    unsafe_allow_html=True,
)
