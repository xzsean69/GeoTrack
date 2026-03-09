"""
GeoTrack – Urban Mobility Dashboard for Stockholm.

This dashboard lets you:
  • Visualise the simulated public-transit network on an interactive map
  • Explore demand-zone grid with colour-coded demand intensity
  • Forecast passenger demand with an XGBoost model (weather + POI features)
  • Plan optimal routes between any two stops
  • Identify congested segments in real time
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

from src.ml.demand_predictor import (
    generate_synthetic_demand, engineer_features, train_demand_model,
    ALL_FEATURE_COLS, BASE_FEATURE_COLS, WEATHER_FEATURE_COLS, POI_FEATURE_COLS,
)
from src.geospatial.geo_processor import create_grid_zones, haversine_distance
from src.network.graph_builder import build_transit_graph, get_network_stats, compute_shortest_path
from src.optimization.route_optimizer import identify_congested_edges
from src.data.smhi_weather import generate_synthetic_weather
from src.data.poi_data import get_poi_dataframe, add_poi_features, compute_poi_scores

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GeoTrack – Urban Mobility Stockholm",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────
_SESSION_DEFAULTS = {
    "route_path":   None,
    "route_length": None,
    "route_src":    None,
    "route_dst":    None,
}
for _key, _val in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

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
    .app-header .title    { font-size: 1.55rem; font-weight: 700; margin: 0; }
    .app-header .subtitle { font-size: 0.83rem; opacity: 0.82; margin-top: 0.25rem; }
    .app-header .tagline  { font-size: 0.78rem; opacity: 0.68; margin-top: 0.15rem; }

    /* Onboarding info card */
    .onboard-card {
        background: #eef6ff;
        border: 1px solid #b8d4f5;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        font-size: 0.86rem;
        line-height: 1.7;
        margin-bottom: 0.8rem;
    }
    .onboard-card b { color: #1a2f4e; }

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
    st.caption(
        "Adjust the simulation parameters below. Changes apply to all tabs immediately."
    )

    with st.expander("🗺️ Map Options", expanded=True):
        map_style = st.selectbox(
            "Tile Layer",
            ["CartoDB Positron", "OpenStreetMap", "CartoDB Dark Matter"],
            index=0,
            help="Background style for the geographic map",
        )
        show_heatmap = st.checkbox(
            "Demand Heatmap",
            value=True,
            help="Overlay a colour-coded heatmap showing relative passenger volumes at each stop",
        )
        show_routes  = st.checkbox(
            "Transit Routes",
            value=True,
            help="Draw route lines between connected stops on the geographic map",
        )

    with st.expander("📊 Simulation", expanded=True):
        n_zones = st.slider(
            "Number of Stops", 5, 30, 15,
            help="Total number of transit stops in the simulated Stockholm network",
        )
        n_hours = st.slider(
            "Simulation Hours", 24, 336, 168,
            help="Length of the synthetic demand history used for ML training (1 week = 168 h)",
        )
        congestion_threshold = st.slider(
            "Congestion Threshold (pax/hr)", 100, 2000, 500,
            help="Segments with load above this value are flagged as congested in the Route Planner tab",
        )

    st.markdown("---")
    st.markdown(
        "<small>🚇 <b>GeoTrack – Urban Mobility Stockholm</b><br>"
        "Built with Streamlit · Folium · Plotly · XGBoost<br>"
        "<i>All data is synthetic and for demonstration only.</i></small>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Cached data helpers
# ──────────────────────────────────────────────
@st.cache_data
def get_weather_data(n_hours):
    """Return synthetic SMHI-style weather data for the simulation period."""
    return generate_synthetic_weather(n_hours=n_hours)


@st.cache_data
def get_demand_data(n_zones, n_hours):
    weather_df = get_weather_data(n_hours)
    return generate_synthetic_demand(
        n_zones=n_zones, n_hours=n_hours, weather_df=weather_df
    )


@st.cache_data
def get_trained_model(n_zones, n_hours):
    weather_df = get_weather_data(n_hours)
    df = get_demand_data(n_zones, n_hours)
    df_feat = engineer_features(df, weather_df=weather_df)
    model, metrics = train_demand_model(df_feat, ALL_FEATURE_COLS)
    return model, metrics, ALL_FEATURE_COLS


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
def get_stops_with_poi(n_zones):
    """Return stops DataFrame with appended POI proximity scores."""
    stops = get_stops_df(n_zones)
    return add_poi_features(stops)


@st.cache_data
def get_stop_poi_scores(n_zones) -> dict[str, dict[str, float]]:
    """Return per-stop POI scores keyed by zone_id (zone_0 … zone_{n-1})."""
    stops = get_stops_df(n_zones)
    result: dict[str, dict[str, float]] = {}
    for i, row in stops.iterrows():
        zone_id = f"zone_{i}"
        result[zone_id] = compute_poi_scores(row["stop_lat"], row["stop_lon"])
    return result


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

# ──────────────────────────────────────────────
# SL (Stockholms Lokaltrafik) Metro – real station data
# ──────────────────────────────────────────────
# Approximate geographic coordinates for each station on each line.
# Transfer stations (shared by multiple lines) use the same coordinates.
_SL_METRO_LINES: dict = {
    "T10 · Blå linje": {
        "color": "#0052A3",
        "stations": [
            ("Hjulsta",            59.3667, 17.8707),
            ("Vällingby",          59.3621, 17.8706),
            ("Råcksta",            59.3610, 17.8921),
            ("Skälby",             59.3594, 17.9080),
            ("Islandstorget",      59.3494, 17.9182),
            ("Ängbyplan",          59.3455, 17.9233),
            ("Åkeshov",            59.3408, 17.9316),
            ("Brommaplan",         59.3362, 17.9393),
            ("Thorildsplan",       59.3331, 18.0147),
            ("Fridhemsplan",       59.3322, 18.0278),
            ("St. Eriksplan",      59.3370, 18.0229),
            ("Odenplan",           59.3414, 18.0276),
            ("Rådhuset",           59.3326, 18.0441),
            ("T-Centralen",        59.3310, 18.0590),
            ("Kungsträdgården",    59.3321, 18.0717),
        ],
    },
    "T11 · Blå linje": {
        "color": "#0052A3",
        "stations": [
            ("Akalla",             59.4157, 17.9274),
            ("Husby",              59.4117, 17.9387),
            ("Kista",              59.4033, 17.9507),
            ("Hallonbergen",       59.3740, 17.9620),
            ("Näckrosen",          59.3659, 17.9864),
            ("Solna centrum",      59.3604, 18.0014),
            ("Västra skogen",      59.3502, 18.0025),
            ("Stadshagen",         59.3362, 18.0090),
            ("Thorildsplan",       59.3331, 18.0147),
            ("Fridhemsplan",       59.3322, 18.0278),
            ("St. Eriksplan",      59.3370, 18.0229),
            ("Odenplan",           59.3414, 18.0276),
            ("Rådhuset",           59.3326, 18.0441),
            ("T-Centralen",        59.3310, 18.0590),
            ("Kungsträdgården",    59.3321, 18.0717),
        ],
    },
    "T13 · Röd linje": {
        "color": "#E8212F",
        "stations": [
            ("Norsborg",           59.2577, 17.8313),
            ("Alby",               59.2604, 17.8483),
            ("Hallunda",           59.2667, 17.8651),
            ("Fittja",             59.2714, 17.8779),
            ("Masmo",              59.2784, 17.8878),
            ("Vårby gård",         59.2824, 17.8993),
            ("Vårberg",            59.2852, 17.9097),
            ("Skärholmen",         59.2773, 17.9230),
            ("Sätra",              59.2942, 17.9450),
            ("Bredäng",            59.2987, 17.9582),
            ("Mälarhöjden",        59.3014, 17.9716),
            ("Axelsberg",          59.3036, 17.9861),
            ("Örnsberg",           59.3068, 18.0016),
            ("Liljeholmen",        59.3108, 18.0195),
            ("Hornstull",          59.3161, 18.0342),
            ("Zinkensdamm",        59.3172, 18.0462),
            ("Mariatorget",        59.3183, 18.0535),
            ("Slussen",            59.3196, 18.0715),
            ("Gamla stan",         59.3237, 18.0686),
            ("T-Centralen",        59.3310, 18.0590),
            ("Östermalmstorg",     59.3354, 18.0750),
            ("Karlaplan",          59.3401, 18.0844),
            ("Gärdet",             59.3423, 18.1010),
            ("Ropsten",            59.3578, 18.1085),
        ],
    },
    "T14 · Röd linje": {
        "color": "#E8212F",
        "stations": [
            ("Fruängen",           59.2991, 17.9771),
            ("Västertorp",         59.3039, 17.9900),
            ("Hägerstensåsen",     59.3068, 18.0033),
            ("Telefonplan",        59.3099, 18.0092),
            ("Midsommarkransen",   59.3132, 18.0194),
            ("Liljeholmen",        59.3108, 18.0195),
            ("Hornstull",          59.3161, 18.0342),
            ("Zinkensdamm",        59.3172, 18.0462),
            ("Mariatorget",        59.3183, 18.0535),
            ("Slussen",            59.3196, 18.0715),
            ("Gamla stan",         59.3237, 18.0686),
            ("T-Centralen",        59.3310, 18.0590),
            ("Östermalmstorg",     59.3354, 18.0750),
            ("Stadion",            59.3444, 18.0820),
            ("Tekniska Högskolan", 59.3471, 18.0716),
            ("Universitetet",      59.3660, 18.0574),
            ("Danderyds sjukhus",  59.3930, 18.0369),
            ("Mörby centrum",      59.4010, 18.0311),
        ],
    },
    "T17 · Grön linje": {
        "color": "#00A650",
        "stations": [
            ("Hässelby strand",    59.3630, 17.8283),
            ("Hässelby gård",      59.3634, 17.8467),
            ("Johannelund",        59.3534, 17.8700),
            ("Vällingby",          59.3621, 17.8706),
            ("Råcksta",            59.3610, 17.8921),
            ("Skälby",             59.3594, 17.9080),
            ("Islandstorget",      59.3494, 17.9182),
            ("Ängbyplan",          59.3455, 17.9233),
            ("Åkeshov",            59.3408, 17.9316),
            ("Brommaplan",         59.3362, 17.9393),
            ("Abrahamsberg",       59.3337, 17.9624),
            ("Stora mossen",       59.3328, 17.9783),
            ("Alvik",              59.3318, 17.9894),
            ("Kristineberg",       59.3340, 18.0060),
            ("Thorildsplan",       59.3331, 18.0147),
            ("T-Centralen",        59.3310, 18.0590),
            ("Gamla stan",         59.3237, 18.0686),
            ("Slussen",            59.3196, 18.0715),
            ("Skanstull",          59.3106, 18.0727),
            ("Gullmarsplan",       59.3003, 18.0813),
            ("Skärmarbrink",       59.2989, 18.0897),
            ("Globen",             59.2943, 18.0823),
            ("Enskede gård",       59.2889, 18.0706),
            ("Sockenplan",         59.2826, 18.0590),
            ("Stureby",            59.2780, 18.0494),
            ("Björkhagen",         59.2729, 18.0623),
            ("Kärrtorp",           59.2685, 18.0737),
            ("Hagsätra",           59.2643, 18.0329),
        ],
    },
    "T18 · Grön linje": {
        "color": "#00A650",
        "stations": [
            ("Hässelby strand",    59.3630, 17.8283),
            ("Hässelby gård",      59.3634, 17.8467),
            ("Johannelund",        59.3534, 17.8700),
            ("Vällingby",          59.3621, 17.8706),
            ("Råcksta",            59.3610, 17.8921),
            ("Skälby",             59.3594, 17.9080),
            ("Islandstorget",      59.3494, 17.9182),
            ("Ängbyplan",          59.3455, 17.9233),
            ("Åkeshov",            59.3408, 17.9316),
            ("Brommaplan",         59.3362, 17.9393),
            ("Abrahamsberg",       59.3337, 17.9624),
            ("Stora mossen",       59.3328, 17.9783),
            ("Alvik",              59.3318, 17.9894),
            ("Kristineberg",       59.3340, 18.0060),
            ("Thorildsplan",       59.3331, 18.0147),
            ("T-Centralen",        59.3310, 18.0590),
            ("Gamla stan",         59.3237, 18.0686),
            ("Slussen",            59.3196, 18.0715),
            ("Skanstull",          59.3106, 18.0727),
            ("Gullmarsplan",       59.3003, 18.0813),
            ("Skärmarbrink",       59.2989, 18.0897),
            ("Globen",             59.2943, 18.0823),
            ("Enskede gård",       59.2889, 18.0706),
            ("Sockenplan",         59.2826, 18.0590),
            ("Stureby",            59.2780, 18.0494),
            ("Björkhagen",         59.2729, 18.0623),
            ("Kärrtorp",           59.2685, 18.0737),
            ("Farsta strand",      59.2551, 18.1011),
        ],
    },
    "T19 · Grön linje": {
        "color": "#00A650",
        "stations": [
            ("Hässelby strand",    59.3630, 17.8283),
            ("Hässelby gård",      59.3634, 17.8467),
            ("Johannelund",        59.3534, 17.8700),
            ("Vällingby",          59.3621, 17.8706),
            ("Råcksta",            59.3610, 17.8921),
            ("Skälby",             59.3594, 17.9080),
            ("Islandstorget",      59.3494, 17.9182),
            ("Ängbyplan",          59.3455, 17.9233),
            ("Åkeshov",            59.3408, 17.9316),
            ("Brommaplan",         59.3362, 17.9393),
            ("Abrahamsberg",       59.3337, 17.9624),
            ("Stora mossen",       59.3328, 17.9783),
            ("Alvik",              59.3318, 17.9894),
            ("Kristineberg",       59.3340, 18.0060),
            ("Thorildsplan",       59.3331, 18.0147),
            ("T-Centralen",        59.3310, 18.0590),
            ("Gamla stan",         59.3237, 18.0686),
            ("Slussen",            59.3196, 18.0715),
            ("Skanstull",          59.3106, 18.0727),
            ("Skarpnäck",          59.2742, 18.1406),
        ],
    },
}


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


def create_sl_metro_schematic() -> go.Figure:
    """
    Build a schematic Plotly figure that mirrors the SL (Stockholms Lokaltrafik)
    T-bana (metro) network topology and station names.

    • Uses actual geographic coordinates of each station, which naturally produces
      the characteristic SL metro shape.
    • Each line is colour-coded: Blue (#0052A3), Red (#E8212F), Green (#00A650).
    • Transfer stations (served by ≥ 2 lines) are shown as larger dark-bordered
      circles to make interchanges immediately visible.
    • Legend lets the user toggle individual lines on/off.
    """
    # Identify transfer stations across all SL lines
    # Identify transfer stations: only those shared between lines of DIFFERENT colours
    # (e.g. Blue ↔ Green, Red ↔ Green) — not just between variants of the same line.
    station_colors: dict = {}
    for line_data in _SL_METRO_LINES.values():
        color = line_data["color"]
        for name, _lat, _lon in line_data["stations"]:
            station_colors.setdefault(name, set()).add(color)
    transfers = {s for s, colors in station_colors.items() if len(colors) >= 2}

    fig = go.Figure()

    for line_name, line_data in _SL_METRO_LINES.items():
        color = line_data["color"]
        stations = line_data["stations"]

        lons  = [lon  for _, _lat, lon  in stations]
        lats  = [lat  for _, lat,  _lon in stations]
        names = [name for name, _, _    in stations]

        # Line trace
        fig.add_trace(go.Scatter(
            x=lons, y=lats,
            mode="lines",
            line=dict(width=6, color=color),
            name=line_name,
            legendgroup=line_name,
            hoverinfo="none",
        ))

        reg_idx = [j for j, n in enumerate(names) if n not in transfers]
        tr_idx  = [j for j, n in enumerate(names) if n in     transfers]

        if reg_idx:
            fig.add_trace(go.Scatter(
                x=[lons[j] for j in reg_idx],
                y=[lats[j] for j in reg_idx],
                mode="markers+text",
                marker=dict(size=9, color="white", symbol="circle",
                            line=dict(color=color, width=2)),
                text=[names[j] for j in reg_idx],
                textposition="top center",
                textfont=dict(size=7, color="#333333"),
                hovertext=[f"{names[j]}<br><i>{line_name}</i>" for j in reg_idx],
                hoverinfo="text",
                showlegend=False,
                legendgroup=line_name,
            ))

        if tr_idx:
            fig.add_trace(go.Scatter(
                x=[lons[j] for j in tr_idx],
                y=[lats[j] for j in tr_idx],
                mode="markers+text",
                marker=dict(size=15, color="white", symbol="circle",
                            line=dict(color="#1a2f4e", width=3)),
                text=[names[j] for j in tr_idx],
                textposition="top center",
                textfont=dict(size=8, color="#1a2f4e", family="Arial Black"),
                hovertext=[f"⇄ {names[j]}<br><i>Transfer station · {line_name}</i>"
                           for j in tr_idx],
                hoverinfo="text",
                showlegend=False,
                legendgroup=line_name,
            ))

    fig.update_layout(
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#eef2f7",
        height=600,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            title=dict(text="<b>SL Metro Lines</b>", font=dict(size=11)),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#d1d5db",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1.8),
        title=dict(
            text="SL T-bana – Metro Schematic  "
                 "<span style='font-size:11px;color:#888'>"
                 "(large circles = transfer stations · click legend to toggle lines)</span>",
            font=dict(size=14, color="#1a2f4e"),
            x=0.5, xanchor="center",
        ),
        hovermode="closest",
    )
    return fig


@st.cache_data
def get_sl_station_poi_scores() -> list[dict]:
    """
    Return a list of unique SL metro stations with their coordinates and
    POI proximity scores normalised to [0, 1] across all stations (matching
    the scale used during model training via ``add_poi_features``).
    Results are cached across re-renders.
    """
    seen: dict = {}
    for line_name, line_data in _SL_METRO_LINES.items():
        for name, lat, lon in line_data["stations"]:
            if name not in seen:
                scores = compute_poi_scores(lat, lon)
                seen[name] = {
                    "name":  name,
                    "lat":   lat,
                    "lon":   lon,
                    "lines": [line_name],
                    **{f"poi_{k}": v for k, v in scores.items()},
                }
            else:
                if line_name not in seen[name]["lines"]:
                    seen[name]["lines"].append(line_name)

    stations = list(seen.values())

    # Normalise each POI category to [0, 1] across all stations so the scale
    # matches the normalised features the model was trained on.
    poi_cats = ["poi_office", "poi_university", "poi_hospital",
                "poi_shopping", "poi_tourist", "poi_transit_hub"]
    for cat in poi_cats:
        cat_max = max((s[cat] for s in stations), default=0.0)
        if cat_max > 0:
            for s in stations:
                s[cat] = s[cat] / cat_max

    return stations


# ──────────────────────────────────────────────
# App header
# ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="title">🚇 GeoTrack – Urban Mobility Stockholm</div>
  <div class="subtitle">
    Interactive transit analytics &nbsp;·&nbsp;
    Passenger demand forecasting &nbsp;·&nbsp;
    Shortest-path route planning &nbsp;·&nbsp;
    Congestion detection
  </div>
  <div class="tagline">
    Explore the simulated Stockholm transit network: visualise stops &amp; routes on a live map,
    view the demand-zone grid, predict demand with ML (weather + POI features),
    plan optimal journeys, and identify congested segments.
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
    "🗺️ Live Map & Network",
    "📊 Demand Analytics",
    "🔀 Plan My Route",
    "📈 Network Statistics",
])

# ══════════════════════════════════════════════
# Tab 1 – Map  (geographic  OR  metro schematic  OR  grid demand zones)
# ══════════════════════════════════════════════
with tab1:
    G, stops_df, stop_times = get_graph(n_zones)

    st.markdown(
        '<div class="onboard-card">'
        "📌 <b>How to use this tab:</b> Switch between the live geographic map, the "
        "SL metro schematic, the <b>Grid Demand Zones</b> view, and the new "
        "<b>Map Prediction</b> view using the toggle below. "
        "The Grid view shows Stockholm divided into a colour-coded grid where "
        "each cell's colour reflects the aggregated passenger demand – from cool blue "
        "(low demand) through yellow to red (high demand). "
        "The Map Prediction view uses the XGBoost model to forecast demand at every "
        "SL metro station across the city map. "
        "Toggle the heatmap and route overlays in the sidebar."
        "</div>",
        unsafe_allow_html=True,
    )

    view_mode = st.radio(
        "Map view",
        ["📍 Geographic Map", "🚇 Metro Schematic", "🟦 Grid Demand Zones", "🗺️ Map Prediction"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "🚇 Metro Schematic":
        # ── SL Schematic view ───────────────────────
        st.caption(
            "SL T-bana schematic · based on Stockholms Lokaltrafik metro network · "
            "circles = stations · large circles = transfer stations · "
            "click legend items to toggle lines"
        )
        st.plotly_chart(create_sl_metro_schematic(), use_container_width=True)

        st.markdown("##### 📋 SL Metro Line Summary")
        line_summary_rows = []
        for line_name, line_data in _SL_METRO_LINES.items():
            unique_stations = len({name for name, _, _ in line_data["stations"]})
            color_hex = line_data["color"]
            line_summary_rows.append({
                "Line": line_name,
                "Colour": color_hex,
                "Stations": unique_stations,
                "Terminus A": line_data["stations"][0][0],
                "Terminus B": line_data["stations"][-1][0],
            })
        sl_summary_df = pd.DataFrame(line_summary_rows)
        st.dataframe(sl_summary_df, use_container_width=True, hide_index=True)

    elif view_mode == "🟦 Grid Demand Zones":
        # ── Grid Demand Zones view ────────────────
        st.markdown("#### 🟦 Demand Zone Grid – Stockholm")
        st.caption(
            "Each cell represents a geographic grid zone. Colour encodes the average "
            "passenger demand aggregated for all stops within that cell. "
            "Blue = low demand · Yellow = medium · Red = high demand."
        )

        grid_cols = st.columns([3, 1])
        with grid_cols[1]:
            grid_rows = st.slider("Grid rows",    3, 12, 6, key="grid_rows")
            grid_ncols = st.slider("Grid columns", 3, 12, 6, key="grid_ncols")

        # Build grid zones around Stockholm stop bounding box
        pad = 0.015  # degree padding
        minx = stops_df["stop_lon"].min() - pad
        maxx = stops_df["stop_lon"].max() + pad
        miny = stops_df["stop_lat"].min() - pad
        maxy = stops_df["stop_lat"].max() + pad

        grid_gdf = create_grid_zones(
            bounds=(minx, miny, maxx, maxy),
            n_rows=grid_rows,
            n_cols=grid_ncols,
        )

        # Average demand per zone from demand data
        demand_df_grid = get_demand_data(n_zones, n_hours)
        zone_demand = demand_df_grid.groupby("zone_id")["demand"].mean().to_dict()

        # Assign demand to grid cells by counting stops inside each cell
        # and summing their zone demand scores
        import geopandas as gpd
        from shapely.geometry import Point as SPoint

        stops_gdf = gpd.GeoDataFrame(
            stops_df.reset_index(drop=True),
            geometry=[SPoint(r["stop_lon"], r["stop_lat"]) for _, r in stops_df.iterrows()],
            crs="EPSG:4326",
        )
        # Add per-stop average demand using the demand data zone mapping
        stops_gdf["demand_val"] = [
            zone_demand.get(f"zone_{i}", 0.0) for i in range(len(stops_gdf))
        ]
        joined = gpd.sjoin(stops_gdf, grid_gdf, how="left", predicate="within")
        # geopandas renames the right-side 'zone_id' to 'zone_id_right' only when
        # there is a naming conflict; use whichever column name is present.
        zone_id_col = "zone_id" if "zone_id" in joined.columns else "zone_id_right"
        cell_demand = joined.groupby(zone_id_col)["demand_val"].mean()
        grid_gdf["demand"] = grid_gdf["zone_id"].map(cell_demand).fillna(0)

        # Normalise for colour scale
        d_min = grid_gdf["demand"].min()
        d_max = grid_gdf["demand"].max()
        grid_gdf["demand_norm"] = (
            (grid_gdf["demand"] - d_min) / max(d_max - d_min, 1)
        )

        def _demand_color(norm_val: float) -> str:
            """Map normalised demand (0–1) to a hex colour (blue→yellow→red)."""
            if norm_val <= 0.5:
                t = norm_val * 2          # 0→1
                r = int(0   + t * 255)
                g = int(0   + t * 255)
                b = int(200 - t * 200)
            else:
                t = (norm_val - 0.5) * 2  # 0→1
                r = 255
                g = int(255 - t * 255)
                b = 0
            return f"#{r:02x}{g:02x}{b:02x}"

        # Build Folium map with coloured grid rectangles
        with grid_cols[0]:
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2
            gm = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles=get_map_tiles(map_style),
            )

            for _, cell in grid_gdf.iterrows():
                geom = cell["geometry"]
                b = geom.bounds           # (minx, miny, maxx, maxy)
                norm_val = float(cell["demand_norm"])
                color = _demand_color(norm_val)
                demand_val = float(cell["demand"])
                zone_label = cell["zone_id"]

                folium.Rectangle(
                    bounds=[[b[1], b[0]], [b[3], b[2]]],
                    color="#333333",
                    weight=0.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.55,
                    tooltip=(
                        f"<b>{zone_label}</b><br>"
                        f"Avg demand: {demand_val:.0f} pax/hr<br>"
                        f"Intensity: {norm_val:.0%}"
                    ),
                ).add_to(gm)

            # Add stop markers on top
            for _, row in stops_df.iterrows():
                folium.CircleMarker(
                    [row["stop_lat"], row["stop_lon"]],
                    radius=5,
                    color="#1a2f4e",
                    fill=True,
                    fill_color="white",
                    fill_opacity=0.9,
                    weight=2,
                    tooltip=row["stop_name"],
                ).add_to(gm)

            st_folium(
                gm,
                width=None,
                height=500,
                returned_objects=[],
                key=f"grid_map_{n_zones}_{grid_rows}_{grid_ncols}_{map_style}",
            )

        with grid_cols[1]:
            st.markdown("#### 🎨 Demand Scale")
            st.markdown(
                '<div class="legend-card">'
                '<span style="color:#0000c8;font-size:1.2em">■</span> Very low<br>'
                '<span style="color:#7f7f00;font-size:1.2em">■</span> Medium<br>'
                '<span style="color:#ff8000;font-size:1.2em">■</span> High<br>'
                '<span style="color:#ff0000;font-size:1.2em">■</span> Peak demand<br>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.metric("Grid cells", grid_rows * grid_ncols)
            st.metric("Cells with stops", int((grid_gdf["demand"] > 0).sum()))
            st.metric("Peak demand", f"{d_max:.0f} pax/hr")
            st.metric("Avg demand",  f"{grid_gdf['demand'][grid_gdf['demand'] > 0].mean():.0f} pax/hr"
                      if (grid_gdf["demand"] > 0).any() else "—")

        # Zone demand bar chart
        st.markdown("#### 📊 Demand by Grid Zone (top 20 populated cells)")
        top_cells = (
            grid_gdf[grid_gdf["demand"] > 0]
            .sort_values("demand", ascending=False)
            .head(20)
        )
        if not top_cells.empty:
            fig_grid = px.bar(
                top_cells,
                x="zone_id",
                y="demand",
                color="demand",
                color_continuous_scale="RdYlBu_r",
                labels={"zone_id": "Grid Zone", "demand": "Avg Demand (pax/hr)"},
                title="Average Passenger Demand per Grid Zone",
            )
            fig_grid.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=20),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_grid, use_container_width=True)

    elif view_mode == "🗺️ Map Prediction":
        # ── Map Prediction view ────────────────────
        st.markdown("#### 🗺️ Demand Prediction Map – All SL Metro Stations")
        st.markdown(
            '<div class="onboard-card">'
            "The XGBoost demand model predicts passenger demand at <b>every SL metro "
            "station</b> for the time and weather conditions you select below. "
            "Stations are colour-coded from <b style='color:#0000cc'>blue</b> (low demand) "
            "through <b style='color:#e8e800'>yellow</b> to <b style='color:#cc0000'>red</b> "
            "(peak demand). Hover over a station to see its predicted demand."
            "</div>",
            unsafe_allow_html=True,
        )

        # ── User controls ────────────────────────────
        mp_col_controls, mp_col_map = st.columns([1, 3])

        with mp_col_controls:
            st.markdown("##### ⏰ Time")
            mp_hour    = st.slider("Hour",         0, 23,  8,  key="mp_hour")
            mp_dow     = st.slider("Day (0=Mon)",  0,  6,  1,  key="mp_dow")
            mp_month   = st.slider("Month",        1, 12,  6,  key="mp_month")
            mp_weekend = int(mp_dow >= 5)
            st.caption(f"Weekend: {'Yes' if mp_weekend else 'No'}")

            st.markdown("##### 🌦️ Weather")
            mp_temp  = st.slider("Temp (°C)",    -20.0, 35.0, 10.0, 0.5, key="mp_temp")
            mp_rain  = st.slider("Precip (mm)",    0.0, 20.0,  0.0, 0.1, key="mp_rain")
            mp_wind  = st.slider("Wind (m/s)",     0.0, 25.0,  3.0, 0.5, key="mp_wind")
            mp_humid = st.slider("Humidity (%)",    30, 100,    70,       key="mp_humid")
            mp_israiny  = int(mp_rain > 0.1)
            _cold_b     = max(0.0, min(0.2, (10.0 - mp_temp) / 10.0 * 0.1))
            _rain_b     = 0.15 if mp_rain > 0.1 else 0.0
            _wind_b     = 0.05 if mp_wind > 8.0  else 0.0
            _heat_p     = -0.05 if mp_temp > 25.0 else 0.0
            mp_wfactor  = round(min(1.4, max(0.5, 1.0 + _cold_b + _rain_b + _wind_b + _heat_p)), 3)
            st.metric("Demand factor", mp_wfactor)

        with mp_col_map:
            with st.spinner("Computing demand predictions for all SL stations…"):
                _model, _metrics, _feat_cols = get_trained_model(n_zones, n_hours)
                _avg_demand = float(get_demand_data(n_zones, n_hours)["demand"].mean())
                _sl_stations = get_sl_station_poi_scores()

                _predictions: list[dict] = []
                for _stn in _sl_stations:
                    _input_df = pd.DataFrame([{
                        "hour":                  mp_hour,
                        "day_of_week":           mp_dow,
                        "month":                 mp_month,
                        "is_weekend":            mp_weekend,
                        "demand_lag_1":          _avg_demand,
                        "demand_lag_2":          _avg_demand,
                        "demand_lag_3":          _avg_demand,
                        "demand_lag_24":         _avg_demand,
                        "temperature":           mp_temp,
                        "precipitation":         mp_rain,
                        "wind_speed":            mp_wind,
                        "relative_humidity":     mp_humid,
                        "is_rainy":              mp_israiny,
                        "weather_demand_factor": mp_wfactor,
                        "poi_office":            _stn.get("poi_office",      0.0),
                        "poi_university":        _stn.get("poi_university",  0.0),
                        "poi_hospital":          _stn.get("poi_hospital",    0.0),
                        "poi_shopping":          _stn.get("poi_shopping",    0.0),
                        "poi_tourist":           _stn.get("poi_tourist",     0.0),
                        "poi_transit_hub":       _stn.get("poi_transit_hub", 0.0),
                    }])
                    _pred = float(_model.predict(_input_df)[0])
                    _predictions.append({**_stn, "predicted_demand": _pred})

            _pred_min = min(p["predicted_demand"] for p in _predictions)
            _pred_max = max(p["predicted_demand"] for p in _predictions)
            _pred_range = max(_pred_max - _pred_min, 1.0)

            def _pred_color(demand: float) -> str:
                """Map predicted demand to a hex colour (blue → yellow → red)."""
                norm = (demand - _pred_min) / _pred_range
                if norm <= 0.5:
                    t = norm * 2
                    r = int(t * 255)
                    g = int(t * 255)
                    b = int(200 - t * 200)
                else:
                    t = (norm - 0.5) * 2
                    r = 255
                    g = int(255 - t * 255)
                    b = 0
                return f"#{r:02x}{g:02x}{b:02x}"

            # Build Folium map
            _mp_map = folium.Map(
                location=[59.3310, 18.0590],
                zoom_start=11,
                tiles=get_map_tiles(map_style),
            )

            # Draw SL metro line segments as light polylines for context
            for _line_name, _line_data in _SL_METRO_LINES.items():
                _lc = _line_data["color"]
                _coords = [(_lat, _lon) for _, _lat, _lon in _line_data["stations"]]
                folium.PolyLine(
                    _coords, weight=3, color=_lc, opacity=0.45,
                    tooltip=_line_name,
                ).add_to(_mp_map)

            # Add heatmap layer
            _heat_pts = [
                [p["lat"], p["lon"], p["predicted_demand"]]
                for p in _predictions
            ]
            HeatMap(
                _heat_pts, radius=30, blur=20, max_zoom=13,
                gradient={0.35: "blue", 0.6: "lime", 0.8: "yellow", 1: "red"},
            ).add_to(_mp_map)

            # Station markers coloured by predicted demand
            for _p in _predictions:
                _color = _pred_color(_p["predicted_demand"])
                _lines_str = " · ".join(_p["lines"])
                folium.CircleMarker(
                    location=[_p["lat"], _p["lon"]],
                    radius=8,
                    color="#333333",
                    weight=1,
                    fill=True,
                    fill_color=_color,
                    fill_opacity=0.9,
                    tooltip=(
                        f"<b>{_p['name']}</b><br>"
                        f"Predicted demand: <b>{_p['predicted_demand']:.0f} pax/hr</b><br>"
                        f"Lines: {_lines_str}"
                    ),
                ).add_to(_mp_map)

            st_folium(
                _mp_map,
                width=None,
                height=560,
                returned_objects=[],
                key=f"mp_map_{mp_hour}_{mp_dow}_{mp_month}_{round(mp_temp, 1)}_{round(mp_rain, 1)}_{round(mp_wind, 1)}",
            )

        # Summary table below the map
        _pred_df = pd.DataFrame([
            {
                "Station": p["name"],
                "Lines":   " · ".join(p["lines"]),
                "Predicted Demand (pax/hr)": round(p["predicted_demand"], 1),
            }
            for p in sorted(_predictions, key=lambda x: -x["predicted_demand"])
        ])
        st.markdown("##### 📊 Station Demand Ranking")
        _p1, _p2, _p3 = st.columns(3)
        _p1.metric("🏆 Highest demand", _pred_df.iloc[0]["Station"],
                   f"{_pred_df.iloc[0]['Predicted Demand (pax/hr)']:.0f} pax/hr")
        _p2.metric("📉 Lowest demand",  _pred_df.iloc[-1]["Station"],
                   f"{_pred_df.iloc[-1]['Predicted Demand (pax/hr)']:.0f} pax/hr")
        _p3.metric("📊 Network average",
                   f"{_pred_df['Predicted Demand (pax/hr)'].mean():.0f} pax/hr")
        st.dataframe(_pred_df, use_container_width=True, hide_index=True)

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

            map_data = st_folium(
                m,
                width=None,
                height=500,
                returned_objects=["last_object_clicked"],
                key=f"geo_map_{n_zones}_{map_style}_{show_heatmap}_{show_routes}",
            )

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
    st.markdown(
        '<div class="onboard-card">'
        "📌 <b>How to use this tab:</b> Explore synthetic passenger demand generated for "
        "the simulated network, modulated by <b>SMHI weather</b> and <b>POI proximity</b>. "
        "Charts update automatically when you change the "
        "<b>Number of Stops</b> or <b>Simulation Hours</b> sliders in the sidebar. "
        "Scroll down to use the <b>ML Demand Predictor</b> – enter time, weather, and POI "
        "context and click <em>Predict</em> to get a forecast."
        "</div>",
        unsafe_allow_html=True,
    )

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

    # ── Weather (SMHI) section ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌦️ SMHI Weather Influence on Demand")
    st.markdown(
        '<div class="onboard-card">'
        "Weather data sourced from the <b>SMHI (Swedish Meteorological and "
        "Hydrological Institute)</b> open data API. "
        "Cold temperatures, rain, and strong wind all increase transit demand "
        "(more people choose public transport over walking/cycling). "
        "The <em>weather demand factor</em> is a composite multiplier applied "
        "to the base demand signal."
        "</div>",
        unsafe_allow_html=True,
    )

    weather_df_tab = get_weather_data(n_hours).copy()
    weather_df_tab["hour"] = weather_df_tab["timestamp"].dt.hour
    weather_df_tab["date"] = weather_df_tab["timestamp"].dt.date

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("🌡️ Avg Temp",    f"{weather_df_tab['temperature'].mean():.1f} °C")
    w2.metric("🌧️ Rainy Hours", int(weather_df_tab["is_rainy"].sum()))
    w3.metric("💨 Avg Wind",    f"{weather_df_tab['wind_speed'].mean():.1f} m/s")
    w4.metric("📈 Avg Factor",  f"{weather_df_tab['weather_demand_factor'].mean():.3f}")

    wcol1, wcol2 = st.columns(2)

    with wcol1:
        fig_temp = px.line(
            weather_df_tab.head(72),  # first 3 days
            x="timestamp", y="temperature",
            title="🌡️ Temperature (first 72 h)",
            labels={"timestamp": "Time", "temperature": "°C"},
            color_discrete_sequence=["#e63946"],
        )
        fig_temp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with wcol2:
        fig_factor = px.area(
            weather_df_tab.head(72),
            x="timestamp", y="weather_demand_factor",
            title="📈 Weather Demand Factor (first 72 h)",
            labels={"timestamp": "Time", "weather_demand_factor": "Factor"},
            color_discrete_sequence=["#0077b6"],
        )
        fig_factor.add_hline(y=1.0, line_dash="dot", line_color="#888888",
                             annotation_text="Baseline (1.0)")
        fig_factor.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_factor, use_container_width=True)

    # Precipitation bar chart
    precip_daily = (
        weather_df_tab.groupby("date")["precipitation"].sum().reset_index()
    )
    precip_daily.columns = ["date", "precip_mm"]
    fig_precip = px.bar(
        precip_daily,
        x="date", y="precip_mm",
        title="🌧️ Daily Precipitation (mm)",
        labels={"date": "Date", "precip_mm": "Precipitation (mm)"},
        color="precip_mm",
        color_continuous_scale="Blues",
    )
    fig_precip.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_precip, use_container_width=True)

    # ── POI section ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📍 Points of Interest (POI) Influence")
    st.markdown(
        '<div class="onboard-card">'
        "POI proximity scores measure how close each transit stop is to key "
        "demand generators: <b>offices</b> (morning/evening peaks), "
        "<b>universities</b> (off-peak demand), <b>hospitals</b> (all-day), "
        "<b>shopping centres</b> (afternoon peaks), "
        "<b>tourist attractions</b> (weekend daytime), and "
        "<b>transit hubs</b> (transfer demand multiplier). "
        "Higher scores → more demand. These scores are used as features in the ML model."
        "</div>",
        unsafe_allow_html=True,
    )

    poi_df = get_poi_dataframe()
    stops_poi_df = get_stops_with_poi(n_zones)
    poi_cols = [c for c in stops_poi_df.columns if c.startswith("poi_")]

    # Map of POI locations
    pcol1, pcol2 = st.columns([2, 1])
    with pcol1:
        poi_map = folium.Map(
            location=[59.3293, 18.0686],
            zoom_start=11,
            tiles=get_map_tiles(map_style),
        )
        _poi_cat_colors = {
            "office":      "#0077b6",
            "university":  "#2dc653",
            "hospital":    "#e63946",
            "shopping":    "#f4a261",
            "tourist":     "#9d4edd",
            "transit_hub": "#e76f51",
        }
        for _, poi_row in poi_df.iterrows():
            cat   = poi_row["category"]
            color = _poi_cat_colors.get(cat, "#888888")
            folium.CircleMarker(
                [poi_row["lat"], poi_row["lon"]],
                radius=6 + poi_row["weight"],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=(
                    f"<b>{poi_row['name']}</b><br>"
                    f"Category: {cat}<br>"
                    f"Weight: {poi_row['weight']}"
                ),
            ).add_to(poi_map)
        # Add stops as small markers
        for _, row in stops_poi_df.iterrows():
            folium.CircleMarker(
                [row["stop_lat"], row["stop_lon"]],
                radius=4,
                color="#1a2f4e",
                fill=True, fill_color="white", fill_opacity=0.9, weight=2,
                tooltip=row["stop_name"],
            ).add_to(poi_map)
        st_folium(poi_map, width=None, height=420, returned_objects=[],
                  key=f"poi_map_{n_zones}_{map_style}")

    with pcol2:
        st.markdown("#### 🎨 POI Legend")
        st.markdown(
            '<div class="legend-card">'
            '<span style="color:#0077b6;font-size:1.1em">●</span> Office<br>'
            '<span style="color:#2dc653;font-size:1.1em">●</span> University<br>'
            '<span style="color:#e63946;font-size:1.1em">●</span> Hospital<br>'
            '<span style="color:#f4a261;font-size:1.1em">●</span> Shopping<br>'
            '<span style="color:#9d4edd;font-size:1.1em">●</span> Tourist<br>'
            '<span style="color:#e76f51;font-size:1.1em">●</span> Transit Hub<br>'
            "<hr style='margin:6px 0'>"
            "⬜ Transit stop"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.metric("Total POIs", len(poi_df))
        for cat, color in _poi_cat_colors.items():
            cnt = int((poi_df["category"] == cat).sum())
            st.markdown(
                f'<span style="color:{color}">■</span> {cat.title()}: **{cnt}**',
                unsafe_allow_html=True,
            )

    # POI score radar/bar chart per stop
    if not stops_poi_df.empty and poi_cols:
        st.markdown("#### 📊 POI Proximity Scores per Stop")
        poi_melt = stops_poi_df[["stop_name"] + poi_cols].melt(
            id_vars="stop_name",
            var_name="POI Category",
            value_name="Score",
        )
        poi_melt["POI Category"] = poi_melt["POI Category"].str.replace("poi_", "").str.title()
        fig_poi = px.bar(
            poi_melt,
            x="stop_name", y="Score", color="POI Category",
            barmode="stack",
            title="Cumulative POI Proximity Score by Stop",
            labels={"stop_name": "Stop", "Score": "Proximity Score (normalised)"},
            color_discrete_sequence=list(_poi_cat_colors.values()),
        )
        fig_poi.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_poi, use_container_width=True)

    # ML section
    st.markdown("---")
    st.markdown("### 🤖 ML Demand Prediction")
    st.markdown(
        '<div class="onboard-card">'
        "The XGBoost model is trained on <b>time features</b> (hour, day, month, weekend), "
        "<b>lag features</b> (demand 1 h, 2 h, 3 h, and 24 h ago), "
        "<b>SMHI weather features</b> (temperature, precipitation, wind, humidity), and "
        "<b>POI proximity scores</b> (office, university, hospital, shopping, tourist, transit hub). "
        "Enter values below to generate a demand forecast."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Training XGBoost model…"):
        model, metrics, feat_cols = get_trained_model(n_zones, n_hours)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",       f"{metrics['mae']:.2f}",   help="Mean Absolute Error")
    m2.metric("RMSE",      f"{metrics['rmse']:.2f}",  help="Root Mean Squared Error")
    m3.metric("R² Score",  f"{metrics['r2']:.4f}",    help="Coefficient of Determination")
    m4.metric("Features",  len(feat_cols))

    st.markdown("#### 🔮 Demand Prediction")

    with st.expander("⏰ Time & Lag features", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        with p1: pred_hour    = st.number_input("Hour (0–23)",         0, 23,  8,  key="pred_hour")
        with p2: pred_dow     = st.number_input("Day of Week (0=Mon)", 0,  6,  1,  key="pred_dow")
        with p3: pred_month   = st.number_input("Month (1–12)",        1, 12,  3,  key="pred_month")
        with p4: pred_weekend = st.selectbox("Weekend?", [0, 1], index=0, key="pred_we")

        l1, l2, l3, l4 = st.columns(4)
        with l1: pred_lag1  = st.number_input("Demand t-1 (pax)",  0, 1000, 120, key="lag1")
        with l2: pred_lag2  = st.number_input("Demand t-2 (pax)",  0, 1000, 115, key="lag2")
        with l3: pred_lag3  = st.number_input("Demand t-3 (pax)",  0, 1000, 110, key="lag3")
        with l4: pred_lag24 = st.number_input("Demand t-24 (pax)", 0, 1000, 125, key="lag24")

    with st.expander("🌦️ SMHI Weather features", expanded=True):
        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            pred_temp  = st.slider("Temperature (°C)", -20.0, 35.0, 10.0, 0.5, key="pred_temp")
            pred_rain  = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0, 0.1, key="pred_rain")
        with wf2:
            pred_wind  = st.slider("Wind speed (m/s)", 0.0, 25.0, 3.0, 0.5, key="pred_wind")
            pred_humid = st.slider("Relative Humidity (%)", 30, 100, 70, key="pred_humid")
        with wf3:
            pred_israiny = int(pred_rain > 0.1)
            st.metric("🌧️ Is Rainy?", "Yes" if pred_israiny else "No")
            # Compute weather factor from inputs
            cold_b = max(0, min(0.2, (10 - pred_temp) / 10 * 0.1))
            rain_b = 0.15 if pred_rain > 0.1 else 0.0
            wind_b = 0.05 if pred_wind > 8 else 0.0
            heat_p = -0.05 if pred_temp > 25 else 0.0
            pred_wfactor = round(min(1.4, max(0.5, 1.0 + cold_b + rain_b + wind_b + heat_p)), 4)
            st.metric("📈 Demand Factor", pred_wfactor)

    with st.expander("📍 POI proximity features", expanded=True):
        pp1, pp2, pp3 = st.columns(3)
        with pp1:
            pred_poi_office  = st.slider("Office score",      0.0, 1.0, 0.5, 0.05, key="poi_off")
            pred_poi_uni     = st.slider("University score",  0.0, 1.0, 0.3, 0.05, key="poi_uni")
        with pp2:
            pred_poi_hosp    = st.slider("Hospital score",    0.0, 1.0, 0.2, 0.05, key="poi_hosp")
            pred_poi_shop    = st.slider("Shopping score",    0.0, 1.0, 0.4, 0.05, key="poi_shop")
        with pp3:
            pred_poi_tourist = st.slider("Tourist score",     0.0, 1.0, 0.1, 0.05, key="poi_tour")
            pred_poi_hub     = st.slider("Transit hub score", 0.0, 1.0, 0.6, 0.05, key="poi_hub")

    if st.button("🔮 Predict", type="primary"):
        input_df = pd.DataFrame([{
            "hour":                    pred_hour,
            "day_of_week":             pred_dow,
            "month":                   pred_month,
            "is_weekend":              pred_weekend,
            "demand_lag_1":            pred_lag1,
            "demand_lag_2":            pred_lag2,
            "demand_lag_3":            pred_lag3,
            "demand_lag_24":           pred_lag24,
            "temperature":             pred_temp,
            "precipitation":           pred_rain,
            "wind_speed":              pred_wind,
            "relative_humidity":       pred_humid,
            "is_rainy":                pred_israiny,
            "weather_demand_factor":   pred_wfactor,
            "poi_office":              pred_poi_office,
            "poi_university":          pred_poi_uni,
            "poi_hospital":            pred_poi_hosp,
            "poi_shopping":            pred_poi_shop,
            "poi_tourist":             pred_poi_tourist,
            "poi_transit_hub":         pred_poi_hub,
        }])
        prediction = model.predict(input_df)[0]
        weather_note = "☔ rainy, " if pred_israiny else ""
        poi_note     = "near transit hub, " if pred_poi_hub > 0.7 else ""
        st.success(
            f"🎯 Predicted Demand: **{prediction:.0f} passengers**  \n"
            f"*Context: hour {pred_hour:02d}:00 · {weather_note}{poi_note}"
            f"weather factor {pred_wfactor}*"
        )

# ══════════════════════════════════════════════
# Tab 3 – Route Planner
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🔀 Plan My Route")
    st.markdown(
        '<div class="onboard-card">'
        "📌 <b>How to use this tab:</b> Choose an <b>origin</b> and a <b>destination</b> "
        "stop from the dropdowns on the left, then click <em>Find Optimal Route</em>. "
        "The shortest path (by travel time) is highlighted on the map and shown as "
        "step-by-step directions below. Your last result is kept visible until you "
        "search for a new route. Scroll down to view the <b>Congestion Analysis</b> "
        "for the current network."
        "</div>",
        unsafe_allow_html=True,
    )

    G, stops_df, _ = get_graph(n_zones)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📍 Select Origin & Destination")
        stop_options = {
            f"{row['stop_name']} ({row['stop_id']})": row["stop_id"]
            for _, row in stops_df.iterrows()
        }
        source_display = st.selectbox("🟢 From (origin stop)", list(stop_options.keys()), index=0)
        target_display = st.selectbox("🔴 To (destination stop)", list(stop_options.keys()),
                                      index=min(4, len(stop_options) - 1))
        source = stop_options[source_display]
        target = stop_options[target_display]
        find_route = st.button("🔍 Find Optimal Route", type="primary",
                               use_container_width=True)

        if find_route:
            path, length = compute_shortest_path(G, source, target)
            # Persist results in session state so they survive re-renders
            st.session_state.route_path   = path
            st.session_state.route_length = length
            st.session_state.route_src    = source_display
            st.session_state.route_dst    = target_display

    with col2:
        path   = st.session_state.route_path
        length = st.session_state.route_length

        if path is not None:
            if path:
                route_label = (
                    f"**{st.session_state.route_src}** → "
                    f"**{st.session_state.route_dst}**"
                )
                st.success(
                    f"✅ Route found · {route_label} · "
                    f"**{len(path) - 1} stops** · "
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

                st_folium(
                    route_map,
                    width=None,
                    height=380,
                    returned_objects=[],
                    key="route_map_" + "_".join(str(s) for s in path),
                )

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
            st.info("👆 Select an origin and destination on the left, then click **Find Optimal Route**.")
            preview_map = folium.Map(
                location=[stops_df["stop_lat"].mean(), stops_df["stop_lon"].mean()],
                zoom_start=13, tiles=get_map_tiles(map_style),
            )
            for _, row in stops_df.iterrows():
                folium.CircleMarker(
                    [row["stop_lat"], row["stop_lon"]],
                    radius=5, color="#0077b6", fill=True,
                    tooltip=row["stop_name"],
                    popup=row["stop_name"],
                ).add_to(preview_map)
            st_folium(
                preview_map,
                width=None,
                height=380,
                returned_objects=[],
                key=f"route_preview_{n_zones}",
            )

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
    st.markdown(
        '<div class="onboard-card">'
        "📌 <b>How to use this tab:</b> View graph-level metrics for the simulated "
        "transit network. The <em>Stop Connectivity Distribution</em> histogram shows "
        "how many connections each stop has. The <em>Network Graph</em> plots each stop "
        "geographically – larger, darker nodes have more connections. "
        "Use the slider in the sidebar to change the number of stops and see how the "
        "network topology evolves."
        "</div>",
        unsafe_allow_html=True,
    )

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
    "🚇 GeoTrack – Urban Mobility Stockholm &nbsp;·&nbsp; "
    "Streamlit · Folium · Plotly · XGBoost · SMHI Weather · Stockholm POIs &nbsp;·&nbsp; "
    "<i>All data is synthetic and for demonstration purposes only.</i>"
    "</div>",
    unsafe_allow_html=True,
)
