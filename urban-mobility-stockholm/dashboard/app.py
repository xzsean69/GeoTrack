"""
Interactive Streamlit dashboard with map visualizations for urban mobility.
Enhanced UI with Stockholm map as the centerpiece.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

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
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🚌 Urban Mobility Stockholm</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time transit analytics · Demand forecasting · Route optimization</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("---")
    st.markdown("**🗺️ Map Options**")
    map_style = st.selectbox(
        "Map Style",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark Matter"],
        index=0
    )
    show_heatmap = st.checkbox("Show Demand Heatmap", value=True)
    show_routes = st.checkbox("Show Transit Routes", value=True)
    
    st.markdown("---")
    st.markdown("**📊 Simulation**")
    n_zones = st.slider("Number of Stops", 5, 30, 15)
    n_hours = st.slider("Simulation Hours", 24, 336, 168, help="Duration to simulate (hours)")
    congestion_threshold = st.slider("Congestion Threshold", 100, 2000, 500)
    
    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.caption("Urban Mobility Stockholm provides analytics for public transit optimization using ML and network analysis.")

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
def get_stops_df(n_zones):
    """Generate realistic Stockholm stop locations."""
    # Stockholm central area coordinates
    center_lat, center_lon = 59.3293, 18.0686
    
    # Generate stops in a realistic pattern around Stockholm
    np.random.seed(42)
    
    # Famous Stockholm locations as anchor points
    landmarks = [
        ("T-Centralen", 59.3310, 18.0590),
        ("Gamla Stan", 59.3250, 18.0710),
        ("Södermalm", 59.3150, 18.0720),
        ("Östermalm", 59.3370, 18.0890),
        ("Kungsholmen", 59.3340, 18.0280),
        ("Djurgården", 59.3260, 18.1150),
        ("Norrmalm", 59.3380, 18.0550),
        ("Vasastan", 59.3450, 18.0520),
        ("Hornstull", 59.3160, 18.0340),
        ("Slussen", 59.3200, 18.0720),
    ]
    
    stops = []
    for i in range(n_zones):
        if i < len(landmarks):
            name, lat, lon = landmarks[i]
            stops.append({
                "stop_id": f"S{i}",
                "stop_name": name,
                "stop_lat": lat + np.random.uniform(-0.002, 0.002),
                "stop_lon": lon + np.random.uniform(-0.002, 0.002),
            })
        else:
            # Generate additional stops around the center
            angle = (i / n_zones) * 2 * np.pi
            radius = np.random.uniform(0.01, 0.04)
            stops.append({
                "stop_id": f"S{i}",
                "stop_name": f"Station {i}",
                "stop_lat": center_lat + radius * np.cos(angle) + np.random.uniform(-0.005, 0.005),
                "stop_lon": center_lon + radius * np.sin(angle) * 1.5 + np.random.uniform(-0.005, 0.005),
            })
    
    return pd.DataFrame(stops)

@st.cache_data
def get_graph(n_zones):
    stops = get_stops_df(n_zones)
    
    # Create more realistic route connections
    stop_times_data = []
    trip_id = 0
    
    # Create several transit lines
    n_lines = max(3, n_zones // 4)
    for line in range(n_lines):
        trip_id += 1
        # Each line connects a subset of stops
        np.random.seed(line + 100)
        line_stops = np.random.choice(n_zones, size=min(n_zones, np.random.randint(4, 8)), replace=False)
        line_stops = sorted(line_stops)
        
        for seq, stop_idx in enumerate(line_stops):
            stop_times_data.append({
                "trip_id": f"Line{line}",
                "stop_id": f"S{stop_idx}",
                "stop_sequence": seq,
            })
    
    # Add a circular line connecting all stops
    for seq in range(n_zones):
        stop_times_data.append({
            "trip_id": "CircleLine",
            "stop_id": f"S{seq}",
            "stop_sequence": seq,
        })
    # Connect back to start
    stop_times_data.append({
        "trip_id": "CircleLine",
        "stop_id": "S0",
        "stop_sequence": n_zones,
    })
    
    stop_times = pd.DataFrame(stop_times_data)
    return build_transit_graph(stops, stop_times), stops

def get_map_tiles(style):
    """Get the appropriate tile layer based on style selection."""
    tiles_map = {
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB Positron": "CartoDB positron",
        "CartoDB Dark Matter": "CartoDB dark_matter",
    }
    return tiles_map.get(style, "OpenStreetMap")

# ──────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Map", 
    "📊 Demand Analytics", 
    "🔀 Route Planner",
    "📈 Network Stats"
])

# ══════════════════════════════════════════════
# Tab 1: Interactive Map (Main Feature)
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🗺️ Stockholm Transit Network")
    
    G, stops_df = get_graph(n_zones)
    demand_df = get_demand_data(n_zones, n_hours)
    
    # Calculate demand per zone for the heatmap
    zone_demand = demand_df.groupby("zone_id")["demand"].mean().reset_index()
    zone_demand["zone_idx"] = zone_demand["zone_id"].str.extract(r'zone_(\d+)').astype(int)
    
    # Create the Folium map centered on Stockholm
    center_lat = stops_df["stop_lat"].mean()
    center_lon = stops_df["stop_lon"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=get_map_tiles(map_style),
    )
    
    # Add demand heatmap layer
    if show_heatmap:
        heat_data = []
        for _, row in stops_df.iterrows():
            # Generate synthetic demand intensity for heatmap
            idx = int(row["stop_id"].replace("S", ""))
            intensity = np.random.uniform(0.3, 1.0) * (1 + np.sin(idx))
            heat_data.append([row["stop_lat"], row["stop_lon"], intensity])
        
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            max_zoom=13,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
        ).add_to(m)
    
    # Add transit routes
    if show_routes:
        # Draw edges between connected stops
        edge_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        
        for idx, (u, v, data) in enumerate(G.edges(data=True)):
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            if u_data.get("lat") and v_data.get("lat"):
                color = edge_colors[idx % len(edge_colors)]
                folium.PolyLine(
                    locations=[
                        [u_data["lat"], u_data["lon"]],
                        [v_data["lat"], v_data["lon"]]
                    ],
                    weight=3,
                    color=color,
                    opacity=0.7,
                ).add_to(m)
    
    # Add stop markers with custom icons
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in stops_df.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; width: 150px;">
            <h4 style="margin: 0; color: #1E3A5F;">🚏 {row['stop_name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0;"><b>ID:</b> {row['stop_id']}</p>
            <p style="margin: 2px 0;"><b>Lat:</b> {row['stop_lat']:.4f}</p>
            <p style="margin: 2px 0;"><b>Lon:</b> {row['stop_lon']:.4f}</p>
        </div>
        """
        
        folium.Marker(
            location=[row["stop_lat"], row["stop_lon"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=row["stop_name"],
            icon=folium.Icon(color='blue', icon='info-sign'),
        ).add_to(marker_cluster)
    
    # Display the map
    col1, col2 = st.columns([3, 1])
    
    with col1:
        map_data = st_folium(m, width=None, height=500, returned_objects=["last_object_clicked"])
    
    with col2:
        st.markdown("#### 📍 Quick Stats")
        stats = get_network_stats(G)
        
        st.metric("🚏 Total Stops", stats["num_stops"])
        st.metric("🔗 Connections", stats["num_connections"])
        st.metric("📊 Avg Degree", f"{stats['average_degree']:.1f}")
        
        st.markdown("---")
        st.markdown("#### 🎯 Selected Stop")
        
        if map_data and map_data.get("last_object_clicked"):
            clicked = map_data["last_object_clicked"]
            st.info(f"📍 Lat: {clicked.get('lat', 'N/A'):.4f}\n\nLon: {clicked.get('lng', 'N/A'):.4f}")
        else:
            st.caption("Click a stop on the map for details")
        
        st.markdown("---")
        st.markdown("#### 🚦 Legend")
        st.markdown("""
        - 🔵 **Blue markers**: Transit stops
        - 🌈 **Colored lines**: Transit routes
        - 🔥 **Heatmap**: Passenger demand
        """)

# ══════════════════════════════════════════════
# Tab 2: Demand Analytics
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Passenger Demand Analytics")
    
    demand_df = get_demand_data(n_zones, n_hours)
    demand_df = demand_df.copy()
    demand_df["hour"] = pd.to_datetime(demand_df["timestamp"]).dt.hour
    demand_df["day"] = pd.to_datetime(demand_df["timestamp"]).dt.date
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly demand pattern
        hourly = demand_df.groupby("hour")["demand"].mean().reset_index()
        fig1 = px.area(
            hourly, x="hour", y="demand",
            title="📈 Average Demand by Hour",
            labels={"hour": "Hour of Day", "demand": "Avg Passengers"},
            color_discrete_sequence=["#3498db"],
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Demand by day of week
        demand_df["day_of_week"] = pd.to_datetime(demand_df["timestamp"]).dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = demand_df.groupby("day_of_week")["demand"].mean().reindex(dow_order).reset_index()
        
        fig2 = px.bar(
            daily, x="day_of_week", y="demand",
            title="📅 Demand by Day of Week",
            labels={"day_of_week": "Day", "demand": "Avg Passengers"},
            color="demand",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Demand Heatmap (Zone x Hour)
    st.markdown("#### 🔥 Demand Heatmap")
    pivot = demand_df.pivot_table(index="zone_id", columns="hour", values="demand", aggfunc="mean")
    fig3 = px.imshow(
        pivot,
        title="Zone × Hour Demand Distribution",
        labels={"x": "Hour", "y": "Zone", "color": "Passengers"},
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # ML Model Performance
    st.markdown("---")
    st.markdown("### 🤖 ML Demand Prediction Model")
    
    with st.spinner("Training XGBoost model..."):
        model, metrics, feat_cols = get_trained_model(n_zones, n_hours)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📉 MAE", f"{metrics['mae']:.2f}", help="Mean Absolute Error")
    col2.metric("📊 RMSE", f"{metrics['rmse']:.2f}", help="Root Mean Squared Error")
    col3.metric("🎯 R² Score", f"{metrics['r2']:.4f}", help="Coefficient of Determination")
    col4.metric("🔧 Features", len(feat_cols))
    
    # Prediction interface
    st.markdown("#### 🔮 Make a Prediction")
    
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    with pred_col1:
        pred_hour = st.number_input("Hour (0-23)", 0, 23, 8)
    with pred_col2:
        pred_dow = st.number_input("Day of Week (0=Mon)", 0, 6, 1)
    with pred_col3:
        pred_month = st.number_input("Month (1-12)", 1, 12, 3)
    with pred_col4:
        pred_weekend = st.selectbox("Weekend?", [0, 1], index=0)
    
    if st.button("🔮 Predict Demand", type="primary"):
        input_data = pd.DataFrame([{
            "hour": pred_hour,
            "day_of_week": pred_dow,
            "month": pred_month,
            "is_weekend": pred_weekend,
            "demand_lag_1": 120.0,
            "demand_lag_2": 115.0,
            "demand_lag_3": 110.0,
            "demand_lag_24": 125.0,
        }])
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Predicted Demand: **{prediction:.0f} passengers**")

# ══════════════════════════════════════════════
# Tab 3: Route Planner
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🔀 Route Planner")
    
    G, stops_df = get_graph(n_zones)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📍 Select Route")
        
        stop_options = {f"{row['stop_name']} ({row['stop_id']})": row['stop_id'] 
                       for _, row in stops_df.iterrows()}
        
        source_display = st.selectbox("🟢 From", list(stop_options.keys()), index=0)
        target_display = st.selectbox("🔴 To", list(stop_options.keys()), 
                                      index=min(4, len(stop_options) - 1))
        
        source = stop_options[source_display]
        target = stop_options[target_display]
        
        find_route = st.button("🔍 Find Optimal Route", type="primary", use_container_width=True)
    
    with col2:
        if find_route:
            path, length = compute_shortest_path(G, source, target)
            
            if path:
                st.success(f"✅ Route found! Travel time: **{length:.1f} min**")
                
                # Create route map
                route_map = folium.Map(
                    location=[stops_df["stop_lat"].mean(), stops_df["stop_lon"].mean()],
                    zoom_start=13,
                    tiles=get_map_tiles(map_style),
                )
                
                # Draw the route
                route_coords = []
                for stop_id in path:
                    node = G.nodes[stop_id]
                    route_coords.append([node["lat"], node["lon"]])
                
                # Add route line
                folium.PolyLine(
                    route_coords,
                    weight=5,
                    color='#e74c3c',
                    opacity=0.8,
                ).add_to(route_map)
                
                # Add markers for start, end, and intermediate stops
                for i, stop_id in enumerate(path):
                    node = G.nodes[stop_id]
                    if i == 0:
                        icon = folium.Icon(color='green', icon='play')
                        label = "Start"
                    elif i == len(path) - 1:
                        icon = folium.Icon(color='red', icon='stop')
                        label = "End"
                    else:
                        icon = folium.Icon(color='blue', icon='info-sign')
                        label = f"Stop {i}"
                    
                    folium.Marker(
                        [node["lat"], node["lon"]],
                        popup=f"{label}: {node.get('name', stop_id)}",
                        tooltip=node.get('name', stop_id),
                        icon=icon,
                    ).add_to(route_map)
                
                st_folium(route_map, width=None, height=400)
                
                # Route details
                st.markdown("#### 📋 Route Details")
                route_names = [G.nodes[s].get('name', s) for s in path]
                st.info(" → ".join(route_names))
                
            else:
                st.error("❌ No route found between selected stops.")
        else:
            # Show empty map placeholder
            st.info("👆 Select your origin and destination, then click 'Find Optimal Route'")
            
            preview_map = folium.Map(
                location=[stops_df["stop_lat"].mean(), stops_df["stop_lon"].mean()],
                zoom_start=13,
                tiles=get_map_tiles(map_style),
            )
            for _, row in stops_df.iterrows():
                folium.CircleMarker(
                    [row["stop_lat"], row["stop_lon"]],
                    radius=5,
                    color='#3498db',
                    fill=True,
                    popup=row["stop_name"],
                ).add_to(preview_map)
            
            st_folium(preview_map, width=None, height=400)
    
    # Congestion Analysis
    st.markdown("---")
    st.markdown("### 🚦 Congestion Analysis")
    
    # Assign mock loads
    for u, v in G.edges():
        G[u][v]["load"] = np.random.uniform(0, 1000)
    
    congested = identify_congested_edges(G, threshold=congestion_threshold)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("⚠️ Congested Segments", len(congested))
        st.caption(f"Threshold: {congestion_threshold} passengers")
        
        if congested:
            cong_df = pd.DataFrame(
                [(G.nodes[u].get('name', u), G.nodes[v].get('name', v), int(d.get("load", 0))) 
                 for u, v, d in congested],
                columns=["From", "To", "Load"]
            )
            st.dataframe(cong_df, use_container_width=True, hide_index=True)
    
    with col2:
        if congested:
            # Congestion visualization
            fig_cong = px.bar(
                cong_df.head(10),
                x="Load",
                y=[f"{r['From']} → {r['To']}" for _, r in cong_df.head(10).iterrows()],
                orientation='h',
                title="🚦 Top Congested Routes",
                color="Load",
                color_continuous_scale="Reds",
            )
            fig_cong.update_layout(yaxis_title="Route", showlegend=False)
            st.plotly_chart(fig_cong, use_container_width=True)
        else:
            st.success("✅ No congested routes detected!")

# ══════════════════════════════════════════════
# Tab 4: Network Statistics
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Network Statistics")
    
    G, stops_df = get_graph(n_zones)
    stats = get_network_stats(G)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚏 Stops", stats["num_stops"])
    col2.metric("🔗 Connections", stats["num_connections"])
    col3.metric("📊 Avg Degree", f"{stats['average_degree']:.2f}")
    col4.metric("🔄 Connected", "Yes" if stats["is_weakly_connected"] else "No")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Degree distribution
        degrees = [d for _, d in G.degree()]
        fig_deg = px.histogram(
            x=degrees,
            nbins=20,
            title="📊 Stop Connectivity Distribution",
            labels={"x": "Number of Connections", "y": "Count"},
            color_discrete_sequence=["#3498db"],
        )
        st.plotly_chart(fig_deg, use_container_width=True)
    
    with col2:
        # Network visualization using Plotly
        edge_x = []
        edge_y = []
        for u, v in G.edges():
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            edge_x.extend([u_data.get("lon", 0), v_data.get("lon", 0), None])
            edge_y.extend([u_data.get("lat", 0), v_data.get("lat", 0), None])
        
        node_x = [G.nodes[n].get("lon", 0) for n in G.nodes()]
        node_y = [G.nodes[n].get("lat", 0) for n in G.nodes()]
        node_text = [G.nodes[n].get("name", n) for n in G.nodes()]
        node_degree = [G.degree(n) for n in G.nodes()]
        
        fig_network = go.Figure()
        
        fig_network.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none'
        ))
        
        fig_network.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=[10 + d * 2 for d in node_degree],
                color=node_degree,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections"),
            ),
            text=node_text,
            textposition="top center",
            hoverinfo='text',
        ))
        
        fig_network.update_layout(
            title="🕸️ Network Graph",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Stops table
    st.markdown("#### 📋 All Stops")
    display_df = stops_df.copy()
    display_df["Connections"] = display_df["stop_id"].apply(lambda x: G.degree(x))
    display_df.columns = ["ID", "Name", "Latitude", "Longitude", "Connections"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>🚌 Urban Mobility Stockholm | Built with Streamlit, Folium & Plotly</p>
        <p>Real-time transit analytics for smarter urban planning</p>
    </div>
    """,
    unsafe_allow_html=True
)
