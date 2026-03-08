<div align="center">

# 🚌 Urban Mobility Stockholm

**Real-time transit analytics platform for Stockholm's public transport network**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-reference) • [Architecture](#-architecture)

</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🗺️ Interactive Map Dashboard
- Real-time Stockholm transit network visualization
- Demand heatmap overlay with color gradients
- Clickable stop markers with detailed popups
- Multiple map styles (Street, Light, Dark)

</td>
<td width="50%">

### 📊 Demand Analytics
- Hourly passenger demand patterns
- Day-of-week trend analysis
- Zone × Hour heatmap visualization
- Historical data exploration

</td>
</tr>
<tr>
<td width="50%">

### 🤖 ML-Powered Predictions
- XGBoost demand forecasting model
- Feature engineering with lag variables
- Real-time prediction interface
- Model performance metrics (MAE, RMSE, R²)

</td>
<td width="50%">

### 🔀 Route Optimization
- Dijkstra's shortest path algorithm
- Visual route display on map
- Congestion detection & alerts
- Alternative route suggestions

</td>
</tr>
</table>

---

## 🎬 Demo

### Live Map View
The main dashboard features an interactive Folium map centered on Stockholm, showing:
- 🔵 **Transit stops** as clickable markers
- 🌈 **Route connections** with colored lines
- 🔥 **Demand heatmap** showing passenger density

### Route Planning
Select origin and destination to find the optimal route with travel time estimates.

### Demand Prediction
Input time parameters to get ML-powered passenger demand forecasts.

---

## 📁 Project Structure

```
urban-mobility-stockholm/
│
├── 📊 dashboard/
│   └── app.py                 # Streamlit dashboard with Folium maps
│
├── 📓 notebooks/
│   └── exploration.ipynb      # Jupyter notebook for data exploration
│
├── 🐍 src/
│   ├── api/
│   │   └── main.py            # FastAPI REST endpoints
│   │
│   ├── ingestion/
│   │   └── gtfs_loader.py     # GTFS data parsing & loading
│   │
│   ├── geospatial/
│   │   └── geo_processor.py   # Spatial joins & zone creation
│   │
│   ├── network/
│   │   └── graph_builder.py   # NetworkX graph construction
│   │
│   ├── ml/
│   │   └── demand_predictor.py # XGBoost demand forecasting
│   │
│   └── optimization/
│       └── route_optimizer.py  # Route & congestion analysis
│
├── 🐳 docker-compose.yml       # Container orchestration
├── 🐳 Dockerfile               # Container image definition
├── 📋 requirements.txt         # Python dependencies
└── 📖 README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker & Docker Compose

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/urban-mobility-stockholm.git
cd urban-mobility-stockholm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# macOS only: Install OpenMP for XGBoost
brew install libomp
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `geopandas`, `shapely` | Geospatial processing |
| `networkx` | Graph algorithms |
| `xgboost`, `scikit-learn` | Machine learning |
| `fastapi`, `uvicorn` | REST API |
| `streamlit`, `streamlit-folium` | Dashboard UI |
| `folium`, `plotly` | Visualizations |

---

## 💻 Usage

### 🎨 Run the Dashboard (Recommended)

The interactive dashboard is the best way to explore the platform:

```bash
# Set Python path and run Streamlit
PYTHONPATH=$(pwd) streamlit run dashboard/app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

#### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🗺️ **Live Map** | Interactive Stockholm map with stops, routes, and demand heatmap |
| 📊 **Demand Analytics** | Charts showing demand patterns and ML model performance |
| 🔀 **Route Planner** | Find optimal routes and view congestion analysis |
| 📈 **Network Stats** | Network connectivity metrics and stop information |

---

### 🔌 Run the REST API

For programmatic access:

```bash
PYTHONPATH=$(pwd) uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Access the interactive API docs at **http://localhost:8000/docs**

---

### 🐳 Run with Docker

```bash
docker-compose up --build
```

Services:
- Dashboard: http://localhost:8501
- API: http://localhost:8000

---

## 📡 API Reference

### Endpoints

#### `POST /predict-demand`
Predict passenger demand for a given time period.

**Request:**
```json
{
  "hour": 8,
  "day_of_week": 1,
  "month": 3,
  "is_weekend": 0,
  "demand_lag_1": 120.0,
  "demand_lag_2": 115.0,
  "demand_lag_3": 110.0,
  "demand_lag_24": 125.0
}
```

**Response:**
```json
{
  "predicted_demand": 142.5,
  "confidence": 0.89
}
```

#### `POST /optimal-route`
Find the shortest path between two stops.

**Request:**
```json
{
  "source": "S0",
  "target": "S5"
}
```

**Response:**
```json
{
  "path": ["S0", "S2", "S5"],
  "total_time": 12.5,
  "num_stops": 3
}
```

#### `GET /network-stats`
Get transit network statistics.

**Response:**
```json
{
  "num_stops": 15,
  "num_connections": 28,
  "average_degree": 3.73,
  "is_connected": true
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   Streamlit     │    │    FastAPI      │                     │
│  │   Dashboard     │    │    REST API     │                     │
│  │   (Port 8501)   │    │   (Port 8000)   │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
├───────────┴──────────────────────┴──────────────────────────────┤
│                      Core Modules                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Ingestion│  │Geospatial│  │ Network  │  │   ML Prediction  │ │
│  │  (GTFS)  │  │Processing│  │  Graph   │  │    (XGBoost)     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Route Optimization                       │   │
│  │         (Dijkstra, Congestion Detection)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🐍 Python Quick Start

```python
from src.ml.demand_predictor import (
    generate_synthetic_demand,
    engineer_features,
    train_demand_model
)

# Generate synthetic data (10 zones, 1 week)
df = generate_synthetic_demand(n_zones=10, n_hours=168)

# Feature engineering
df_feat = engineer_features(df)

# Train XGBoost model
feature_cols = [
    "hour", "day_of_week", "month", "is_weekend",
    "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_24"
]
model, metrics = train_demand_model(df_feat, feature_cols)

print(f"📊 Model Performance:")
print(f"   MAE:  {metrics['mae']:.2f}")
print(f"   RMSE: {metrics['rmse']:.2f}")
print(f"   R²:   {metrics['r2']:.4f}")
```

---

## 🗺️ Stockholm Coverage

The platform simulates transit coverage across key Stockholm areas:

| Area | Coordinates | Description |
|------|-------------|-------------|
| T-Centralen | 59.331°N, 18.059°E | Central station hub |
| Gamla Stan | 59.325°N, 18.071°E | Old Town |
| Södermalm | 59.315°N, 18.072°E | Southern island |
| Östermalm | 59.337°N, 18.089°E | Eastern district |
| Kungsholmen | 59.334°N, 18.028°E | City island |
| Djurgården | 59.326°N, 18.115°E | Royal park |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stockholm Public Transport (SL)](https://sl.se) for transit data inspiration
- [OpenStreetMap](https://www.openstreetmap.org) for map tiles
- [Streamlit](https://streamlit.io) for the amazing dashboard framework
- [Folium](https://python-visualization.github.io/folium/) for interactive maps

---

<div align="center">

**Built with ❤️ for smarter urban mobility**

[⬆ Back to Top](#-urban-mobility-stockholm)

</div>
