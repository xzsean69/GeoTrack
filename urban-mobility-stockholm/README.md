# Urban Mobility Stockholm

A comprehensive urban mobility analytics platform for Stockholm's public transport system.

## Overview

This project implements end-to-end analytics for urban transit data, including:

- **GTFS Data Ingestion** – Download, parse, and clean GTFS public transport feeds
- **Geospatial Processing** – Spatial joins, zone grid creation, and demand heatmaps using GeoPandas
- **Network Graph Analysis** – Build weighted directed transit graphs and compute shortest paths with NetworkX
- **ML Demand Prediction** – Hourly passenger demand forecasting using XGBoost
- **Route Optimization** – Congestion detection and alternative route suggestion
- **REST API** – FastAPI service for real-time predictions and route queries
- **Interactive Dashboard** – Streamlit dashboard with Plotly visualizations

## Project Structure

```
urban-mobility-stockholm/
├── data/                    # Raw and processed data
├── notebooks/
│   └── exploration.ipynb    # Exploratory analysis
├── src/
│   ├── ingestion/           # GTFS data loading
│   ├── geospatial/          # Spatial processing
│   ├── network/             # Graph construction
│   ├── ml/                  # Demand prediction model
│   ├── optimization/        # Route optimization
│   └── api/                 # FastAPI service
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /predict-demand` – Predict passenger demand
- `POST /optimal-route` – Find optimal route between stops
- `GET /network-stats` – Get transit network statistics

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

### Docker

```bash
docker-compose up
```

## API Example

```bash
curl -X POST http://localhost:8000/predict-demand \
  -H "Content-Type: application/json" \
  -d '{"hour": 8, "day_of_week": 1, "month": 3, "is_weekend": 0,
       "demand_lag_1": 120.0, "demand_lag_2": 115.0,
       "demand_lag_3": 110.0, "demand_lag_24": 125.0}'
```

## Quick Start (Python)

```python
from src.ml.demand_predictor import generate_synthetic_demand, engineer_features, train_demand_model

df = generate_synthetic_demand(n_zones=10, n_hours=168)
df_feat = engineer_features(df)
feat_cols = ["hour", "day_of_week", "month", "is_weekend",
             "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_24"]
model, metrics = train_demand_model(df_feat, feat_cols)
print(f"Model R²: {metrics['r2']:.4f}")
```
